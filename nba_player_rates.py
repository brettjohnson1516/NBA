"""
nba_player_rates.py
-------------------
Builds per-player, STRICTLY-PRIOR-TO-GAME quality estimates from ESPN play-by-play.

For every (game_id, athlete_id) pair where the player appears on the floor, writes
that player's rates using ONLY games that finished before that game's date.

Two layers:
  1. Box rates    - per-100-possession production, shooting, fouls, usage.
  2. RAPM         - ridge-regressed adjusted plus-minus from stint data
                    (offensive and defensive coefficients estimated separately).

Output: <data-root>\\features\\nba_player_rates_<season>.parquet

Usage (PowerShell):
  python nba_player_rates.py --seasons 2021 2022 2023 2024 2025 2026
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsqr

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

DEFAULT_DATA_ROOT = r"C:\Users\saint\OneDrive\Documents\NBA_AUG_2026\data"

# The 30 NBA franchise codes as they appear in the pbp files. Anything else
# (STARS / STRIPES etc.) is an exhibition and is dropped -- note that All-Star
# games are tagged season_type=2, so the season_type filter alone is not enough.
NBA_TEAMS = {
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY",
    "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH",
    # NJ is the New Jersey Nets, who existed through 2011-12 before becoming
    # BKN. Leaving it out silently deleted every Nets game from the 2012 file.
    "NJ", "NOH", "NOK", "CHO", "SEA",
}

REGULAR_SEASON = 2

# league-average points per 100 possessions is solved from the data, but we
# keep a floor on how much prior-possession volume a player needs before his
# own numbers are trusted at all.
BOX_PRIOR_K = 400.0      # pseudo-possessions of league-average regression
RAPM_LAMBDA = 2000.0     # ridge penalty (in possession-weighted units)
SEASON_DECAY = 0.75      # weight on stints from N seasons back = decay ** N


# --------------------------------------------------------------------------
# play parsing
# --------------------------------------------------------------------------

def tag_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean/int event columns derived from type_text and text."""
    tt = df["type_text"].fillna("")
    tx = df["text"].fillna("")
    tx_l = tx.str.lower()

    is_ft = tt.str.contains("Free Throw", case=False, na=False)
    made = tx_l.str.contains("makes", na=False)
    missed = tx_l.str.contains("misses", na=False)

    # A blocked shot is its own row reading "X blocks Y's layup" - no "makes"
    # and no "misses" - so it was previously dropped from fga entirely. It IS
    # an attempt, and athlete_id_1 on these rows is the SHOOTER (verified at
    # 99.7-99.9% across seasons by checking which lineup he appears in), so the
    # attempt lands on the right player. athlete_id_2 is the blocker.
    is_blk_row = (~is_ft) & (~made) & (~missed) & tx_l.str.contains("block", na=False)

    df["is_ft"] = is_ft
    df["is_ft_made"] = is_ft & made
    df["is_ft_att"] = is_ft & (made | missed)

    is_fga = (~is_ft) & (made | missed | is_blk_row)
    df["is_fga"] = is_fga
    df["is_fgm"] = is_fga & made

    # score_value is only trusted as a three-point tiebreaker where the column
    # actually holds a per-play point value; in several older files it does not.
    sv = pd.to_numeric(df["score_value"], errors="coerce")
    is_three = tx_l.str.contains("three point", na=False)
    if bool(sv.notna().any() and sv.max(skipna=True) <= 4):
        is_three = is_three | (sv == 3)
    df["is_fg3a"] = is_fga & is_three
    df["is_fg3m"] = is_fga & is_three & made

    # Two-pointers are every field-goal attempt that is not a three. Blocked
    # shots are attempts (is_blk_row above) and are never makes, so they land
    # in fg2a and not fg2m, which is correct.
    df["is_fg2a"] = is_fga & ~is_three
    df["is_fg2m"] = df["is_fgm"] & ~is_three

    # Points are rebuilt from the made shots rather than read off score_value,
    # which is not reliably populated in the older ESPN feeds.
    df["pts"] = (
        3.0 * (df["is_fg3m"]).astype(float)
        + 2.0 * (df["is_fgm"].astype(float) - df["is_fg3m"].astype(float))
        + 1.0 * df["is_ft_made"].astype(float)
    )

    df["is_tov"] = tt.str.contains("Turnover", case=False, na=False)
    df["is_foul"] = tt.str.contains("Foul", case=False, na=False) & (
        ~tt.str.contains("Turnover", case=False, na=False)
    )

    reb = tt.str.contains("Rebound", case=False, na=False)
    df["is_oreb"] = reb & tt.str.contains("Offensive", case=False, na=False)
    df["is_dreb"] = reb & tt.str.contains("Defensive", case=False, na=False)
    # team rebounds have no athlete attached
    df["is_team_reb"] = reb & df["athlete_id_1"].isna()

    df["is_ast"] = tx_l.str.contains("assists", na=False)
    df["is_sub"] = tt.str.fullmatch("Substitution", case=False, na=False)

    return df


def possessions_from_flags(sub: pd.DataFrame) -> float:
    """Standard possession estimate for one team's plays."""
    fga = sub["is_fga"].sum()
    fta = sub["is_ft_att"].sum()
    oreb = (sub["is_oreb"] & sub["athlete_id_1"].notna()).sum()
    tov = sub["is_tov"].sum()
    return float(fga - oreb + tov + 0.44 * fta)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def resolve_pbp_dir(root: Path, override: str | None) -> Path:
    """
    Find the directory holding pbp_<season>.parquet. Tries an explicit
    override first, then any subdirectory of the data root whose name
    contains 'pbp' and which actually holds matching files.
    """
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = root / p
        if not p.is_dir():
            raise FileNotFoundError(f"--pbp-dir not a directory: {p}")
        return p

    candidates = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.is_dir() and "pbp" in d.name.lower():
            if list(d.glob("pbp_*.parquet")):
                candidates.append(d)
    if not candidates:
        raise FileNotFoundError(
            f"no directory under {root} contains pbp_<season>.parquet files; "
            f"pass --pbp-dir explicitly"
        )
    if len(candidates) > 1:
        print(f"  multiple pbp directories found {[c.name for c in candidates]}, "
              f"using {candidates[0].name}")
    return candidates[0]


def load_pbp(pbp_dir: Path, season: int) -> pd.DataFrame:
    path = pbp_dir / f"pbp_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing pbp file: {path}")
    df = pd.read_parquet(path)

    n0 = df["game_id"].nunique()
    df = df[df["season_type"] == REGULAR_SEASON]
    df = df[df["home_team"].isin(NBA_TEAMS) & df["away_team"].isin(NBA_TEAMS)]
    df = df[df["lineup_ok"]]
    n1 = df["game_id"].nunique()
    print(f"  {season}: {n1} regular-season games kept of {n0} in file "
          f"({len(df):,} plays)")

    df = df.sort_values(["game_id", "play_number"]).reset_index(drop=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return tag_plays(df)


# --------------------------------------------------------------------------
# stint construction
# --------------------------------------------------------------------------

def build_stints(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (game, unbroken 10-man lineup) with points and possessions
    for each side. Substitution rows carry the OLD lineup on ESPN feeds only
    at the moment of the sub, so we key the stint on the lineup columns
    directly and let the boundary fall where the strings change.
    """
    key = (
        df["game_id"].astype(str)
        + "|" + df["home_lineup"].astype(str)
        + "|" + df["away_lineup"].astype(str)
    )
    df = df.copy()
    df["stint_id"] = (key != key.shift()).cumsum()

    home_actor = df["actor_team_id"] == df["home_team_id"]
    df["pts_home"] = np.where(home_actor, df["pts"], 0.0)
    df["pts_away"] = np.where(~home_actor & df["actor_team_id"].notna(),
                              df["pts"], 0.0)

    away_actor = (~home_actor) & df["actor_team_id"].notna()
    player_oreb = df["is_oreb"] & df["athlete_id_1"].notna()

    for side, mask in (("home", home_actor), ("away", away_actor)):
        m = mask.to_numpy()
        df[f"_fga_{side}"] = df["is_fga"].to_numpy() & m
        df[f"_fta_{side}"] = df["is_ft_att"].to_numpy() & m
        df[f"_oreb_{side}"] = player_oreb.to_numpy() & m
        df[f"_tov_{side}"] = df["is_tov"].to_numpy() & m

    agg = df.groupby("stint_id", sort=False).agg(
        game_id=("game_id", "first"),
        game_date=("game_date", "first"),
        season=("season", "first"),
        home_lineup=("home_lineup", "first"),
        away_lineup=("away_lineup", "first"),
        pts_home=("pts_home", "sum"),
        pts_away=("pts_away", "sum"),
        sec_min=("seconds_elapsed", "min"),
        sec_max=("seconds_elapsed", "max"),
        fga_home=("_fga_home", "sum"), fta_home=("_fta_home", "sum"),
        oreb_home=("_oreb_home", "sum"), tov_home=("_tov_home", "sum"),
        fga_away=("_fga_away", "sum"), fta_away=("_fta_away", "sum"),
        oreb_away=("_oreb_away", "sum"), tov_away=("_tov_away", "sum"),
    ).reset_index()

    for side in ("home", "away"):
        agg[f"poss_{side}"] = (
            agg[f"fga_{side}"] - agg[f"oreb_{side}"] + agg[f"tov_{side}"]
            + 0.44 * agg[f"fta_{side}"]
        ).astype(float)

    agg["seconds"] = (agg["sec_max"] - agg["sec_min"]).clip(lower=0).astype(float)
    agg["pts_home"] = agg["pts_home"].astype(float)
    agg["pts_away"] = agg["pts_away"].astype(float)
    agg["season"] = agg["season"].astype(int)

    keep = (agg["poss_home"] > 0) | (agg["poss_away"] > 0)
    cols = ["stint_id", "game_id", "game_date", "season", "home_lineup",
            "away_lineup", "pts_home", "pts_away", "poss_home", "poss_away",
            "seconds"]
    return agg.loc[keep, cols].reset_index(drop=True)


# --------------------------------------------------------------------------
# RAPM
# --------------------------------------------------------------------------

def fit_rapm(stints: pd.DataFrame, weights: np.ndarray, lam: float):
    """
    Two observations per stint (home offense, away offense).
    y = points per 100 possessions.
    Design: +1 for each offensive player, +1 for each defensive player in a
    separate defensive block, plus a home-offense indicator and an intercept.
    Returns dicts player -> off coef, player -> def coef.
    """
    players = sorted({
        int(p)
        for col in ("home_lineup", "away_lineup")
        for s in stints[col]
        for p in str(s).split(",") if p
    })
    idx = {p: i for i, p in enumerate(players)}
    n_p = len(players)

    rows, cols, vals, ys, ws = [], [], [], [], []
    r = 0

    def add_obs(off_ids, def_ids, pts, poss, w, is_home_off):
        nonlocal r
        if poss <= 0:
            return
        for p in off_ids:
            rows.append(r); cols.append(idx[p]); vals.append(1.0)
        for p in def_ids:
            rows.append(r); cols.append(n_p + idx[p]); vals.append(1.0)
        rows.append(r); cols.append(2 * n_p); vals.append(1.0 if is_home_off else 0.0)
        rows.append(r); cols.append(2 * n_p + 1); vals.append(1.0)
        ys.append(100.0 * pts / poss)
        ws.append(w * poss)
        r += 1

    h_ids = [[int(x) for x in str(s).split(",") if x] for s in stints["home_lineup"]]
    a_ids = [[int(x) for x in str(s).split(",") if x] for s in stints["away_lineup"]]
    ph = stints["pts_home"].to_numpy()
    pa = stints["pts_away"].to_numpy()
    qh = stints["poss_home"].to_numpy()
    qa = stints["poss_away"].to_numpy()

    for i in range(len(stints)):
        add_obs(h_ids[i], a_ids[i], ph[i], qh[i], weights[i], True)
        add_obs(a_ids[i], h_ids[i], pa[i], qa[i], weights[i], False)

    n_col = 2 * n_p + 2
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(r, n_col))
    y = np.asarray(ys)
    w = np.sqrt(np.asarray(ws))

    Xw = X.multiply(w[:, None]).tocsr()
    yw = y * w

    # ridge via augmented system; intercept and home term left unpenalised
    pen = np.full(n_col, np.sqrt(lam))
    pen[2 * n_p] = 0.0
    pen[2 * n_p + 1] = 0.0
    A = sparse.vstack([Xw, sparse.diags(pen)]).tocsr()
    b = np.concatenate([yw, np.zeros(n_col)])

    sol = lsqr(A, b, atol=1e-8, btol=1e-8, iter_lim=500)[0]

    off = {p: float(sol[idx[p]]) for p in players}
    dfn = {p: float(sol[n_p + idx[p]]) for p in players}
    return off, dfn


# --------------------------------------------------------------------------
# box aggregation
# --------------------------------------------------------------------------

def player_game_box(df: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """Per (game_id, athlete) counting stats, plus on-court possessions."""
    d = df[df["athlete_id_1"].notna()].copy()
    d["athlete_id"] = d["athlete_id_1"].astype("int64")
    d["is_home_actor"] = d["actor_team_id"] == d["home_team_id"]

    agg = d.groupby(["game_id", "athlete_id"]).agg(
        game_date=("game_date", "first"),
        season=("season", "first"),
        is_home=("is_home_actor", "mean"),
        pts=("pts", "sum"),
        fga=("is_fga", "sum"),
        fgm=("is_fgm", "sum"),
        fg3a=("is_fg3a", "sum"),
        fg3m=("is_fg3m", "sum"),
        fg2a=("is_fg2a", "sum"),
        fg2m=("is_fg2m", "sum"),
        fta=("is_ft_att", "sum"),
        ftm=("is_ft_made", "sum"),
        tov=("is_tov", "sum"),
        oreb=("is_oreb", "sum"),
        dreb=("is_dreb", "sum"),
        fouls=("is_foul", "sum"),
    ).reset_index()
    agg["is_home"] = (agg["is_home"] > 0.5).astype(int)

    # on-court possessions per player, exploded from the lineup strings
    parts = []
    for side, other in (("home", "away"), ("away", "home")):
        t = stints[["game_id", f"{side}_lineup", f"poss_{side}",
                    f"poss_{other}", "seconds"]].copy()
        t.columns = ["game_id", "lineup", "poss_off", "poss_def", "seconds"]
        t["athlete_id"] = t["lineup"].str.split(",")
        t = t.explode("athlete_id")
        t["oc_is_home"] = 1 if side == "home" else 0
        parts.append(t.drop(columns="lineup"))
    oc = pd.concat(parts, ignore_index=True)
    oc = oc[oc["athlete_id"].astype(str).str.len() > 0]
    oc["athlete_id"] = oc["athlete_id"].astype("int64")
    oc = oc.groupby(["game_id", "athlete_id"], as_index=False).agg(
        poss_off=("poss_off", "sum"),
        poss_def=("poss_def", "sum"),
        seconds=("seconds", "sum"),
        oc_is_home=("oc_is_home", "max"),
    )

    out = oc.merge(agg, on=["game_id", "athlete_id"], how="left")

    # Points the feed never recorded for this player's own team in this game.
    # ESPN dropped roughly 2.4 made field goals per game from the 2016 file and
    # a smaller number from 2013. Nothing in the tagging recovers a play that
    # was never written, so the shortfall is measured and carried forward
    # instead, letting a model drop or downweight holed history.
    hact = df["actor_team_id"] == df["home_team_id"]
    gm = df.assign(
        _ph=np.where(hact, df["pts"], 0.0),
        _pa=np.where(~hact & df["actor_team_id"].notna(), df["pts"], 0.0),
    ).groupby("game_id").agg(
        built_home=("_ph", "sum"),
        built_away=("_pa", "sum"),
        home_final=("home_final", "first"),
        away_final=("away_final", "first"),
    ).reset_index()
    gm["gap_home"] = gm["home_final"] - gm["built_home"]
    gm["gap_away"] = gm["away_final"] - gm["built_away"]

    out = out.merge(gm[["game_id", "gap_home", "gap_away"]],
                    on="game_id", how="left")
    out["pts_gap"] = np.where(out["oc_is_home"] == 1,
                              out["gap_home"], out["gap_away"])
    out = out.drop(columns=["gap_home", "gap_away", "oc_is_home"])
    for c in ["pts", "fga", "fgm", "fg3a", "fg3m", "fg2a", "fg2m", "fta", "ftm",
              "tov", "oreb", "dreb", "fouls", "pts_gap"]:
        out[c] = out[c].fillna(0.0)
    gmeta = df.groupby("game_id")[["game_date", "season"]].first()
    out["game_date"] = pd.Series(
        gmeta["game_date"].reindex(out["game_id"]).to_numpy(), index=out.index)
    out["season"] = pd.Series(
        gmeta["season"].reindex(out["game_id"]).to_numpy(), index=out.index).astype(int)
    out["is_home"] = out["is_home"].fillna(0).astype(int)
    out["minutes"] = out["seconds"] / 60.0
    return out


# --------------------------------------------------------------------------
# main build
# --------------------------------------------------------------------------

def build(args):
    root = Path(args.data_root)
    pbp_dir = resolve_pbp_dir(root, args.pbp_dir)
    print(f"pbp dir:   {pbp_dir}")
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_seasons = sorted(set(args.seasons))
    load_from = min(all_seasons) - args.history_seasons

    pbp_cache, box_cache, stint_cache = {}, {}, {}
    for s in range(load_from, max(all_seasons) + 1):
        path = pbp_dir / f"pbp_{s}.parquet"
        if not path.exists():
            if s in all_seasons:
                raise FileNotFoundError(f"missing pbp file: {path}")
            print(f"  {s}: no pbp file, skipping history season")
            continue
        df = load_pbp(pbp_dir, s)
        pbp_cache[s] = df
        st = build_stints(df)
        stint_cache[s] = st
        box_cache[s] = player_game_box(df, st)
        print(f"      {len(st):,} stints, {len(box_cache[s]):,} player-games")

    box_all = pd.concat(box_cache.values(), ignore_index=True)
    stint_all = pd.concat(stint_cache.values(), ignore_index=True)
    box_all = box_all.sort_values("game_date").reset_index(drop=True)
    stint_all = stint_all.sort_values("game_date").reset_index(drop=True)

    for season in all_seasons:
        print(f"\nbuilding rates for {season}")
        target = box_cache[season][["game_id", "game_date", "athlete_id",
                                    "is_home"]].drop_duplicates()
        target = target.sort_values("game_date").reset_index(drop=True)

        game_dates = sorted(target["game_date"].unique())
        refit_dates = game_dates[:: max(args.refit_every, 1)]
        if game_dates and refit_dates[-1] != game_dates[-1]:
            refit_dates.append(game_dates[-1])

        rapm_by_date = {}
        for d in refit_dates:
            hist = stint_all[stint_all["game_date"] < d]
            if len(hist) < args.min_stints:
                rapm_by_date[d] = ({}, {})
                continue
            newest = hist["season"].max()
            w = np.power(args.season_decay, newest - hist["season"].to_numpy())
            off, dfn = fit_rapm(hist, w, args.ridge_lambda)
            rapm_by_date[d] = (off, dfn)
            print(f"  refit {pd.Timestamp(d).date()}  stints={len(hist):,}  "
                  f"players={len(off)}")

        refit_arr = np.array(refit_dates, dtype="datetime64[ns]")

        # ---- strictly-prior cumulative totals per player -----------------
        cnt = ["pts", "fga", "fgm", "fg3a", "fg3m", "fg2a", "fg2m", "fta", "ftm",
               "tov", "oreb", "dreb", "fouls", "poss_off", "minutes", "pts_gap"]
        hist = box_all[box_all["game_date"] <= target["game_date"].max()].copy()
        hist = hist.sort_values(["athlete_id", "game_date", "game_id"])
        g = hist.groupby("athlete_id", sort=False)
        for c in cnt:
            hist["prior_" + c] = g[c].cumsum() - hist[c]
        hist["prior_gp"] = g.cumcount()

        # ---- strictly-prior league totals by date ------------------------
        day = box_all.groupby("game_date")[["pts", "poss_off", "fg3m", "fg3a",
                                            "fg2m", "fg2a",
                                            "ftm", "fta"]].sum().sort_index()
        prior_day = day.cumsum() - day
        lg_rate_s = 100.0 * prior_day["pts"] / prior_day["poss_off"].clip(lower=1)
        lg_fg3_s = prior_day["fg3m"] / prior_day["fg3a"].clip(lower=1)
        lg_fg2_s = prior_day["fg2m"] / prior_day["fg2a"].clip(lower=1)
        lg_ft_s = prior_day["ftm"] / prior_day["fta"].clip(lower=1)
        lg_rate_s = lg_rate_s.replace(0.0, np.nan).fillna(110.0)
        lg_fg3_s = lg_fg3_s.replace(0.0, np.nan).fillna(0.36)
        lg_fg2_s = lg_fg2_s.replace(0.0, np.nan).fillna(0.52)
        lg_ft_s = lg_ft_s.replace(0.0, np.nan).fillna(0.78)

        out = target.merge(
            hist[["game_id", "athlete_id", "prior_gp"]
                 + ["prior_" + c for c in cnt]],
            on=["game_id", "athlete_id"], how="left")

        out["season"] = season
        lg_rate = out["game_date"].map(lg_rate_s)
        lg_fg3 = out["game_date"].map(lg_fg3_s)
        lg_fg2 = out["game_date"].map(lg_fg2_s)
        lg_ft = out["game_date"].map(lg_ft_s)

        poss = out["prior_poss_off"].fillna(0.0)
        k = BOX_PRIOR_K
        out["prior_min"] = out["prior_minutes"].fillna(0.0)
        out["prior_poss"] = poss
        out["prior_gp"] = out["prior_gp"].fillna(0).astype(int)
        out["pts_per100"] = (100.0 * out["prior_pts"] + k * lg_rate * 0.2) / (poss + k)
        out["usage_per100"] = 100.0 * (out["prior_fga"] + 0.44 * out["prior_fta"]
                                       + out["prior_tov"]) / poss.clip(lower=1)
        out["ts_pct"] = out["prior_pts"] / (2.0 * (out["prior_fga"]
                                                   + 0.44 * out["prior_fta"])).clip(lower=1)
        out["fg3a_rate"] = out["prior_fg3a"] / out["prior_fga"].clip(lower=1)
        out["fg3_pct"] = ((out["prior_fg3m"] + k * 0.05 * lg_fg3)
                          / (out["prior_fg3a"] + k * 0.05))
        out["fg2_pct"] = ((out["prior_fg2m"] + k * 0.05 * lg_fg2)
                          / (out["prior_fg2a"] + k * 0.05))
        out["fg2a_rate"] = out["prior_fg2a"] / out["prior_fga"].clip(lower=1)
        out["ft_pct"] = ((out["prior_ftm"] + k * 0.05 * lg_ft)
                         / (out["prior_fta"] + k * 0.05))
        out["ft_rate"] = out["prior_fta"] / out["prior_fga"].clip(lower=1)
        out["tov_per100"] = 100.0 * out["prior_tov"] / poss.clip(lower=1)
        out["foul_per100"] = 100.0 * out["prior_fouls"] / poss.clip(lower=1)
        out["reb_per100"] = 100.0 * (out["prior_oreb"] + out["prior_dreb"]) / poss.clip(lower=1)

        # players below the prior-volume floor get NaN rather than a rate
        # computed off a handful of possessions (ts_pct of 1.5, usage of 100)
        no_hist = poss < args.min_prior_poss
        for c in ["pts_per100", "usage_per100", "ts_pct", "fg3a_rate",
                  "fg2a_rate", "ft_rate", "tov_per100", "foul_per100",
                  "reb_per100"]:
            out.loc[no_hist, c] = np.nan
        out.loc[no_hist, "fg3_pct"] = lg_fg3[no_hist]
        out.loc[no_hist, "fg2_pct"] = lg_fg2[no_hist]
        out.loc[no_hist, "ft_pct"] = lg_ft[no_hist]

        # ---- RAPM as of the last refit at or before each game date -------
        pos = np.searchsorted(refit_arr,
                              out["game_date"].to_numpy(dtype="datetime64[ns]"),
                              side="right") - 1
        pos = np.clip(pos, 0, len(refit_dates) - 1)
        off_map = [rapm_by_date[refit_dates[i]][0] for i in range(len(refit_dates))]
        def_map = [rapm_by_date[refit_dates[i]][1] for i in range(len(refit_dates))]
        pids = out["athlete_id"].to_numpy()
        out["rapm_off"] = [off_map[p].get(int(a), np.nan) for p, a in zip(pos, pids)]
        out["rapm_def"] = [def_map[p].get(int(a), np.nan) for p, a in zip(pos, pids)]
        out["rapm_net"] = out["rapm_off"] - out["rapm_def"]

        # Data quality: points the feed never recorded across this player's
        # prior games. pts_missing_pct is the share of his team's true prior
        # scoring that is absent, so pts_per100 and ts_pct on this row are
        # understated by roughly that much. 2016 rows run near 2.6%, 2013 near
        # 0.4%, everything else near zero.
        out["prior_pts_missing"] = out["prior_pts_gap"].fillna(0.0)
        true_prior = (out["prior_pts"] + out["prior_pts_gap"]).replace(0.0, np.nan)
        out["pts_missing_pct"] = 100.0 * out["prior_pts_gap"] / true_prior

        out = out.drop(columns=[c for c in out.columns
                                if c.startswith("prior_")
                                and c not in ("prior_gp", "prior_min",
                                              "prior_poss", "prior_pts_missing")])
        path = out_dir / f"nba_player_rates_{season}.parquet"
        out.to_parquet(path, index=False)
        print(f"  wrote {path}  rows={len(out):,}  "
              f"players={out['athlete_id'].nunique()}  "
              f"rapm_cov={out['rapm_net'].notna().mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",
                    default=os.environ.get("NBA_DATA_ROOT", DEFAULT_DATA_ROOT))
    ap.add_argument("--pbp-dir", default=os.environ.get("NBA_PBP_DIR"),
                    help="directory holding pbp_<season>.parquet; auto-discovered if omitted")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=[2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--history-seasons", type=int, default=2,
                    help="extra earlier seasons to load for prior-only rates")
    ap.add_argument("--refit-every", type=int, default=7,
                    help="refit RAPM every N distinct game dates")
    ap.add_argument("--ridge-lambda", type=float, default=RAPM_LAMBDA)
    ap.add_argument("--season-decay", type=float, default=SEASON_DECAY)
    ap.add_argument("--min-stints", type=int, default=5000)
    ap.add_argument("--min-prior-poss", type=float, default=50.0,
                    help="below this many prior possessions, box rates are NaN")
    args = ap.parse_args()

    print(f"data root: {args.data_root}")
    build(args)


if __name__ == "__main__":
    main()
