"""
nba_wp_features.py

Builds the per-event feature table for the NBA live moneyline win-probability model.

One row per play-by-play event = one scoreable game state.
Target = home_won (final result, OT included).

Usage (PowerShell):
  python nba_wp_features.py --seasons 2021 2022 2023 2024 2025 2026

Paths are resolved from the NBA_ROOT environment variable if set, otherwise from
--root. Nothing is hardcoded beyond the directory layout you gave me.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

import nba_sim as SIM

# ----------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------

DEFAULT_ROOT = os.environ.get(
    "NBA_ROOT", r"C:\Users\saint\OneDrive\Documents\NBA_AUG_2026"
)

# ESPN labels All-Star / Rising Stars games as season_type 2, so the season_type
# filter alone does not remove them. Games are additionally required to be
# between two of the 30 franchises.
NBA_TEAMS = {
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY",
    "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH",
}

REG_SECONDS = 2880.0   # 4 x 12:00
OT_SECONDS = 300.0     # 5:00


def resolve_paths(root):
    data = os.path.join(root, "data")
    return {
        "root": root,
        "data": data,
        "pbp": os.path.join(data, "pbp_espn"),
        "odds": os.path.join(data, "odds"),
        "features": os.path.join(data, "features"),
        "kalshi": os.path.join(data, "kalshi"),
        "models": os.path.join(root, "models"),
    }


def find_one(directory, patterns, what):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(directory, pat)))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not find {what} in {directory} (tried: {', '.join(patterns)})"
    )


# ----------------------------------------------------------------------------
# loaders
# ----------------------------------------------------------------------------

def load_pbp(paths, season):
    f = find_one(
        paths["pbp"],
        [f"pbp_{season}.parquet", f"*{season}*.parquet"],
        f"pbp for season {season}",
    )
    df = pd.read_parquet(f)
    print(f"  pbp {season}: {len(df):,} rows / {df['game_id'].nunique():,} games  ({os.path.basename(f)})")
    return df


def load_odds(paths):
    ml = pd.read_parquet(find_one(paths["odds"], ["closing_lines_pinnacle.parquet"], "closing moneylines"))
    sp = pd.read_parquet(find_one(paths["odds"], ["closing_spreads_pinnacle.parquet"], "closing spreads"))
    to = pd.read_parquet(find_one(paths["odds"], ["closing_totals_pinnacle.parquet"], "closing totals"))

    ml = ml[["espn_game_id", "p_home_close", "home_close_ml", "away_close_ml"]].copy()
    sp = sp[["espn_game_id", "home_spread_close"]].copy()
    to = to[["espn_game_id", "total_close"]].copy()

    odds = ml.merge(sp, on="espn_game_id", how="outer").merge(to, on="espn_game_id", how="outer")
    odds["espn_game_id"] = odds["espn_game_id"].astype(str)
    odds = odds.drop_duplicates(subset=["espn_game_id"], keep="first")
    print(f"  odds: {len(odds):,} games with at least one closing market")
    return odds


def load_player_rates(paths, season):
    """Per (game_id, athlete_id) season-to-date rates BEFORE that game."""
    try:
        f = find_one(
            paths["features"],
            [f"nba_player_rates_{season}.parquet", f"*player_rates*{season}*.parquet"],
            f"player rates for {season}",
        )
    except FileNotFoundError:
        print(f"  WARNING: no player rates file for {season} - shot-quality baselines "
              f"will fall back to the league prior for every shooter")
        return None
    r = pd.read_parquet(f)
    keep = ["game_id", "athlete_id", "fg3_pct", "ft_pct", "prior_min", "prior_poss",
            "fg3a_rate", "rapm_net", "rapm_off", "rapm_def", "prior_gp"]
    keep = [c for c in keep if c in r.columns]
    r = r[keep].copy()
    r["game_id"] = r["game_id"].astype(str)
    r["athlete_id"] = pd.to_numeric(r["athlete_id"], errors="coerce")
    r = r.drop_duplicates(subset=["game_id", "athlete_id"], keep="first")
    return r


# ----------------------------------------------------------------------------
# event parsing
# ----------------------------------------------------------------------------

RE_MAKES = re.compile(r"\bmakes\b", re.I)
RE_MISSES = re.compile(r"\bmisses\b", re.I)
RE_THREE = re.compile(r"three point", re.I)
RE_FT_LAST = re.compile(r"free throw (\d+) of (\d+)", re.I)
RE_DIST = re.compile(r"(\d+)-foot", re.I)

# Shot families, coarse enough to hold sample in every distance bucket.
SHOT_FAMILIES = [
    ("dunk", re.compile(r"dunk", re.I)),
    ("layup", re.compile(r"layup|finger roll|tip shot|putback", re.I)),
    ("hook", re.compile(r"hook", re.I)),
    ("fade", re.compile(r"fade away|turnaround|step back", re.I)),
    ("float", re.compile(r"floating|driving", re.I)),
    ("pullup", re.compile(r"pullup|running", re.I)),
    ("jump", re.compile(r"jump|shot", re.I)),
]
DIST_BINS = [0, 4, 7, 10, 14, 18, 22, 24, 27, 30, 100]

TURNOVER_PAT = re.compile(r"turnover", re.I)
STEAL_PAT = re.compile(r"steal", re.I)
DEF_REB_PAT = re.compile(r"defensive rebound", re.I)
OFF_REB_PAT = re.compile(r"offensive rebound", re.I)
FOUL_PAT = re.compile(r"foul", re.I)


def parse_events(df):
    """Adds shot / free-throw / foul flags parsed from type_text and text."""
    tt = df["type_text"].fillna("")
    tx = df["text"].fillna("")

    is_ft = tt.str.startswith("Free Throw")
    makes = tx.str.contains(RE_MAKES)
    misses = tx.str.contains(RE_MISSES)

    df["is_fga"] = (~is_ft) & (makes | misses)
    df["is_fg3a"] = df["is_fga"] & tx.str.contains(RE_THREE)
    df["is_fg3m"] = df["is_fg3a"] & makes
    df["is_fg2a"] = df["is_fga"] & ~df["is_fg3a"]
    df["is_fg2m"] = df["is_fg2a"] & makes

    df["is_fta"] = is_ft & (makes | misses)
    df["is_ftm"] = df["is_fta"] & makes

    # last free throw of a trip (possession changes on a make)
    ft_idx = tx.str.extract(RE_FT_LAST)
    df["_ft_n"] = pd.to_numeric(ft_idx[0], errors="coerce")
    df["_ft_of"] = pd.to_numeric(ft_idx[1], errors="coerce")
    df["is_ft_last"] = df["is_fta"] & (df["_ft_n"] == df["_ft_of"])

    # shot quality: distance and family, so a contested 26-footer and a dunk are
    # not treated as the same event when deciding what regresses
    dist = pd.to_numeric(tx.str.extract(RE_DIST)[0], errors="coerce")
    df["shot_dist"] = dist
    fam = pd.Series("other", index=df.index, dtype=object)
    assigned = pd.Series(False, index=df.index)
    for name, pat in SHOT_FAMILIES:
        hit = (~assigned) & (tt.str.contains(pat) | tx.str.contains(pat))
        fam = fam.mask(hit, name)
        assigned = assigned | hit
    df["shot_family"] = fam
    df["dist_bin"] = np.clip(np.digitize(dist.fillna(-1).values, DIST_BINS) - 1,
                             -1, len(DIST_BINS) - 2)

    df["is_turnover"] = tt.str.contains(TURNOVER_PAT) | tx.str.contains(TURNOVER_PAT)
    df["is_steal"] = tx.str.contains(STEAL_PAT)
    df["is_def_reb"] = tt.str.contains(DEF_REB_PAT)
    df["is_off_reb"] = tt.str.contains(OFF_REB_PAT)
    df["is_foul"] = tt.str.contains(FOUL_PAT) & ~df["is_turnover"]

    return df


def derive_possession(df):
    """
    Causal possession inference from the event that just happened.

     1 = home has the ball, -1 = away, 0 = unknown / live ball.

    Rules (applied to the actor team of the event):
      defensive rebound     -> rebounder's team
      offensive rebound     -> rebounder's team
      steal                 -> stealing team (actor on a steal row is the stealer)
      turnover              -> opponent of the actor
      made FG               -> opponent of the shooter
      made last free throw  -> opponent of the shooter
      missed shot / missed last FT -> unknown until the rebound
      everything else       -> carry the previous known value forward
    """
    home_id = df["home_team_id"].values
    actor = df["actor_team_id"].values
    actor_is_home = np.where(np.isnan(actor), np.nan, (actor == home_id).astype(float))
    actor_side = np.where(np.isnan(actor_is_home), np.nan, np.where(actor_is_home == 1, 1.0, -1.0))

    poss = np.full(len(df), np.nan)

    made_fg = df["is_fga"].values & (df["is_fg3m"].values | df["is_fg2m"].values)
    made_ft_last = df["is_ft_last"].values & df["is_ftm"].values
    missed_shot = (df["is_fga"].values & ~made_fg) | (df["is_ft_last"].values & ~df["is_ftm"].values)

    keep = df["is_def_reb"].values | df["is_off_reb"].values | df["is_steal"].values
    flip = df["is_turnover"].values | made_fg | made_ft_last

    poss = np.where(keep, actor_side, poss)
    poss = np.where(flip, -actor_side, poss)
    poss = np.where(missed_shot, 0.0, poss)   # live ball, rebound pending

    s = pd.Series(poss, index=df.index)
    # 0 means "genuinely unknown right now"; NaN means "no information from this
    # event" and should inherit the last known state within the game.
    s = s.replace(0.0, np.nan).groupby(df["game_id"]).ffill()
    known = s.notna()
    df["possession"] = s.fillna(0.0).values
    df["poss_known"] = known.astype(int).values
    return df




# ----------------------------------------------------------------------------
# lineup / RAPM features
# ----------------------------------------------------------------------------

def _parse_lineup(sr):
    """'a,b,c' -> list of float ids. Returns a Series of lists."""
    return sr.fillna("").map(
        lambda x: [float(v) for v in x.split(",") if v.strip() != ""] if x else []
    )


def build_lineup_features(df, rates, args):
    """
    Three distinct quantities, because the on-court five is NOT who plays the
    rest of the game:

      oncourt_rapm_diff  - the ten players actually on the floor right now
      rest_rapm_diff     - the whole active roster, weighted by each player's
                           prior minutes per game, i.e. who is expected to play
                           the remaining minutes
      oncourt_edge       - oncourt minus rest, i.e. how much better or worse the
                           current five are than that team's expected mix

    Only rest_rapm_diff is scaled by time remaining; the on-court edge applies
    to the next stretch of play, not the rest of the game.
    """
    if rates is None:
        for c in LINEUP_FEATURES:
            df[c] = np.nan
        print("    WARNING: no player rates - lineup features are all NaN")
        return df

    r = rates[["game_id", "athlete_id", "rapm_net", "prior_min", "prior_gp"]].copy()
    r["rapm_net"] = r["rapm_net"].fillna(0.0)          # unknown = league average
    mpg = np.where(r["prior_gp"].fillna(0) > 0,
                   r["prior_min"].fillna(0) / r["prior_gp"].replace(0, np.nan), 0.0)
    r["mpg"] = np.nan_to_num(mpg)
    rapm_by = {(g, a): v for g, a, v in zip(r["game_id"], r["athlete_id"], r["rapm_net"])}
    mpg_by = {(g, a): v for g, a, v in zip(r["game_id"], r["athlete_id"], r["mpg"])}

    out = {}
    for side in ("home", "away"):
        col = f"{side}_lineup"
        # one calculation per distinct (game, lineup) instead of per row
        uniq = df[["game_id", col]].drop_duplicates()
        ids = _parse_lineup(uniq[col])
        vals, miss = [], []
        for g, lst in zip(uniq["game_id"], ids):
            v = [rapm_by.get((g, a)) for a in lst]
            known = [x for x in v if x is not None]
            vals.append(float(np.sum(known)) if known else np.nan)
            miss.append(len(lst) - len(known))
        uniq[f"_{side}_oncourt_rapm"] = vals
        uniq[f"_{side}_oncourt_missing"] = miss
        df = df.merge(uniq, on=["game_id", col], how="left")

        # active roster = every player who appears on the floor at some point in
        # the game, which is what the inactive list tells a bettor pregame
        roster = {}
        for g, lst in zip(uniq["game_id"], ids):
            roster.setdefault(g, set()).update(lst)
        rest = {}
        for g, players in roster.items():
            w = np.array([mpg_by.get((g, a), 0.0) for a in players], dtype=float)
            v = np.array([rapm_by.get((g, a), 0.0) for a in players], dtype=float)
            if w.sum() <= 0:
                rest[g] = np.nan
                continue
            # shares sum to 5, matching the five slots the on-court number covers
            rest[g] = float(np.dot(v, w / w.sum()) * 5.0)
        out[side] = rest

    df["home_rest_rapm"] = df["game_id"].map(out["home"])
    df["away_rest_rapm"] = df["game_id"].map(out["away"])

    df["oncourt_rapm_diff"] = df["_home_oncourt_rapm"] - df["_away_oncourt_rapm"]
    df["rest_rapm_diff"] = df["home_rest_rapm"] - df["away_rest_rapm"]
    df["oncourt_edge"] = df["oncourt_rapm_diff"] - df["rest_rapm_diff"]
    df["home_oncourt_edge"] = df["_home_oncourt_rapm"] - df["home_rest_rapm"]
    df["away_oncourt_edge"] = df["_away_oncourt_rapm"] - df["away_rest_rapm"]
    df["rest_rapm_diff_left"] = df["rest_rapm_diff"] * df["frac_left"]

    # Roster strength is ALREADY inside the closing spread, so the raw RAPM
    # differential is collinear with it and picks up the wrong sign when both
    # are fed to the edge model. What is actually informative is the part the
    # market has NOT priced: regress the roster differential on the pregame
    # expected margin (one row per game, not per event) and keep the residual.
    per_game = df.drop_duplicates("game_id")[
        ["game_id", "rest_rapm_diff", "home_spread_close"]].copy()
    per_game["exp_margin_full"] = -per_game["home_spread_close"]
    ok = per_game["rest_rapm_diff"].notna() & per_game["exp_margin_full"].notna()
    if ok.sum() >= 50:
        x = per_game.loc[ok, "exp_margin_full"].values
        y = per_game.loc[ok, "rest_rapm_diff"].values
        A = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = per_game["rest_rapm_diff"].values - (
            coef[0] + coef[1] * per_game["exp_margin_full"].values)
        corr = float(np.corrcoef(x, y)[0, 1])
        print(f"    roster vs market: corr(rest_rapm_diff, exp_margin) {corr:+.3f}; "
              f"residual sd {np.nanstd(resid):.3f} of {np.nanstd(y):.3f}")
        per_game["rest_rapm_resid"] = resid
    else:
        per_game["rest_rapm_resid"] = np.nan
        print("    WARNING: too few games to orthogonalise rest_rapm_diff")
    df = df.merge(per_game[["game_id", "rest_rapm_resid"]], on="game_id", how="left")
    df["rest_rapm_resid_left"] = df["rest_rapm_resid"] * df["frac_left"]
    df["oncourt_rapm_missing"] = (df["_home_oncourt_missing"].fillna(0)
                                  + df["_away_oncourt_missing"].fillna(0))
    return df


LINEUP_FEATURES = [
    # rest_rapm_resid / rest_rapm_resid_left are still COMPUTED (the printed
    # roster-vs-market correlation is useful) but are NOT model inputs: they are
    # built from the same collinear RAPM family as rest_rapm_diff and cost over
    # 1.7 ROI points when the trees had access to them.
    "oncourt_rapm_diff",
    "rest_rapm_diff",
    "rest_rapm_diff_left",
    "oncourt_edge",
    "home_oncourt_edge",
    "away_oncourt_edge",
]


def add_sim_features(df, paths, args):
    """
    Possession-level rest-of-game win probability from nba_sim.py.

    n_poss  - total possessions left, from the measured possession length for
              this exact game state (late-game urgency and fouling are in it)
    adv     - the decayed pregame expected margin converted to a per-possession
              efficiency edge, so the structural model carries the team-quality
              term instead of the trees having to
    """
    tf = os.path.join(paths["features"], "sim_tables.npz")
    pf = os.path.join(paths["features"], "sim_params.json")
    if not (os.path.exists(tf) and os.path.exists(pf)):
        raise SystemExit(
            f"missing {os.path.basename(tf)} / {os.path.basename(pf)} - "
            f"run: python nba_sim.py --rebuild"
        )
    z = np.load(tf)
    W = z["W"]
    params = json.load(open(pf))
    grid = np.asarray(params["sec_grid"])
    tl_bins = np.asarray(params["tl_bins"], dtype=float)
    mg_bins = np.asarray(params["mg_bins"], dtype=float)

    off_margin = np.where(df["possession"].values > 0,
                          df["score_margin"].values, -df["score_margin"].values)
    ti = np.clip(np.digitize(df["seconds_left_horizon"].values, tl_bins) - 1,
                 0, grid.shape[0] - 1)
    mi = np.clip(np.digitize(off_margin, mg_bins) - 1, 0, grid.shape[1] - 1)
    sec_per_poss = grid[ti, mi]

    # Split the closing total between PACE and EFFICIENCY using the measured
    # relationship (possessions regressed on the total), instead of loading all
    # of it onto possession count. Whatever the possessions do not explain is
    # points per possession, which is what sets the variance of the rest of the
    # game and therefore how safe a lead is.
    league_ppp = float(np.dot(np.arange(len(params["ppd"])), params["ppd"]))
    pf = params.get("pace_fit")
    tot = df["total_close"].values.astype(float)
    if pf:
        exp_game_poss = pf[0] + pf[1] * tot
    else:
        exp_game_poss = tot / max(league_ppp, 0.5)
    exp_game_poss = np.where(np.isfinite(exp_game_poss) & (exp_game_poss > 40),
                             exp_game_poss, REG_SECONDS / max(params["overall_sec"], 1.0))
    league_game_poss = REG_SECONDS / max(params["overall_sec"], 1.0)
    pace_scale = np.clip(league_game_poss / exp_game_poss, 0.75, 1.35)
    sec_per_poss = sec_per_poss * pace_scale

    n_poss = np.clip(df["seconds_left_horizon"].values / np.maximum(sec_per_poss, 1.0),
                     0, SIM.MAX_POSS)

    half = np.maximum(n_poss / 2.0, 1.0)

    # The per-possession edge comes from the fitted rest-of-game edge model when
    # one exists, so the simulation is told what the GAME has revealed and not
    # just what the pregame line said. Falls back to the decayed spread alone.
    ef = os.path.join(paths["features"], "edge_model.json")
    if os.path.exists(ef):
        em = json.load(open(ef))
        adv = np.full(len(df), float(em["intercept"]))
        missing = []
        for name, c in zip(em["predictors"], em["coefs"]):
            if name == "exp_margin_full":
                v = -df["home_spread_close"].values
            elif name in df.columns:
                v = df[name].values
            else:
                missing.append(name)
                continue
            adv = adv + c * np.nan_to_num(v.astype(float), nan=0.0)
        if missing:
            raise SystemExit(f"edge_model.json references columns not in the feature "
                             f"build: {missing}")
        print(f"    edge model: {os.path.basename(ef)} "
              f"(fitted on {em['seasons']}, R2 {em['r2']:.5f})")
    else:
        print("    no edge_model.json - sim edge falls back to the decayed spread "
              "alone; run nba_edge_model.py to fit it")
        adv = df["exp_margin_left"].values / half

    adv = np.clip(np.nan_to_num(adv, nan=0.0), SIM.ADV_GRID[0], SIM.ADV_GRID[-1])
    home_ball = (df["possession"].values > 0).astype(int)

    df["sim_n_poss"] = n_poss
    df["sim_adv"] = adv
    df["sim_wp"] = SIM.win_prob(W, adv, n_poss, df["score_margin"].values, home_ball)
    p = np.clip(df["sim_wp"].values, 1e-6, 1 - 1e-6)
    df["sim_logit"] = np.log(p / (1 - p))
    # the same state with the ball on the other side, i.e. what possession is worth
    df["sim_ball_value"] = df["sim_wp"].values - SIM.win_prob(
        W, adv, n_poss, df["score_margin"].values, 1 - home_ball)
    # how far the current score is from what the pregame line projected
    df["sim_wp_neutral"] = SIM.win_prob(
        W, np.zeros_like(adv), n_poss, df["score_margin"].values, home_ball)
    df["sim_pregame_lift"] = df["sim_wp"].values - df["sim_wp_neutral"].values
    return df


SIM_FEATURES = [
    "sim_wp", "sim_logit", "sim_n_poss", "sim_adv",
    "sim_ball_value", "sim_pregame_lift",
]


# ----------------------------------------------------------------------------
# league priors (computed from --prior-seasons only, never from backtest data)
# ----------------------------------------------------------------------------

def build_league_priors(paths, prior_seasons, season_type_keep):
    out = {}
    fg3a = fg3m = fta = ftm = fouls = fg2a = fg2m = 0
    shot_frames = []
    for s in prior_seasons:
        try:
            df = load_pbp(paths, s)
        except FileNotFoundError:
            print(f"  WARNING: prior season {s} not found, skipping")
            continue
        df = df[df["season_type"].isin(season_type_keep)]
        df = df[df["home_team"].isin(NBA_TEAMS) & df["away_team"].isin(NBA_TEAMS)]
        df = parse_events(df)
        fg3a += int(df["is_fg3a"].sum())
        fg3m += int(df["is_fg3m"].sum())
        fg2a += int(df["is_fg2a"].sum())
        fg2m += int(df["is_fg2m"].sum())
        fta += int(df["is_fta"].sum())
        ftm += int(df["is_ftm"].sum())
        fouls += int(df["is_foul"].sum())
        sh = df[df["is_fga"]].copy()
        if len(sh):
            sh["is3"] = sh["is_fg3a"].astype(int)
            sh["made"] = (sh["is_fg3m"] | sh["is_fg2m"]).astype(int)
            shot_frames.append(sh[["shot_family", "dist_bin", "is3", "made"]])
    if fg3a == 0:
        raise RuntimeError("No prior-season events found; cannot build league priors.")

    shots = pd.concat(shot_frames, ignore_index=True) if shot_frames else pd.DataFrame()
    shot_table = {}
    if len(shots):
        g = shots.groupby(["shot_family", "dist_bin", "is3"], observed=True)["made"]
        agg = g.agg(["mean", "size"])
        overall3 = float(shots.loc[shots["is3"] == 1, "made"].mean())
        overall2 = float(shots.loc[shots["is3"] == 0, "made"].mean())
        for (famname, db, is3), r in agg.iterrows():
            if r["size"] >= 500:
                shot_table[f"{famname}|{int(db)}|{int(is3)}"] = float(r["mean"])
        out["shot_table"] = shot_table
        out["shot_fallback"] = {"2": overall2, "3": overall3}
        print(f"  shot-quality table: {len(shot_table)} cells with 500+ attempts; "
              f"league 2P {overall2:.4f}, 3P {overall3:.4f}")
    out["league_fg3_pct"] = fg3m / fg3a
    out["league_fg2_pct"] = fg2m / fg2a
    out["league_ft_pct"] = ftm / fta
    out["n_prior_fg3a"] = fg3a
    out["prior_seasons"] = list(prior_seasons)
    return out


# ----------------------------------------------------------------------------
# per-season feature build
# ----------------------------------------------------------------------------

def build_season(paths, season, priors, args):
    df = load_pbp(paths, season)

    keep_types = set(args.season_types)
    n0 = df["game_id"].nunique()
    df = df[df["season_type"].isin(keep_types)].copy()
    print(f"    season_type filter {sorted(keep_types)}: {n0:,} -> {df['game_id'].nunique():,} games")

    n1 = df["game_id"].nunique()
    df = df[df["home_team"].isin(NBA_TEAMS) & df["away_team"].isin(NBA_TEAMS)].copy()
    n2 = df["game_id"].nunique()
    if n2 < n1:
        print(f"    franchise filter (drops All-Star / exhibition): {n1:,} -> {n2:,} games")

    df["game_id"] = df["game_id"].astype(str)
    df = df.sort_values(["game_id", "play_number"]).reset_index(drop=True)

    # drop games with no decided result
    bad = df["home_won"].isna() | df["home_final"].isna() | (df["home_final"] == df["away_final"])
    if bad.any():
        drop_games = df.loc[bad, "game_id"].unique()
        df = df[~df["game_id"].isin(drop_games)]
        print(f"    dropped {len(drop_games)} games with missing or tied final scores")

    df = parse_events(df)
    df = derive_possession(df)

    # ---------------- clock ----------------
    df["is_ot"] = (df["period"] > 4).astype(int)
    df["ot_number"] = np.maximum(0, df["period"] - 4)
    df["period_len"] = np.where(df["period"] <= 4, 720.0, OT_SECONDS)

    # horizon = end of regulation, or end of the current overtime period
    horizon_left = np.where(
        df["period"] <= 4,
        df["seconds_left_reg"].values,
        df["pc_seconds_left"].values,
    )
    horizon_total = np.where(df["period"] <= 4, REG_SECONDS, OT_SECONDS)
    df["seconds_left_horizon"] = horizon_left
    df["frac_left"] = np.clip(horizon_left / horizon_total, 0.0, 1.0)
    df["frac_elapsed"] = 1.0 - df["frac_left"]
    df["seconds_left_period"] = df["pc_seconds_left"]

    # ---------------- odds ----------------
    odds = load_odds(paths)
    df = df.merge(odds, left_on="game_id", right_on="espn_game_id", how="left")
    miss = df["p_home_close"].isna().groupby(df["game_id"]).first()
    n_miss = int(miss.sum())
    if n_miss:
        print(f"    WARNING: {n_miss} games have no closing moneyline (kept, feature = NaN)")

    p = df["p_home_close"].clip(1e-4, 1 - 1e-4)
    df["logit_close"] = np.log(p / (1 - p))
    df["home_spread_close"] = df["home_spread_close"].astype(float)
    df["total_close"] = df["total_close"].astype(float)

    # expected remaining point differential from the pregame spread, scaled by
    # how much of the game is left. Trees get the raw pieces and the products.
    # The pregame line never enters the model undecayed. Every pregame term is
    # multiplied by frac_left raised to a power; the powers are supplied so the
    # trees pick the decay shape instead of me assuming one.
    exp_full = -df["home_spread_close"]
    for k in args.decay_pows:
        w = np.power(df["frac_left"], k)
        tag = str(k).replace(".", "p")
        df[f"exp_margin_left_p{tag}"] = exp_full * w
        df[f"logit_close_p{tag}"] = df["logit_close"] * w
    # This one feeds the simulation's per-possession edge, so it uses the decay
    # MEASURED by nba_decay_fit.py (k ~ 0.8 across 2021-2025), not the first
    # entry of DECAY_POWS. Using k=0.5 here handed the favourite more remaining
    # edge than the data supports.
    df["exp_margin_left"] = exp_full * np.power(df["frac_left"], args.sim_decay_pow)
    # Projected final margin. Kept because trees cannot form a sum on their own,
    # but its pregame content is ~15% of its variance, so the weights table would
    # under-report the pregame share without the split below.
    df["margin_plus_exp_left"] = df["score_margin"] + df["exp_margin_left"]
    # Same quantity expressed per unit of remaining uncertainty, so the score and
    # the pregame projection are separable in the contribution report.
    sl = np.sqrt(np.maximum(df["seconds_left_horizon"], 1.0))
    df["exp_margin_left_per_sec"] = df["exp_margin_left"] / sl

    # scoring environment: the pregame total decays the same way, and realized
    # pace (a game-state quantity) is what is left to carry the information.
    for k in args.decay_pows:
        tag = str(k).replace(".", "p")
        df[f"total_left_p{tag}"] = df["total_close"] * np.power(df["frac_left"], k)

    # ---------------- shooter baselines for luck ----------------
    rates = load_player_rates(paths, season)
    lg3 = priors["league_fg3_pct"]
    lgft = priors["league_ft_pct"]
    lg2 = priors["league_fg2_pct"]

    df["shooter_id"] = pd.to_numeric(df["athlete_id_1"], errors="coerce")
    if rates is not None:
        df = df.merge(
            rates.rename(columns={"athlete_id": "shooter_id"}),
            on=["game_id", "shooter_id"],
            how="left",
        )
    else:
        df["fg3_pct"] = np.nan
        df["ft_pct"] = np.nan
        df["prior_min"] = np.nan

    # shrink the player's prior rate toward the league rate. --shrink-min is the
    # pseudo-count in prior minutes; a player with that many prior minutes gets
    # half his own rate and half the league rate.
    pm = df["prior_min"].fillna(0.0).clip(lower=0.0)
    w = pm / (pm + args.shrink_min)
    df["exp_fg3_pct"] = w * df["fg3_pct"].fillna(lg3) + (1 - w) * lg3
    df["exp_ft_pct"] = w * df["ft_pct"].fillna(lgft) + (1 - w) * lgft

    is_home_actor = (df["actor_team_id"] == df["home_team_id"]).astype(float)
    is_away_actor = (df["actor_team_id"] == df["away_team_id"]).astype(float)

    def side_sum(mask, side_flag, col=None):
        v = (mask.astype(float) * side_flag) if col is None else (mask.astype(float) * side_flag * col)
        return v.groupby(df["game_id"]).cumsum()

    # 3-point luck: actual made threes minus what an average outcome for those
    # exact shooters would have produced, in points.
    h_fg3a = side_sum(df["is_fg3a"], is_home_actor)
    a_fg3a = side_sum(df["is_fg3a"], is_away_actor)
    h_fg3m = side_sum(df["is_fg3m"], is_home_actor)
    a_fg3m = side_sum(df["is_fg3m"], is_away_actor)
    h_x3 = side_sum(df["is_fg3a"], is_home_actor, df["exp_fg3_pct"])
    a_x3 = side_sum(df["is_fg3a"], is_away_actor, df["exp_fg3_pct"])

    h_fta = side_sum(df["is_fta"], is_home_actor)
    a_fta = side_sum(df["is_fta"], is_away_actor)
    h_ftm = side_sum(df["is_ftm"], is_home_actor)
    a_ftm = side_sum(df["is_ftm"], is_away_actor)
    h_xft = side_sum(df["is_fta"], is_home_actor, df["exp_ft_pct"])
    a_xft = side_sum(df["is_fta"], is_away_actor, df["exp_ft_pct"])

    h_fg2a = side_sum(df["is_fg2a"], is_home_actor)
    a_fg2a = side_sum(df["is_fg2a"], is_away_actor)
    h_fg2m = side_sum(df["is_fg2m"], is_home_actor)
    a_fg2m = side_sum(df["is_fg2m"], is_away_actor)

    h_foul = side_sum(df["is_foul"], is_home_actor)
    a_foul = side_sum(df["is_foul"], is_away_actor)

    df["fg3a_home"], df["fg3a_away"] = h_fg3a, a_fg3a
    df["fg3_luck_home_pts"] = 3.0 * (h_fg3m - h_x3)
    df["fg3_luck_away_pts"] = 3.0 * (a_fg3m - a_x3)
    df["fg3_luck_diff_pts"] = df["fg3_luck_home_pts"] - df["fg3_luck_away_pts"]

    df["ft_luck_diff_pts"] = (h_ftm - h_xft) - (a_ftm - a_xft)

    df["fg2_luck_diff_pts"] = 2.0 * ((h_fg2m - lg2 * h_fg2a) - (a_fg2m - lg2 * a_fg2a))

    df["fta_diff"] = h_fta - a_fta
    df["foul_diff"] = h_foul - a_foul          # positive = home fouling more
    df["team_foul_diff_period"] = df["home_team_fouls_period"] - df["away_team_fouls_period"]

    df["luck_diff_pts"] = df["fg3_luck_diff_pts"] + df["ft_luck_diff_pts"] + df["fg2_luck_diff_pts"]
    df["margin_ex_luck"] = df["score_margin"] - df["luck_diff_pts"]

    # regression opportunity depends on time left
    df["fg3_luck_x_fracleft"] = df["fg3_luck_diff_pts"] * df["frac_left"]
    df["luck_x_fracleft"] = df["luck_diff_pts"] * df["frac_left"]
    df["fta_diff_x_fracleft"] = df["fta_diff"] * df["frac_left"]

    # shot volume so far, for context on how meaningful the luck term is
    df["fg3a_diff"] = h_fg3a - a_fg3a
    df["fga_total"] = h_fg3a + a_fg3a + h_fg2a + a_fg2a

    # realized pace vs the pregame total
    pts_total = df["home_score"] + df["away_score"]
    df["pts_total"] = pts_total
    with np.errstate(divide="ignore", invalid="ignore"):
        df["pace_ratio"] = np.where(
            df["frac_elapsed"] > 0.02,
            pts_total / (df["total_close"] * df["frac_elapsed"]),
            np.nan,
        )

    df["margin_x_fracleft"] = df["score_margin"] * df["frac_left"]
    with np.errstate(divide="ignore", invalid="ignore"):
        df["margin_per_sec_left"] = np.where(
            df["seconds_left_horizon"] > 0,
            df["score_margin"] / np.sqrt(np.maximum(df["seconds_left_horizon"], 1.0)),
            df["score_margin"] * 1.0,
        )

    df = build_lineup_features(df, rates, args)
    df = add_sim_features(df, paths, args)

    df["home_won"] = df["home_won"].astype(int)

    cols = list(dict.fromkeys(ID_COLS + FEATURES + ["home_won"]))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"internal error: missing columns {missing}")

    out = df[cols].copy()
    return out


ID_COLS = [
    "game_id", "season", "game_date", "home_team", "away_team",
    "period", "play_number", "wallclock", "pc_seconds_left",
    "home_score", "away_score", "home_final", "away_final",
    "home_lineup", "away_lineup",
    "logit_close",   # carried for the backtest's fav/dog view, NOT a model input
    "total_close",       # reference only, NOT a model input
    "home_spread_close", # reference only, NOT a model input
    "home_final", "away_final",
]

FEATURES = [
    # game state
    "score_margin",
    "seconds_left_period",
    "seconds_left_horizon",
    "frac_left",
    "frac_elapsed",
    "margin_x_fracleft",
    "margin_per_sec_left",
    # pregame skill - ONLY in decayed form, never raw. The _p<k> suffix is the
    # power of frac_left applied; DECAY_POWS below controls which are built.
    "exp_margin_left",
    "exp_margin_left_per_sec",
    "margin_plus_exp_left",
    # scoring environment - decayed pregame total plus realized pace
    "pts_total",
    "pace_ratio",
    # how we got here / regression opportunity
    "fg3_luck_diff_pts",
    "fg3_luck_x_fracleft",
    "fg2_luck_diff_pts",
    "ft_luck_diff_pts",
    "luck_diff_pts",
    "luck_x_fracleft",
    "margin_ex_luck",
    "fg3a_diff",
    "fga_total",
    "fta_diff",
    "fta_diff_x_fracleft",
    "foul_diff",
    "team_foul_diff_period",
]

FEATURES += LINEUP_FEATURES
FEATURES += SIM_FEATURES

# powers of frac_left applied to the pregame terms. Changing this changes the
# feature set, so rebuild the feature tables and retrain together.
# Measured from 2021-2025 with nba_decay_fit.py, not assumed: the share of the
# pregame edge still to be realised runs consistently ABOVE frac_left, i.e. the
# line decays slower than linearly. Implied k clustered at 0.75-0.9 and never
# exceeded ~1.02. These four bracket that range.
DECAY_POWS = [0.5, 0.75, 1.0, 1.25]

for _k in DECAY_POWS:
    _t = str(_k).replace(".", "p")
    FEATURES.append(f"exp_margin_left_p{_t}")
    FEATURES.append(f"logit_close_p{_t}")
    FEATURES.append(f"total_left_p{_t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--prior-seasons", nargs="+", type=int, default=None,
                    help="seasons used to compute league shooting baselines. "
                         "Defaults to the earliest 3 seasons passed to --seasons.")
    ap.add_argument("--season-types", nargs="+", type=int, default=[2],
                    help="ESPN season_type values to keep (2 = regular season)")
    ap.add_argument("--shrink-min", type=float, default=500.0,
                    help="pseudo-count in prior MINUTES for shrinking a shooter's "
                         "prior 3P%%/FT%% toward the league rate")
    ap.add_argument("--decay-pows", nargs="+", type=float, default=DECAY_POWS,
                    help="powers of frac_left applied to the pregame line. The "
                         "pregame terms are NEVER exposed undecayed.")
    ap.add_argument("--sim-decay-pow", type=float, default=0.8,
                    help="decay power for the pregame margin that feeds the "
                         "simulation. Default is the value measured by "
                         "nba_decay_fit.py on 2021-2025.")
    ap.add_argument("--rebuild-priors", action="store_true")
    args = ap.parse_args()

    if [float(x) for x in args.decay_pows] != [float(x) for x in DECAY_POWS]:
        raise SystemExit(
            f"--decay-pows {args.decay_pows} does not match DECAY_POWS {DECAY_POWS} "
            f"at the top of this file, so FEATURES would not line up. Edit "
            f"DECAY_POWS and rerun."
        )

    paths = resolve_paths(args.root)
    os.makedirs(paths["features"], exist_ok=True)

    prior_seasons = args.prior_seasons or sorted(args.seasons)[:3]
    prior_file = os.path.join(paths["features"], "league_priors.json")

    if os.path.exists(prior_file) and not args.rebuild_priors:
        priors = json.load(open(prior_file))
        print(f"League priors loaded from {prior_file} (seasons {priors.get('prior_seasons')})")
    else:
        print(f"Building league priors from seasons {prior_seasons} ...")
        priors = build_league_priors(paths, prior_seasons, set(args.season_types))
        json.dump(priors, open(prior_file, "w"), indent=2)
        print(f"  wrote {prior_file}")
    print(f"  league 3P% {priors['league_fg3_pct']:.4f}  2P% {priors['league_fg2_pct']:.4f}  FT% {priors['league_ft_pct']:.4f}")

    for s in args.seasons:
        print(f"\n=== season {s} ===")
        out = build_season(paths, s, priors, args)
        f = os.path.join(paths["features"], f"nba_wp_features_{s}.parquet")
        out.to_parquet(f, index=False)
        print(f"    wrote {f}  ({len(out):,} rows / {out['game_id'].nunique():,} games / {len(FEATURES)} features)")


if __name__ == "__main__":
    main()
