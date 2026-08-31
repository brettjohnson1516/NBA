"""
nba_wp_features_v22.py

Builds the per-event feature table for the NBA live moneyline win-probability model.

One row per play-by-play event = one scoreable game state.
Target = home_won (final result, OT included).

Usage (PowerShell):
  python nba_wp_features_v22.py --seasons 2021 2022 2023 2024 2025 2026

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

# v22. nba_sim_v2 fixes the ADV_GRID ceiling lookup in win_prob. Nothing
# else in this file changes from v19.
import nba_sim_v2 as SIM

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
        "espn_raw": os.path.join(data, "espn_raw"),
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


def load_game_rosters(paths, season):
    """
    Pregame availability, one row per (game, player) from the ESPN box-score
    roster.

    This replaces the roster the build used to infer from the play-by-play. That
    one was the set of players who ACTUALLY TOOK THE FLOOR, which is an outcome:
    deep bench players only appear once a game is decided, and the losing side
    empties its bench harder. rest_rapm_diff was therefore pulled toward the
    eventual winner at every row of the game, including the opening tip.

    Availability rules, from inspecting the files:
      * `active` is NOT an availability flag - it sums to exactly 10 per game
        and agrees with `starter` only 58% of the time. Ignored.
      * `reason` defaults to COACH'S DECISION for every row and is only
        overwritten with a real injury / rest / suspension note. So a DNP whose
        reason still says coach's decision was AVAILABLE and simply not used -
        he stays in the roster. Dropping him would rebuild the same leak.
      * a DNP with any other reason was unavailable and is removed.
      * a DNP with a missing reason is kept. Ambiguous, and keeping is the side
        that cannot leak.
    """
    f = find_one(
        paths["espn_raw"],
        [f"game_rosters_{season}.parquet", f"*game_rosters*{season}*.parquet"],
        f"game rosters for {season}",
    )
    r = pd.read_parquet(f)
    need = ["game_id", "team_id", "athlete_id", "did_not_play", "reason"]
    missing = [c for c in need if c not in r.columns]
    if missing:
        raise SystemExit(f"{os.path.basename(f)} is missing {missing}")
    r = r[need].copy()

    n0 = len(r)
    r["game_id"] = r["game_id"].astype(str)
    r["athlete_id"] = pd.to_numeric(r["athlete_id"], errors="coerce")
    r["team_id"] = pd.to_numeric(r["team_id"], errors="coerce")
    r = r.dropna(subset=["athlete_id", "team_id"])
    n_bad = n0 - len(r)

    dnp = r["did_not_play"].fillna(False).astype(bool)
    # na=True: a missing reason counts as coach's decision, i.e. still available
    coach = r["reason"].str.contains("coach", case=False, na=True)
    unavailable = dnp & ~coach
    r = r[~unavailable].drop_duplicates(["game_id", "athlete_id"], keep="first")

    per = r.groupby(["game_id", "team_id"]).size()
    print(f"    rosters {season}: {len(r):,} available player-games from "
          f"{r['game_id'].nunique():,} games "
          f"({os.path.basename(f)}); {int(unavailable.sum()):,} removed as "
          f"injury/rest/suspension, {per.mean():.1f} available per team-game"
          + (f"; DROPPED {n_bad:,} rows with a null athlete_id or team_id"
             if n_bad else ""))
    return r[["game_id", "team_id", "athlete_id"]]


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
    keep = ["game_id", "athlete_id", "fg3_pct", "fg2_pct", "ft_pct",
            "prior_min", "prior_poss",
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


def build_lineup_features(df, rates, rosters, args):
    """
    Three distinct quantities, because the on-court five is NOT who plays the
    rest of the game:

      oncourt_rapm_diff  - the ten players actually on the floor right now
      rest_rapm_diff     - the players AVAILABLE for this game, weighted by each
                           player's prior minutes per game, i.e. who is expected
                           to play the remaining minutes
      oncourt_edge       - oncourt minus rest, i.e. how much better or worse the
                           current five are than that team's expected mix

    Only rest_rapm_diff is scaled by time remaining; the on-court edge applies
    to the next stretch of play, not the rest of the game.

    rest_rapm_diff is built from the pregame box-score roster (see
    load_game_rosters), NOT from the players observed on the floor. The old
    version used the observed set, which is an outcome and leaked the result
    into a game-constant feature.
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

    # ---- who is available, split home / away ----
    # The roster file carries team_id but its home_away column is empty, so the
    # side comes from the pbp's own home/away team ids.
    sides = df.drop_duplicates("game_id")[
        ["game_id", "home_team_id", "away_team_id"]].copy()
    sides["game_id"] = sides["game_id"].astype(str)
    for c in ("home_team_id", "away_team_id"):
        sides[c] = pd.to_numeric(sides[c], errors="coerce")

    av = rosters.merge(sides, on="game_id", how="inner")
    av["side"] = np.where(av["team_id"] == av["home_team_id"], "home",
                          np.where(av["team_id"] == av["away_team_id"], "away", None))
    n_unmatched = int(av["side"].isna().sum())
    if n_unmatched:
        print(f"    WARNING: {n_unmatched:,} roster rows have a team_id matching "
              f"neither side of the game and are dropped")
        av = av[av["side"].notna()]

    games_pbp = set(sides["game_id"])
    games_ros = set(av["game_id"])
    if games_pbp - games_ros:
        print(f"    WARNING: {len(games_pbp - games_ros):,} games have no roster "
              f"rows; their rest_rapm_diff will be NaN")

    n_known = sum(1 for g, a in zip(av["game_id"], av["athlete_id"])
                  if (g, a) in rapm_by)
    print(f"    availability roster: {len(av):,} player-games, "
          f"{100.0 * n_known / max(len(av), 1):.1f}% found in the player-rates file")

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

        rest = {}
        for g, sub in av[av["side"] == side].groupby("game_id"):
            players = sub["athlete_id"].values
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
    # v19. Prefer the gradient-boosted edge model when one exists AND its
    # sidecar records that it beat the linear fit on the same holdout. The
    # per-possession edge is not linear in the pregame line: two linear
    # interaction terms (v16, v18) were both priced at ~0 by the fit, while the
    # GBM picked up 0.12165 vs 0.11703 on the same games. This is that test.
    #
    # The sidecar is written by nba_edge_model_v4.py. If it is absent, or says
    # the GBM lost, the linear path below runs exactly as before - a losing
    # experiment cannot ship itself.
    gsf = os.path.join(paths["features"], "edge_model_gbm.json")
    gbm_used = False
    if os.path.exists(gsf):
        gm = json.load(open(gsf))
        gbf = os.path.join(paths["features"], gm.get("booster", "edge_model_gbm.ubj"))
        if gm.get("beats_linear") and os.path.exists(gbf):
            import xgboost as _xgb
            cols = []
            gmissing = []
            for name in gm["predictors"]:
                if name == "exp_margin_full":
                    cols.append(-df["home_spread_close"].values.astype(float))
                elif name in df.columns:
                    cols.append(df[name].values.astype(float))
                else:
                    gmissing.append(name)
            if gmissing:
                print(f"    WARNING: edge_model_gbm.json names columns this build "
                      f"does not write: {gmissing}. Ignoring it and using the "
                      f"linear edge model.")
            else:
                Xg = np.column_stack([np.nan_to_num(c, nan=0.0) for c in cols])
                _b = _xgb.Booster()
                _b.load_model(gbf)
                dg = _xgb.DMatrix(Xg.astype(np.float32),
                                  feature_names=list(gm["predictors"]))
                adv = _b.predict(dg, iteration_range=(0, int(gm["best_iteration"]) + 1))
                gbm_used = True
                print(f"    edge model: {os.path.basename(gbf)} GBM "
                      f"(fitted on {gm['seasons']}, holdout R2 {gm['r2_holdout_gbm']:.5f} "
                      f"vs linear {gm['r2_holdout_linear']:.5f})")

    ef = os.path.join(paths["features"], "edge_model.json")
    if gbm_used:
        pass
    elif os.path.exists(ef):
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
            # v18. This used to be a SystemExit, and that is what deadlocked the
            # v17 run: a stale edge_model.json naming a column this build no
            # longer writes killed the feature build, and the refit that would
            # have replaced the json needed the parquet the build was supposed to
            # produce. Fall back to the decayed spread for this pass instead. The
            # WARNING below is loud on purpose - if you see it on the SECOND
            # feature build of a sequence, the refit did not take and the numbers
            # from that run are not testing what you think they are.
            print(f"    WARNING: edge_model.json names columns this build does not "
                  f"write: {missing}. Ignoring the file and falling back to the "
                  f"decayed spread for this pass. Refit with nba_edge_model_v3.py, "
                  f"then rebuild features.")
            adv = df["exp_margin_left"].values / half
        else:
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

def fit_decay_pow(dd, k_lo=0.30, k_hi=1.80):
    """
    Measures how fast the pregame line decays, instead of taking it as a
    constant. This used to be a 0.8 typed into the argparse default, measured
    once by nba_decay_fit.py on 2021-2025 and never refreshed.

    For every state, holding the CURRENT score fixed:

        rest_margin = a + b * exp_margin * frac_left**k + c * score_margin

    where rest_margin is what is still to be scored and exp_margin is the
    pregame expected full-game margin. k is found by grid search on the sum of
    squared residuals; b and c are ordinary least squares at each k.

    Reported but NOT applied:
      b - the share of the pregame edge that actually shows up. 1.0 means the
          closing line is fair once decayed. exp_margin_left keeps b = 1.
      a - residual home-court points over the REST of the game after the
          closing spread and the current score. Non-zero means the closing
          line is not the whole story live.
    """
    d = dd.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["rest_margin", "exp_margin", "score_margin", "frac_left"])
    # frac_left near 1 carries no information about k (fl**k == 1 for every k)
    # and near 0 the term is numerically dead; both stay in the fit, they just
    # contribute nothing to separating the candidates.
    d = d[(d["frac_left"] >= 0.0) & (d["frac_left"] <= 1.0)]
    n = len(d)
    if n < 50_000:
        raise SystemExit(
            f"only {n:,} states have both a closing spread and a final score - "
            f"not enough to measure the pregame decay. Check that "
            f"closing_spreads_pinnacle.parquet covers the prior seasons.")

    y = d["rest_margin"].values.astype(float)
    e = d["exp_margin"].values.astype(float)
    fl = d["frac_left"].values.astype(float)
    sm = d["score_margin"].values.astype(float)
    ones = np.ones(n)

    def sse_at(k):
        X = np.column_stack([ones, e * np.power(fl, k), sm])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r), beta

    # coarse grid, then refine around the winner
    best_k, best_sse, best_beta = None, None, None
    for k in np.round(np.arange(k_lo, k_hi + 1e-9, 0.05), 4):
        s, b = sse_at(float(k))
        if best_sse is None or s < best_sse:
            best_k, best_sse, best_beta = float(k), s, b
    for k in np.round(np.arange(best_k - 0.05, best_k + 0.05 + 1e-9, 0.01), 4):
        k = float(np.clip(k, k_lo, k_hi))
        s, b = sse_at(k)
        if s < best_sse:
            best_k, best_sse, best_beta = k, s, b

    if best_k <= k_lo + 1e-9 or best_k >= k_hi - 1e-9:
        print(f"    WARNING: fitted decay power {best_k:.2f} is at the edge of the "
              f"search range [{k_lo}, {k_hi}] - the fit is not identified, treat "
              f"it as suspect")

    a, b, c = float(best_beta[0]), float(best_beta[1]), float(best_beta[2])
    print(f"    pregame decay fit: k {best_k:.2f} on {n:,} states / "
          f"{d['game_id'].nunique():,} games")
    print(f"      share of the pregame edge realised (b) {b:.3f}  "
          f"[1.000 = closing line is fair once decayed, NOT applied]")
    print(f"      residual home edge over the rest of the game {a:+.3f} pts, "
          f"current-score coefficient {c:+.4f}")
    return {"sim_decay_pow": round(best_k, 4),
            "sim_decay_b": b,
            "sim_decay_home_resid": a,
            "sim_decay_score_coef": c,
            "sim_decay_n": int(n)}


def build_league_priors(paths, prior_seasons, season_type_keep):
    out = {}
    fg3a = fg3m = fta = ftm = fouls = fg2a = fg2m = 0
    shot_frames = []
    decay_frames = []
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

        # Rows for the pregame-decay fit. Same pbp read, no extra pass. Only
        # regulation states: an overtime state has no pregame margin left to
        # decay, frac_left is defined against the OT period, and including them
        # biases k.
        dd = df[["game_id", "period", "seconds_left_reg", "score_margin",
                 "home_final", "away_final"]].copy()
        dd = dd[dd["period"] <= 4]
        dd = dd[dd["home_final"].notna() & dd["away_final"].notna()
                & (dd["home_final"] != dd["away_final"])]
        dd["game_id"] = dd["game_id"].astype(str)
        dd["frac_left"] = np.clip(
            dd["seconds_left_reg"].astype(float) / REG_SECONDS, 0.0, 1.0)
        dd["rest_margin"] = ((dd["home_final"] - dd["away_final"])
                             - dd["score_margin"]).astype(float)
        decay_frames.append(dd[["game_id", "frac_left", "score_margin", "rest_margin"]])
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

    # ---- pregame decay, measured here instead of hardcoded ----
    if not decay_frames:
        raise SystemExit("no prior-season states available for the decay fit")
    dd = pd.concat(decay_frames, ignore_index=True)
    sp = load_odds(paths)[["espn_game_id", "home_spread_close"]].dropna()
    sp["espn_game_id"] = sp["espn_game_id"].astype(str)
    dd = dd.merge(sp, left_on="game_id", right_on="espn_game_id", how="inner")
    dd["exp_margin"] = -dd["home_spread_close"].astype(float)
    out.update(fit_decay_pow(dd))
    out["sim_decay_seasons"] = list(prior_seasons)
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
    # This one feeds the simulation's per-possession edge, so it uses a single
    # measured decay rather than the DECAY_POWS bracket the trees get. The power
    # is fitted in build_league_priors() from the prior seasons every time the
    # priors are rebuilt - it is NOT a constant and NOT imported from
    # nba_decay_fit.py, which is a standalone diagnostic that nothing reads.
    df["exp_margin_left"] = exp_full * np.power(df["frac_left"], args.sim_decay_pow)

    # ---- how far the game has already contradicted the pregame line ----
    #
    # v17. v16 tried this as exp_margin_left * line_surprise and the fit gave it
    # a coefficient of -0.000097 (12th of 18 by contribution, edge-model R2
    # 0.11887 -> 0.11918) and the backtest lost 0.36 ROI points. Two reasons it
    # could not work, both mine:
    #
    #   1. It was multiplied by exp_margin_LEFT, which is the pregame edge
    #      decayed by frac_left**0.76. That factor goes to zero at exactly the
    #      point in the game where the disagreement is largest and where the
    #      losing bets are, so the term switched itself off before it could bite.
    #   2. adv is a per-possession RATE, and the edge model carries the pregame
    #      prior as a constant rate (exp_margin_full, coef ~0.0107). A raw
    #      points-times-points product cannot express "shrink that rate by a
    #      fraction"; the coefficient has to mean something different at every
    #      possession count, so the fit settled on ~nothing.
    #
    # v17 fixes both. line_surprise is still the margin banked minus the margin
    # the closing line implied should be banked by now, home-positive. It is then
    # extrapolated to a full game (divided by the fraction of the game played) so
    # it stays on one scale from Q1 to Q4, and interacted with the UNDECAYED
    # pregame margin - the same term the edge model already prices as a constant
    # rate. A negative coefficient then means exactly "shrink the pregame prior
    # in proportion to how wrong the game says it is", at any time of night.
    #
    # Still not offered on its own. A standalone score term counts the current
    # score twice - once as the DP's position, once as a per-possession rate -
    # which is what margin_ex_luck did in v7. Interaction only, and it is 0
    # whenever the game is running to script.
    #
    # The 0.15 floor and the +/-30 clip are numerical guards, not fitted: below
    # 15% of the game played the extrapolation divides by almost nothing, and a
    # 60-point extrapolated surprise off two possessions is not information.
    df["line_surprise"] = df["score_margin"] - exp_full * (
        1.0 - np.power(df["frac_left"], args.sim_decay_pow))
    df["line_surprise_rate"] = np.clip(
        df["line_surprise"] / np.clip(df["frac_elapsed"], 0.15, 1.0), -30.0, 30.0)
    df["exp_margin_x_surprise_rate"] = exp_full * df["line_surprise_rate"]
    # v18. Also kept: the v16 form. It is not a predictor any more, but a
    # leftover edge_model.json written by nba_edge_model_v2.py still names it,
    # and the feature build has to be able to resolve that file in order to
    # produce the parquet that the refit reads. Dropping the column created a
    # deadlock - features would not run without the refit and the refit would
    # not run without features - which is exactly what killed the v17 attempt.
    df["exp_margin_left_x_surprise"] = df["exp_margin_left"] * df["line_surprise"]
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
    # Two-point expectation from the SHOOTER's own prior rate, shrunk toward
    # league by prior minutes exactly like the 3P and FT terms. It used to be a
    # flat league 2P% for everyone, which gave a rim-finishing big and a pull-up
    # guard the same expectation — the rotation-player spread is 0.33 to 0.72.
    if "fg2_pct" in df.columns:
        df["exp_fg2_pct"] = w * df["fg2_pct"].fillna(lg2) + (1 - w) * lg2
    else:
        df["exp_fg2_pct"] = lg2
        if season == args.seasons[0]:
            print("    WARNING: no fg2_pct in the player rates file - two-point "
                  "luck falls back to a flat league rate")

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

    h_x2 = side_sum(df["is_fg2a"], is_home_actor, df["exp_fg2_pct"])
    a_x2 = side_sum(df["is_fg2a"], is_away_actor, df["exp_fg2_pct"])
    df["fg2_luck_diff_pts"] = 2.0 * ((h_fg2m - h_x2) - (a_fg2m - a_x2))

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

    rosters = load_game_rosters(paths, season)
    df = build_lineup_features(df, rates, rosters, args)
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
    # v16. Written so nba_edge_model_v2.py can read them out of the parquet.
    # Deliberately NOT in FEATURES: the tree feature set is unchanged from v15,
    # so anything this run moves is attributable to the sim edge alone.
    "line_surprise",
    "line_surprise_rate",
    "exp_margin_x_surprise_rate",
    "exp_margin_left_x_surprise",
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
    ap.add_argument("--sim-decay-pow", type=float, default=None,
                    help="OVERRIDE ONLY. The decay power for the pregame margin "
                         "that feeds the simulation is measured from the prior "
                         "seasons and stored in league_priors.json. Passing a "
                         "value here overrides the measurement; leave it unset "
                         "in normal use.")
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

    # The decay power is measured, never typed in. A league_priors.json written
    # by an older build has no measurement in it, so it is refused rather than
    # quietly falling back to a constant.
    if args.sim_decay_pow is not None:
        print(f"  sim decay power: {args.sim_decay_pow} FROM THE COMMAND LINE - "
              f"this OVERRIDES the measured value "
              f"{priors.get('sim_decay_pow', 'n/a')}")
    else:
        if "sim_decay_pow" not in priors:
            raise SystemExit(
                f"{prior_file} has no measured sim_decay_pow (it was written by "
                f"an older feature build). Rerun with --rebuild-priors so the "
                f"decay is measured instead of assumed.")
        args.sim_decay_pow = float(priors["sim_decay_pow"])
        print(f"  sim decay power: {args.sim_decay_pow:.2f}, measured on seasons "
              f"{priors.get('sim_decay_seasons', priors.get('prior_seasons'))} "
              f"({priors.get('sim_decay_n', 0):,} states)")

    for s in args.seasons:
        print(f"\n=== season {s} ===")
        out = build_season(paths, s, priors, args)
        f = os.path.join(paths["features"], f"nba_wp_features_{s}.parquet")
        out.to_parquet(f, index=False)
        # The feature list must travel with the DATA, not through a python
        # import. The trainer used to do `from nba_wp_features import FEATURES`,
        # so running a versioned build (nba_wp_features_v10mel.py) still trained
        # on whatever list the plain module held — a new column was written to
        # the parquet and then silently ignored.
        flist = os.path.join(paths["features"], "feature_list.json")
        json.dump({"features": FEATURES, "n": len(FEATURES),
                   "written_by": os.path.basename(__file__)},
                  open(flist, "w"), indent=2)
        print(f"    wrote {f}  ({len(out):,} rows / {out['game_id'].nunique():,} games / {len(FEATURES)} features)")


if __name__ == "__main__":
    main()
