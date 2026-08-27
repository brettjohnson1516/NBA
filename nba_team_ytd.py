"""
nba_team_ytd.py
---------------
Builds team-level, STRICTLY-PRIOR-TO-GAME (year-to-date) rate stats from the
ESPN play-by-play parquet files written by nba_espn_pbp.py.

One row per (game_id, team). Every stat column describes what that team had
accumulated in that same regular season BEFORE the game began. The row for a
team's first game of the season has games_played = 0 and NaN rates.

Two different possession notions are used and both are reported:

  * FORMULA possessions - fga - oreb + tov + 0.44*fta, computed per team.
    This is the denominator for every per-100 rate and for pace, and it is the
    same formula nba_player_rates.py uses, so the two feature sets line up.

  * SEGMENTED possessions - the play-by-play is walked and split into runs of
    plays belonging to one offense. These are used ONLY for the two timing
    metrics (seconds per possession, seconds until the first shot), because
    the formula count has no clock attached to it. Segmented and formula
    counts will not be identical; the segmented count is reported alongside so
    the gap is visible.

Output: <data-root>\\<out-dir>\\nba_team_ytd_<season>.parquet

Usage (PowerShell):
  python nba_team_ytd.py --seasons 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

DEFAULT_DATA_ROOT = r"C:\Users\saint\OneDrive\Documents\NBA_AUG_2026\data"

# Franchise codes as they appear in the pbp files. NJ is the New Jersey Nets,
# who existed through the 2011-12 season before becoming BKN; leaving it out
# silently deletes every Nets game from 2012. NOH/NOK are the pre-2014 New
# Orleans codes and CHO the 2015-2025 Charlotte code; they are included so the
# same omission cannot happen again if ESPN's abbreviations vary by season.
NBA_TEAMS = {
    "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS",
    "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN", "NO", "NY",
    "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC", "TOR", "UTAH", "WSH",
    "NJ", "NOH", "NOK", "CHO", "SEA",
}

REGULAR_SEASON = 2
REGULATION_PERIODS = 4
PERIOD_SECONDS = 720
OT_SECONDS = 300

# Timing sanity bounds, in seconds of game clock.
MAX_POSS_SECONDS = 60.0     # possessions longer than this are clock artifacts
MAX_SHOT_SECONDS = 40.0     # time from possession start to first field goal


# --------------------------------------------------------------------------
# play tagging  (same rules as nba_player_rates.py, plus steals and blocks)
# --------------------------------------------------------------------------

def tag_plays(df: pd.DataFrame) -> pd.DataFrame:
    tt = df["type_text"].fillna("")
    tx_l = df["text"].fillna("").str.lower()

    is_ft = tt.str.contains("Free Throw", case=False, na=False)
    made = tx_l.str.contains("makes", na=False)
    missed = tx_l.str.contains("misses", na=False)

    # A blocked shot is its own row and reads "X blocks Y's layup" - no "makes"
    # and no "misses". It is still a field goal attempt by the SHOOTING team,
    # and actor_team_id on these rows is the shooter's team (verified against
    # which side rebounds next). Without this the attempt vanishes from fga and
    # from the possession count, and FG% comes out several points too high.
    is_blk_row = (~is_ft) & (~made) & (~missed) & tx_l.str.contains("block", na=False)

    df["is_ft"] = is_ft
    df["is_ft_made"] = is_ft & made
    df["is_ft_att"] = is_ft & (made | missed)

    is_fga = (~is_ft) & (made | missed | is_blk_row)
    df["is_fga"] = is_fga
    df["is_fgm"] = is_fga & made

    # score_value is only trusted as a three-point tiebreaker when the column
    # actually looks like a per-play point value. In files where it is blank or
    # holds something else, the shot text alone decides.
    sv = pd.to_numeric(df["score_value"], errors="coerce")
    sv_is_play_value = bool(sv.notna().any() and sv.max(skipna=True) <= 4)
    is_three = tx_l.str.contains("three point", na=False)
    if sv_is_play_value:
        is_three = is_three | (sv == 3)
    df["is_fg3a"] = is_fga & is_three
    df["is_fg3m"] = is_fga & is_three & made

    df["is_tov"] = tt.str.contains("Turnover", case=False, na=False)
    df["is_foul"] = tt.str.contains("Foul", case=False, na=False) & (
        ~tt.str.contains("Turnover", case=False, na=False)
    )

    reb = tt.str.contains("Rebound", case=False, na=False)
    has_player = df["athlete_id_1"].notna()
    df["is_oreb"] = reb & tt.str.contains("Offensive", case=False, na=False)
    df["is_dreb"] = reb & tt.str.contains("Defensive", case=False, na=False)
    df["is_oreb_player"] = df["is_oreb"] & has_player
    df["is_dreb_player"] = df["is_dreb"] & has_player

    df["is_ast"] = tx_l.str.contains("assists", na=False)

    # A steal shows up on the TURNOVER row, credited to the other team.
    # A block shows up on the MISSED SHOT row, credited to the other team.
    df["tov_stolen"] = df["is_tov"] & (
        tx_l.str.contains("steal", na=False)
        | tt.str.contains("Steal", case=False, na=False)
    )
    df["fga_blocked"] = is_blk_row
    return df


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def resolve_pbp_dir(root: Path, override: str | None) -> Path:
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
    """
    Regular-season NBA-vs-NBA plays only.

    lineup_ok is deliberately NOT filtered here. That flag marks periods where
    the five-man lineup could not be resolved; it says nothing about whether
    the shot, rebound or turnover on the row is real. Dropping those rows would
    silently remove real events from team totals.
    """
    path = pbp_dir / f"pbp_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing pbp file: {path}")
    df = pd.read_parquet(path)

    n0 = df["game_id"].nunique()
    df = df[df["season_type"] == REGULAR_SEASON]
    df = df[df["home_team"].isin(NBA_TEAMS) & df["away_team"].isin(NBA_TEAMS)]
    n1 = df["game_id"].nunique()
    print(f"  {season}: {n1} regular-season games kept of {n0} in file "
          f"({len(df):,} plays)")

    df = df.sort_values(["game_id", "play_number"]).reset_index(drop=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return tag_plays(df)


# --------------------------------------------------------------------------
# per-team, per-game box totals
# --------------------------------------------------------------------------

def team_game_box(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (game_id, team_id) with that team's own totals.

    Everything is attributed to actor_team_id, which is the team that performed
    the action: the shooter's team on a shot, the rebounder's team on a rebound
    (so defensive rebounds land on the defending team, as they should), and the
    fouling team on a foul.

    Steals and blocks are not their own rows in this feed, so they are counted
    from the opponent's turnover and missed-shot rows and joined back on.
    """
    d = df[df["actor_team_id"].notna()].copy()
    d["actor_team_id"] = d["actor_team_id"].astype("int64")

    # Points are rebuilt from the made shots themselves rather than read off
    # score_value. score_value is not reliably populated in the older ESPN
    # feeds, and summing it undercounts scoring badly in the 2013-2020 files.
    d["pts"] = (
        3.0 * d["is_fg3m"].astype(float)
        + 2.0 * (d["is_fgm"].astype(float) - d["is_fg3m"].astype(float))
        + 1.0 * d["is_ft_made"].astype(float)
    )

    g = d.groupby(["game_id", "actor_team_id"], sort=False)
    box = g.agg(
        pts=("pts", "sum"),
        fga=("is_fga", "sum"),
        fgm=("is_fgm", "sum"),
        fg3a=("is_fg3a", "sum"),
        fg3m=("is_fg3m", "sum"),
        fta=("is_ft_att", "sum"),
        ftm=("is_ft_made", "sum"),
        oreb=("is_oreb_player", "sum"),
        dreb=("is_dreb_player", "sum"),
        tov=("is_tov", "sum"),
        ast=("is_ast", "sum"),
        fouls=("is_foul", "sum"),
        tov_stolen=("tov_stolen", "sum"),
        fga_blocked=("fga_blocked", "sum"),
    ).reset_index().rename(columns={"actor_team_id": "team_id"})

    meta = df.groupby("game_id", sort=False).agg(
        season=("season", "first"),
        game_date=("game_date", "first"),
        home_team=("home_team", "first"),
        away_team=("away_team", "first"),
        home_team_id=("home_team_id", "first"),
        away_team_id=("away_team_id", "first"),
        home_final=("home_final", "first"),
        away_final=("away_final", "first"),
        max_period=("period", "max"),
    ).reset_index()
    meta["home_team_id"] = meta["home_team_id"].astype("int64")
    meta["away_team_id"] = meta["away_team_id"].astype("int64")
    meta["game_seconds"] = (
        REGULATION_PERIODS * PERIOD_SECONDS
        + np.maximum(meta["max_period"].astype(int) - REGULATION_PERIODS, 0) * OT_SECONDS
    ).astype(float)

    box = box.merge(meta, on="game_id", how="inner")
    box = box[(box["team_id"] == box["home_team_id"])
              | (box["team_id"] == box["away_team_id"])].copy()

    is_home = box["team_id"] == box["home_team_id"]
    box["is_home"] = is_home.astype(int)
    box["team"] = np.where(is_home, box["home_team"], box["away_team"])
    box["opponent"] = np.where(is_home, box["away_team"], box["home_team"])
    box["opp_team_id"] = np.where(is_home, box["away_team_id"], box["home_team_id"])
    box["team_final"] = np.where(is_home, box["home_final"], box["away_final"])
    box["opp_final"] = np.where(is_home, box["away_final"], box["home_final"])
    box["won"] = (box["team_final"] > box["opp_final"]).astype(int)

    box["poss"] = (
        box["fga"] - box["oreb"] + box["tov"] + 0.44 * box["fta"]
    ).astype(float)

    # How many of this team's points are missing from the feed for this game.
    # ESPN dropped roughly 2.4 made field goals per game from the 2016 file and
    # a smaller number from 2013; nothing in the tagging can recover a play
    # that was never written. Carrying the shortfall forward lets a model drop
    # or downweight teams whose accumulated history is built on holed games.
    box["pts_gap"] = (box["team_final"] - box["pts"]).astype(float)

    # opponent side, joined on the same game
    opp_cols = ["pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb",
                "dreb", "tov", "ast", "fouls", "tov_stolen", "fga_blocked",
                "poss"]
    opp = box[["game_id", "team_id"] + opp_cols].copy()
    opp.columns = ["game_id", "opp_team_id"] + [f"opp_{c}" for c in opp_cols]
    box = box.merge(opp, on=["game_id", "opp_team_id"], how="left")

    # a team's steals and blocks are recorded on the opponent's rows
    box["stl"] = box["opp_tov_stolen"].fillna(0.0)
    box["blk"] = box["opp_fga_blocked"].fillna(0.0)
    box["reb"] = box["oreb"] + box["dreb"]
    box["opp_reb"] = box["opp_oreb"] + box["opp_dreb"]
    box["opp_stl"] = box["tov_stolen"]
    box["opp_blk"] = box["fga_blocked"]

    return box


# --------------------------------------------------------------------------
# possession segmentation (for the timing metrics only)
# --------------------------------------------------------------------------

def period_start_seconds(period: np.ndarray) -> np.ndarray:
    period = period.astype(float)
    return np.where(
        period <= REGULATION_PERIODS,
        (period - 1) * PERIOD_SECONDS,
        REGULATION_PERIODS * PERIOD_SECONDS
        + (period - REGULATION_PERIODS - 1) * OT_SECONDS,
    )


def possession_timing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Split each period into runs of plays belonging to one offense and measure
    how long each run lasted and how long it took to get a shot up.

    The offense is read off the plays where it is unambiguous - a shot, a free
    throw, a turnover and an offensive rebound belong to the offense, a
    defensive rebound belongs to the defense - and carried across the plays in
    between (fouls, substitutions, timeouts) by forward then backward fill
    inside the period.

    A possession starts when the previous one ended, or at the tip for the
    first possession of a period, and ends on its last play. Returns per
    (game_id, team_id) sums and counts so that the year-to-date averages can be
    possession-weighted rather than game-weighted.
    """
    d = df[["game_id", "period", "play_number", "seconds_elapsed",
            "actor_team_id", "home_team_id", "away_team_id",
            "is_fga", "is_ft_att", "is_tov", "is_oreb", "is_dreb"]].copy()

    actor = d["actor_team_id"]
    home = d["home_team_id"].astype("int64")
    away = d["away_team_id"].astype("int64")
    other = np.where(actor.to_numpy() == home.to_numpy(),
                     away.to_numpy(), home.to_numpy())

    off_side = (d["is_fga"] | d["is_ft_att"] | d["is_tov"] | d["is_oreb"]) \
        & actor.notna()
    def_side = d["is_dreb"] & actor.notna()

    d["_off"] = np.select(
        [off_side.to_numpy(), def_side.to_numpy()],
        [actor.to_numpy(), other],
        default=np.nan,
    )
    grp = d.groupby(["game_id", "period"], sort=False)["_off"]
    d["_off"] = grp.ffill()
    d["_off"] = d.groupby(["game_id", "period"], sort=False)["_off"].bfill()
    d = d[d["_off"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["game_id", "team_id", "poss_seg",
                                     "sum_poss_sec", "n_poss_sec",
                                     "sum_shot_sec", "n_shot_sec"])

    key = (d["game_id"].astype(str) + "|" + d["period"].astype(str)
           + "|" + d["_off"].astype("int64").astype(str))
    d["poss_id"] = (key != key.shift()).cumsum()

    d["_fga_sec"] = np.where(d["is_fga"].to_numpy(),
                             d["seconds_elapsed"].to_numpy(), np.nan)

    p = d.groupby("poss_id", sort=True).agg(
        game_id=("game_id", "first"),
        period=("period", "first"),
        team_id=("_off", "first"),
        t_end=("seconds_elapsed", "max"),
        t_first_fga=("_fga_sec", "min"),
    ).reset_index()
    p["team_id"] = p["team_id"].astype("int64")

    p["prev_end"] = p.groupby(["game_id", "period"], sort=False)["t_end"].shift(1)
    p["t_start"] = p["prev_end"].fillna(
        pd.Series(period_start_seconds(p["period"].to_numpy()), index=p.index)
    )

    p["dur"] = p["t_end"] - p["t_start"]
    p["shot_sec"] = p["t_first_fga"] - p["t_start"]

    n_all = len(p)
    ok_dur = p["dur"].between(0.0, MAX_POSS_SECONDS)
    ok_shot = p["shot_sec"].between(0.0, MAX_SHOT_SECONDS)
    print(f"      segmented {n_all:,} possessions, "
          f"{(~ok_dur).sum():,} outside 0-{MAX_POSS_SECONDS:.0f}s dropped from "
          f"the duration average")

    p["dur_ok"] = np.where(ok_dur, p["dur"], np.nan)
    p["shot_ok"] = np.where(ok_shot, p["shot_sec"], np.nan)

    out = p.groupby(["game_id", "team_id"], sort=False).agg(
        poss_seg=("poss_id", "size"),
        sum_poss_sec=("dur_ok", "sum"),
        n_poss_sec=("dur_ok", "count"),
        sum_shot_sec=("shot_ok", "sum"),
        n_shot_sec=("shot_ok", "count"),
    ).reset_index()
    return out


# --------------------------------------------------------------------------
# year-to-date accumulation
# --------------------------------------------------------------------------

# counting columns that get carried forward as running season totals
SUM_COLS = [
    "pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb", "dreb", "reb",
    "tov", "ast", "fouls", "stl", "blk", "poss",
    "opp_pts", "opp_fga", "opp_fgm", "opp_fg3a", "opp_fg3m", "opp_fta",
    "opp_ftm", "opp_oreb", "opp_dreb", "opp_reb", "opp_tov", "opp_ast",
    "opp_fouls", "opp_stl", "opp_blk", "opp_poss",
    "game_seconds", "poss_seg", "pts_gap",
    "sum_poss_sec", "n_poss_sec", "sum_shot_sec", "n_shot_sec",
]

# per-100-possession rates: offense over own possessions, defense over
# opponent possessions
PER100_OFF = ["pts", "fga", "fgm", "fg3a", "fg3m", "fta", "ftm", "oreb",
              "dreb", "reb", "tov", "ast", "stl", "blk", "fouls"]


def build_ytd(box: pd.DataFrame) -> pd.DataFrame:
    df = box.sort_values(["season", "team_id", "game_date", "game_id"]) \
            .reset_index(drop=True)

    g = df.groupby(["season", "team_id"], sort=False)
    prior = g[SUM_COLS].cumsum() - df[SUM_COLS]
    prior.columns = [f"c_{c}" for c in SUM_COLS]
    df = pd.concat([df, prior], axis=1)
    df["games_played"] = g.cumcount()

    out = df[["game_id", "season", "game_date", "team", "team_id", "opponent",
              "opp_team_id", "is_home", "games_played"]].copy()

    poss = df["c_poss"].replace(0.0, np.nan)
    opp_poss = df["c_opp_poss"].replace(0.0, np.nan)
    minutes = df["c_game_seconds"] / 60.0
    minutes = minutes.replace(0.0, np.nan)

    # headline efficiency and pace
    out["off_ppp"] = df["c_pts"] / poss
    out["def_ppp"] = df["c_opp_pts"] / opp_poss
    out["net_ppp"] = out["off_ppp"] - out["def_ppp"]
    out["poss_per_48"] = ((poss + opp_poss) / 2.0) / minutes * 48.0

    # timing, possession-weighted across all prior games
    out["sec_per_poss"] = df["c_sum_poss_sec"] / df["c_n_poss_sec"].replace(0.0, np.nan)
    out["sec_to_first_shot"] = df["c_sum_shot_sec"] / df["c_n_shot_sec"].replace(0.0, np.nan)
    out["poss_segmented"] = df["c_poss_seg"]
    out["poss_formula"] = df["c_poss"]

    # Data quality: points the feed never recorded across the prior games this
    # row is built from. pts_missing_pct is the share of the team's true prior
    # scoring that is absent, so off_ppp and every points-based rate on this
    # row are understated by roughly that much.
    out["prior_pts_missing"] = df["c_pts_gap"]
    true_prior_pts = (df["c_pts"] + df["c_pts_gap"]).replace(0.0, np.nan)
    out["pts_missing_pct"] = 100.0 * df["c_pts_gap"] / true_prior_pts

    # shooting and ball control
    out["fg_pct"] = df["c_fgm"] / df["c_fga"].replace(0.0, np.nan)
    out["fg3_pct"] = df["c_fg3m"] / df["c_fg3a"].replace(0.0, np.nan)
    out["ft_pct"] = df["c_ftm"] / df["c_fta"].replace(0.0, np.nan)
    out["efg_pct"] = (df["c_fgm"] + 0.5 * df["c_fg3m"]) / df["c_fga"].replace(0.0, np.nan)
    out["fg3a_rate"] = df["c_fg3a"] / df["c_fga"].replace(0.0, np.nan)
    out["ft_rate"] = df["c_fta"] / df["c_fga"].replace(0.0, np.nan)
    out["tov_pct"] = 100.0 * df["c_tov"] / poss

    out["opp_fg_pct"] = df["c_opp_fgm"] / df["c_opp_fga"].replace(0.0, np.nan)
    out["opp_fg3_pct"] = df["c_opp_fg3m"] / df["c_opp_fg3a"].replace(0.0, np.nan)
    out["opp_ft_pct"] = df["c_opp_ftm"] / df["c_opp_fta"].replace(0.0, np.nan)
    out["opp_efg_pct"] = (df["c_opp_fgm"] + 0.5 * df["c_opp_fg3m"]) \
        / df["c_opp_fga"].replace(0.0, np.nan)
    out["opp_fg3a_rate"] = df["c_opp_fg3a"] / df["c_opp_fga"].replace(0.0, np.nan)
    out["opp_ft_rate"] = df["c_opp_fta"] / df["c_opp_fga"].replace(0.0, np.nan)
    out["opp_tov_pct"] = 100.0 * df["c_opp_tov"] / opp_poss

    # rebounding shares
    oreb_chances = (df["c_oreb"] + df["c_opp_dreb"]).replace(0.0, np.nan)
    dreb_chances = (df["c_dreb"] + df["c_opp_oreb"]).replace(0.0, np.nan)
    out["oreb_pct"] = df["c_oreb"] / oreb_chances
    out["dreb_pct"] = df["c_dreb"] / dreb_chances
    out["reb_pct"] = df["c_reb"] / (df["c_reb"] + df["c_opp_reb"]).replace(0.0, np.nan)

    # per-100 rates
    for c in PER100_OFF:
        out[f"off_{c}_per100"] = 100.0 * df[f"c_{c}"] / poss
        out[f"def_{c}_per100"] = 100.0 * df[f"c_opp_{c}"] / opp_poss

    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root",
                    default=os.environ.get("NBA_DATA_ROOT", DEFAULT_DATA_ROOT))
    ap.add_argument("--pbp-dir", default=os.environ.get("NBA_PBP_DIR"),
                    help="directory holding pbp_<season>.parquet; "
                         "auto-discovered under the data root if omitted")
    ap.add_argument("--out-dir", default="features")
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=[2021, 2022, 2023, 2024, 2025, 2026],
                    help="explicit list of ESPN season numbers (ENDING year)")
    args = ap.parse_args()

    root = Path(args.data_root)
    pbp_dir = resolve_pbp_dir(root, args.pbp_dir)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"data root: {root}")
    print(f"pbp dir:   {pbp_dir}")
    print(f"out dir:   {out_dir}")

    for season in sorted(set(args.seasons)):
        df = load_pbp(pbp_dir, season)
        box = team_game_box(df)
        timing = possession_timing(df)
        box = box.merge(timing, on=["game_id", "team_id"], how="left")
        for c in ("poss_seg", "sum_poss_sec", "n_poss_sec",
                  "sum_shot_sec", "n_shot_sec"):
            box[c] = box[c].fillna(0.0)

        # Reconcile the rebuilt points against the official final scores. If
        # this drifts from 1.000 the play tagging has stopped matching the
        # feed's wording and every rate in the file is suspect.
        chk = box[["game_id", "pts", "team_final"]].copy()
        agg = chk.groupby("game_id").agg(built=("pts", "sum"),
                                         official=("team_final", "sum"))
        ratio = (agg["built"] / agg["official"].replace(0.0, np.nan))
        exact = float((agg["built"] == agg["official"]).mean())
        print(f"      scoring check: built/official = {ratio.mean():.4f}, "
              f"{exact:.1%} of games match exactly")
        # FG% and blocks per team-game. League FG% sits in the .440-.480 band
        # and blocks land near 4.5-5.5 per team-game. FG% far below that band
        # with blocks far above it would mean blocked shots are being counted
        # twice in this season's file.
        n_tg = max(len(box), 1)
        print(f"      shooting check: fg%={box['fgm'].sum() / max(box['fga'].sum(), 1):.3f}, "
              f"blk/team-game={box['blk'].sum() / n_tg:.2f}, "
              f"fga/team-game={box['fga'].sum() / n_tg:.1f}")

        ytd = build_ytd(box)
        path = out_dir / f"nba_team_ytd_{season}.parquet"
        ytd.to_parquet(path, index=False)

        played = ytd[ytd["games_played"] > 0]
        print(f"    wrote {path.name}: {len(ytd):,} team-games, "
              f"{ytd['team'].nunique()} teams, "
              f"{len(ytd) - len(played):,} season-opening rows with no history")
        if len(played):
            print(f"      mean off_ppp={played['off_ppp'].mean():.4f}  "
                  f"poss/48={played['poss_per_48'].mean():.2f}  "
                  f"sec/poss={played['sec_per_poss'].mean():.2f}  "
                  f"sec to 1st shot={played['sec_to_first_shot'].mean():.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
