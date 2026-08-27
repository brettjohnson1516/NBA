#!/usr/bin/env python
"""
Stage 3c - merge the two closing-line sources into the single pregame prior
table the model trains on.

Pinnacle wins wherever it exists; SBR fills in below 2020-06-06 and anywhere a
Pinnacle pull has not been run. Every row keeps `line_source`, so the model can
tell the two apart instead of treating a multi-book consensus and a Pinnacle
close as the same measurement. They are not: SBR's median overround is about
3.8% against Pinnacle's typical 2-3%, and a softer, more heavily vigged
composite carries different information about a game than a sharp close does.
Blending them into one anonymous number is the mistake this column exists to
prevent.

`has_line` is emitted explicitly so downstream code cannot quietly read a
missing prior as a pick'em. A game with no closing line is not a 50/50 game -
it is a game where the prior is unknown, and the model has to be told which.

Inputs (either may be absent):
  <outdir>/odds/closing_lines_sbr.parquet
  <outdir>/odds/closing_lines_pinnacle.parquet

Output:
  <outdir>/odds/closing_lines.parquet   one row per ESPN game_id

  season, game_date, home_team, away_team, espn_game_id
  home_close_ml, away_close_ml, p_home_close, overround_close
  line_source     'pinnacle' | 'sbr_composite'
  has_line        False when neither source priced the game
  snapshot_lag_seconds   Pinnacle only; null for SBR rows

Setup
-----
  pip install pandas pyarrow

Usage
-----
  python nba_merge_odds.py --outdir data
  python nba_merge_odds.py --outdir data --seasons 2012 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

NON_TEAM_CODES = {"EAST", "WEST", "DUR", "LEB", "GIA", "USA", "WLD"}

COLS = ["season", "game_date", "home_team", "away_team", "espn_game_id",
        "home_close_ml", "away_close_ml", "p_home_close", "overround_close",
        "line_source", "snapshot_lag_seconds"]


def load(path: Path, label: str) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  {label:9s} not found at {path} - skipping")
        return None
    df = pd.read_parquet(path)
    if "snapshot_lag_seconds" not in df.columns:
        df["snapshot_lag_seconds"] = pd.NA
    df = df[df["espn_game_id"].notna()].copy()
    df["espn_game_id"] = df["espn_game_id"].astype(str)
    print(f"  {label:9s} {len(df):,} games, "
          f"seasons {int(df['season'].min())}-{int(df['season'].max())}, "
          f"{int(df['p_home_close'].notna().sum()):,} priced")
    return df[[c for c in COLS if c in df.columns]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    default=None,
                    help="Optional ESPN season range to restrict the output to.")
    args = ap.parse_args()

    odds_dir = args.outdir / "odds"
    raw_dir = args.outdir / "odds_raw"

    print("Sources:")
    sbr = load(odds_dir / "closing_lines_sbr.parquet", "SBR")
    pin = load(odds_dir / "closing_lines_pinnacle.parquet", "Pinnacle")
    if sbr is None and pin is None:
        print("Neither source is present.", file=sys.stderr)
        return 1

    frames = [f for f in (pin, sbr) if f is not None]
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["p_home_close"].notna()]

    # Pinnacle first, so drop_duplicates keeps it over SBR for the same game.
    combined["_rank"] = (combined["line_source"] == "pinnacle").map(
        {True: 0, False: 1}
    )
    combined = combined.sort_values(["espn_game_id", "_rank"])
    n_before = len(combined)
    merged = combined.drop_duplicates(subset=["espn_game_id"], keep="first")
    overlap = n_before - len(merged)
    merged = merged.drop(columns="_rank")

    # Every scheduled game, so games with no line are present and flagged
    # rather than silently absent.
    seasons = sorted(merged["season"].dropna().astype(int).unique())
    if args.seasons:
        seasons = [s for s in range(args.seasons[0], args.seasons[1] + 1)]

    sched_frames = []
    for season in seasons:
        p = raw_dir / f"nba_schedule_{season}.parquet"
        if not p.exists():
            print(f"  no schedule file for {season}; games without a line "
                  f"cannot be listed for that season")
            continue
        s = pd.read_parquet(p, columns=["game_id", "game_date",
                                        "home_abbreviation", "away_abbreviation"])
        s = s[~s["home_abbreviation"].isin(NON_TEAM_CODES)
              & ~s["away_abbreviation"].isin(NON_TEAM_CODES)]
        s = s.rename(columns={"game_id": "espn_game_id",
                              "home_abbreviation": "home_team",
                              "away_abbreviation": "away_team"})
        s["espn_game_id"] = s["espn_game_id"].astype(str)
        s["game_date"] = pd.to_datetime(s["game_date"]).dt.date
        s["season"] = season
        sched_frames.append(s)

    if sched_frames:
        sched = pd.concat(sched_frames, ignore_index=True)
        priced = merged.drop(columns=["season", "game_date",
                                      "home_team", "away_team"])
        out = sched.merge(priced, on="espn_game_id", how="left")
    else:
        out = merged.copy()

    out["has_line"] = out["p_home_close"].notna()
    if args.seasons:
        out = out[out["season"].between(args.seasons[0], args.seasons[1])]

    out = out.sort_values(["game_date", "espn_game_id"]).reset_index(drop=True)
    out_path = odds_dir / "closing_lines.parquet"
    out.to_parquet(out_path, index=False)

    # -- report --------------------------------------------------------------
    print(f"\nWrote {out_path}: {len(out):,} games")
    print(f"  games priced by both sources (Pinnacle kept) : {overlap:,}")

    print("\nCoverage by season:")
    print(f"  {'season':>7} {'games':>7} {'pinnacle':>9} {'sbr':>7} "
          f"{'no line':>8} {'covered':>8}")
    for season in sorted(out["season"].dropna().unique()):
        sub = out[out["season"] == season]
        n_pin = int((sub["line_source"] == "pinnacle").sum())
        n_sbr = int((sub["line_source"] == "sbr_composite").sum())
        n_none = int((~sub["has_line"]).sum())
        print(f"  {int(season):>7} {len(sub):>7,} {n_pin:>9,} {n_sbr:>7,} "
              f"{n_none:>8,} {(len(sub) - n_none) / max(1, len(sub)):>7.1%}")

    print("\nOverround by source:")
    print(out.groupby("line_source")["overround_close"]
          .describe(percentiles=[0.5]).round(4).to_string())

    print("\nMean devigged P(home) by source:")
    print(out.groupby("line_source")["p_home_close"].agg(["size", "mean"])
          .round(4).to_string())

    gaps = out[~out["has_line"]]
    if len(gaps):
        print(f"\n{len(gaps):,} games have no closing line. Downstream code "
              f"must treat has_line=False as UNKNOWN, not as 0.5.")
        by_season = gaps.groupby("season").size()
        for s, n in by_season.items():
            print(f"    {int(s)}: {n:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
