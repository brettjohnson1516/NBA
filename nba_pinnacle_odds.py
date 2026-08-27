#!/usr/bin/env python
"""
Stage 3b - pull Pinnacle closing moneylines from The Odds API historical
snapshots, for the seasons SBR does not cover and any season you want on
Pinnacle rather than the composite.

WHY THIS EXISTS
---------------
The SBR archive stops at the 2021-22 season and is a multi-book consensus, not
Pinnacle. The Odds API carries Pinnacle back to 2020-06-06, so ESPN seasons
2021 onward can use the sharper source. Below that, SBR is all there is.

THE SNAPSHOT GRID - WHY YOU CANNOT HAVE "ONE MINUTE BEFORE TIP"
--------------------------------------------------------------
Historical odds are stored as periodic snapshots, not as a continuous feed:
10-minute intervals from June 2020, tightening to 5 minutes from September
2022. A request returns the closest snapshot AT OR BEFORE the timestamp you
ask for.

So a true one-minute-before-tip price does not exist in this data. What you
actually get is the last snapshot before tip, which is somewhere between 0 and
10 minutes stale depending on the era. This script asks for tip-off minus 60
seconds and records, per game, the real staleness in `snapshot_lag_seconds`,
so the feature can be trusted or filtered on rather than assumed to be a true
close.

QUOTA - READ THIS BEFORE RUNNING
--------------------------------
Historical calls are metered and the per-call cost depends on your plan and on
markets x regions. This script does NOT assume a cost. Instead:

  * `--dry-run` plans the whole pull and prints how many requests it would
    make, per season, WITHOUT spending anything.
  * `--probe` makes exactly ONE request, dumps the raw JSON shape and the
    quota headers, and stops.
  * `--max-requests N` is a hard stop.
  * Every run prints x-requests-remaining before and after.

Snapshots are shared: every game tipping at 7:00pm ET appears in the same
snapshot, so the plan groups games by their target snapshot and asks once.
That is the difference between roughly 200 requests a season and 1,300.

Each snapshot is cached to disk as raw JSON, so a re-run costs nothing and an
interrupted run resumes.

TEAM NAMES
----------
The Odds API uses full names ("Golden State Warriors"); ESPN uses its own
abbreviations ("GS"). The map below covers the current 30. Any name that does
not map is reported by count and example rather than dropped silently.

Output: <outdir>/odds/closing_lines_pinnacle.parquet

  season, game_date, home_team, away_team, espn_game_id
  home_close_ml, away_close_ml     American odds
  p_home_close                     devigged, proportional
  overround_close
  snapshot_time                    the snapshot actually used
  commence_time                    ESPN tip-off
  snapshot_lag_seconds             tip-off minus snapshot; how stale the price is
  line_source                      always 'pinnacle'

Setup
-----
  pip install pandas pyarrow requests

Usage
-----
  python nba_pinnacle_odds.py --seasons 2021 2026 --outdir data --dry-run
  python nba_pinnacle_odds.py --seasons 2021 2026 --outdir data --probe
  python nba_pinnacle_odds.py --seasons 2021 2026 --outdir data --max-requests 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

SCHEDULE_URL = ("https://github.com/sportsdataverse/sportsdataverse-data/releases"
                "/download/espn_nba_schedules/nba_schedule_{season}.parquet")

# The Odds API full name -> ESPN abbreviation.
TEAM_MAP = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GS", "Houston Rockets": "HOU",
    "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO",
    "New York Knicks": "NY", "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SA", "Toronto Raptors": "TOR", "Utah Jazz": "UTAH",
    "Washington Wizards": "WSH",
}

NON_TEAM_CODES = {"EAST", "WEST", "DUR", "LEB", "GIA", "USA", "WLD"}

# The snapshot grid is DISCOVERED, not assumed.
#
# The published description - 10-minute snapshots from June 2020, 5-minute from
# September 2022 - reads as though they land on :00, :10, :20. They do not. A
# probe at 2020-12-22T23:50Z came back with the 23:45 snapshot, previous 23:35,
# next 23:55: 10-minute spacing PHASED five minutes off the hour.
#
# Flooring to an assumed :00/:10 grid therefore asks for timestamps that never
# exist. The API quietly serves the preceding snapshot instead, so every price
# lands up to a full step staler than it needed to be, and games that share a
# real snapshot get split across separate requests.
#
# One cheap request per season reads `timestamp` and `previous_timestamp` off
# the response and derives the real spacing and phase from them.
DEFAULT_STEP = 300


def floor_to_grid(when: datetime, step: int, phase: int) -> datetime:
    epoch = int(when.timestamp())
    return datetime.fromtimestamp(
        epoch - ((epoch - phase) % step), tz=timezone.utc
    )


def american_to_prob(o) -> float:
    o = float(o)
    return (-o) / ((-o) + 100.0) if o < 0 else 100.0 / (o + 100.0)


def load_schedule(season: int, raw_dir: Path) -> Optional[pd.DataFrame]:
    path = raw_dir / f"nba_schedule_{season}.parquet"
    if not path.exists():
        r = requests.get(SCHEDULE_URL.format(season=season), timeout=180)
        if r.status_code != 200:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    df = pd.read_parquet(path, columns=["game_id", "game_date", "game_date_time",
                                        "home_abbreviation", "away_abbreviation"])
    df = df[~df["home_abbreviation"].isin(NON_TEAM_CODES)
            & ~df["away_abbreviation"].isin(NON_TEAM_CODES)].copy()
    # The schedule calls these *_abbreviation; everything downstream, including
    # the merge against the extracted Pinnacle rows, uses home_team/away_team.
    df = df.rename(columns={"home_abbreviation": "home_team",
                            "away_abbreviation": "away_team"})
    df["game_id"] = df["game_id"].astype(str)
    df["season"] = season
    # game_date_time carries a UTC offset; normalize to UTC.
    df["commence_time"] = pd.to_datetime(df["game_date_time"], utc=True)
    df = df[df["commence_time"].notna()]
    return df


def fetch_snapshot(when: datetime, api_key: str, cache_dir: Path,
                   session: requests.Session,
                   stats: dict) -> Optional[dict]:
    """One historical snapshot, cached to disk so a re-run is free."""
    cache = cache_dir / f"{when.strftime('%Y%m%dT%H%M%SZ')}.json"
    if cache.exists() and cache.stat().st_size > 0:
        stats["cached"] += 1
        try:
            return json.loads(cache.read_text())
        except json.JSONDecodeError:
            cache.unlink()

    params = {
        "apiKey": api_key,
        "regions": "us,eu",
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
        "bookmakers": "pinnacle",
        "date": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for attempt in range(5):
        try:
            r = session.get(f"{API_BASE}/historical/sports/{SPORT}/odds",
                            params=params, timeout=60)
        except requests.RequestException as e:
            time.sleep(min(30.0, 2 ** attempt))
            stats["errors"] += 1
            if attempt == 4:
                print(f"  request failed for {when}: {e}")
            continue

        stats["requests"] += 1
        rem = r.headers.get("x-requests-remaining")
        if rem is not None:
            stats["remaining"] = rem
        used = r.headers.get("x-requests-used")
        if used is not None:
            stats["used"] = used

        if r.status_code == 401:
            print(f"  HTTP 401 - the key was rejected: {r.text[:200]}")
            return None
        if r.status_code == 422:
            print(f"  HTTP 422 for {when}: {r.text[:200]}")
            return None
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(30.0, 2 ** attempt))
            continue
        if r.status_code != 200:
            print(f"  HTTP {r.status_code} for {when}: {r.text[:200]}")
            return None

        try:
            payload = r.json()
        except ValueError:
            return None
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
        return payload
    return None


def calibrate_grid(sample: datetime, api_key: str, cache_dir: Path,
                   session: requests.Session, stats: dict) -> tuple[int, int]:
    """
    Discover (step_seconds, phase_seconds) for the snapshot grid near `sample`.

    `previous_timestamp` and `next_timestamp` bracket the served snapshot, so
    the gap between them is the real spacing and the served timestamp modulo
    that spacing is the phase. Falls back to a 5-minute zero-phase grid when
    the response carries neither neighbour - requests then still resolve
    backwards to the nearest real snapshot, just less efficiently.
    """
    payload = fetch_snapshot(sample, api_key, cache_dir, session, stats)
    if payload is None:
        return DEFAULT_STEP, 0

    ts = payload.get("timestamp")
    if not ts:
        return DEFAULT_STEP, 0
    t = pd.Timestamp(ts).to_pydatetime()

    step = None
    for other in (payload.get("previous_timestamp"), payload.get("next_timestamp")):
        if other:
            d = abs(int((pd.Timestamp(other).to_pydatetime() - t).total_seconds()))
            if d > 0:
                step = d if step is None else min(step, d)
    if not step:
        step = DEFAULT_STEP

    # Snap to the nearest real cadence. Measured gaps come back a second or two
    # off - 298 rather than 300 - and an unsnapped value makes the derived
    # phase drift between runs, which changes every requested timestamp and
    # therefore misses the entire snapshot cache. A re-run then costs full
    # price instead of nothing.
    step = min((300, 600), key=lambda c: abs(c - step))

    # Round the phase to the nearest minute for the same reason.
    phase = int(t.timestamp()) % step
    phase = int(round(phase / 60.0) * 60) % step
    return step, phase


def extract_pinnacle(payload: dict, report: dict) -> list[dict]:
    """
    Pull Pinnacle h2h prices out of one snapshot.

    The payload wraps the standard odds array in snapshot metadata, so the
    games are under `data`, and `timestamp` is the snapshot actually served -
    which is at or before what was asked for, never after.
    """
    snap_ts = payload.get("timestamp")
    games = payload.get("data") or []
    out = []
    for ev in games:
        home_raw, away_raw = ev.get("home_team"), ev.get("away_team")
        home = TEAM_MAP.get(home_raw)
        away = TEAM_MAP.get(away_raw)
        if home is None or away is None:
            report["unmapped_names"].add(
                home_raw if home is None else away_raw
            )
            continue

        price = {}
        for bk in ev.get("bookmakers") or []:
            if bk.get("key") != "pinnacle":
                continue
            for mk in bk.get("markets") or []:
                if mk.get("key") != "h2h":
                    continue
                for oc in mk.get("outcomes") or []:
                    nm = TEAM_MAP.get(oc.get("name"))
                    if nm is not None and oc.get("price") is not None:
                        price[nm] = float(oc["price"])
        if home not in price or away not in price:
            report["no_pinnacle_price"] += 1
            continue

        out.append({
            "snapshot_time": snap_ts,
            "commence_time_api": ev.get("commence_time"),
            "home_team": home,
            "away_team": away,
            "home_close_ml": price[home],
            "away_close_ml": price[away],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=None,
                    help="Falls back to the ODDS_API_KEY env var.")
    ap.add_argument("--seasons", nargs=2, type=int, metavar=("FIRST", "LAST"),
                    default=[2021, 2026],
                    help="ESPN season numbers (ENDING year), inclusive. "
                         "Pinnacle history starts 2020-06-06, so 2021 is the "
                         "earliest fully covered season.")
    ap.add_argument("--outdir", type=Path, default=Path("data"))
    ap.add_argument("--lead-seconds", type=int, default=60,
                    help="Ask for the snapshot this many seconds before tip-off. "
                         "Default 60.")
    ap.add_argument("--max-lag-seconds", type=int, default=1800,
                    help="Reject a matched snapshot staler than this. Default 1800.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan the pull and print the request count per season. "
                         "Spends nothing.")
    ap.add_argument("--probe", action="store_true",
                    help="Make exactly ONE request, print the raw JSON shape "
                         "and quota headers, then stop.")
    ap.add_argument("--max-requests", type=int, default=0,
                    help="Hard stop after N requests. 0 means no cap.")
    ap.add_argument("--sleep", type=float, default=0.2)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("ODDS_API_KEY", "")
    if not api_key and not args.dry_run:
        print("Need --api-key or the ODDS_API_KEY env var.", file=sys.stderr)
        return 1

    raw_dir = args.outdir / "odds_raw"
    cache_dir = raw_dir / "pinnacle_snapshots"
    out_dir = args.outdir / "odds"
    out_dir.mkdir(parents=True, exist_ok=True)

    first, last = args.seasons

    # -- plan ----------------------------------------------------------------
    games = []
    for season in range(first, last + 1):
        sched = load_schedule(season, raw_dir)
        if sched is None:
            print(f"  season {season}: no schedule file")
            continue
        games.append(sched)
    if not games:
        print("No schedules loaded.", file=sys.stderr)
        return 1
    sched = pd.concat(games, ignore_index=True)

    sched["target"] = (sched["commence_time"]
                       - pd.Timedelta(seconds=args.lead_seconds))

    # Calibrate the snapshot grid per season before planning. Costs one
    # request per season and it is cached, so the second run is free.
    session = requests.Session()
    stats = {"requests": 0, "cached": 0, "errors": 0,
             "remaining": "?", "used": "?"}

    grids: dict[int, tuple[int, int]] = {}
    if args.dry_run or not api_key:
        for season in sorted(sched["season"].unique()):
            grids[int(season)] = (DEFAULT_STEP, 0)
        print(f"Dry run: assuming a {DEFAULT_STEP}s grid with no phase. The "
              f"real grid is measured from the API when actually fetching, so "
              f"the live counts can differ from these.")
    else:
        print("Calibrating the snapshot grid (one request per season):")
        for season in sorted(sched["season"].unique()):
            sample = sched.loc[sched["season"] == season, "target"].min()
            step, phase = calibrate_grid(
                pd.Timestamp(sample).to_pydatetime(), api_key,
                cache_dir, session, stats
            )
            grids[int(season)] = (step, phase)
            print(f"  {int(season)}: {step}s spacing, "
                  f"phase +{phase}s ({phase // 60}m past the step)")

    sched["snapshot_req"] = [
        floor_to_grid(pd.Timestamp(t).to_pydatetime(), *grids[int(s)])
        for t, s in zip(sched["target"], sched["season"])
    ]

    plan = sched.groupby("season")["snapshot_req"].nunique()
    total_snaps = sched["snapshot_req"].nunique()

    print("Plan - distinct snapshots to request (games sharing a tip time "
          "share one request):")
    print(f"  {'season':>7} {'games':>7} {'snapshots':>10}")
    for season, n in plan.items():
        print(f"  {int(season):>7} {int((sched['season'] == season).sum()):>7,} {int(n):>10,}")
    print(f"  {'TOTAL':>7} {len(sched):>7,} {total_snaps:>10,}")
    print("\nThat is the REQUEST count, not the credit cost. Historical calls "
          "are metered per markets x regions and the multiplier depends on "
          "your plan - the run prints your remaining balance before and after "
          "so you can see the real cost.")

    if args.dry_run:
        return 0

    report = {"unmapped_names": set(), "no_pinnacle_price": 0,
              "no_snapshot": 0, "too_stale": 0,
              "no_snapshot_before_tip": 0}

    # -- probe ---------------------------------------------------------------
    if args.probe:
        when = sorted(sched["snapshot_req"].unique())[0]
        when = pd.Timestamp(when).to_pydatetime()
        print(f"\nProbe: one request at {when}")
        payload = fetch_snapshot(when, api_key, cache_dir, session, stats)
        if payload is None:
            print("  no payload returned")
            return 1
        print(f"  top-level keys : {list(payload.keys())}")
        print(f"  timestamp      : {payload.get('timestamp')}")
        print(f"  previous       : {payload.get('previous_timestamp')}")
        print(f"  next           : {payload.get('next_timestamp')}")
        data = payload.get("data") or []
        print(f"  games in data  : {len(data)}")
        if data:
            print("\n  first event:")
            print(json.dumps(data[0], indent=2)[:2000])
        print(f"\n  x-requests-remaining : {stats['remaining']}")
        print(f"  x-requests-used      : {stats['used']}")
        return 0

    # -- fetch ---------------------------------------------------------------
    targets = sorted(pd.Timestamp(t).to_pydatetime()
                     for t in sched["snapshot_req"].unique())
    rows: list[dict] = []
    for i, when in enumerate(targets, 1):
        if args.max_requests and stats["requests"] >= args.max_requests:
            print(f"\nStopping: hit --max-requests {args.max_requests}. "
                  f"Re-run to continue; snapshots already fetched are cached.")
            break
        payload = fetch_snapshot(when, api_key, cache_dir, session, stats)
        if payload is None:
            report["no_snapshot"] += 1
            continue
        rows.extend(extract_pinnacle(payload, report))
        if i % 50 == 0 or i == len(targets):
            print(f"  [{i}/{len(targets)}] requests={stats['requests']:,} "
                  f"cached={stats['cached']:,} rows={len(rows):,} "
                  f"remaining={stats['remaining']}")
        time.sleep(args.sleep)

    if not rows:
        print("No Pinnacle prices extracted.", file=sys.stderr)
        return 1

    odds = pd.DataFrame(rows)
    odds["snapshot_time"] = pd.to_datetime(odds["snapshot_time"], utc=True)
    odds["commence_time_api"] = pd.to_datetime(odds["commence_time_api"], utc=True)

    # A snapshot contains every game live at that moment, including ones that
    # tip hours later, so the same game appears in many snapshots. Keep, per
    # game, the LATEST snapshot at or before its own tip-off.
    merged = sched.merge(odds, on=["home_team", "away_team"], how="left")
    merged = merged[merged["snapshot_time"].notna()]
    merged["lag"] = (merged["commence_time"]
                     - merged["snapshot_time"]).dt.total_seconds()
    merged = merged[merged["lag"] >= 0]

    n_with_snapshot = merged["game_id"].nunique()
    report["no_snapshot_before_tip"] = int(
        sched["game_id"].nunique() - n_with_snapshot
    )
    merged = merged.sort_values("lag").groupby("game_id", as_index=False).first()

    stale = merged["lag"] > args.max_lag_seconds
    report["too_stale"] = int(stale.sum())
    merged = merged[~stale]

    p_home = merged["home_close_ml"].map(american_to_prob)
    p_away = merged["away_close_ml"].map(american_to_prob)
    tot = p_home + p_away
    merged["p_home_close"] = np.where(tot > 0, p_home / tot, np.nan)
    merged["overround_close"] = tot - 1.0
    merged["line_source"] = "pinnacle"

    out = merged.rename(columns={"game_id": "espn_game_id",
                                 "lag": "snapshot_lag_seconds"})
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    out = out[["season", "game_date", "home_team", "away_team", "espn_game_id",
               "home_close_ml", "away_close_ml", "p_home_close",
               "overround_close", "snapshot_time", "commence_time",
               "snapshot_lag_seconds", "line_source"]]

    out_path = out_dir / "closing_lines_pinnacle.parquet"
    out.to_parquet(out_path, index=False)

    # -- report --------------------------------------------------------------
    print(f"\nWrote {out_path}: {len(out):,} games")
    print(f"  requests made        : {stats['requests']:,}")
    print(f"  snapshots from cache : {stats['cached']:,}")
    print(f"  quota remaining      : {stats['remaining']}")
    print(f"  quota used           : {stats['used']}")

    print("\nCoverage against the ESPN schedule:")
    print(f"  {'season':>7} {'games':>7} {'priced':>7} {'rate':>7}")
    for season in sorted(sched["season"].unique()):
        n_all = int((sched["season"] == season).sum())
        n_ok = int((out["season"] == season).sum())
        print(f"  {int(season):>7} {n_all:>7,} {n_ok:>7,} "
              f"{n_ok / max(1, n_all):>6.1%}")

    print("\nSnapshot staleness (tip-off minus snapshot, seconds):")
    print(out["snapshot_lag_seconds"].describe(
        percentiles=[0.5, 0.9, 0.99]).round(0).to_string())
    print("\nOverround:")
    print(out["overround_close"].describe(
        percentiles=[0.05, 0.5, 0.95]).round(4).to_string())
    print(f"\nMean devigged P(home): {out['p_home_close'].mean():.4f}")

    print("\nIssues:")
    print(f"  games matched to no snapshot at or before tip : "
          f"{report['no_snapshot_before_tip']:,}")
    print(f"  dropped for a snapshot staler than "
          f"{args.max_lag_seconds}s : {report['too_stale']:,}")
    print(f"  events with no Pinnacle h2h price             : "
          f"{report['no_pinnacle_price']:,}")
    if report["unmapped_names"]:
        print(f"  UNMAPPED TEAM NAMES: {sorted(report['unmapped_names'])}")
        print("  Add these to TEAM_MAP - games involving them were skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
