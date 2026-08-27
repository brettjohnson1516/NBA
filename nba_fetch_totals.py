"""
nba_fetch_totals.py
-------------------
Pulls Pinnacle closing OVER/UNDER totals and writes
<odds-dir>\\closing_totals_pinnacle.parquet, keyed by espn_game_id so it joins
straight onto the feature tables (nba_wp_features.py already globs the odds
directory for *total*.parquet and reads espn_game_id + total_close).

Design notes
------------
* Games and snapshot times come from your EXISTING closing_lines_pinnacle.parquet
  rather than being re-derived. That guarantees the total is priced at the same
  moment as the moneyline close, and it means no separate schedule matching.

* The historical endpoint returns a snapshot of EVERY event at one timestamp, so
  games sharing a snapshot minute cost one call between them. Snapshots are
  rounded to --snap-round minutes to increase sharing.

* Every API response is cached to <odds-dir>\\totals_cache\\. Re-runs read the
  cache and cost nothing, so an interrupted run resumes for free.

* Output is MERGED with any existing closing_totals_pinnacle.parquet, not
  overwritten. Running for one season will not wipe the others.

Requires an Odds API key in the ODDS_API_KEY environment variable:
  $env:ODDS_API_KEY = "<your-key>"

Usage (PowerShell):
  python nba_fetch_totals.py --dry-run
  python nba_fetch_totals.py --seasons 2026
  python nba_fetch_totals.py --seasons 2021 2022 2023 2024 2025 2026
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

DEFAULT_DATA_ROOT = r"C:\Users\saint\OneDrive\Documents\NBA_AUG_2026\data"
API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"

# Odds API full names -> the codes used in the pbp / closing-line files
TEAM_TO_CODE = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GS", "Houston Rockets": "HOU",
    "Indiana Pacers": "IND", "Los Angeles Clippers": "LAC",
    "LA Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NO", "New York Knicks": "NY",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI", "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SA", "Toronto Raptors": "TOR",
    "Utah Jazz": "UTAH", "Washington Wizards": "WSH",
}


def resolve_odds_dir(root: Path, override):
    if override:
        p = Path(override)
        if not p.is_absolute():
            p = root / p
        if not p.is_dir():
            raise FileNotFoundError(f"--odds-dir not a directory: {p}")
        return p
    cands = [d for d in sorted(root.iterdir())
             if d.is_dir() and "odds" in d.name.lower()
             and list(d.glob("*closing*.parquet"))]
    if not cands:
        raise FileNotFoundError(
            f"no directory under {root} matching *odds* holds a "
            f"*closing*.parquet; pass --odds-dir")
    if len(cands) > 1:
        print(f"  multiple odds dirs {[c.name for c in cands]}, "
              f"using {cands[0].name}")
    return cands[0]


def load_games(odds_dir: Path, seasons):
    cands = (sorted(odds_dir.glob("*closing*pinnacle*.parquet"))
             or sorted(odds_dir.glob("*closing*.parquet")))
    src = cands[0]
    df = pd.read_parquet(src)
    print(f"  game index: {src.name}  {len(df):,} rows")
    need = {"espn_game_id", "season", "home_team", "away_team",
            "commence_time", "snapshot_time"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"{src.name} missing columns: {sorted(missing)}")
    df = df[df["season"].isin(seasons)].copy()
    df["espn_game_id"] = df["espn_game_id"].astype(str)
    for c in ("commence_time", "snapshot_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def snap_key(ts, round_min):
    ts = ts.floor(f"{round_min}min")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_snapshot(key, cache_dir: Path, api_key, bookmaker, regions,
                   timeout, retries, sleep_s):
    """One historical snapshot of all NBA totals. Cached to disk."""
    cpath = cache_dir / f"{key.replace(':', '')}.json"
    if cpath.exists():
        with open(cpath) as f:
            return json.load(f), True

    url = f"{API_BASE}/historical/sports/{SPORT}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "totals",
        "oddsFormat": "american",
        "date": key,
    }
    if bookmaker:
        params["bookmakers"] = bookmaker

    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except requests.RequestException as e:
            last = str(e)
            time.sleep(sleep_s * (attempt + 1))
            continue
        if r.status_code == 200:
            data = r.json()
            with open(cpath, "w") as f:
                json.dump(data, f)
            rem = r.headers.get("x-requests-remaining")
            return data, False if rem is None else rem
        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}"
            time.sleep(sleep_s * (attempt + 1))
            continue
        raise SystemExit(f"API error {r.status_code} on {key}: {r.text[:300]}")
    raise SystemExit(f"failed on {key} after {retries} attempts: {last}")


def extract(data, home_code, away_code, commence, tol_minutes):
    """Find this game in a snapshot and pull its total line and prices."""
    events = data.get("data", data if isinstance(data, list) else [])
    for ev in events:
        h = TEAM_TO_CODE.get(ev.get("home_team"))
        a = TEAM_TO_CODE.get(ev.get("away_team"))
        if h != home_code or a != away_code:
            continue
        ct = pd.to_datetime(ev.get("commence_time"), utc=True, errors="coerce")
        if pd.notna(ct) and abs((ct - commence).total_seconds()) > tol_minutes * 60:
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "totals":
                    continue
                over = under = point = None
                for o in mk.get("outcomes", []):
                    if o.get("name") == "Over":
                        over, point = o.get("price"), o.get("point")
                    elif o.get("name") == "Under":
                        under = o.get("price")
                        point = point if point is not None else o.get("point")
                if point is not None:
                    return {
                        "total_close": float(point),
                        "over_price": over,
                        "under_price": under,
                        "book": bk.get("key"),
                        "odds_api_event_id": ev.get("id"),
                    }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root",
                    default=os.environ.get("NBA_DATA_ROOT", DEFAULT_DATA_ROOT))
    ap.add_argument("--odds-dir", default=os.environ.get("NBA_ODDS_DIR"))
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=[2021, 2022, 2023, 2024, 2025, 2026])
    ap.add_argument("--bookmaker", default="pinnacle",
                    help="empty string fetches all books")
    ap.add_argument("--regions", default="eu")
    ap.add_argument("--snap-round", type=int, default=5,
                    help="round snapshot times to N minutes so nearby games "
                         "share one API call")
    ap.add_argument("--tol-minutes", type=int, default=90,
                    help="commence_time tolerance when matching an event")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N API calls (0 = no limit)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report how many snapshots and calls are needed, "
                         "then exit without calling the API")
    args = ap.parse_args()

    root = Path(args.data_root)
    odds_dir = resolve_odds_dir(root, args.odds_dir)
    cache_dir = odds_dir / "totals_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = odds_dir / "closing_totals_pinnacle.parquet"

    seasons = sorted(set(args.seasons))
    print(f"data root: {root}\nodds dir:  {odds_dir}\nseasons:   {seasons}")

    games = load_games(odds_dir, seasons)
    games["snap"] = games["snapshot_time"].map(
        lambda t: snap_key(t, args.snap_round))
    print(f"  {len(games):,} games, {games['snap'].nunique():,} distinct "
          f"snapshots after rounding to {args.snap_round}min")

    cached = {p.stem for p in cache_dir.glob("*.json")}
    todo = [s for s in sorted(games["snap"].unique())
            if s.replace(":", "") not in cached]
    print(f"  {len(cached):,} snapshots already cached, "
          f"{len(todo):,} still to fetch")

    if args.dry_run:
        print("\n--dry-run: no API calls made. Each snapshot is one historical "
              "request against your Odds API quota.")
        for s in seasons:
            sub = games[games["season"] == s]
            n = sub["snap"].nunique()
            miss = len([x for x in sub["snap"].unique()
                        if x.replace(":", "") not in cached])
            print(f"    {s}: {len(sub):,} games, {n:,} snapshots, "
                  f"{miss:,} to fetch")
        return

    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        raise SystemExit("set ODDS_API_KEY first:  "
                         '$env:ODDS_API_KEY = "<your-key>"')

    calls = 0
    for i, key in enumerate(todo, 1):
        if args.limit and calls >= args.limit:
            print(f"  --limit {args.limit} reached, stopping")
            break
        _, hit = fetch_snapshot(key, cache_dir, api_key, args.bookmaker,
                                args.regions, args.timeout, args.retries,
                                args.sleep)
        if hit is not True:
            calls += 1
            if calls % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} snapshots, {calls} API calls"
                      + (f", {hit} requests remaining" if hit else ""))

    # ---- assemble ------------------------------------------------------
    rows, misses = [], []
    snap_cache = {}
    for g in games.itertuples(index=False):
        cpath = cache_dir / f"{g.snap.replace(':', '')}.json"
        if not cpath.exists():
            misses.append((g.espn_game_id, "no snapshot"))
            continue
        if g.snap not in snap_cache:
            with open(cpath) as f:
                snap_cache[g.snap] = json.load(f)
        rec = extract(snap_cache[g.snap], g.home_team, g.away_team,
                      g.commence_time, args.tol_minutes)
        if rec is None:
            misses.append((g.espn_game_id, "event/total not in snapshot"))
            continue
        rec.update({
            "espn_game_id": g.espn_game_id,
            "season": int(g.season),
            "game_date": g.game_date if hasattr(g, "game_date") else None,
            "home_team": g.home_team,
            "away_team": g.away_team,
            "snapshot_time": g.snapshot_time,
        })
        rows.append(rec)

    new = pd.DataFrame(rows)
    print(f"\n  resolved {len(new):,} of {len(games):,} games "
          f"({len(new) / max(len(games), 1):.1%})")
    if misses:
        print(f"  {len(misses):,} unresolved; first few: {misses[:5]}")

    if new.empty:
        print("  nothing to write")
        return

    if out_path.exists():
        old = pd.read_parquet(out_path)
        old["espn_game_id"] = old["espn_game_id"].astype(str)
        before = len(old)
        merged = (pd.concat([old, new], ignore_index=True)
                  .drop_duplicates("espn_game_id", keep="last"))
        print(f"  merging with existing file: {before:,} rows -> "
              f"{len(merged):,} (existing seasons preserved)")
    else:
        merged = new

    merged = merged.sort_values(["season", "game_date"]
                                if "game_date" in merged.columns else ["season"])
    merged.to_parquet(out_path, index=False)
    print(f"  wrote {out_path}")
    print("\ncoverage by season:")
    print(merged.groupby("season").agg(
        games=("espn_game_id", "nunique"),
        total_min=("total_close", "min"),
        total_med=("total_close", "median"),
        total_max=("total_close", "max")).to_string())


if __name__ == "__main__":
    main()
