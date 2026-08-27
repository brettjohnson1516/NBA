"""
nba_wp_backtest.py

Backtests the NBA live WP model against the Kalshi trade tape.

Mechanics (as specified):
  * step wall clock every --grid seconds through each game
  * at each step take the game state in force at that instant; skip the step if
    the game clock has not changed since the previous step
  * anchor = that wall-clock instant + --lag seconds
  * fill = a Kalshi trade printed in that exact second for that game
  * a print at X on one side means the other side is available at (101 - X)
  * bet on either side whose net edge, after the taker fee, clears --min-edge.
    The edge test is evaluated at the nominal --contracts size, so bet selection
    is unchanged.
  * size = edge-scaled. Once the book of qualifying bets is fixed, each bet's
    edge is converted to a z-score across the whole book, the size multiplier is
    1 + --size-slope * z clipped to [--size-min, --size-max], and the book is
    rescaled so the AVERAGE bet is exactly --contracts contracts. Bigger edge ->
    bigger bet, same average size as flat sizing. --size-slope 0 = flat sizing.

Usage (PowerShell):
  python nba_wp_backtest.py --seasons 2026 --tag v1
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from nba_wp_features import FEATURES, DEFAULT_ROOT, resolve_paths
# v11. Imports from the VERSIONED trainer. The plain nba_wp_train.py on
# disk has an apply_calibrator that does not know the tail_iso kind and
# would silently return the uncalibrated probability instead of failing.
from nba_wp_train_v12 import apply_calibrator, predict_ensemble

KALSHI_COLS = [
    "game_date", "away_team", "home_team", "ticker",
    "timestamp_unix", "is_home_yes", "yes_price_cents",
    "home_yes_price_cents", "count_fp", "taker_side",
]


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

def load_model(paths, tag):
    mf = os.path.join(paths["models"], f"nba_wp_{tag}_meta.json")
    meta = json.load(open(mf))
    boosters = []
    for m in meta["members"]:
        b = xgb.Booster()
        b.load_model(os.path.join(paths["models"], m))
        boosters.append(b)
    print(f"  model {tag}: {len(boosters)} member(s), {len(meta['features'])} features, "
          f"calibrator {meta['calibrator']['kind']}")
    _known = {"raw", "platt", "isotonic", "tail_iso"}
    if meta["calibrator"]["kind"] not in _known:
        raise SystemExit(f"model meta names calibrator kind "
                         f"{meta['calibrator']['kind']!r}, which this backtest does "
                         f"not know how to apply. Refusing to run rather than "
                         f"silently scoring uncalibrated probabilities.")
    return meta, boosters


# ----------------------------------------------------------------------------
# kalshi tape
# ----------------------------------------------------------------------------

# ESPN and Kalshi use different abbreviations for the same nine franchises.
# Everything is mapped to a single canonical code before joining.
TEAM_ALIASES = {
    "GS": "GSW", "GSW": "GSW",
    "NO": "NOP", "NOP": "NOP", "NOH": "NOP",
    "NY": "NYK", "NYK": "NYK",
    "SA": "SAS", "SAS": "SAS",
    "UTAH": "UTA", "UTA": "UTA",
    "WSH": "WAS", "WAS": "WAS",
    "PHX": "PHX", "PHO": "PHX",
    "BKN": "BKN", "BRK": "BKN", "NJN": "BKN",
    "CHA": "CHA", "CHO": "CHA",
}


def norm_team(s):
    t = (s.astype(str).str.strip().str.upper()
         .str.replace(r"[^A-Z]", "", regex=True))
    return t.map(lambda x: TEAM_ALIASES.get(x, x))


def load_kalshi(paths, dates_needed, spread=1.0, legacy=False):
    f = None
    for pat in ["*.parquet"]:
        hits = sorted(glob.glob(os.path.join(paths["kalshi"], pat)),
                      key=os.path.getsize, reverse=True)
        if hits:
            f = hits[0]
            break
    if f is None:
        raise FileNotFoundError(f"no parquet trade tape found in {paths['kalshi']}")
    print(f"  kalshi tape: {os.path.basename(f)}")

    k = pd.read_parquet(f, columns=KALSHI_COLS)
    print(f"    {len(k):,} trades loaded")

    k["game_date"] = pd.to_datetime(k["game_date"]).dt.date
    k = k[k["game_date"].isin(dates_needed)]
    k["home_team_n"] = norm_team(k["home_team"])
    k["away_team_n"] = norm_team(k["away_team"])

    # ---- price BOTH sides, using which side the taker actually lifted ----
    #
    # A print tells you one price on one side. Which price it is depends on the
    # taker:
    #   taker bought  -> the print IS the ask on that side, and the other side's
    #                    ask is 100 - (this side's bid) = 100 + spread - print
    #   taker sold    -> the print is the BID on that side, so that side's ask is
    #                    print + spread, and the other side's ask is 100 - print
    #
    # The old version assumed every print was the home ask and derived the away
    # side as 101 - it. On this tape home_yes_price_cents is always exactly
    # 100 - yes_price_cents, so that assumption held for only 46% of prints; on
    # the other 54% it made the home side a cent too cheap and the away side a
    # cent too dear. Home bets were flattered and away bets penalised, on the
    # same rows.
    yes_px = pd.to_numeric(k["yes_price_cents"], errors="coerce")
    is_home_yes = k["is_home_yes"].astype(bool).values
    ts_raw = k["taker_side"].astype(str).str.strip().str.lower()
    taker_bought = ts_raw.eq("yes").values
    known_taker = ts_raw.isin(["yes", "no"]).values
    n_unknown = int((~known_taker).sum())
    if n_unknown:
        print(f"    WARNING: {n_unknown:,} trades ({100.0 * n_unknown / max(len(k), 1):.2f}%) "
              f"have no usable taker_side; treated as taker-bought")

    P = yes_px.values.astype(float)
    s = float(spread)
    ask_printed = np.where(taker_bought, P, P + s)
    ask_other = np.where(taker_bought, 100.0 + s - P, 100.0 - P)
    k["px_home"] = np.where(is_home_yes, ask_printed, ask_other)
    k["px_away"] = np.where(is_home_yes, ask_other, ask_printed)

    if legacy:
        home_px = pd.to_numeric(k["home_yes_price_cents"], errors="coerce")
        fallback = np.where(is_home_yes, yes_px, 101.0 - yes_px)
        k["px_home"] = home_px.fillna(pd.Series(fallback, index=k.index))
        k["px_away"] = 101.0 - k["px_home"]
        print("    PRICE MODE: legacy (every print treated as the home ask, "
              "away = 101 - home)")
    else:
        n_ok = int((is_home_yes == taker_bought).sum())
        print(f"    price mode: taker (spread {s:.0f}c); "
              f"{100.0 * n_ok / max(len(k), 1):.1f}% of prints priced the same as "
              f"legacy, {100.0 * (len(k) - n_ok) / max(len(k), 1):.1f}% corrected")

    # timestamp_unix can arrive as object/str/float depending on how the tape
    # was written; go through float64 so the cast to int64 is always safe
    k["ts"] = pd.to_numeric(k["timestamp_unix"], errors="coerce").astype("float64")
    k = k.dropna(subset=["px_home", "px_away", "ts"])
    k["ts"] = k["ts"].round().astype("int64")
    k = k[(k["px_home"] > 0) & (k["px_home"] < 101)
          & (k["px_away"] > 0) & (k["px_away"] < 101)]
    print(f"    {len(k):,} trades on the backtest dates")
    return k


def join_kalshi_to_games(k, games):
    """
    games: DataFrame with game_id, game_date, home_team, away_team (ESPN codes).
    Matches on normalised team pair with a +/- 1 day tolerance on the date.
    Anything unmatched is reported, never silently dropped.
    """
    g = games.copy()
    g["home_team_n"] = norm_team(g["home_team"])
    g["away_team_n"] = norm_team(g["away_team"])
    g["game_date"] = pd.to_datetime(g["game_date"]).dt.date

    lookup = {}
    for _, r in g.iterrows():
        for off in (0, -1, 1):
            d = r["game_date"] + pd.Timedelta(days=off)
            lookup.setdefault((d.date() if hasattr(d, "date") else d,
                               r["home_team_n"], r["away_team_n"]), r["game_id"])

    keys = list(zip(k["game_date"], k["home_team_n"], k["away_team_n"]))
    k = k.copy()
    k["game_id"] = [lookup.get(x) for x in keys]

    matched = k["game_id"].notna()
    if not matched.all():
        bad = (k.loc[~matched, ["game_date", "home_team", "away_team"]]
               .drop_duplicates().head(25))
        print(f"    WARNING: {(~matched).sum():,} trades ({(~matched).mean():.2%}) "
              f"did not match a game. Unmatched date/team combos (first 25):")
        print(bad.to_string(index=False))
    k = k[matched]
    print(f"    matched trades: {len(k):,} across {k['game_id'].nunique():,} games")
    return k


# ----------------------------------------------------------------------------
# wall-clock grid
# ----------------------------------------------------------------------------

def build_grid(feat, grid_seconds=None):
    """
    One scoreable state per distinct wall-clock SECOND that has a play-by-play
    event. Events are sparse and irregular -- median gap about 11 seconds, 75th
    percentile 22, some past 60 -- so a fixed grid mostly re-scores the same
    event, while filtering on the game clock throws away states where the score
    moved with the clock stopped (every free throw, ~20% of event-seconds).

    Where several events share a second the LAST one is kept: that is the state
    as of the end of that second.
    """
    f = feat.sort_values(["game_id", "play_number"]).copy()
    f["wc"] = pd.to_datetime(f["wallclock"], utc=True, errors="coerce")
    f = f.dropna(subset=["wc"])
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    f["wc_unix"] = (f["wc"] - epoch).dt.total_seconds().astype("float64")
    f = f.dropna(subset=["wc_unix"])
    f["wc_unix"] = f["wc_unix"].round().astype("int64")

    n_events = len(f)
    f = f.sort_values(["game_id", "wc_unix", "play_number"])
    grid_df = f.drop_duplicates(["game_id", "wc_unix"], keep="last").copy()
    grid_df["grid_unix"] = grid_df["wc_unix"]

    gaps = grid_df.groupby("game_id")["wc_unix"].diff().dropna()
    print(f"  states: {len(grid_df):,} distinct wall-clock seconds from "
          f"{n_events:,} events across {grid_df['game_id'].nunique():,} games "
          f"({len(grid_df) / max(grid_df['game_id'].nunique(), 1):.0f} per game)")
    if len(gaps):
        print(f"  gap between states: median {gaps.median():.0f}s, "
              f"mean {gaps.mean():.1f}s, 90th pct {gaps.quantile(0.9):.0f}s")
    return grid_df


# ----------------------------------------------------------------------------
# fees
# ----------------------------------------------------------------------------

def taker_fee(contracts, price_dollars):
    """Kalshi taker fee, rounded up to the cent, at the order level."""
    raw = 0.07 * contracts * price_dollars * (1.0 - price_dollars)
    return np.ceil(raw * 100.0) / 100.0


def edge_scaled_size(edge, target_avg, slope, lo, hi):
    """
    Contracts per bet, scaled by how large the edge is relative to the rest of
    the bet book, with the mean pinned to target_avg.

    z          = (edge - mean(edge)) / sd(edge)   across all qualifying bets
    multiplier = clip(1 + slope * z, lo, hi)
    contracts  = round(target_avg * multiplier / mean(multiplier)), floor of 1

    Rounding and the floor of 1 both push the realised mean off target, so the
    scale factor is solved for in a short loop instead of applied once.
    Returns (contracts, z).
    """
    e = np.asarray(edge, dtype="float64")
    sd = float(np.std(e))
    if slope == 0.0 or not np.isfinite(sd) or sd <= 0.0:
        z = np.zeros_like(e)
    else:
        z = (e - float(np.mean(e))) / sd

    mult = np.clip(1.0 + slope * z, lo, hi)
    m = float(np.mean(mult))
    if not np.isfinite(m) or m <= 0.0:
        mult = np.ones_like(e)
        m = 1.0
    mult = mult / m

    scale = float(target_avg)
    contracts = np.maximum(1.0, np.rint(scale * mult))
    for _ in range(40):
        realised = float(np.mean(contracts))
        if realised <= 0 or abs(realised - target_avg) < 1e-9:
            break
        scale *= target_avg / realised
        new = np.maximum(1.0, np.rint(scale * mult))
        if np.array_equal(new, contracts):
            break
        contracts = new
    return contracts, z


# ----------------------------------------------------------------------------
# views
# ----------------------------------------------------------------------------

def summarize(bets, by, label):
    if bets.empty:
        return pd.DataFrame()
    g = bets.groupby(by, dropna=False, observed=True)
    t = pd.DataFrame({
        "bets": g.size(),
        "games": g["game_id"].nunique(),
        "staked": g["staked"].sum(),
        "pnl": g["pnl"].sum(),
        "avg_size": g["contracts"].mean(),
    })
    t["roi_pct"] = 100.0 * t["pnl"] / t["staked"]
    t = t.reset_index()
    tot = pd.DataFrame([{
        by: "TOTAL",
        "bets": len(bets),
        "games": bets["game_id"].nunique(),
        "staked": bets["staked"].sum(),
        "pnl": bets["pnl"].sum(),
        "avg_size": bets["contracts"].mean(),
        "roi_pct": 100.0 * bets["pnl"].sum() / bets["staked"].sum(),
    }])
    t = pd.concat([t, tot], ignore_index=True)
    print(f"\n--- {label} ---")
    print(t.to_string(index=False,
                      formatters={"staked": "{:,.0f}".format,
                                  "pnl": "{:,.0f}".format,
                                  "avg_size": "{:,.0f}".format,
                                  "roi_pct": "{:,.2f}".format}))
    return t


EDGE_BINS = [0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 1.0]
EDGE_LAB = ["2-3%", "3-5%", "5-7.5%", "7.5-10%", "10-15%", "15%+"]
PX_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101]
PX_LAB = ["0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80-89", "90-100"]
MG_BINS = [-999, -14.5, -9.5, -4.5, -0.5, 0.5, 4.5, 9.5, 14.5, 999]
MG_LAB = ["trail 15+", "trail 10-14", "trail 5-9", "trail 1-4", "tied",
          "lead 1-4", "lead 5-9", "lead 10-14", "lead 15+"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--grid", type=int, default=None,
                    help="deprecated and ignored: states are now every distinct "
                         "wall-clock second that has an event")
    ap.add_argument("--lag", type=int, default=10,
                    help="execution latency: seconds after the state before a "
                         "fill can happen. Not a tuning knob.")
    ap.add_argument("--window", type=int, default=5,
                    help="if no print lands exactly on anchor+lag, take the first "
                         "print within this many seconds AFTER it. Forward only. "
                         "0 = exact second only.")
    ap.add_argument("--contracts", type=int, default=100,
                    help="AVERAGE contracts per bet; individual bets are scaled by edge")
    ap.add_argument("--size-slope", type=float, default=1.0,
                    help="contracts multiplier = 1 + slope * z(edge). 0 = flat sizing")
    ap.add_argument("--size-min", type=float, default=0.20,
                    help="floor on the size multiplier before rescaling")
    ap.add_argument("--size-max", type=float, default=3.00,
                    help="cap on the size multiplier before rescaling")
    ap.add_argument("--min-edge", type=float, default=0.02)
    ap.add_argument("--price-mode", default="taker", choices=["taker", "legacy"],
                    help="taker = price both sides from taker_side (correct). "
                         "legacy = reproduce v9, every print treated as the home "
                         "ask with away = 101 - home. For A/B only.")
    ap.add_argument("--spread", type=float, default=1.0,
                    help="assumed book width in cents")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    paths = resolve_paths(args.root)
    meta, boosters = load_model(paths, args.tag)
    feats = meta["features"]

    frames = []
    for s in args.seasons:
        f = os.path.join(paths["features"], f"nba_wp_features_{s}.parquet")
        frames.append(pd.read_parquet(f))
    feat = pd.concat(frames, ignore_index=True)
    print(f"  features: {len(feat):,} rows / {feat['game_id'].nunique():,} games")

    grid = build_grid(feat)

    X = grid[feats].values.astype(np.float32)
    bm = meta.get("base_margin", "none")
    margin = None
    if bm and bm != "none":
        if bm not in grid.columns:
            raise SystemExit(f"model was trained with base margin {bm}, which is not in "
                             f"the feature tables; rebuild with nba_wp_features.py")
        margin = np.nan_to_num(grid[bm].values.astype(float), nan=0.0,
                               posinf=8.0, neginf=-8.0)
    p = predict_ensemble(boosters, X, feats, margin=margin)
    grid["p_home"] = np.clip(apply_calibrator(meta["calibrator"], p), 1e-6, 1 - 1e-6)

    games = feat[["game_id", "game_date", "home_team", "away_team"]].drop_duplicates("game_id")
    dates = set(pd.to_datetime(games["game_date"]).dt.date)
    for off in (-1, 1):
        dates |= {d + pd.Timedelta(days=off) for d in list(dates)}
    dates = {d.date() if hasattr(d, "date") else d for d in dates}

    k = load_kalshi(paths, dates, spread=args.spread,
                    legacy=(args.price_mode == "legacy"))
    k = join_kalshi_to_games(k, games)

    # one print per (game, second); first print in that second wins
    k = k.sort_values(["game_id", "ts"]).drop_duplicates(["game_id", "ts"], keep="first")
    k_idx = k.set_index(["game_id", "ts"])[["px_home", "px_away", "count_fp", "taker_side"]]

    grid["anchor"] = grid["grid_unix"] + args.lag
    grid = grid.reset_index(drop=True)

    # Try the exact anchor second, then walk FORWARD one second at a time up to
    # --window. Forward only: a print before the anchor is a price that existed
    # before the execution lag had elapsed, so it could not have been hit.
    for c in ["px_home", "px_away", "count_fp"]:
        grid[c] = np.nan
    grid["fill_delay"] = np.nan
    need = np.ones(len(grid), dtype=bool)
    for off in range(0, args.window + 1):
        if not need.any():
            break
        idx = np.flatnonzero(need)
        keys = pd.MultiIndex.from_arrays(
            [grid["game_id"].values[idx],
             (grid["anchor"].values[idx] + off).astype("int64")])
        got = k_idx.reindex(keys)
        hit = got["px_home"].notna().values
        if hit.any():
            rows = idx[hit]
            for c in ["px_home", "px_away", "count_fp"]:
                grid.loc[rows, c] = got[c].values[hit]
            grid.loc[rows, "fill_delay"] = off
            need[rows] = False
    filled = grid["px_home"].notna()
    print(f"  fills: {filled.sum():,} of {len(grid):,} states ({filled.mean():.2%})"
          + (f", within +{args.window}s of the anchor" if args.window else ""))
    if args.window and filled.any():
        d = grid.loc[filled, "fill_delay"]
        print(f"  fill delay past the anchor: {100 * (d == 0).mean():.1f}% exact, "
              f"median {d.median():.0f}s, mean {d.mean():.2f}s")
    g = grid[filled].copy()
    if g.empty:
        print("no fills - nothing to backtest")
        return

    # model vs market at the states we could actually have traded
    mkt = g["px_home"].values / 100.0
    act = (g["home_final"] > g["away_final"]).astype(int).values
    pm = g["p_home"].values
    diag = pd.DataFrame({"model": pm, "market": mkt, "actual": act})
    diag["q"] = pd.qcut(diag["model"], 10, labels=False, duplicates="drop")
    dt = diag.groupby("q").agg(n=("actual", "size"), model=("model", "mean"),
                               market=("market", "mean"), actual=("actual", "mean"))
    dt["model_gap"] = dt["actual"] - dt["model"]
    dt["market_gap"] = dt["actual"] - dt["market"]
    print("\n--- model vs market at filled states (decile of model home WP) ---")
    print(dt.round(4).to_string())
    print(f"  mean |model-actual| {np.abs(pm - act).mean():.4f}   "
          f"mean |market-actual| {np.abs(mkt - act).mean():.4f}")

    C = args.contracts
    rows = []
    for side in ("home", "away"):
        px = g[f"px_{side}"].values / 100.0
        pw = g["p_home"].values if side == "home" else 1.0 - g["p_home"].values
        fee = taker_fee(C, px)
        edge = pw - px - fee / C
        take = edge >= args.min_edge
        if not take.any():
            continue
        s = g[take].copy()
        s["side"] = side
        s["price"] = px[take] * 100.0
        s["p_win"] = pw[take]
        s["edge"] = edge[take]
        won = (s["home_final"] > s["away_final"]).values if side == "home" else (s["away_final"] > s["home_final"]).values
        s["won"] = won.astype(int)
        rows.append(s)

    if not rows:
        print("no bets cleared the edge filter")
        return
    bets = pd.concat(rows, ignore_index=True)

    # sized after the whole book is built, so the z-score spans every bet rather
    # than being taken within the home leg and the away leg separately
    n, zscore = edge_scaled_size(bets["edge"].to_numpy(), float(C),
                                 args.size_slope, args.size_min, args.size_max)
    bets["contracts"] = n
    bets["edge_z"] = zscore
    bpx = bets["price"].to_numpy() / 100.0
    bets["fee"] = taker_fee(n, bpx)
    bets["staked"] = n * bpx
    bets["pnl"] = np.where(bets["won"].to_numpy() == 1,
                           n * (1.0 - bpx), -n * bpx) - bets["fee"].to_numpy()
    print(f"\nsizing: edge-scaled, slope {args.size_slope:g}, multiplier clipped to "
          f"[{args.size_min:.2f}, {args.size_max:.2f}]")
    print(f"        mean {n.mean():.1f} contracts  min {n.min():.0f}  "
          f"median {np.median(n):.0f}  max {n.max():.0f}")

    bets["game_date"] = pd.to_datetime(bets["game_date"])
    bets["month"] = bets["game_date"].dt.strftime("%Y-%m")
    bets["quarter"] = np.where(bets["period"] <= 4, "Q" + bets["period"].astype(str), "OT")
    bets["edge_bucket"] = pd.cut(bets["edge"], EDGE_BINS, labels=EDGE_LAB, right=False)
    bets["price_bucket"] = pd.cut(bets["price"], PX_BINS, labels=PX_LAB, right=False)
    bet_margin = np.where(bets["side"] == "home",
                          bets["score_margin"], -bets["score_margin"])
    bets["margin_bucket"] = pd.cut(bet_margin, MG_BINS, labels=MG_LAB)
    is_home_fav = bets["logit_close"] >= 0
    bet_is_home = bets["side"] == "home"
    bet_is_fav = np.where(bet_is_home, is_home_fav, ~is_home_fav)
    bets["role"] = np.where(bet_is_home,
                            np.where(bet_is_fav, "home fav", "home dog"),
                            np.where(bet_is_fav, "road fav", "road dog"))
    bets["side_label"] = np.where(bet_is_home, "home", "away")
    bets["favdog"] = np.where(bet_is_fav, "pregame fav", "pregame dog")

    print(f"\n=== BACKTEST {args.tag}  seasons {args.seasons} "
          f"(avg {C} contracts, min edge {args.min_edge:.1%}, lag {args.lag}s, "
          f"window +{args.window}s) ===")

    views = {
        "edge_bucket": "by edge range",
        "side_label": "by home vs away",
        "favdog": "by pregame fav vs dog",
        "price_bucket": "by execution price",
        "quarter": "by quarter",
        "margin_bucket": "by score differential (from the bet side)",
        "role": "by home/road x fav/dog",
        "month": "by month",
        "season": "by season",
    }
    out = {}
    for col, label in views.items():
        out[col] = summarize(bets, col, label)

    tot_staked = bets["staked"].sum()
    print(f"\n=== TOTAL: {len(bets):,} bets / {bets['game_id'].nunique():,} games / "
          f"${tot_staked:,.0f} staked / ${bets['pnl'].sum():,.0f} PNL / "
          f"{100.0 * bets['pnl'].sum() / tot_staked:.2f}% ROI ===")

    sfx = args.out_suffix
    bf = os.path.join(paths["root"], f"nba_backtest_bets_{args.tag}{sfx}.csv")
    keep = ["game_id", "game_date", "home_team", "away_team", "season", "period",
            "pc_seconds_left", "grid_unix", "anchor", "side", "price", "p_win",
            "edge", "edge_z", "contracts", "fee", "staked", "won", "pnl",
            "score_margin", "logit_close",
            "fill_delay",
            "edge_bucket", "price_bucket", "margin_bucket", "role", "quarter", "month"]
    bets[keep].to_csv(bf, index=False)
    vf = os.path.join(paths["root"], f"nba_backtest_views_{args.tag}{sfx}.csv")
    with open(vf, "w", newline="") as fh:
        for col, label in views.items():
            fh.write(f"# {label}\n")
            out[col].to_csv(fh, index=False)
            fh.write("\n")
    print(f"\nwrote {bf}\nwrote {vf}")


if __name__ == "__main__":
    main()
