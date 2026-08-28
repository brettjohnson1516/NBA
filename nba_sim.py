"""
nba_sim.py

Possession-level model of the rest of an NBA game.

XGBoost on a flat feature vector has to learn the shape of a basketball game
from scratch, and it is worst exactly where the money is: the last few minutes,
where possession, the discreteness of 2s and 3s, and intentional fouling
dominate. This builds that structure explicitly:

  1. Measure, from play-by-play, the distribution of points per possession and
     how long a possession takes as a function of game state (a trailing team
     late in a game plays much faster and fouls).
  2. Backward-induct the exact win probability over remaining possessions,
     tracking who has the ball, on a grid of per-possession efficiency edges.

The result, sim_wp, is a calibrated win probability that needs no fitting. It
is then used as the base margin for the trees, so they only model the residual.

Usage (PowerShell):
  python nba_sim.py --rebuild --prior-seasons 2021 2022 2023
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

# nba_wp_features imports this module for the win-probability tables, so the
# pbp helpers are imported lazily inside the functions that need them.
DEFAULT_ROOT = os.environ.get(
    "NBA_ROOT", r"C:\Users\saint\OneDrive\Documents\NBA_AUG_2026"
)

MAX_PTS = 4                 # points scorable on one possession (and-1 three)
DIFF_CAP = 60               # score margins beyond this are decided anyway
ADV_GRID = np.round(np.arange(-0.30, 0.3001, 0.02), 4)   # per-possession edge
MAX_POSS = 250              # TOTAL possessions left (both teams); a full game is ~200


# ----------------------------------------------------------------------------
# measurement
# ----------------------------------------------------------------------------

def possession_segments(df):
    """
    One row per possession: points scored, seconds used, state at its start.

    Every score column in the pbp is POST-event, and a possession is often a
    single row (the made shot). So a possession's points are the score at the
    END of its segment minus the score at the end of the PREVIOUS segment --
    start-minus-start silently returns zero for every one-row possession.
    """
    df = df.sort_values(["game_id", "play_number"]).reset_index(drop=True).copy()
    # The flag flips ON the made shot / turnover, so the team on offence DURING
    # a row is the flag carried into it.
    df["possession"] = df.groupby("game_id")["possession"].shift(1).fillna(0.0)
    poss = df["possession"].values
    gid = df["game_id"].values
    new = np.ones(len(df), dtype=bool)
    new[1:] = (poss[1:] != poss[:-1]) | (gid[1:] != gid[:-1])
    df["_seg"] = np.cumsum(new) - 1

    g = df.groupby("_seg")
    out = pd.DataFrame({
        "game_id": g["game_id"].first(),
        "possession": g["possession"].first(),
        "home_end": g["home_score"].last(),
        "away_end": g["away_score"].last(),
        "elapsed_end": g["seconds_elapsed"].last(),
        "seconds_left_reg": g["seconds_left_reg"].first(),
        "period": g["period"].first(),
    }).reset_index(drop=True)

    prev_h = out.groupby("game_id")["home_end"].shift(1)
    prev_a = out.groupby("game_id")["away_end"].shift(1)
    prev_e = out.groupby("game_id")["elapsed_end"].shift(1)
    out["home_start"] = prev_h.fillna(0.0)
    out["away_start"] = prev_a.fillna(0.0)
    out["margin_start"] = out["home_start"] - out["away_start"]

    home_gain = out["home_end"] - out["home_start"]
    away_gain = out["away_end"] - out["away_start"]
    off_home = out["possession"] > 0
    out["points"] = np.where(off_home, home_gain, away_gain)
    out["opp_points"] = np.where(off_home, away_gain, home_gain)
    out["duration"] = out["elapsed_end"] - prev_e.fillna(0.0)

    out = out[(out["points"] >= 0) & (out["points"] <= MAX_PTS)]
    out = out[(out["duration"] >= 0) & (out["duration"] <= 60)]
    out = out[out["possession"] != 0]
    return out


def measure(paths, seasons, season_types):
    from nba_wp_features import (NBA_TEAMS, derive_possession, load_pbp,
                                 parse_events)
    segs = []
    for s in seasons:
        df = load_pbp(paths, s)
        df = df[df["season_type"].isin(season_types)]
        df = df[df["home_team"].isin(NBA_TEAMS) & df["away_team"].isin(NBA_TEAMS)]
        df = parse_events(df)
        df = derive_possession(df)
        segs.append(possession_segments(df))
    seg = pd.concat(segs, ignore_index=True)
    print(f"  {len(seg):,} possessions measured from seasons {seasons}")

    pts = np.bincount(seg["points"].astype(int).values, minlength=MAX_PTS + 1)[:MAX_PTS + 1]
    ppd = pts / pts.sum()
    print("  points-per-possession distribution: " +
          ", ".join(f"{i}pt {p:.4f}" for i, p in enumerate(ppd)) +
          f"  -> mean {float(np.dot(np.arange(MAX_PTS + 1), ppd)):.4f}")

    # how long a possession takes, by time left and by the offense's deficit.
    # This is where late-game urgency and intentional fouling show up.
    reg = seg[seg["period"] <= 4].copy()
    off_margin = np.where(reg["possession"] > 0, reg["margin_start"], -reg["margin_start"])
    reg["off_margin"] = off_margin
    tl_bins = [0, 30, 60, 120, 300, 720, 3000]
    mg_bins = [-200, -15, -9, -5, -2, 0, 2, 5, 9, 15, 200]
    reg["tl"] = pd.cut(reg["seconds_left_reg"], tl_bins, labels=False, right=False)
    reg["mg"] = pd.cut(reg["off_margin"], mg_bins, labels=False, right=False)
    tab = reg.groupby(["tl", "mg"], observed=True)["duration"].agg(["mean", "size"])
    grid = np.full((len(tl_bins) - 1, len(mg_bins) - 1), np.nan)
    for (t, m), r in tab.iterrows():
        if r["size"] >= 200:
            grid[int(t), int(m)] = r["mean"]
    overall = float(reg["duration"].mean())
    grid = np.where(np.isnan(grid), overall, grid)
    print(f"  mean possession length {overall:.2f}s; "
          f"under 30s left, trailing 5-9: {grid[0, 2]:.2f}s, leading 5-9: {grid[0, 7]:.2f}s")

    # How much of a high total is extra possessions vs extra efficiency?
    # Regress possessions per game on the closing total; whatever the possessions
    # do not explain is efficiency.
    poss_per_game = seg.groupby("game_id").size()
    pace_fit = None
    try:
        from nba_wp_features import load_odds, resolve_paths as _rp
        odds = load_odds(paths)
        odds["espn_game_id"] = odds["espn_game_id"].astype(str)
        pg = poss_per_game.rename("poss").reset_index()
        pg["game_id"] = pg["game_id"].astype(str)
        pg = pg.merge(odds[["espn_game_id", "total_close"]],
                      left_on="game_id", right_on="espn_game_id", how="inner").dropna()
        if len(pg) > 200:
            A = np.column_stack([np.ones(len(pg)), pg["total_close"].values])
            c, *_ = np.linalg.lstsq(A, pg["poss"].values, rcond=None)
            pace_fit = [float(c[0]), float(c[1])]
            lo, hi = pg["total_close"].quantile([0.05, 0.95])
            print(f"  possessions vs total: {c[0]:.1f} + {c[1]:.3f} x total "
                  f"({c[0] + c[1] * lo:.0f} poss at total {lo:.0f}, "
                  f"{c[0] + c[1] * hi:.0f} at {hi:.0f}); "
                  f"the rest of the total is efficiency")
    except Exception as e:
        print(f"  WARNING: could not fit possessions vs total ({e}); "
              f"the environment split will fall back to pace only")

    return {"ppd": ppd.tolist(), "sec_grid": grid.tolist(), "pace_fit": pace_fit,
            "tl_bins": tl_bins, "mg_bins": mg_bins,
            "overall_sec": overall, "prior_seasons": list(seasons)}


# ----------------------------------------------------------------------------
# backward induction
# ----------------------------------------------------------------------------

def tilt(ppd, adv):
    """
    Shift the points-per-possession distribution so its mean moves by `adv`,
    keeping the shape. Weights are exponentially tilted, which is the minimal
    distortion for a given mean shift.
    """
    k = np.arange(MAX_PTS + 1, dtype=float)
    base = np.asarray(ppd, dtype=float)
    target = float(np.dot(k, base)) + adv
    if target <= 0.02:
        target = 0.02
    lo, hi = -5.0, 5.0
    for _ in range(80):
        th = 0.5 * (lo + hi)
        w = base * np.exp(th * k)
        w = w / w.sum()
        if np.dot(k, w) < target:
            lo = th
        else:
            hi = th
    w = base * np.exp(0.5 * (lo + hi) * k)
    return w / w.sum()


def build_tables(ppd, verbose=True):
    """
    W[a, n, d, h] = P(home wins), where

      n = TOTAL possessions left in the game, both teams combined
      d = current margin, home minus away (index-shifted by DIFF_CAP)
      h = 1 if home has the ball on the next possession

    Possessions alternate, so each step hands the ball to the other team and
    decrements n. Counting per-team instead of total is what made a trailing
    team look worse with more time left.
    """
    D = 2 * DIFF_CAP + 1
    diffs = np.arange(-DIFF_CAP, DIFF_CAP + 1)
    W = np.zeros((len(ADV_GRID), MAX_POSS + 1, D, 2), dtype=np.float32)

    for ai, adv in enumerate(ADV_GRID):
        p_home = tilt(ppd, adv / 2.0)
        p_away = tilt(ppd, -adv / 2.0)

        w0 = np.where(diffs > 0, 1.0, np.where(diffs < 0, 0.0, 0.5))
        W[ai, 0, :, 0] = w0
        W[ai, 0, :, 1] = w0

        for n in range(1, MAX_POSS + 1):
            nxt = W[ai, n - 1, :, 0]
            acc = np.zeros(D)
            for j, pj in enumerate(p_home):
                if pj == 0:
                    continue
                if j == 0:
                    acc += pj * nxt
                else:
                    sh = np.roll(nxt, -j)
                    sh[-j:] = nxt[-1]
                    acc += pj * sh
            W[ai, n, :, 1] = acc

            nxt = W[ai, n - 1, :, 1]
            acc = np.zeros(D)
            for j, pj in enumerate(p_away):
                if pj == 0:
                    continue
                if j == 0:
                    acc += pj * nxt
                else:
                    sh = np.roll(nxt, j)
                    sh[:j] = nxt[0]
                    acc += pj * sh
            W[ai, n, :, 0] = acc
        if verbose and ai % 10 == 0:
            print(f"    adv {adv:+.2f} done")
    return W


def win_prob(W, adv, n_poss, margin, home_ball):
    ai = np.clip(np.searchsorted(ADV_GRID, adv), 0, len(ADV_GRID) - 1)
    n = np.clip(np.round(n_poss).astype(int), 0, MAX_POSS)
    d = np.clip(np.round(margin).astype(int), -DIFF_CAP, DIFF_CAP) + DIFF_CAP
    h = np.where(home_ball > 0, 1, 0)
    return W[ai, n, d, h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--prior-seasons", nargs="+", type=int, default=[2021, 2022, 2023])
    ap.add_argument("--season-types", nargs="+", type=int, default=[2])
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    from nba_wp_features import resolve_paths
    paths = resolve_paths(args.root)
    os.makedirs(paths["features"], exist_ok=True)
    mf = os.path.join(paths["features"], "sim_params.json")
    tf = os.path.join(paths["features"], "sim_tables.npz")

    if args.rebuild or not os.path.exists(mf):
        params = measure(paths, args.prior_seasons, set(args.season_types))
        json.dump(params, open(mf, "w"), indent=2)
        print(f"  wrote {mf}")
    else:
        params = json.load(open(mf))

    if args.rebuild or not os.path.exists(tf):
        print("  building win-probability tables ...")
        W = build_tables(params["ppd"])
        np.savez_compressed(tf, W=W, adv_grid=ADV_GRID)
        print(f"  wrote {tf}  shape {W.shape}")


if __name__ == "__main__":
    main()
