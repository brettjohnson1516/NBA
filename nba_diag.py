"""
nba_diag.py

Post-mortem on a backtest, from the bets CSV that nba_wp_backtest.py already
wrote. No model rerun, no Kalshi reload.

Answers two questions the v5 run raised:
  1. Road favourites lose while every other role wins. Is the MODEL wrong about
     them, or is it the market, or is it neither and just a few games?
  2. April is -22%. Is that the same defect concentrated, or something else?

Usage (PowerShell):
  python nba_diag.py --tag v5
"""

import argparse
import os

import numpy as np
import pandas as pd

from nba_wp_features import DEFAULT_ROOT, resolve_paths


def table(df, by, label, extra=None):
    g = df.groupby(by, dropna=False, observed=True)
    t = pd.DataFrame({
        "bets": g.size(),
        "games": g["game_id"].nunique(),
        "staked": g["staked"].sum(),
        "pnl": g["pnl"].sum(),
        "model_p": g["p_win"].mean(),
        "market_p": g["price"].mean() / 100.0,
        "actual": g["won"].mean(),
    })
    t["roi_pct"] = 100.0 * t["pnl"] / t["staked"]
    t["model_err"] = t["model_p"] - t["actual"]
    t["market_err"] = t["market_p"] - t["actual"]
    print(f"\n--- {label} ---")
    print(t.reset_index().to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--top-games", type=int, default=15)
    args = ap.parse_args()

    paths = resolve_paths(args.root)
    f = os.path.join(paths["root"], f"nba_backtest_bets_{args.tag}.csv")
    b = pd.read_csv(f)
    b["game_date"] = pd.to_datetime(b["game_date"])
    print(f"  {len(b):,} bets / {b['game_id'].nunique():,} games from {os.path.basename(f)}")

    print("\n================ ROLE OVERVIEW ================")
    table(b, "role", "every role: is the model or the market wrong?")

    rf = b[b["role"] == "road fav"].copy()
    print(f"\n================ ROAD FAVOURITES ({len(rf):,} bets) ================")
    table(rf, "quarter", "road fav by quarter")
    table(rf, "price_bucket", "road fav by execution price")
    table(rf, "margin_bucket", "road fav by score differential")
    table(rf, "month", "road fav by month")

    # is the loss broad or a handful of games?
    per = rf.groupby("game_id").agg(bets=("pnl", "size"), pnl=("pnl", "sum"),
                                    staked=("staked", "sum")).sort_values("pnl")
    print(f"\n--- road fav: concentration ---")
    print(f"  games with road-fav bets: {len(per):,}")
    print(f"  total pnl {per['pnl'].sum():,.0f}")
    print(f"  worst {args.top_games} games contribute {per['pnl'].head(args.top_games).sum():,.0f}")
    print(f"  median game pnl {per['pnl'].median():,.0f}; "
          f"share of games losing {100.0 * (per['pnl'] < 0).mean():.1f}%")

    print("\n================ MONTHS ================")
    table(b, "month", "all bets by month")
    apr = b[b["game_date"].dt.strftime("%Y-%m") == b["month"].max()].copy()
    if len(apr):
        print(f"\n================ WORST MONTH ({b['month'].max()}) ================")
        table(apr, "role", "worst month by role")
        table(apr, "price_bucket", "worst month by execution price")
        aper = apr.groupby("game_id").agg(bets=("pnl", "size"), pnl=("pnl", "sum")).sort_values("pnl")
        print(f"\n--- worst month: concentration ---")
        print(f"  games {len(aper):,}, total pnl {aper['pnl'].sum():,.0f}, "
              f"worst {args.top_games} contribute {aper['pnl'].head(args.top_games).sum():,.0f}")
        print(f"  share of games losing {100.0 * (aper['pnl'] < 0).mean():.1f}%")


if __name__ == "__main__":
    main()
