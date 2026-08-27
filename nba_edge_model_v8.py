"""
nba_edge_model_v8.py

Fits the per-possession efficiency edge for the REST of the game.

nba_sim.py turns an edge into a win probability exactly. But the edge it is
handed today is just the decayed pregame spread, which ignores everything the
game has revealed: who is on the floor, how the roster is being used, whether
the score came from shooting that will regress. The trees then patch the final
probability with a flat additive term, which is the wrong place to correct --
an edge error means something different with 90 possessions left than with 9.

This fits the edge itself against what actually happened:

    realized_edge = (final margin - current margin) / (possessions left / 2)

and writes coefficients that nba_wp_features.py uses to build sim_adv. Fitting
happens on TRAIN seasons only; the coefficients are then applied to every season
including the backtest one.

The only change is in load(): the feature parquets are read through a ladder of
read strategies instead of a single call.

nba_wp_train_v12.py reads nba_wp_features_2022.parquet and counts 574,318 rows.
pq.ParquetFile on the same path cannot parse its footer, and so does
pd.read_parquet with a columns= list. Those are different code paths inside
pyarrow - the plain full read goes through one, the projected and metadata-first
reads go through others - and on this file they do not agree.

So load() stops asserting which one should work and just tries them in order,
starting with the one the trainer uses, and returns the first that comes back
with data. It only raises if all four fail, and then it names the file and the
season and prints what each path said.

Usage (PowerShell):
  python nba_edge_model_v8.py --seasons 2021 2022 2023 2024 2025
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb

from nba_wp_features import DEFAULT_ROOT, resolve_paths

# Candidate predictors of the remaining per-possession edge. All are already
# oriented home-positive. Kept deliberately small: this is a linear fit whose
# output feeds a structural model, not a second black box.
PREDICTORS = [
    "exp_margin_full",        # the pregame line, undecayed - the fit learns the decay
    "exp_margin_left",        # and the decayed version, so it can pick the mix
    "rest_rapm_diff",         # who actually plays the remaining minutes
    "oncourt_rapm_diff",      # who is on the floor right now
    "home_oncourt_edge",
    "away_oncourt_edge",
    "fg3_luck_diff_pts",      # score built on shooting that may regress
    "ft_luck_diff_pts",
    "fg2_luck_diff_pts",
    "fta_diff",               # whistle so far
    "foul_diff",
    "pace_ratio",
    # Regression opportunity depends on how much game is left to regress in,
    # which is the interaction his spec called for. The luck terms above are
    # stocks; these are the same stocks weighted by time remaining.
    "fg3_luck_x_fracleft",
    "luck_x_fracleft",
    "fta_diff_x_fracleft",
    "rest_rapm_diff_left",
    # v8. THE CLOCK.
    #
    # Every fit before this one had no time input at all. Its predictors were
    # the line, the RAPM ratings, the shooting-luck terms, fouls and pace -
    # nothing that says how much game is left. So it could not lean less on the
    # pregame line in the second half even in principle: it fitted ONE blend of
    # line-versus-game-evidence and applied that same blend from tip to buzzer.
    #
    # The interaction terms in the list above (the _x_fracleft ones) let a few
    # individual terms fade, but the pregame line itself had no way to fade,
    # because frac_left was never a predictor in its own right.
    #
    # With the GBM this is enough on its own: trees split on frac_left and can
    # therefore learn a different weighting of exp_margin_full early and late,
    # rather than being forced to pick one number for both. It LEARNS the time
    # profile; nothing here imposes one. In the linear fit it can only shift the
    # intercept, so expect it to matter far more to the GBM.
    "frac_left",
    # v17. The pregame prior, shrunk by how much the game has already
    # contradicted it: exp_margin_full x (full-game-extrapolated surprise).
    # Replaces v16's exp_margin_left_x_surprise, which decayed to zero exactly
    # where it needed to bite and was not on a rate scale, and which the fit
    # duly priced at ~0. This one sits on the same constant-rate scale as
    # exp_margin_full itself, so a negative coefficient means "shrink the prior
    # in proportion to how wrong the game says it is". Zero when the game runs
    # to script. Expect a NEGATIVE coefficient with real contribution.
    "exp_margin_x_surprise_rate",
    # NOT margin_ex_luck: it is score_margin minus luck, and the DP already
    # takes score margin as its state. Including it here counts the current
    # score twice — once as position, once as a per-possession rate — which
    # made the model over-confident and cost 0.41 ROI points in v7.
]


def _read_full(f, need):
    """Exactly what nba_wp_train_v12.py does. Tried first because it is the one
    path that is known to read the 2022 file."""
    return pd.read_parquet(f)


def _read_projected(f, need):
    """Column projection. Cheaper when it works."""
    return pd.read_parquet(f, columns=need)


def _read_row_groups(f, need):
    """Row group at a time through ParquetFile, concatenated. A different
    reader path again, and it survives a footer that the dataset API chokes on."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(f)
    parts = [pf.read_row_group(i).to_pandas()
             for i in range(pf.metadata.num_row_groups)]
    if not parts:
        raise ValueError("no row groups")
    return pd.concat(parts, ignore_index=True)


def _read_dataset(f, need):
    """The dataset API, reading the file as a one-file dataset."""
    import pyarrow.dataset as ds
    return ds.dataset(f, format="parquet").to_table().to_pandas()


READERS = [
    ("pd.read_parquet(path)", _read_full),
    ("pd.read_parquet(path, columns=...)", _read_projected),
    ("ParquetFile row groups", _read_row_groups),
    ("pyarrow.dataset", _read_dataset),
]


def read_features(f, need, season):
    """
    Read a feature parquet using whichever pyarrow path actually works on it.

    These four are not wrappers around one another - pyarrow dispatches a plain
    full read, a projected read, a row-group read and a dataset scan through
    different code, and on a file with a marginal footer they disagree about
    whether it is readable. Rather than pick one and have the run die, try them
    in order and take the first that returns.
    """
    errors = []
    for name, fn in READERS:
        try:
            d = fn(f, need)
        except Exception as e:
            errors.append(f"      {name}: {type(e).__name__}: {str(e).strip()}")
            continue
        missing = [c for c in need if c not in d.columns]
        if missing:
            raise SystemExit(f"{os.path.basename(f)} is missing {missing} - rebuild "
                             f"with nba_wp_features.py (and run nba_sim.py first)")
        if name != READERS[0][0]:
            print(f"    {os.path.basename(f)}: read via {name}")
        return d

    raise SystemExit(
        f"every reader failed on {f}\n"
        + "\n".join(errors)
        + f"\n  Rebuild season {season}:\n"
          f"    python nba_wp_features_v23.py --seasons {season}"
    )


def load(paths, seasons):
    # exp_margin_full is derived here, not stored, so it is not read from disk
    # exp_margin_left is always loaded even when it has been dropped as a
    # PREDICTOR, because the "current sim_adv" baseline line below is computed
    # from it. Without this, --drop-predictors exp_margin_left crashes on a
    # missing column instead of running.
    need = ["game_id", "season", "score_margin", "frac_left", "period",
            "home_final", "away_final", "sim_n_poss", "home_spread_close",
            "exp_margin_left"] + \
           [c for c in PREDICTORS if c != "exp_margin_full"]
    need = list(dict.fromkeys(need))
    frames = []
    for s in seasons:
        f = os.path.join(paths["features"], f"nba_wp_features_{s}.parquet")
        d = read_features(f, need, s)
        frames.append(d[need])
    df = pd.concat(frames, ignore_index=True)
    df["exp_margin_full"] = -df["home_spread_close"]
    return df


def prepare(df, min_poss):
    df = df.copy()
    df["rest_margin"] = (df["home_final"] - df["away_final"]) - df["score_margin"]
    df["n_half"] = df["sim_n_poss"] / 2.0
    df = df[df["n_half"] >= min_poss]
    df["realized_edge"] = df["rest_margin"] / df["n_half"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["realized_edge"] + PREDICTORS)
    return df


def wls(y, X, w, alpha=0.0, mu=None, sd=None):
    """
    Weighted least squares with an optional ridge penalty.

    The RAPM terms correlate ~0.88 with the closing spread and with each other,
    which is why plain OLS gave them large opposite signs that did not survive
    from one season to five. Ridge shrinks correlated predictors together
    instead. Standardising first is required, otherwise the penalty falls on
    whichever predictors happen to have the largest units.
    """
    if mu is None:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 1e-9, sd, 1.0)
    Z = (X - mu) / sd
    Zd = np.column_stack([np.ones(len(Z)), Z])
    sw = np.sqrt(w)
    A = Zd * sw[:, None]
    b = y * sw
    if alpha > 0:
        P = np.eye(Zd.shape[1]) * np.sqrt(alpha)
        P[0, 0] = 0.0                      # never penalise the intercept
        A = np.vstack([A, P])
        b = np.concatenate([b, np.zeros(Zd.shape[1])])
    beta_z, *_ = np.linalg.lstsq(A, b, rcond=None)
    # convert back to the original units so the saved coefficients apply directly
    beta = np.empty_like(beta_z)
    beta[1:] = beta_z[1:] / sd
    beta[0] = beta_z[0] - float(np.dot(beta[1:], mu))
    return beta, mu, sd


def r2(y, X, w, beta):
    pred = np.column_stack([np.ones(len(X)), X]) @ beta
    ss_res = float(np.sum(w * (y - pred) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    return 1.0 - ss_res / ss_tot


def fit_gbm(df, y, w, args, band_fn=None):
    """
    Same target, gradient boosted instead of linear. Split by GAME so the
    holdout R2 is honest, and early-stop on it.
    """
    games = df["game_id"].unique()
    rng = np.random.default_rng(11)
    g = np.array(sorted(games))
    rng.shuffle(g)
    n = int(round(len(g) * args.holdout_frac))
    hold = set(g[:n])
    m_h = df["game_id"].isin(hold).values
    X = df[PREDICTORS].values.astype(np.float32)

    dtr = xgb.DMatrix(X[~m_h], label=y[~m_h], weight=w[~m_h], feature_names=PREDICTORS)
    dho = xgb.DMatrix(X[m_h], label=y[m_h], weight=w[m_h], feature_names=PREDICTORS)
    params = {"objective": "reg:squarederror", "eval_metric": "rmse",
              "max_depth": args.gbm_depth, "eta": args.gbm_eta,
              "min_child_weight": args.gbm_min_child, "subsample": 0.8,
              "colsample_bytree": 0.8, "lambda": 5.0, "tree_method": "hist",
              "seed": 11, "nthread": 0}
    bst = xgb.train(params, dtr, num_boost_round=args.gbm_rounds,
                    evals=[(dho, "holdout")], early_stopping_rounds=30,
                    verbose_eval=False)
    pred = bst.predict(dho, iteration_range=(0, bst.best_iteration + 1))
    ss_res = float(np.sum(w[m_h] * (y[m_h] - pred) ** 2))
    ss_tot = float(np.sum(w[m_h] * (y[m_h] - np.average(y[m_h], weights=w[m_h])) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    print(f"  GBM edge model (holdout):               R2 {r2:.5f}  "
          f"best_iter {bst.best_iteration}")
    full = np.zeros(len(y))
    full[m_h] = pred
    if band_fn is not None:
        print(f"  GBM by band (holdout):    {band_fn(m_h, full)}")
    return bst, r2, m_h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--min-poss", type=float, default=5.0,
                    help="skip states with fewer than this many possessions left "
                         "per team; the edge is meaningless and the target explodes")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gbm", action="store_true",
                    help="fit a gradient-boosted edge model instead of the linear one")
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.0],
                    help="ridge penalties to try; the holdout picks the winner. "
                         "Default [0.0] = plain weighted least squares, which is "
                         "what the 1.33% configuration used. Pass several values "
                         "to search.")
    ap.add_argument("--gbm-depth", type=int, default=5)
    ap.add_argument("--gbm-eta", type=float, default=0.05)
    ap.add_argument("--gbm-min-child", type=float, default=2000.0)
    ap.add_argument("--gbm-rounds", type=int, default=2000)
    ap.add_argument("--drop-predictors", nargs="*", default=[],
                    help="predictors to remove from BOTH the linear and GBM edge "
                         "fits. Use to test whether the pregame line is worth "
                         "what the fit gives it: dropping exp_margin_full and "
                         "exp_margin_left leaves the line reaching adv only "
                         "through exp_margin_x_surprise_rate and the RAPM terms.")
    ap.add_argument("--weight-pow", type=float, default=0.5,
                    help="fit weight is (possessions left per team) ** this. "
                         "1.0 is the statistically efficient WLS weight for a "
                         "rate target and is what every version through v22 "
                         "used; it also gives a Q1 state ~10x the weight of a Q4 "
                         "state, so the edge model has effectively never been fit "
                         "where the money is lost. 0.5 keeps the direction of the "
                         "correction without crushing late states. 0.0 = uniform.")
    args = ap.parse_args()

    paths = resolve_paths(args.root)

    if args.drop_predictors:
        global PREDICTORS
        unknown = [c for c in args.drop_predictors if c not in PREDICTORS]
        if unknown:
            raise SystemExit(f"--drop-predictors names terms that are not in the "
                             f"predictor list: {unknown}")
        kept = [c for c in PREDICTORS if c not in set(args.drop_predictors)]
        if len(kept) < 2:
            raise SystemExit("refusing to drop almost every predictor")
        print(f"  DROPPED predictors: {sorted(set(args.drop_predictors))}")
        print(f"  {len(kept)} predictors remain")
        PREDICTORS = kept

    df = prepare(load(paths, args.seasons), args.min_poss)
    print(f"  {len(df):,} states / {df['game_id'].nunique():,} games, seasons {args.seasons}")

    y = df["realized_edge"].values
    # Weight by possessions left, raised to --weight-pow. At pow 1.0 (the old
    # behaviour) this is the efficient WLS weight, because Var(realized_edge)
    # scales as 1/n_half - but it also means the fit is dominated by early-game
    # states, and the road-favourite money is lost in Q3/Q4 close games. Lowering
    # the power trades statistical efficiency for relevance.
    w = np.power(df["n_half"].values, args.weight_pow)
    print(f"  fit weight: n_half ** {args.weight_pow} "
          f"(range {w.min():.2f}-{w.max():.2f})")

    # Bands used for every R2 printed below, so the effect of the weighting is
    # visible where it matters instead of only in a single pooled number.
    nh = df["n_half"].values
    BANDS = [("late   n_half<15", nh < 15),
             ("mid    15-40", (nh >= 15) & (nh < 40)),
             ("early  40+", nh >= 40)]

    # baseline: what the model does TODAY, edge = decayed pregame spread / n_half
    today = df["exp_margin_left"].values / df["n_half"].values
    ss_res = float(np.sum(w * (y - today) ** 2))
    ss_tot = float(np.sum(w * (y - np.average(y, weights=w)) ** 2))
    print(f"  current sim_adv (decayed spread only):  R2 {1.0 - ss_res / ss_tot:.5f}")

    X = df[PREDICTORS].values

    # Split by GAME and select the ridge penalty on the holdout. Every number
    # printed below is out-of-sample; nothing ships on an in-sample fit again.
    g = np.array(sorted(df["game_id"].unique()))
    np.random.default_rng(11).shuffle(g)
    hold = set(g[: int(round(len(g) * args.holdout_frac))])
    m_h = df["game_id"].isin(hold).values
    print(f"  holdout: {len(hold):,} games / {int(m_h.sum()):,} states")

    def band_r2(mask, pred_all):
        """Unweighted R2 inside each band, so a band is judged on its own terms."""
        out = []
        for lab, bm in BANDS:
            m = mask & bm
            if m.sum() < 1000:
                out.append(f"{lab} n/a")
                continue
            yy, pp = y[m], pred_all[m]
            ss = float(np.sum((yy - pp) ** 2))
            st = float(np.sum((yy - yy.mean()) ** 2))
            out.append(f"{lab} {1.0 - ss / st:+.5f}")
        return "   ".join(out)

    base = float(np.sum(w[m_h] * (y[m_h] - np.average(y[m_h], weights=w[m_h])) ** 2))
    best = None
    for a in args.alphas:
        b_a, mu, sd = wls(y[~m_h], X[~m_h], w[~m_h], alpha=a)
        pred = np.column_stack([np.ones(int(m_h.sum())), X[m_h]]) @ b_a
        ss = float(np.sum(w[m_h] * (y[m_h] - pred) ** 2))
        r2_h = 1.0 - ss / base
        print(f"    alpha {a:>10,.0f}: holdout R2 {r2_h:.5f}")
        if best is None or r2_h > best[1]:
            best = (a, r2_h)
    alpha = best[0]
    print(f"  chosen alpha {alpha:,.0f} (holdout R2 {best[1]:.5f})")
    b_h, _, _ = wls(y[~m_h], X[~m_h], w[~m_h], alpha=alpha)
    lin_hold_pred = np.zeros(len(y))
    lin_hold_pred[m_h] = np.column_stack([np.ones(int(m_h.sum())), X[m_h]]) @ b_h
    print(f"  linear by band (holdout): {band_r2(m_h, lin_hold_pred)}")

    # refit on everything at the chosen penalty
    beta, mu, sd = wls(y, X, w, alpha=alpha)
    print(f"  fitted edge model (full sample):         R2 {r2(y, X, w, beta):.5f}")

    print("\n--- coefficients (per-possession edge, home-positive) ---")
    rows = [{"term": "intercept", "coef": beta[0], "sd_x": np.nan, "contrib_sd": abs(beta[0])}]
    for i, name in enumerate(PREDICTORS):
        sd = float(np.std(X[:, i]))
        rows.append({"term": name, "coef": beta[i + 1], "sd_x": sd,
                     "contrib_sd": abs(beta[i + 1]) * sd})
    t = pd.DataFrame(rows).sort_values("contrib_sd", ascending=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:,.6f}"))

    if args.gbm:
        # linear R2 above is in-sample; score it on the same holdout for a fair
        # comparison before deciding which one ships
        bst, r2_gbm, m_h = fit_gbm(df, y, w, args, band_fn=band_r2)
        Xh = np.column_stack([np.ones(int(m_h.sum())), X[m_h]])
        lin_pred = Xh @ beta
        ss_res = float(np.sum(w[m_h] * (y[m_h] - lin_pred) ** 2))
        ss_tot = float(np.sum(w[m_h] * (y[m_h] - np.average(y[m_h], weights=w[m_h])) ** 2))
        print(f"  linear model on the SAME holdout:       R2 {1.0 - ss_res / ss_tot:.5f}")
        # Deliberately does NOT touch edge_model.json. The GBM is an experiment;
        # the feature build only ever reads the linear coefficients, so a losing
        # experiment cannot silently become the shipped model.
        gf = os.path.join(paths["features"], "edge_model_gbm.ubj")
        bst.save_model(gf)
        # v4. The booster alone is not enough for the feature build to use it:
        # it needs the predictor ORDER, and it needs to know whether the GBM
        # actually beat the linear model on the same holdout. Both go in a
        # sidecar next to the booster. nba_wp_features_v19.py reads the sidecar,
        # not this script, so a losing experiment still cannot ship itself.
        r2_lin_holdout = 1.0 - ss_res / ss_tot
        sf = os.path.join(paths["features"], "edge_model_gbm.json")
        json.dump({"predictors": PREDICTORS,
                   "dropped_predictors": sorted(set(args.drop_predictors)),
                   "booster": os.path.basename(gf),
                   "seasons": list(args.seasons),
                   "min_poss": args.min_poss,
                   "weight_pow": args.weight_pow,
                   "best_iteration": int(bst.best_iteration),
                   "r2_holdout_gbm": float(r2_gbm),
                   "r2_holdout_linear": float(r2_lin_holdout),
                   "beats_linear": bool(r2_gbm > r2_lin_holdout)}, open(sf, "w"), indent=2)
        print(f"\nwrote {gf}")
        print(f"wrote {sf}  (gbm {r2_gbm:.5f} vs linear {r2_lin_holdout:.5f} on the "
              f"same holdout, beats_linear={r2_gbm > r2_lin_holdout})")
        print("edge_model.json untouched - the linear coefficients are still on disk")
        return

    out = args.out or os.path.join(paths["features"], "edge_model.json")
    json.dump({"predictors": PREDICTORS,
               "intercept": float(beta[0]),
               "coefs": [float(b) for b in beta[1:]],
               "seasons": list(args.seasons),
               "min_poss": args.min_poss,
               "alpha": float(alpha),
               "r2_holdout": float(best[1]),
               "r2": float(r2(y, X, w, beta))}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
