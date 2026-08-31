"""
nba_wp_train_v12.py

Trains the NBA live win-probability model (XGBoost, binary:logistic) on the
feature tables written by nba_wp_features.py.

Splits are made by GAME, never by row, so no game appears in two folds.
A shared EVAL fold is held out from every ensemble member and is never trained
or early-stopped on; it is the only clean number reported.

Usage (PowerShell):
  python nba_wp_train_v12.py --seasons 2021 2022 2023 2024 2025 --tag v1
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from nba_wp_features import FEATURES as _FEATURES_FALLBACK
from nba_wp_features import DEFAULT_ROOT, resolve_paths


def load_feature_list(paths):
    """
    Read the feature list written by whichever feature build produced the
    parquets. Importing it from nba_wp_features instead means a versioned
    build's new columns get silently dropped at training time.
    """
    f = os.path.join(paths["features"], "feature_list.json")
    if os.path.exists(f):
        d = json.load(open(f))
        print(f"  feature list from {os.path.basename(f)}: {d['n']} features "
              f"(written by {d.get('written_by', 'unknown')})")
        return list(d["features"])
    print("  WARNING: no feature_list.json - falling back to the FEATURES list "
          "imported from nba_wp_features.py. If you ran a versioned feature "
          "build, its new columns will be IGNORED. Rebuild features first.")
    return list(_FEATURES_FALLBACK)


def load_features(paths, seasons):
    frames = []
    for s in seasons:
        f = os.path.join(paths["features"], f"nba_wp_features_{s}.parquet")
        if not os.path.exists(f):
            raise FileNotFoundError(f"missing {f} - run nba_wp_features.py --seasons {s}")
        d = pd.read_parquet(f)
        frames.append(d)
        print(f"  {s}: {len(d):,} rows / {d['game_id'].nunique():,} games")
    df = pd.concat(frames, ignore_index=True)
    print(f"  total: {len(df):,} rows / {df['game_id'].nunique():,} games")
    return df


def game_split(games, frac, seed, outcome=None):
    """
    Split by game. When `outcome` (game_id -> home_won) is supplied the split is
    stratified on it, so both sides carry the same home-win base rate. Without
    stratification a random split drifts by 2+ points, which shows up as a
    uniform calibration gap and makes run-to-run comparison unreliable.
    """
    rng = np.random.default_rng(seed)
    g = np.array(sorted(games))
    if outcome is None:
        rng.shuffle(g)
        n = int(round(len(g) * frac))
        return set(g[:n]), set(g[n:])

    a, b = set(), set()
    y = np.array([outcome.get(x, -1) for x in g])
    for lab in np.unique(y):
        sub = g[y == lab]
        rng.shuffle(sub)
        n = int(round(len(sub) * frac))
        a.update(sub[:n])
        b.update(sub[n:])
    return a, b


def fit_calibrator(kind, p, y):
    """
    v12 adds tail_iso.

    The auto-selection has picked `raw` on every run, and the EVAL longshot
    table shows why that is not the same as being calibrated: under 0.05 the
    model says 0.0116 and the truth is 0.0081, a 43% overstatement, and that is
    measured out-of-sample on the EVAL fold with no Kalshi data involved. It is
    also where the 0-9c bets come from, and that bucket has run -21% to -33% in
    every version so far.

    Full isotonic loses on overall logloss because it distorts the bulk, where
    the model is already fine, and the bulk is 90% of the rows. tail_iso fits
    the same isotonic curve but only applies it in the extremes, ramping
    linearly to zero effect by `lo` from below and by `hi` from above, so the
    middle of the distribution is returned untouched. Several lo/hi pairs are
    offered to the same cross-fit selection that already runs - nothing is
    hardcoded, and if the tail is not really miscalibrated then raw still wins.
    """
    if kind.startswith("tail_iso"):
        lo = float(kind.split("_")[-1])
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        return {"kind": "tail_iso", "lo": lo, "hi": 1.0 - lo,
                "x": [float(v) for v in iso.X_thresholds_],
                "y": [float(v) for v in iso.y_thresholds_]}
    if kind == "platt":
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        lr.fit(z.reshape(-1, 1), y)
        return {"kind": "platt", "a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}
    if kind == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        iso.fit(p, y)
        return {"kind": "isotonic",
                "x": [float(v) for v in iso.X_thresholds_],
                "y": [float(v) for v in iso.y_thresholds_]}
    return {"kind": "raw"}


def apply_calibrator(cal, p):
    if cal["kind"] == "tail_iso":
        lo, hi = float(cal["lo"]), float(cal["hi"])
        iso = np.interp(p, cal["x"], cal["y"])
        # full weight inside the tail, ramping to zero effect by 2*lo (and by
        # 1-2*(1-hi) from above). A plain ramp that starts at 0 weight right at
        # `lo` only half-corrects the tail it is supposed to fix; this corrects
        # it and puts the blend in the band just outside instead.
        w_lo = np.clip((2.0 * lo - p) / max(lo, 1e-9), 0.0, 1.0)
        w_hi = np.clip((p - (2.0 * hi - 1.0)) / max(1.0 - hi, 1e-9), 0.0, 1.0)
        w = np.maximum(w_lo, w_hi)
        return w * iso + (1.0 - w) * p
    if cal["kind"] == "platt":
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return 1.0 / (1.0 + np.exp(-(cal["a"] * z + cal["b"])))
    if cal["kind"] == "isotonic":
        return np.interp(p, cal["x"], cal["y"])
    return p


def predict_ensemble(boosters, X, feature_names, margin=None):
    dm = xgb.DMatrix(X, feature_names=feature_names)
    if margin is not None:
        dm.set_base_margin(margin)
    logits = []
    for b in boosters:
        p = b.predict(dm, iteration_range=(0, b.best_iteration + 1))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        logits.append(np.log(p / (1 - p)))
    z = np.mean(logits, axis=0)
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--seasons", nargs="+", type=int, required=True)
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--seeds", nargs="+", type=int, default=[17, 41, 99, 123, 271])
    ap.add_argument("--eval-frac", type=float, default=0.15)
    ap.add_argument("--stop-frac", type=float, default=0.15)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--min-child-weight", type=float, default=200.0)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample", type=float, default=0.8)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--num-round", type=int, default=3000)
    ap.add_argument("--early-stopping", type=int, default=50)
    ap.add_argument("--season-decay", type=float, default=1.0,
                    help="weight a row from season s by D**(newest-s). 1.0 = off")
    ap.add_argument("--calibrator", default="auto")
    ap.add_argument("--tail-cuts", nargs="+", type=float, default=[0.05, 0.10, 0.20],
                    help="lo cut points offered to the tail_iso calibrator; the "
                         "cross-fit picks one, or picks raw/platt/isotonic instead")
    ap.add_argument("--drop-features", nargs="*", default=[])
    ap.add_argument("--bet-weights", default=None,
                    help="path to a nba_backtest_bets_<tag>.csv. Training rows are "
                         "reweighted toward the states that actually produced bets, "
                         "so the trees spend their capacity where you trade instead "
                         "of on the bulk of the distribution.")
    ap.add_argument("--bet-weight-strength", type=float, default=1.0,
                    help="0 = off, 1 = match the betting distribution")
    ap.add_argument("--base-margin", default="sim_logit",
                    help="feature column used as the XGBoost base margin, so the "
                         "trees fit only the residual to it. 'none' disables.")
    args = ap.parse_args()

    paths = resolve_paths(args.root)
    os.makedirs(paths["models"], exist_ok=True)

    print("Loading features ...")
    df = load_features(paths, args.seasons)

    all_features = load_feature_list(paths)
    feats = [f for f in all_features if f not in set(args.drop_features)]
    unknown = [f for f in args.drop_features if f not in all_features]
    if args.drop_features:
        print(f"  dropped: {sorted(set(args.drop_features) & set(all_features))}")
    if unknown:
        print(f"  WARNING: --drop-features names not in the feature set: {unknown}")
    print(f"  {len(feats)} features")

    if args.base_margin != "none":
        if args.base_margin not in df.columns:
            raise SystemExit(f"--base-margin {args.base_margin} is not a column in the "
                             f"feature tables; rebuild with nba_wp_features.py")
        margin_all = np.nan_to_num(df[args.base_margin].values.astype(float),
                                   nan=0.0, posinf=8.0, neginf=-8.0)
        feats = [f for f in feats if f != args.base_margin]
        print(f"  base margin: {args.base_margin} (removed from the feature list)")
    else:
        margin_all = None

    y_all = df["home_won"].values
    games = df["game_id"].unique()
    outcome = df.drop_duplicates("game_id").set_index("game_id")["home_won"].to_dict()

    eval_games, rest_games = game_split(games, args.eval_frac, seed=1234, outcome=outcome)
    is_eval = df["game_id"].isin(eval_games).values
    print(f"  EVAL fold: {len(eval_games):,} games / {is_eval.sum():,} rows "
          f"(home_won {y_all[is_eval].mean():.4f})")
    print(f"  pool:      {len(rest_games):,} games / {(~is_eval).sum():,} rows "
          f"(home_won {y_all[~is_eval].mean():.4f})")

    newest = max(args.seasons)
    w_all = np.power(args.season_decay, newest - df["season"].values).astype(float)
    if args.season_decay != 1.0:
        print(f"  season decay {args.season_decay}: " +
              ", ".join(f"{s}={args.season_decay ** (newest - s):.3f}" for s in sorted(args.seasons)))

    if args.bet_weights:
        bets = pd.read_csv(args.bet_weights, usecols=["p_win", "side", "staked"])
        # p_win is from the bet side; convert back to a home-win probability so it
        # lives on the same axis as the model's own output
        p_home_bet = np.where(bets["side"].values == "home",
                              bets["p_win"].values, 1.0 - bets["p_win"].values)
        edges = np.linspace(0.0, 1.0, 21)
        bet_hist, _ = np.histogram(p_home_bet, bins=edges,
                                   weights=bets["staked"].values)
        model_p = np.clip(df["sim_wp"].values.astype(float), 0.0, 1.0)
        base_hist, _ = np.histogram(model_p, bins=edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(base_hist > 0, bet_hist / np.maximum(base_hist, 1), 0.0)
        ratio = ratio / max(ratio.mean(), 1e-9)
        idx = np.clip(np.digitize(model_p, edges) - 1, 0, len(ratio) - 1)
        bw = ratio[idx]
        bw = 1.0 + args.bet_weight_strength * (bw - 1.0)
        bw = np.clip(np.nan_to_num(bw, nan=1.0), 0.05, 20.0)
        w_all = w_all * bw
        print(f"  bet weighting from {os.path.basename(args.bet_weights)}: "
              f"weight range {bw.min():.2f}-{bw.max():.2f}, "
              f"{100.0 * (bw > 1).mean():.0f}% of rows upweighted")

    X_all = df[feats].values.astype(np.float32)
    d_eval = xgb.DMatrix(X_all[is_eval], label=y_all[is_eval], feature_names=feats)
    if margin_all is not None:
        d_eval.set_base_margin(margin_all[is_eval])

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": args.max_depth,
        "eta": args.eta,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample,
        "lambda": args.reg_lambda,
        "tree_method": "hist",
        "nthread": 0,
    }

    boosters, members = [], []
    for i, seed in enumerate(args.seeds):
        stop_g, train_g = game_split(rest_games, args.stop_frac, seed=seed, outcome=outcome)
        m_stop = df["game_id"].isin(stop_g).values & ~is_eval
        m_train = df["game_id"].isin(train_g).values & ~is_eval

        dtr = xgb.DMatrix(X_all[m_train], label=y_all[m_train], weight=w_all[m_train], feature_names=feats)
        dst = xgb.DMatrix(X_all[m_stop], label=y_all[m_stop], weight=w_all[m_stop], feature_names=feats)
        if margin_all is not None:
            dtr.set_base_margin(margin_all[m_train])
            dst.set_base_margin(margin_all[m_stop])

        p = dict(params, seed=seed)
        bst = xgb.train(p, dtr, num_boost_round=args.num_round,
                        evals=[(dst, "stop")],
                        early_stopping_rounds=args.early_stopping,
                        verbose_eval=False)
        pe = bst.predict(d_eval, iteration_range=(0, bst.best_iteration + 1))
        ll = log_loss(y_all[is_eval], pe)
        print(f"  member {i} seed {seed}: best_iter {bst.best_iteration}  EVAL logloss {ll:.5f}")

        f = os.path.join(paths["models"], f"nba_wp_{args.tag}_m{i}.json")
        bst.save_model(f)
        boosters.append(bst)
        members.append(os.path.basename(f))

    p_eval = predict_ensemble(boosters, X_all[is_eval], feats,
                              margin=None if margin_all is None else margin_all[is_eval])
    y_eval = y_all[is_eval]
    print(f"\n  ENSEMBLE EVAL: logloss {log_loss(y_eval, p_eval):.5f}  AUC {roc_auc_score(y_eval, p_eval):.5f}")

    # calibrator selection cross-fits on halves of EVAL
    eval_g = np.array(sorted(eval_games))
    rng = np.random.default_rng(7)
    rng.shuffle(eval_g)
    half = set(eval_g[: len(eval_g) // 2])
    m_a = df.loc[is_eval, "game_id"].isin(half).values
    m_b = ~m_a

    if args.calibrator == "auto":
        scores = {}
        for kind in (["raw", "platt", "isotonic"] +
                     [f"tail_iso_{c}" for c in args.tail_cuts]):
            tot, n = 0.0, 0
            for fit_m, sc_m in [(m_a, m_b), (m_b, m_a)]:
                cal = fit_calibrator(kind, p_eval[fit_m], y_eval[fit_m])
                tot += log_loss(y_eval[sc_m], np.clip(apply_calibrator(cal, p_eval[sc_m]), 1e-6, 1 - 1e-6)) * sc_m.sum()
                n += sc_m.sum()
            scores[kind] = tot / n
            print(f"  calibrator {kind}: cross-fit logloss {scores[kind]:.5f}")
        chosen = min(scores, key=scores.get)
    else:
        chosen = args.calibrator
    cal = fit_calibrator(chosen, p_eval, y_eval) if chosen != "raw" else {"kind": "raw"}
    if chosen.startswith("tail_iso"):
        m_t = (p_eval < cal["lo"]) | (p_eval > cal["hi"])
        print(f"  tail_iso lo {cal['lo']:.2f}: reshapes {100.0 * m_t.mean():.1f}% of "
              f"EVAL rows, bulk returned unchanged")
    print(f"  calibrator: {chosen}")

    p_cal = apply_calibrator(cal, p_eval)
    print("\n  calibration on EVAL (decile of predicted home WP):")
    q = pd.qcut(p_cal, 10, labels=False, duplicates="drop")
    tab = pd.DataFrame({"pred": p_cal, "act": y_eval, "q": q}).groupby("q").agg(
        n=("act", "size"), pred=("pred", "mean"), actual=("act", "mean"))
    tab["gap"] = tab["actual"] - tab["pred"]
    print(tab.round(4).to_string())

    print("\n  longshot tail on EVAL (this is where the 0-9c bets come from):")
    for hi in (0.02, 0.05, 0.10):
        m = p_cal < hi
        if m.sum():
            print(f"    pred < {hi:.2f}: n {m.sum():>7,}  mean pred {p_cal[m].mean():.4f}  "
                  f"actual {y_eval[m].mean():.4f}")
        m = p_cal > 1 - hi
        if m.sum():
            print(f"    pred > {1-hi:.2f}: n {m.sum():>7,}  mean pred {p_cal[m].mean():.4f}  "
                  f"actual {y_eval[m].mean():.4f}")

    meta = {
        "tag": args.tag,
        "features": feats,
        "members": members,
        "calibrator": cal,
        "seasons": args.seasons,
        "seeds": args.seeds,
        "params": params,
        "season_decay": args.season_decay,
        "base_margin": args.base_margin,
        "eval_logloss": float(log_loss(y_eval, p_cal)),
        "eval_auc": float(roc_auc_score(y_eval, p_eval)),
    }
    mf = os.path.join(paths["models"], f"nba_wp_{args.tag}_meta.json")
    json.dump(meta, open(mf, "w"), indent=2)
    print(f"\n  wrote {mf}")


if __name__ == "__main__":
    main()
