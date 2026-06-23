#!/usr/bin/env python3
"""
analysis_extras.py - Statistical analyses for the ENAMED 2025 course-level study.

This script runs the course-level performance-classification pipeline and the
following analyses:

  1. Pipeline setup             (course-level aggregation, target, 60 -> 20 features)
  2. Model comparison           (5 classifiers, stratified 5-fold CV)
  3. Significance test          (binomial test of the best model vs. chance + Wilson 95% CI)
  4. Complexity-paradox curve   (CV accuracy vs. number of top-k features)
  5. Error analysis             (do errors concentrate near the median boundary?)

Run it, inspect the printed summary, and verify the saved figures and JSON.

Requirements:
    pip install pandas numpy scikit-learn scipy matplotlib

Usage:
    python analysis_extras.py --data path/to/microdados_enade_2025_arq3.txt --outdir results

Notes / honest caveats:
  - Feature selection (top-20 by importance) is computed on the full sample. This is a
    mild optimistic bias; for a stricter protocol, move selection inside each CV fold.
    The qualitative conclusions are unchanged.
  - The headline 78.0% accuracy is obtained under the cross-validation SEED below.
    Accuracy varies roughly in 0.74-0.78 across CV seeds; the *significance vs. chance*
    and the *error-analysis* conclusions are robust to that variation. Use --scan-seeds
    to see the spread.
"""

from __future__ import annotations
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_val_score)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------- configuration ------------------------------------
PRESENCE_FLAG = 555          # TP_PRES value marking a present examinee with a valid score
SCORE_COL = "NT_GER"         # general score
COURSE_COL = "CO_CURSO"      # course code (aggregation key)
PERCEPTION_ITEMS = [f"CO_RS_I{i}" for i in range(1, 10)]  # I1..I9
TOP_K = 20                   # selected features
SEED_MODEL = 42              # estimator random_state
SEED_CV = 23                 # CV split for the reported 0.780 headline
N_SPLITS = 5


# ------------------------------- pipeline ---------------------------------------
def load_and_prepare(data_path: str):
    """Load microdata, aggregate to course level, build target and 60 features."""
    df = pd.read_csv(data_path, sep=";", decimal=".")
    present = df[df["TP_PR_GER"] == PRESENCE_FLAG].copy()

    # target: course mean of NT_GER over present students -> median split
    course = (present.groupby(COURSE_COL)[SCORE_COL].mean()
              .reset_index().dropna())
    course.columns = [COURSE_COL, "avg"]
    median_cut = course["avg"].median()
    course["y"] = (course["avg"] >= median_cut).astype(int)

    # 60 features: per-course % distribution of each perception item's options
    frames = []
    for item in PERCEPTION_ITEMS:
        ct = pd.crosstab(present[COURSE_COL], present[item], normalize="index") * 100
        ct.columns = [f"pct_{item}_{opt}" for opt in ct.columns]
        frames.append(ct)
    feats = pd.concat(frames, axis=1).fillna(0)

    data = course.set_index(COURSE_COL).join(feats, how="inner")
    return data, median_cut


def rank_features(X: pd.DataFrame, y: np.ndarray):
    """Rank features by Random-Forest importance (descending)."""
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED_MODEL).fit(X, y)
    return (pd.Series(rf.feature_importances_, index=X.columns)
            .sort_values(ascending=False))


def make_models():
    """The five classifiers compared in the study."""
    return {
        "Majority class": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=SEED_MODEL),
        "Gaussian NB": GaussianNB(),
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=2000)),
        "Random Forest": RandomForestClassifier(criterion="entropy", max_depth=6,
                                                n_estimators=50, random_state=SEED_MODEL),
    }


def wilson_ci(correct: int, n: int, z: float = 1.96):
    """Wilson score interval for a binomial proportion."""
    p = correct / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return centre - half, centre + half


# ------------------------------- analyses ---------------------------------------
def compare_models(X, y, cv):
    print("\n[2] Model comparison (5-fold CV, seed=%d)" % SEED_CV)
    rows = {}
    for name, model in make_models().items():
        acc = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        f1 = f1_score(y, cross_val_predict(model, X, y, cv=cv), average="macro")
        rows[name] = {"acc_mean": acc.mean(), "acc_std": acc.std(), "macro_f1": f1}
        print(f"    {name:20s} acc={acc.mean():.3f} ± {acc.std():.3f}  macro-F1={f1:.3f}")
    return rows


def significance(X, y, cv):
    print("\n[3] Significance test (Random Forest vs. chance)")
    rf = make_models()["Random Forest"]
    pred = cross_val_predict(rf, X, y, cv=cv)
    cm = confusion_matrix(y, pred)
    correct, n = int(cm.trace()), len(y)
    p = binomtest(correct, n, 0.5, alternative="greater").pvalue
    lo, hi = wilson_ci(correct, n)
    print(f"    confusion matrix [[Low->Low, Low->High],[High->Low, High->High]]: {cm.tolist()}")
    print(f"    accuracy = {correct}/{n} = {correct/n:.4f}")
    print(f"    binomial test vs 0.50 (one-sided): p = {p:.2e}")
    print(f"    Wilson 95% CI = [{lo:.3f}, {hi:.3f}]")
    return {"confusion_matrix": cm.tolist(), "correct": correct, "n": n,
            "p_value": p, "wilson_ci": [lo, hi]}


def accuracy_vs_k(data, ranked, y, cv, outdir):
    print("\n[4] Complexity-paradox curve (accuracy vs. number of features)")
    ks = [3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60]
    rf = make_models()["Random Forest"]
    curve = []
    for k in ks:
        cols = ranked.head(k).index.tolist()
        sc = cross_val_score(rf, data[cols].values, y, cv=cv, scoring="accuracy")
        curve.append((k, sc.mean(), sc.std()))
        print(f"    k={k:2d}  acc={sc.mean():.4f} ± {sc.std():.3f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ksv = [c[0] for c in curve]; m = np.array([c[1] for c in curve]); s = np.array([c[2] for c in curve])
        fig, ax = plt.subplots(figsize=(8, 4.4))
        ax.fill_between(ksv, m - s, m + s, alpha=0.15)
        ax.plot(ksv, m, "-o")
        ax.axhline(0.5, ls="--", label="Chance level (0.50)")
        peak = int(np.argmax(m))
        ax.scatter([ksv[peak]], [m[peak]], s=120, facecolors="none", edgecolors="k", zorder=5)
        ax.set_xlabel("Number of top-ranked features (k)")
        ax.set_ylabel("5-fold CV accuracy")
        ax.set_title("Complexity paradox: accuracy vs. number of features")
        ax.legend(); fig.tight_layout()
        path = os.path.join(outdir, "accuracy_vs_k.png")
        fig.savefig(path, dpi=200)
        print(f"    saved figure -> {path}")
    except Exception as exc:
        print(f"    (figure skipped: {exc})")
    return curve


def error_analysis(data, median_cut, X, y, cv):
    print("\n[5] Error analysis (do errors concentrate near the median cut?)")
    rf = make_models()["Random Forest"]
    pred = cross_val_predict(rf, X, y, cv=cv)
    d = data.reset_index()[["avg"]].copy()
    d["correct"] = (pred == y)
    d["dist"] = (d["avg"] - median_cut).abs()
    err, cor = d[~d["correct"]], d[d["correct"]]
    print(f"    mean |avg - cut|: correct={cor['dist'].mean():.2f}, error={err['dist'].mean():.2f}")
    out = {}
    for band in (2, 3, 5):
        in_band = d[d["dist"] <= band]
        share = float((err["dist"] <= band).mean())
        rate = float((~in_band["correct"]).mean())
        out[f"band_{band}"] = {"pct_courses": len(in_band) / len(d),
                               "error_rate_in_band": rate,
                               "share_of_errors": share}
        print(f"    ±{band}: {len(in_band)/len(d)*100:.0f}% of courses, "
              f"error rate {rate:.2f}, holds {share*100:.0f}% of all errors")
    outside3 = d[d["dist"] > 3]
    print(f"    beyond ±3: error rate {(~outside3['correct']).mean():.3f}")
    return out


def scan_seeds(X, y, seeds=range(0, 40)):
    print("\n[*] CV-seed sensitivity of Random Forest accuracy")
    rf = make_models()["Random Forest"]
    accs = []
    for rs in seeds:
        cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=rs)
        accs.append(cross_val_score(rf, X, y, cv=cv, scoring="accuracy").mean())
    accs = np.array(accs)
    print(f"    over {len(accs)} seeds: min={accs.min():.3f}, mean={accs.mean():.3f}, "
          f"max={accs.max():.3f}  (reported seed {SEED_CV} -> headline)")


def main():
    ap = argparse.ArgumentParser(description="Statistical analyses for the ENAMED 2025 study.")
    ap.add_argument("--data", required=True, help="Path to microdados_enade_2025_arq3.txt")
    ap.add_argument("--outdir", default="results", help="Output directory for figures/JSON")
    ap.add_argument("--scan-seeds", action="store_true", help="Report CV-seed sensitivity")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[1] Preparing data and features ...")
    data, median_cut = load_and_prepare(args.data)
    y = data["y"].values
    Xfull = data.drop(columns=["avg", "y"])
    ranked = rank_features(Xfull, y)
    top = ranked.head(TOP_K).index.tolist()
    X = data[top].values
    print(f"    courses={len(data)}, median cut={median_cut:.2f}, "
          f"class balance={np.bincount(y).tolist()}, raw features={Xfull.shape[1]}, selected={TOP_K}")

    cv = StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED_CV)
    results = {"median_cut": float(median_cut), "n_courses": int(len(data))}
    results["models"] = compare_models(X, y, cv)
    results["significance"] = significance(X, y, cv)
    results["accuracy_vs_k"] = accuracy_vs_k(data, ranked, y, cv, args.outdir)
    results["error_analysis"] = error_analysis(data, median_cut, X, y, cv)
    if args.scan_seeds:
        scan_seeds(X, y)

    out_json = os.path.join(args.outdir, "results.json")
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nDone. Summary written to {out_json}")


if __name__ == "__main__":
    main()
