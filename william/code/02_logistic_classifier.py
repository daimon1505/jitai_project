"""
Milestone 2 - Logistic-regression stress classifier (RELAX dataset).

Label   : EMA item ifb-2 "I feel: excited(1) ... calm(7)"; stress = score <= 3.
Inputs  : HRV + accelerometer features from the window before each EMA
          (produced by 01_etl_features.py).
Protocol: leave-one-subject-out (LOSO). Inside each training fold, C is tuned
          with an inner GroupKFold over the *training* subjects; imputation and
          scaling are fit on training subjects only (sklearn Pipeline).

Usage: python 02_logistic_classifier.py <outputs dir>
"""
import sys
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, GridSearchCV
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore", category=UserWarning)
RNG = np.random.default_rng(0)

HRV = ["mean_rr", "sdnn", "rmssd", "pnn50", "mean_hr", "sd_hr", "cv_rr",
       "lf", "hf", "lf_hf"]
ACC = ["acc_mag_mean", "acc_mag_std", "acc_enmo", "acc_active_frac",
       "acc_axis_std"]
STRESS_CUTOFF = 3

# palette (dataviz reference instance, light mode)
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9890"
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.edgecolor": "#d6d5ce", "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10})


def load(out_dir, window):
    d = pd.read_parquet(os.path.join(out_dir, f"ema_windows_features_{window}min.parquet"))
    n_all = len(d)
    d = d[d["mean_rr"].notna()].copy()           # inclusion: valid HRV window
    d["y"] = (d["ifb_2"] <= STRESS_CUTOFF).astype(int)
    return d.reset_index(drop=True), n_all


def person_normalize(d, cols):
    """z-score each feature within subject (unsupervised: labels not used)."""
    d = d.copy()
    g = d.groupby("user")[cols]
    d[cols] = (d[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    return d


def make_model():
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=5000)),
    ])
    return GridSearchCV(pipe, {"lr__C": [0.001, 0.01, 0.1, 1, 10]},
                        scoring="roc_auc", cv=GroupKFold(n_splits=5))


def loso(d, feats, model="lr"):
    X, y, g = d[feats].to_numpy(), d["y"].to_numpy(), d["user"].to_numpy()
    prob = np.zeros(len(d))
    chosen_c = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if model == "lr":
            m = make_model()
            m.fit(X[tr], y[tr], groups=g[tr])
            chosen_c.append(m.best_params_["lr__C"])
        else:
            m = DummyClassifier(strategy=model, random_state=0,
                                constant=1 if model == "constant" else None)
            m.fit(X[tr], y[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]
    return prob, chosen_c


def metrics(y, prob, groups, thr=0.5):
    pred = (prob >= thr).astype(int)
    out = {
        "n": int(len(y)),
        "auroc": roc_auc_score(y, prob) if len(np.unique(prob)) > 1 else 0.5,
        "bal_acc": balanced_accuracy_score(y, pred),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
    }
    # subject-level bootstrap CI for pooled AUROC
    subs = np.unique(groups)
    idx_by = {s: np.nonzero(groups == s)[0] for s in subs}
    boots = []
    for _ in range(1000):
        idx = np.concatenate([idx_by[s] for s in RNG.choice(subs, len(subs))])
        if len(np.unique(y[idx])) == 2 and len(np.unique(prob[idx])) > 1:
            boots.append(roc_auc_score(y[idx], prob[idx]))
    out["auroc_ci"] = (np.percentile(boots, 2.5), np.percentile(boots, 97.5)) if boots else (np.nan, np.nan)
    return out


def per_subject(d, prob):
    rows = []
    for s, idx in d.groupby("user").groups.items():
        y = d.loc[idx, "y"].to_numpy()
        p = prob[idx]
        rows.append({"user": s, "n": len(y), "n_stress": int(y.sum()),
                     "auroc": roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
                     "bal_acc": balanced_accuracy_score(y, p >= .5) if 0 < y.sum() < len(y) else np.nan})
    return pd.DataFrame(rows)


def main():
    out_dir = sys.argv[1]
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    results, store = [], {}

    for window in (10, 30):
        d, n_all = load(out_dir, window)
        dn = person_normalize(d, HRV + ACC)
        print(f"\n=== window {window} min: {len(d)}/{n_all} EMAs usable, "
              f"{d.user.nunique()} subjects, stress rate {d.y.mean():.3f}")
        configs = [
            ("Dummy: always 'stress'", d, HRV, "constant"),
            ("Dummy: stratified random", d, HRV, "stratified"),
            ("LR: HRV only, raw", d, HRV, "lr"),
            ("LR: HRV + ACC, raw", d, HRV + ACC, "lr"),
            ("LR: HRV only, per-person z", dn, HRV, "lr"),
            ("LR: HRV + ACC, per-person z", dn, HRV + ACC, "lr"),
        ]
        for name, data, feats, model in configs:
            prob, cs = loso(data, feats, model)
            m = metrics(data["y"].to_numpy(), prob, data["user"].to_numpy())
            ps = per_subject(data, prob)
            m.update({"window_min": window, "model": name,
                      "subj_auroc_mean": ps["auroc"].mean(),
                      "subj_auroc_sd": ps["auroc"].std(),
                      "subj_auroc_n": int(ps["auroc"].notna().sum()),
                      "C_mode": pd.Series(cs).mode().iloc[0] if cs else None})
            results.append(m)
            store[(window, name)] = (data, prob, ps, feats)
            print(f"{name:32s} AUROC {m['auroc']:.3f} [{m['auroc_ci'][0]:.2f}-{m['auroc_ci'][1]:.2f}]"
                  f"  balAcc {m['bal_acc']:.3f}  F1 {m['f1']:.3f}  "
                  f"subjAUROC {m['subj_auroc_mean']:.3f}±{m['subj_auroc_sd']:.3f}")

    res = pd.DataFrame(results)
    res["auroc_ci"] = res["auroc_ci"].apply(lambda t: f"{t[0]:.2f}-{t[1]:.2f}")
    res.to_csv(os.path.join(out_dir, "logistic_results.csv"), index=False)

    # ------------------------------------------------------------------
    # Pick the main model = best LOSO AUROC among LR configs (10-min window)
    # ------------------------------------------------------------------
    lr10 = res[(res.window_min == 10) & res.model.str.startswith("LR")]
    main_name = lr10.sort_values("auroc", ascending=False).model.iloc[0]
    data, prob, ps, feats = store[(10, main_name)]
    ps.to_csv(os.path.join(out_dir, "per_subject_main.csv"), index=False)
    print("\nMain model:", main_name)

    # final fit on all subjects for coefficients + permutation importance
    m = make_model()
    X, y, g = data[feats].to_numpy(), data["y"].to_numpy(), data["user"].to_numpy()
    m.fit(X, y, groups=g)
    best = m.best_estimator_
    n_ind = best.named_steps["impute"].indicator_.features_ if best.named_steps["impute"].indicator_ else []
    names = feats + [f"missing_{feats[i]}" for i in n_ind]
    coef = pd.Series(best.named_steps["lr"].coef_[0], index=names)
    pi = permutation_importance(best, X, y, scoring="roc_auc", n_repeats=30, random_state=0)
    imp = pd.DataFrame({"coef_std": coef.reindex(feats).values,
                        "perm_auc_drop": pi.importances_mean,
                        "perm_sd": pi.importances_std}, index=feats)
    imp = imp.sort_values("perm_auc_drop", ascending=False)
    imp.to_csv(os.path.join(out_dir, "feature_importance_main.csv"))
    print(imp.round(4).to_string())
    print("missing-indicator coefs:", coef.drop(feats, errors="ignore").round(3).to_dict())

    # univariate sanity check: stress vs non-stress feature means (person-z)
    dn = person_normalize(data, feats) if "raw" in main_name else data
    uni = dn.groupby("y")[feats].mean().T
    uni.columns = ["non_stress_mean", "stress_mean"]
    uni.to_csv(os.path.join(out_dir, "univariate_means.csv"))

    # ---------------- figures ----------------
    # 1. ROC: main model vs. raw-HRV model
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--")
    other = "LR: HRV only, raw" if main_name != "LR: HRV only, raw" else "LR: HRV + ACC, per-person z"
    for nm, col in ((main_name, BLUE), (other, ORANGE)):
        dd, pp, _, _ = store[(10, nm)]
        fpr, tpr, _ = roc_curve(dd["y"], pp)
        a = res[(res.window_min == 10) & (res.model == nm)].auroc.iloc[0]
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{nm} (AUROC {a:.2f})")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("LOSO ROC, 10-min window", color=INK, loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "roc_loso.png"), dpi=150); plt.close(fig)

    # 2. feature importance (permutation AUROC drop)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    s = imp.sort_values("perm_auc_drop")
    ax.barh(s.index, s.perm_auc_drop, xerr=s.perm_sd, color=BLUE, height=0.6,
            error_kw={"ecolor": INK2, "lw": 1})
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Drop in AUROC when feature is shuffled")
    ax.set_title(f"Permutation importance - {main_name}", color=INK, loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "feature_importance.png"), dpi=150); plt.close(fig)

    # 3. per-subject AUROC
    p2 = ps.dropna(subset=["auroc"]).sort_values("auroc")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axhline(0.5, color=GREY, lw=1, ls="--")
    ax.axhline(0.8, color=INK2, lw=0.8, ls=":")
    ax.scatter(range(len(p2)), p2.auroc, s=14 + 2 * p2.n, color=BLUE,
               edgecolor=SURF, linewidth=1.5, zorder=3)
    ax.set_xticks(range(len(p2))); ax.set_xticklabels(p2.user.astype(str), fontsize=7)
    ax.set_xlabel("Held-out subject (dot size = # EMA windows)"); ax.set_ylabel("AUROC")
    ax.text(len(p2) - 0.5, 0.81, "target 0.80", ha="right", fontsize=8, color=INK2)
    ax.text(len(p2) - 0.5, 0.51, "chance", ha="right", fontsize=8, color=INK2)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-subject AUROC (LOSO) - {main_name}", color=INK, loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "per_subject_auroc.png"), dpi=150); plt.close(fig)

    with open(os.path.join(out_dir, "main_model.json"), "w") as f:
        json.dump({"main_model": main_name, "best_C_full_fit": m.best_params_["lr__C"]}, f)
    print(res.drop(columns=["precision"]).round(3).to_string())


if __name__ == "__main__":
    main()
