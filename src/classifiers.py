

import warnings

import numpy as np
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, roc_curve,
                              balanced_accuracy_score, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 42, 123, 456, 1337]


RF_N_EST = 150
RF_DEPTH = 12
RF_MAX_FEATURES = "sqrt"
RF_CRITERION = "gini"


LR_C = 1.0

N_BOOTSTRAP = 2_000
SHAPIRO_ALPHA = 0.05


RF_PARAM_GRID = {
    "n_estimators": [100, 150, 200],
    "max_depth": [8, 12, 16],
}
LR_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0],
}
GRID_VAL_FRACTION = 0.2

warnings.filterwarnings("ignore")


def _fit_preprocessor(X_fit):
\
\
\
\
\

    X = np.asarray(X_fit, dtype=float).copy()
    if X.ndim != 2:
        raise ValueError("X_fit must be a 2-D array")

    medians = np.empty(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        finite = X[:, j][np.isfinite(X[:, j])]
        medians[j] = float(np.median(finite)) if len(finite) else 0.0
        X[~np.isfinite(X[:, j]), j] = medians[j]

    log_mask = np.zeros(X.shape[1], dtype=bool)
    for j in range(X.shape[1]):
        col = X[:, j]
        if len(col) >= 3 and np.min(col) >= 0:
            try:
                n_sw = min(len(col), 5000)
                _, p_sw = shapiro(col[:n_sw])
                log_mask[j] = bool(p_sw < SHAPIRO_ALPHA)
            except Exception:
                log_mask[j] = False
        if log_mask[j]:
            X[:, j] = np.log1p(np.clip(col, 0, None))

    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    return {"medians": medians, "log_mask": log_mask,
            "mean": mean, "scale": scale}


def _apply_preprocessor(X, state):

    X = np.asarray(X, dtype=float).copy()
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array")
    if X.shape[1] != len(state["medians"]):
        raise ValueError("Feature dimension does not match fitted preprocessor")

    for j, fill in enumerate(state["medians"]):
        X[~np.isfinite(X[:, j]), j] = fill
    if np.any(state["log_mask"]):
        X[:, state["log_mask"]] = np.log1p(
            np.clip(X[:, state["log_mask"]], 0, None))
    return (X - state["mean"]) / state["scale"]


def _preprocess(X_tr, X_te, y_tr=None):
\
\
\
\

    state = _fit_preprocessor(X_tr)
    return _apply_preprocessor(X_tr, state), _apply_preprocessor(X_te, state)


def _fold(df_tr, df_te, cols, fold_seed=None):

    X_tr = df_tr[cols].values.astype(float)
    y_tr = df_tr["label"].values
    X_te = df_te[cols].values.astype(float)
    y_te = df_te["label"].values
    X_tr, X_te = _preprocess(X_tr, X_te, y_tr)
    seed = fold_seed if fold_seed is not None else SEEDS[0]
    rng = np.random.default_rng(seed)
    n = min(np.sum(y_tr == 0), np.sum(y_tr == 1))
    i0 = rng.choice(np.where(y_tr == 0)[0], n, replace=False)
    i1 = rng.choice(np.where(y_tr == 1)[0], n, replace=False)
    idx = np.sort(np.concatenate([i0, i1]))
    return X_tr[idx], y_tr[idx], X_te, y_te


def _rf(seed, n_estimators=RF_N_EST, max_depth=RF_DEPTH):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        max_features=RF_MAX_FEATURES, criterion=RF_CRITERION,
        class_weight="balanced", random_state=seed, n_jobs=-1)


def _lr(seed, C=LR_C):
    return Pipeline([
        ("sc", StandardScaler()),
        ("clf", LogisticRegression(C=C, penalty="l2", max_iter=1000,
                                    class_weight="balanced", solver="lbfgs",
                                    random_state=seed)),
    ])


def _eer(y_true, y_score):
    try:
        fpr, tpr, thr = roc_curve(y_true, y_score, pos_label=1)
        fnr = 1. - tpr
        diff = fpr - fnr
        sc = np.where(np.diff(np.sign(diff)))[0]
        if len(sc) == 0:
            i = np.argmin(np.abs(diff))
            return float((fpr[i] + fnr[i]) / 2), float(thr[i])
        i = sc[0]
        d0, d1 = diff[i], diff[i + 1]
        if abs(d1 - d0) < 1e-12:
            return float((fpr[i] + fnr[i]) / 2), float(thr[i])
        a = d0 / (d0 - d1)
        return float(fpr[i] + a * (fpr[i + 1] - fpr[i])), float(thr[i] + a * (thr[i + 1] - thr[i]))
    except Exception:
        return np.nan, 0.5


def rf_grid_search(X_tr, y_tr, seed, param_grid=None, val_fraction=GRID_VAL_FRACTION):
\
\
\
\
\
\
\
\
\
\
\
\

    param_grid = param_grid or RF_PARAM_GRID
    try:
        X_g, X_v, y_g, y_v = train_test_split(
            X_tr, y_tr, test_size=val_fraction, random_state=seed, stratify=y_tr)
    except ValueError:
        return {"n_estimators": RF_N_EST, "max_depth": RF_DEPTH}

    best_params = {"n_estimators": RF_N_EST, "max_depth": RF_DEPTH}
    best_auc = -np.inf
    for n_est in param_grid["n_estimators"]:
        for depth in param_grid["max_depth"]:
            clf = _rf(seed, n_estimators=n_est, max_depth=depth)
            clf.fit(X_g, y_g)
            yp = clf.predict_proba(X_v)[:, 1]
            try:
                auc = roc_auc_score(y_v, yp)
            except Exception:
                continue

            if auc > best_auc + 1e-6:
                best_auc = auc
                best_params = {"n_estimators": n_est, "max_depth": depth}
            elif abs(auc - best_auc) <= 1e-6:

                if n_est == RF_N_EST and depth == RF_DEPTH:
                    best_params = {"n_estimators": n_est, "max_depth": depth}
    return best_params


def lr_grid_search(X_tr, y_tr, seed, param_grid=None, val_fraction=GRID_VAL_FRACTION):
\

    param_grid = param_grid or LR_PARAM_GRID
    try:
        X_g, X_v, y_g, y_v = train_test_split(
            X_tr, y_tr, test_size=val_fraction, random_state=seed, stratify=y_tr)
    except ValueError:
        return {"C": LR_C}

    best_params = {"C": LR_C}
    best_auc = -np.inf
    for C in param_grid["C"]:
        clf = _lr(seed, C=C)
        clf.fit(X_g, y_g)
        yp = clf.predict_proba(X_v)[:, 1]
        try:
            auc = roc_auc_score(y_v, yp)
        except Exception:
            continue
        if auc > best_auc + 1e-6:
            best_auc = auc
            best_params = {"C": C}
    return best_params


def _multiseed(X_tr, y_tr, X_te, y_te, lr_only=False, grid_search=True):
\
\
\
\
\
\
\
\
\

    aucs, eers, baccs, f1s = [], [], [], []
    seed_metrics = []

    params = None
    if grid_search:
        if lr_only:
            params = lr_grid_search(X_tr, y_tr, SEEDS[0])
        else:
            params = rf_grid_search(X_tr, y_tr, SEEDS[0])

    for s in SEEDS:
        if lr_only:
            clf = _lr(s, **params) if params else _lr(s)
        else:
            clf = _rf(s, **params) if params else _rf(s)
        clf.fit(X_tr, y_tr)
        yp = (clf.predict_proba(X_te)[:, 1]
              if hasattr(clf, "predict_proba") else clf.decision_function(X_te))
        yb = (yp >= 0.5).astype(int)
        try:
            auc = float(roc_auc_score(y_te, yp))
            eer = float(_eer(y_te, yp)[0])
            bacc = float(balanced_accuracy_score(y_te, yb))
            f1 = float(f1_score(y_te, yb, zero_division=0))
            aucs.append(auc)
            eers.append(eer)
            baccs.append(bacc)
            f1s.append(f1)
            seed_metrics.append({"seed": int(s), "auc": auc, "eer": eer,
                                 "balanced_acc": bacc, "f1": f1})
        except Exception:
            pass

    if not aucs:
        return {"auc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "eer": np.nan, "balanced_acc": np.nan, "f1": np.nan,
                "best_params": params, "seed_metrics": []}
    arr = np.array(aucs)
    rng = np.random.default_rng(SEEDS[0])
    boot = rng.choice(arr, (N_BOOTSTRAP, len(arr)), replace=True).mean(axis=1)
    return {"auc": float(np.mean(arr)),
            "ci_lo": float(np.percentile(boot, 2.5)),
            "ci_hi": float(np.percentile(boot, 97.5)),
            "eer": float(np.nanmean(eers)),
            "balanced_acc": float(np.nanmean(baccs)),
            "f1": float(np.nanmean(f1s)),
            "best_params": params,
            "seed_metrics": seed_metrics}
