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

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold


MAILABS_FEAT = "/content/drive/MyDrive/results_paper1_v5/features_mlaad.pkl"
GGMDDC_FEAT  = "/content/gdrive/MyDrive/results_ggmddc_v5/features_ggmddc.pkl"
LOGO_RESULTS = "/content/drive/MyDrive/results_paper1_v5/tables/SQ3_logo_mlaad.csv"
OUTPUT_DIR   = "/content/drive/MyDrive/results_paper1_v5"


SHARED_LANGUAGES = {"en", "es", "fr", "ru"}


GGMDDC_TO_ISO = {
    "english": "en", "spanish": "es", "french": "fr", "russian": "ru"
}

N_FOLDS = 5
N_SEEDS = 5
SEEDS   = [0, 42, 123, 456, 1337]
N_BOOTSTRAP = 2_000


def load_genuine_features(mailabs_path, ggmddc_path):
\
\
\
\
\
\
\

    print("Loading M-AILABS features...")
    dm = pd.read_pickle(mailabs_path)
    dm = dm[dm["label"] == 0].copy()
    dm = dm[dm["language"].isin(SHARED_LANGUAGES)].copy()
    dm["corpus"] = 0
    dm["label"]  = 0

    print("Loading GGMDDC features...")
    dg = pd.read_pickle(ggmddc_path)
    dg = dg[dg["label"] == 0].copy()

    dg["language"] = dg["language"].map(GGMDDC_TO_ISO)
    dg = dg.dropna(subset=["language"])
    dg = dg[dg["language"].isin(SHARED_LANGUAGES)].copy()
    dg["corpus"] = 1
    dg["label"]  = 1

    print(f"  M-AILABS genuine: {len(dm):,} utterances "
          f"({dm['language'].value_counts().to_dict()})")
    print(f"  GGMDDC genuine:   {len(dg):,} utterances "
          f"({dg['language'].value_counts().to_dict()})")


    rng = np.random.default_rng(42)
    balanced_m, balanced_g = [], []
    per_language = {}
    for lang in sorted(SHARED_LANGUAGES):
        ml = dm[dm["language"] == lang]
        gg = dg[dg["language"] == lang]
        n_lang = min(len(ml), len(gg))
        if n_lang == 0:
            continue
        im = rng.choice(ml.index.to_numpy(), n_lang, replace=False)
        ig = rng.choice(gg.index.to_numpy(), n_lang, replace=False)
        balanced_m.append(ml.loc[im])
        balanced_g.append(gg.loc[ig])
        per_language[lang] = int(n_lang)

    if not balanced_m or not balanced_g:
        raise ValueError("No shared language contains genuine speech in both corpora")

    dm = pd.concat(balanced_m, ignore_index=True)
    dg = pd.concat(balanced_g, ignore_index=True)
    df = pd.concat([dm, dg], ignore_index=True)
    n_per_corpus = len(dm)
    print(f"  Per-language matched counts per corpus: {per_language}")
    print(f"  Total after cell balancing: {len(df):,} "
          f"(M-AILABS={len(dm):,}, GGMDDC={len(dg):,})")
    return df, n_per_corpus


def run_provenance_control(df, families, output_dir):
\
\
\
\

    from src.classifiers import _fold, _multiseed
    from src.common import _seed

    _seed(42)
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    y = df["label"].values

    strata = df["corpus"].astype(str) + "_" + df["language"].astype(str)
    cell_counts = strata.value_counts()
    if (cell_counts < N_FOLDS).any():
        bad = cell_counts[cell_counts < N_FOLDS].to_dict()
        raise ValueError(f"Corpus-language cells too small for {N_FOLDS}-fold CV: {bad}")
    results = []
    fold_results = []

    for fam, cols in families.items():
        print(f"\n  Family: {fam} (d={len(cols)})")
        aucs = []

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
        for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(df, strata)):
            dtr = df.iloc[tr_idx].copy()
            dte = df.iloc[te_idx].copy()
            Xtr, ytr, Xte, yte = _fold(dtr, dte, cols, fold_seed=fold_idx)
            rf = _multiseed(Xtr, ytr, Xte, yte, grid_search=True)
            auc = rf["auc"]
            aucs.append(auc)
            fold_row = {
                "family": fam, "fold": fold_idx, "auc": auc,
                "eer": rf["eer"],
                "rf_n_estimators": (rf["best_params"].get("n_estimators")
                                    if rf.get("best_params") else None),
                "rf_max_depth": (rf["best_params"].get("max_depth")
                                 if rf.get("best_params") else None),
            }
            for seed_metric in rf.get("seed_metrics", []):
                seed = int(seed_metric["seed"])
                fold_row[f"auc_seed_{seed}"] = seed_metric["auc"]
                fold_row[f"eer_seed_{seed}"] = seed_metric["eer"]

            test_cells = strata.iloc[te_idx].value_counts().to_dict()
            fold_row["test_cell_counts"] = ";".join(
                f"{k}:{v}" for k, v in sorted(test_cells.items()))
            fold_results.append(fold_row)
            print(f"    Fold {fold_idx}: AUC={auc:.4f}  RF={rf.get('best_params')}")

        mean_auc = float(np.mean(aucs))
        arr = np.array(aucs)
        rng = np.random.default_rng(42)
        boot = rng.choice(arr, (N_BOOTSTRAP, len(arr)), replace=True).mean(axis=1)
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))
        results.append({"family": fam, "provenance_auc": mean_auc,
                         "ci_lo": ci_lo, "ci_hi": ci_hi})
        print(f"  {fam}: AUC={mean_auc:.4f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    df_res = pd.DataFrame(results).sort_values("provenance_auc", ascending=False)
    df_folds = pd.DataFrame(fold_results)


    if LOGO_RESULTS and Path(LOGO_RESULTS).exists():
        dlogo = pd.read_csv(LOGO_RESULTS)
        logo_mean = (dlogo.groupby("family")["auc_rf"].mean()
                     .reset_index().rename(columns={"auc_rf": "logo_auc"}))
        df_res = df_res.merge(logo_mean, on="family", how="left")
        valid = df_res.dropna(subset=["provenance_auc", "logo_auc"])
        if len(valid) >= 3:
            rho, p = spearmanr(valid["provenance_auc"], valid["logo_auc"])
            print(f"\nSpearman rho (provenance vs LOGO AUC): {rho:.3f}  p={p:.4f}  n={len(valid)}")
            df_res["spearman_rho_vs_logo"] = rho
            df_res["spearman_p_vs_logo"] = p

    out_csv = tables_dir / "provenance_control.csv"
    folds_csv = tables_dir / "provenance_control_per_fold.csv"
    df_res.to_csv(out_csv, index=False)
    df_folds.to_csv(folds_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(f"Saved: {folds_csv}")
    print("\n" + df_res[["family", "provenance_auc", "ci_lo", "ci_hi"]].to_string(index=False))
    return df_res


def main():
    from src.features import families_from_columns, EXCLUDED_FEATS
    from src.common import _feat_cols

    META_COLS = {"filepath", "language", "label", "generator",
                 "model_name", "duration", "split", "corpus"}

    df, n_per_corpus = load_genuine_features(MAILABS_FEAT, GGMDDC_FEAT)
    print(f"\nControl set: {len(df):,} genuine utterances "
          f"({n_per_corpus:,} per corpus), "
          f"languages: {sorted(df['language'].unique())}")

    all_cols = _feat_cols(df, META_COLS)
    families = families_from_columns(all_cols)
    total_dims = sum(len(v) for v in families.values())
    print(f"Families: {list(families.keys())} | Total dims: {total_dims}/155")

    run_provenance_control(df, families, OUTPUT_DIR)


if __name__ == "__main__":
    main()
