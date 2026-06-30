import argparse
from pathlib import Path

SHARED_LANGUAGES = {"en", "es", "fr", "ru"}
GGMDDC_TO_ISO = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "russian": "ru",
}
N_FOLDS = 5
N_BOOTSTRAP = 2_000


def existing_file(value):
    path = Path(value).expanduser()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Corpus-provenance control using genuine M-AILABS and genuine "
            "GGMDDC speech in English, Spanish, French, and Russian."
        )
    )
    parser.add_argument(
        "--mailabs-features",
        required=True,
        type=existing_file,
        metavar="PATH",
        help="Path to the M-AILABS feature pickle, for example features_mlaad.pkl.",
    )
    parser.add_argument(
        "--ggmddc-features",
        required=True,
        type=existing_file,
        metavar="PATH",
        help="Path to the GGMDDC feature pickle, for example features_ggmddc.pkl.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        metavar="DIR",
        help="Output directory. Results are written below DIR/tables/.",
    )
    parser.add_argument(
        "--logo-results",
        default=None,
        type=existing_file,
        metavar="PATH",
        help=(
            "Optional path to SQ3_logo_mlaad.csv or its summary file. "
            "When omitted, the provenance experiment runs without Spearman correlation."
        ),
    )
    return parser.parse_args()


def load_genuine_features(mailabs_path, ggmddc_path):
    import numpy as np
    import pandas as pd

    print("Loading M-AILABS features...")
    dm = pd.read_pickle(mailabs_path)
    dm = dm[dm["label"] == 0].copy()
    dm = dm[dm["language"].isin(SHARED_LANGUAGES)].copy()
    dm["corpus"] = 0
    dm["label"] = 0

    print("Loading GGMDDC features...")
    dg = pd.read_pickle(ggmddc_path)
    dg = dg[dg["label"] == 0].copy()
    dg["language"] = dg["language"].map(
        lambda value: GGMDDC_TO_ISO.get(str(value).lower(), str(value).lower())
    )
    dg = dg[dg["language"].isin(SHARED_LANGUAGES)].copy()
    dg["corpus"] = 1
    dg["label"] = 1

    print(
        f"  M-AILABS genuine: {len(dm):,} utterances "
        f"({dm['language'].value_counts().to_dict()})"
    )
    print(
        f"  GGMDDC genuine:   {len(dg):,} utterances "
        f"({dg['language'].value_counts().to_dict()})"
    )

    rng = np.random.default_rng(42)
    balanced_m = []
    balanced_g = []
    per_language = {}

    for language in sorted(SHARED_LANGUAGES):
        ml = dm[dm["language"] == language]
        gg = dg[dg["language"] == language]
        n_language = min(len(ml), len(gg))
        if n_language == 0:
            continue
        ml_idx = rng.choice(ml.index.to_numpy(), n_language, replace=False)
        gg_idx = rng.choice(gg.index.to_numpy(), n_language, replace=False)
        balanced_m.append(ml.loc[ml_idx])
        balanced_g.append(gg.loc[gg_idx])
        per_language[language] = int(n_language)

    if not balanced_m or not balanced_g:
        raise ValueError("No shared language contains genuine speech in both corpora.")

    dm = pd.concat(balanced_m, ignore_index=True)
    dg = pd.concat(balanced_g, ignore_index=True)
    df = pd.concat([dm, dg], ignore_index=True)

    print(f"  Per-language matched counts per corpus: {per_language}")
    print(
        f"  Total after cell balancing: {len(df):,} "
        f"(M-AILABS={len(dm):,}, GGMDDC={len(dg):,})"
    )
    return df, len(dm)


def read_logo_auc(path):
    import pandas as pd

    data = pd.read_csv(path)
    if "family" not in data.columns:
        raise ValueError(f"Missing 'family' column in LOGO results: {path}")

    for column in ("auc_rf", "logo_auc", "auc"):
        if column in data.columns:
            return (
                data.groupby("family", as_index=False)[column]
                .mean()
                .rename(columns={column: "logo_auc"})
            )

    raise ValueError(
        f"No supported AUC column found in LOGO results: {path}. "
        "Expected one of: auc_rf, logo_auc, auc."
    )


def run_provenance_control(df, families, output_dir, logo_results_path=None):
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.model_selection import StratifiedKFold

    from src.classifiers import _fold, _multiseed
    from src.common import _seed

    _seed(42)
    tables_dir = Path(output_dir).expanduser() / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    strata = df["corpus"].astype(str) + "_" + df["language"].astype(str)
    cell_counts = strata.value_counts()
    if (cell_counts < N_FOLDS).any():
        invalid = cell_counts[cell_counts < N_FOLDS].to_dict()
        raise ValueError(
            f"Corpus-language cells too small for {N_FOLDS}-fold CV: {invalid}"
        )

    results = []
    fold_results = []

    for family, columns in families.items():
        print(f"\n  Family: {family} (d={len(columns)})")
        fold_aucs = []
        splitter = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=42,
        )

        for fold_index, (train_index, test_index) in enumerate(
            splitter.split(df, strata)
        ):
            train_df = df.iloc[train_index].copy()
            test_df = df.iloc[test_index].copy()
            x_train, y_train, x_test, y_test = _fold(
                train_df,
                test_df,
                columns,
                fold_seed=fold_index,
            )
            result = _multiseed(
                x_train,
                y_train,
                x_test,
                y_test,
                grid_search=True,
            )
            auc = result["auc"]
            fold_aucs.append(auc)

            best_params = result.get("best_params") or {}
            row = {
                "family": family,
                "fold": fold_index,
                "auc": auc,
                "eer": result["eer"],
                "rf_n_estimators": best_params.get("n_estimators"),
                "rf_max_depth": best_params.get("max_depth"),
            }

            for metric in result.get("seed_metrics", []):
                seed = int(metric["seed"])
                row[f"auc_seed_{seed}"] = metric["auc"]
                row[f"eer_seed_{seed}"] = metric["eer"]

            test_cells = strata.iloc[test_index].value_counts().to_dict()
            row["test_cell_counts"] = ";".join(
                f"{key}:{value}" for key, value in sorted(test_cells.items())
            )
            fold_results.append(row)
            print(
                f"    Fold {fold_index}: AUC={auc:.4f}  "
                f"RF={result.get('best_params')}"
            )

        auc_array = np.asarray(fold_aucs, dtype=float)
        mean_auc = float(np.nanmean(auc_array))
        valid_aucs = auc_array[np.isfinite(auc_array)]

        if valid_aucs.size:
            rng = np.random.default_rng(42)
            bootstrap = rng.choice(
                valid_aucs,
                (N_BOOTSTRAP, valid_aucs.size),
                replace=True,
            ).mean(axis=1)
            ci_low = float(np.percentile(bootstrap, 2.5))
            ci_high = float(np.percentile(bootstrap, 97.5))
        else:
            ci_low = np.nan
            ci_high = np.nan

        results.append(
            {
                "family": family,
                "provenance_auc": mean_auc,
                "ci_lo": ci_low,
                "ci_hi": ci_high,
            }
        )
        print(f"  {family}: AUC={mean_auc:.4f} [{ci_low:.3f}, {ci_high:.3f}]")

    result_table = pd.DataFrame(results).sort_values(
        "provenance_auc",
        ascending=False,
    )
    fold_table = pd.DataFrame(fold_results)

    if logo_results_path is not None:
        logo_auc = read_logo_auc(logo_results_path)
        result_table = result_table.merge(logo_auc, on="family", how="left")
        valid = result_table.dropna(subset=["provenance_auc", "logo_auc"])
        if len(valid) >= 3:
            rho, p_value = spearmanr(
                valid["provenance_auc"],
                valid["logo_auc"],
            )
            print(
                f"\nSpearman rho (provenance vs LOGO AUC): "
                f"{rho:.3f}  p={p_value:.4f}  n={len(valid)}"
            )
            result_table["spearman_rho_vs_logo"] = rho
            result_table["spearman_p_vs_logo"] = p_value

    result_path = tables_dir / "provenance_control.csv"
    fold_path = tables_dir / "provenance_control_per_fold.csv"
    result_table.to_csv(result_path, index=False)
    fold_table.to_csv(fold_path, index=False)

    print(f"\nSaved: {result_path}")
    print(f"Saved: {fold_path}")
    print(
        "\n"
        + result_table[
            ["family", "provenance_auc", "ci_lo", "ci_hi"]
        ].to_string(index=False)
    )
    return result_table


def main():
    args = parse_args()

    from src.common import _feat_cols
    from src.features import families_from_columns

    meta_columns = {
        "filepath",
        "language",
        "label",
        "generator",
        "model_name",
        "duration",
        "split",
        "corpus",
    }

    data, n_per_corpus = load_genuine_features(
        args.mailabs_features,
        args.ggmddc_features,
    )
    print(
        f"\nControl set: {len(data):,} genuine utterances "
        f"({n_per_corpus:,} per corpus), "
        f"languages: {sorted(data['language'].unique())}"
    )

    feature_columns = _feat_cols(data, meta_columns)
    families = families_from_columns(feature_columns)
    total_dimensions = sum(len(columns) for columns in families.values())
    if total_dimensions != 155:
        raise ValueError(
            f"Expected 155 feature dimensions, found {total_dimensions}."
        )
    print(
        f"Families: {list(families.keys())} | "
        f"Total dims: {total_dimensions}/155"
    )

    run_provenance_control(
        data,
        families,
        args.output_dir,
        logo_results_path=args.logo_results,
    )


if __name__ == "__main__":
    main()
