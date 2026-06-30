# Acoustic-Space Geometry for Multilingual Deepfake Speech Detection

Code accompanying **“Understanding When Handcrafted Acoustic Descriptors Generalise: A Geometry-Driven Analysis Across Languages and Synthesis Systems”** (Chakir, Gahi, El-Khatib, 2026), submitted to *Computer Speech & Language*.

Zenodo archive: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20534006.svg)](https://doi.org/10.5281/zenodo.20534006)

## Scope

The repository implements the four diagnostic stages used in the manuscript:

1. projected acoustic-space geometry;
2. significance-based discriminative invariance;
3. held-out generalisation under LOLO and LOGO;
4. dimension-normalised TreeSHAP attribution.

The analysis covers nine handcrafted descriptor families—MFCC, LFCC, CQCC, GD, IF, PD, CPP, Jitter, and Shimmer—forming a joint 155-dimensional representation.

## Repository structure

```text
.
├── pipeline_mlaad.py
├── pipeline_ggmddc.py
├── provenance_control.py
├── experiment_config.json
├── src/
│   ├── __init__.py
│   ├── common.py
│   ├── features.py
│   ├── classifiers.py
│   ├── stats_geometry.py
│   └── sq_pipeline.py
└── requirements.txt
```

## Method-aligned implementation

The current version aligns the executable pipeline with the latest manuscript in the following points:

- PCA geometry and invariance use feature-wise median imputation, the conditional `log1p` transformation, and z-score standardisation.
- `PC1_global` preprocessing and PCA are fitted on the combined genuine–synthetic set.
- `PC1_fake` preprocessing and PCA are fitted on synthetic observations only and then applied unchanged.
- MLAAD geometry and LOGO use the same synthesis-system identifier, `model_name`.
- The exploratory interaction quantity is truncated at zero exactly as defined in the manuscript.
- LOLO and LOGO outputs retain AUC and EER for every classifier seed.
- Family-level macro AUC intervals are generated after macro-averaging held-out units separately for each seed.
- SHAP relative shares, macro profiles, and FP−TN / FN−TP contrasts are generated directly by the pipeline.
- The provenance control balances the two genuine corpora within each shared language and stratifies cross-validation by corpus–language cell.
- Corpus manifests and deterministic fold-definition files are written automatically by the two principal pipelines.

Old SQ1/SQ2 result files and old SQ3 checkpoints that do not contain the new audit columns are ignored automatically and recomputed.

## Installation and execution

```bash
pip install -r requirements.txt

# Edit corpus and output paths near the top of each script.
python pipeline_mlaad.py
python pipeline_ggmddc.py
python provenance_control.py
```

## Main outputs

Outputs are written below `<OUTPUT_DIR>`.

| File | Content |
|---|---|
| `feature_manifest_mlaad.csv` | MLAAD+M-AILABS utterance manifest |
| `feature_manifest_ggmddc.csv` | GGMDDC utterance manifest |
| `fold_definitions_mlaad.csv` | Deterministic LOLO and LOGO held-out definitions |
| `fold_definitions_ggmddc.csv` | Deterministic GGMDDC LOLO definitions |
| `tables/SQ1_representation_space*.csv` | Hellinger, Cohen’s d, rank-based association quantities, ratio CIs, interaction |
| `tables/SQ2_invariance*.csv` | Mann–Whitney, Kruskal–Wallis, BH correction, operational invariance criterion |
| `tables/SQ3_lolo_*.csv` | Fold-level LOLO results with per-seed metrics |
| `tables/SQ3_lolo_*_summary.csv` | Family-level LOLO macro AUC, seed-based CI, and EER |
| `tables/SQ3_logo_mlaad.csv` | Fold-level LOGO results with per-seed metrics |
| `tables/SQ3_logo_mlaad_summary.csv` | Family-level LOGO macro AUC, seed-based CI, and EER |
| `tables/SQ3_delta.csv` | LOGO minus LOLO macro-AUC difference |
| `tables/SQ4_shap_*.csv` | Fold/stratum absolute family attribution and relative shares |
| `tables/SQ4_shap_*_summary.csv` | Macro-averaged SHAP shares by TP/TN/FP/FN stratum |
| `tables/SQ4_delta_shap_*.csv` | Share-scale FP−TN and FN−TP attribution contrasts |
| `tables/provenance_control.csv` | Provenance AUC and exploratory correlation with LOGO |
| `tables/provenance_control_per_fold.csv` | Fold-level provenance results and corpus–language cell audit |

## Important execution parameters

The complete machine-readable configuration is in `experiment_config.json`. In particular, the original feature extraction used:

- 16 kHz mono audio;
- leading/trailing silence trimming with `top_db=30`;
- peak-amplitude normalisation;
- the first 4.0 seconds of each recording;
- rejection of trimmed recordings shorter than 1.0 second;
- a Nyquist-capped CQCC range of 762 bins at 16 kHz, despite a requested nine-octave range.

Jitter descriptors are derived from pYIN F0 trajectories, and Shimmer descriptors are derived from frame-energy variations. They are therefore approximations of strict pitch-synchronous Praat measurements.

## Remaining interpretive limitations

- MLAAD language and synthesis-system factors are not fully crossed; rank-based quantities remain descriptive, not causal variance components.
- Genuine LOGO test examples are sampled from the full M-AILABS pool rather than language-matched to each held-out synthesis system.
- The SHAP model is a fixed single-seed RF and is not the multi-seed, grid-selected RF used in SQ3.
- Preprocessing is fitted on the complete outer-training partition before the internal validation split used for hyperparameter selection; no held-out test data enter this process, but the internal search is not a fully nested preprocessing design.
- Several error-specific SHAP strata can contain very few sampled observations.

## Corpus access

- MLAAD v5: <https://huggingface.co/datasets/mueller91/MLAAD>
- M-AILABS: <https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/>
- GGMDDC: <https://www.kaggle.com/datasets/artharking/ggmddc-original>

## Citation

```bibtex
@article{chakir2026geometry,
  title   = {Understanding When Handcrafted Acoustic Descriptors Generalise:
             A Geometry-Driven Analysis Across Languages and Synthesis Systems},
  author  = {Chakir, Khadija and Gahi, Youssef and El-Khatib, Khalil},
  journal = {Computer Speech & Language},
  year    = {2026},
  note    = {Under review}
}
```
