# SOTA Techniques by Competition Category

## Tabular (try first, always)

| Technique | Expected gain | Notes |
| --- | --- | --- |
| LightGBM + Optuna | Baseline | Use before anything else |
| XGBoost / CatBoost diversity | +0.001–0.003 | Ensemble with LightGBM |
| Target encoding within CV | +0.003–0.010 AUC | High-cardinality categoricals |
| Pseudo-labelling | +0.001–0.005 | High-confidence test predictions as train |
| TabNet / FT-Transformer | Varies | DL for tabular; try when feature count > 200 |
| TabPFN | +0–0.01 | Prior-fitted networks; in-context learning for tabular |
| Adversarial sample weighting | +0.001–0.003 | When train/test distributions differ |

## Time-Series

| Technique | Expected gain | Notes |
| --- | --- | --- |
| Lag + rolling stats | +0.01–0.05 RMSE | Single most impactful feature class |
| N-BEATS / N-HiTS | SOTA on univariate | Pure neural; for multi-step forecasting |
| TFT (Temporal Fusion Transformer) | SOTA on multivariate | Interpretable; good for long horizons |
| TimesFM | Zero-shot baseline | Google foundation model; invoke timesfm skill |
| Calendar/cyclical features | +0.002–0.01 | Hour, day-of-week, month encoded cyclically |

## NLP

| Technique | Expected gain | Notes |
| --- | --- | --- |
| DeBERTa-v3-large fine-tuning | +2–5 F1 points | Best general encoder as of 2024 |
| Sentence-BERT + cosine sim | Fast baseline | For semantic similarity tasks |
| LLM zero-shot (GPT-4o / Claude) | Varies | Best for few-shot label prediction |
| Ensemble BERT + LightGBM | +0.01–0.02 AUC | Combine DL embeddings with GBM features |
| Cross-encoder reranking | +1–3% MAP | For ranking/retrieval tasks |

## Computer Vision

| Technique | Expected gain | Notes |
| --- | --- | --- |
| EfficientNetV2 / ConvNeXt | Strong CNN backbones | Good starting point |
| ViT-L / EVA-02 | Best ViT variants | Classification competitions |
| Test-time augmentation (TTA) | +0.5–2% free | Flips, crops, rotations |
| Mixup / CutMix | +0.5–1% | Regularisation augmentations |
| Pseudo-labelling on test | +0.5–1% | High-confidence predictions added to train |

## Universal Ensemble Techniques

| Technique | Notes |
| --- | --- |
| Rank averaging | Scale-invariant; works across architectures |
| Hill-climbing | Best for large model pools; invoke ensembling skill |
| Stacking (Logistic Regression meta) | Good when base model predictions are well-calibrated |
