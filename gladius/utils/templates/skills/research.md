---
name: research
description: >
  Search ArXiv and Kaggle discussions for state-of-the-art techniques
  specific to this competition's data type and task. Use before major
  architectural decisions and when the score has stagnated.
---

## When to use this skill

- Starting a new competition — discover the best-known approach for this data type.
- Score has stagnated for 3+ iterations — find orthogonal techniques to try.
- Competition is in a specialised domain (genomics, NLP, time-series) with
  dedicated architectures not in your default toolkit.

---

## Step 1 — identify the competition type

From `CLAUDE.md` and `data/train.csv`, determine:

| Signal | Competition type |
| --- | --- |
| ID + many numeric features, no text | Tabular classification/regression |
| `text` / `description` columns | NLP / text classification |
| Image file paths in CSV | Computer vision |
| Date/TimeID column, entities over time | Time-series forecasting |
| User + item IDs | Recommender system / collaborative filtering |
| Graph adjacency | Graph ML |

---

## Step 2 — ArXiv search queries

Use the WebSearch tool with these targeted queries (replace `{TASK}` accordingly):

### Tabular competitions
```
site:arxiv.org tabular deep learning benchmark 2024
site:arxiv.org gradient boosting feature engineering competition winner
site:arxiv.org TabPFN tabular foundation model
site:arxiv.org AutoML tabular ensemble winning solution
```

### NLP competitions
```
site:arxiv.org sentence transformer fine-tuning competition 2024
site:arxiv.org large language model zero-shot classification {DOMAIN}
site:arxiv.org cross-encoder reranking text classification
```

### Time-series competitions
```
site:arxiv.org time series forecasting transformer competition 2024
site:arxiv.org N-BEATS N-HiTS forecasting benchmark
site:arxiv.org temporal fusion transformer probabilistic forecasting
```

### Computer vision competitions
```
site:arxiv.org {DOMAIN} image classification winning solution
site:arxiv.org vision transformer medical image segmentation 2024
site:arxiv.org test-time augmentation ensemble deep learning
```

---

## Step 3 — Kaggle discussion search

Use WebSearch to search the competition's discussion forum:

```
site:kaggle.com/competitions/{COMPETITION_ID}/discussion
site:kaggle.com/competitions/{COMPETITION_ID}/discussion data leak
site:kaggle.com/competitions/{COMPETITION_ID}/discussion feature engineering
site:kaggle.com/competitions/{COMPETITION_ID}/discussion 1st place solution
```

Also search for similar past competitions:
```
site:kaggle.com {TASK_TYPE} competition winner write-up solution summary
```

**What to look for in discussions:**
- Data leaks or shortcut features (high impact, easy to implement)
- Winning solution summaries after competition end
- Novel feature engineering ideas specific to this domain
- External datasets or pretrained models that are allowed

---

## Step 4 — evaluate applicability

For each technique found, assess:

1. **Complexity vs. gain** — is implementation time justified by expected gain?
2. **Data size compatibility** — does it work with the training set size?
3. **Compute budget** — can it finish within the iteration time budget?
4. **Reproducibility** — is there code available (GitHub / Kaggle notebook)?

---

## Step 5 — report findings to planner memory

After research, add a bullet to `.claude/agent-memory/planner/MEMORY.md`:

```markdown
## Patterns & Hypotheses  💡
- [Research] {TECHNIQUE}: found on arxiv.org/{ID} — expected gain {REASON}. Try next.
- [Forum] {COMPETITION}: {USER} reported {TRICK} — easy to replicate.
```

The orchestrator's summarizer will incorporate this into the persistent memory.

---

## High-impact techniques by category (quick reference)

### Tabular (always try first)
- **LightGBM + Optuna** — baseline that beats most alternatives
- **Target encoding within CV folds** — often +0.003–0.010 AUC
- **Pseudo-labelling** — add high-confidence test predictions as training data
- **TabNet / FT-Transformer** — deep learning for tabular; competitive when feature count > 200

### Time-series
- **Lag features + rolling stats** — the single most impactful feature class
- **N-BEATS / N-HiTS** — state-of-the-art pure NN forecasters
- **TFT (Temporal Fusion Transformer)** — interpretable SOTA for multivariate

### NLP
- **DeBERTa-v3-large** — best general-purpose encoder as of 2024
- **Sentence-BERT + cosine similarity** — fast semantic similarity baseline
- **LLM zero-shot via API** — powerful for label prediction with GPT-4o / Claude

### Computer Vision
- **EfficientNetV2 / ConvNeXt** — strong CNN backbones
- **ViT-L / EVA-02** — best ViT variants for classification
- **TTA (test-time augmentation)** — free +0.5–2% with flips/crops
