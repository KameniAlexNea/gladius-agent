---
name: research
description: >
  Search ArXiv and Kaggle for SOTA techniques specific to this competition's
  data type. Use before major architectural decisions and when the score has
  stagnated for 3+ iterations. Provides targeted search query templates by
  competition type and a curated high-impact technique reference.
---

# Research

## Overview

Web search for state-of-the-art techniques is one of the highest-ROI activities when you've hit a plateau. This skill provides targeted query templates for ArXiv and Kaggle by competition type, plus a curated reference of techniques known to win competitions.

## When to Use This Skill

- Starting a new competition — discover the best-known approach for this data type.
- Score has stagnated for 3+ iterations — find orthogonal techniques.
- Competition is in a specialised domain (genomics, NLP, finance) with dedicated architectures outside your default toolkit.
- You need to assess whether external datasets or pretrained models are allowed.

## Core Capabilities

### 1. Identify Competition Type

From `CLAUDE.md` and `data/train.csv`, determine the competition type before searching:

| Signal | Competition type |
| --- | --- |
| ID + many numeric features, no text | Tabular classification/regression |
| `text` / `description` columns | NLP / text classification |
| Image file paths in CSV | Computer vision |
| Date/TimeID column, entities over time | Time-series forecasting |
| User + item IDs | Recommender system |
| Graph adjacency | Graph ML |

### 2. Targeted Web Search

Use the WebSearch tool with queries from `references/search_queries.md`. Replace `{TASK}` and `{COMPETITION_ID}` with the actual values from `CLAUDE.md`.

### 3. Evaluate Applicability

For each technique found:
1. **Complexity vs. gain** — implementation time justified by expected gain?
2. **Data size compatibility** — works with training set size?
3. **Compute budget** — finishes within the iteration time budget?
4. **Reproducibility** — code available on GitHub / Kaggle notebook?

### 4. SOTA Techniques Quick Reference

Read `references/techniques.md` for curated high-impact techniques by competition category: tabular, time-series, NLP, computer vision.

### 5. Record Findings in MEMORY.md

After research, add a bullet to `.claude/agent-memory/planner/MEMORY.md`:

```markdown
## Patterns & Hypotheses 💡
- [Research] {TECHNIQUE}: found on arxiv.org/{ID} — expected gain {REASON}. Try next.
- [Forum] {COMPETITION}: {USER} reported {TRICK} — easy to replicate.
```

## Resources

### references/
- `search_queries.md` — targeted WebSearch query templates by competition type
- `techniques.md` — curated SOTA techniques with expected gains
