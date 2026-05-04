# Resume bullets for this project

Ready-to-paste resume framings, grouped by the kind of role you're
applying to. Numbers should match what you actually ran; primary artefacts:

- **Probe RMSE 0.9491:** `run_recommendation.py` / `MODEL_CARD.md`
- **AWS reference:** [`architecture/AWS_Architecture_Overview.png`](../architecture/AWS_Architecture_Overview.png) + short companion [`ARCHITECTURE.md`](../ARCHITECTURE.md)
- **Dataset scale:** [Netflix Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) — 100,480,507 ratings / 480,189 users / 17,770 movies
- **SLOs / volumes on diagram:** ingestion &lt; 5 min, prediction p95 &lt; 150 ms, raw ~37.5 TB/yr → consumption ~2 TB/yr (as annotated on the architecture figure)

**Pick the framing that matches the role you're applying to.** The repo is
now **one** story: offline Netflix Prize lab + one AWS end-to-end diagram.
Use **Combined** when you want breadth in one breath; use A/B/C when you
need a single-column bullet.

---

## Combined framing — one architecture, one offline lab

> Use this when you want a **single** headline: public **Netflix Prize** data for
> reproducible ETL + recommender + **RFM / clustering**, paired with **one**
> AWS reference diagram (sources → activation, MLOps, security, IaC).

### Resume header ideas

> 1. **AWS ML Systems — ETL to activation (reference architecture)**  
> 2. **From curated data to segments — with an AWS end-to-end design**

### One-line

> Built an offline **hybrid recommender** on Netflix Prize (**probe RMSE 0.9491**) with **ETL to Parquet**, **K-Means + RFM segmentation**, and a **single AWS architecture diagram** covering ingestion, medallion lake, SageMaker MLOps, DynamoDB cache, and activation (Connect / Pinpoint / API Gateway) with SLOs on the figure.

### Two-line

> Shipped Python CLIs + one notebook for the full offline path (load → EDA → recommend → cluster → RFM cross-tab).  
> Documented production intent through **`architecture/AWS_Architecture_Overview.png`** and a Mitchell-style **`MODEL_CARD.md`** — no duplicate long-form API/MLOps PDFs in-repo.

### STAR (short)

> **Situation.** Coursework often stops at notebook metrics.  
> **Task.** Tie offline outputs to a credible AWS rollout story without duplicating five specs.  
> **Action.** One diagram-first architecture + condensed model card + reproducible pipeline code.  
> **Result.** Reviewer sees **one** visual system design and **one** RMSE headline grounded in `run_recommendation.py`.

### Talking points

- **Why one diagram?** It already encodes batch + streaming ingest, four-zone lake, SageMaker Pipelines, drift loop, and activation — the same buckets your ETL/RFM work feeds in a real programme.
- **Honest limit:** nothing here is deployed; the diagram and code are the deliverables.

### Reusable phrases

- *"…probe RMSE 0.9491 on the Netflix Prize probe, plus an AWS end-to-end architecture from sources to activation."*
- *"…ETL and segmentation in Python; production story told through one reference diagram instead of a stack of PDFs."*

---

## Framing A — ML / Data Scientist

### One-line (best for tight resume real estate)

> Built a hybrid SVD + Item-CF recommender on the Netflix Prize dataset (100M ratings) achieving **probe RMSE 0.9491** — **16% better than a global-mean baseline** and within 1% of Netflix's own Cinematch.

### Two-line

> Built and benchmarked a hybrid recommender (SVD + Item-CF residual correction) on the Netflix Prize dataset (100M ratings, 480K users, 17K movies); achieved **probe RMSE 0.9491** vs **0.9632 for SVD alone** and **1.1296 for the global-mean baseline**.
> Paired the offline model with a Mitchell-template model card documenting evaluation slices, ethical considerations (popularity amplification, re-identification risk), and known failure modes — making the model integratable by a downstream team.

### STAR / full bullet (cover letter, interview prep)

> **Situation.** A five-person MLDS 423 team had five weeks to build a recommender on the Netflix Prize dataset (100M ratings, 480K users, 17.7K movies) and present production-grade results.
> **Task.** Own the segmentation track end-to-end and the production-readiness layer for the whole project.
> **Action.** Implemented RFM quintile scoring (9 named segments) and cross-referenced it against K-Means behavioural clusters via a heatmap that ranks which behavioural cohort dominates each value segment. Independently authored a Mitchell-template model card, a Netflix-style AWS production architecture, a REST API contract, and an MLOps spec — none of which were required by the assignment.
> **Result.** Hybrid SVD + Item-CF recommender achieved **probe RMSE 0.9491** (within 1% of Cinematch's 0.9525). The segmentation cross-tab gave the marketing surface a defensible playbook; the architecture and MLOps specs gave a reviewer a complete answer to "how would you ship this?" The repo became my primary portfolio artefact.

### Talking points (for the interview)

- **Why hybrid beat plain SVD**: SVD plus a KNN residual correction on the top-1000 popular movies catches the long-tail items where collaborative filtering still has signal that the latent factors smoothed away. Numerical lift: 0.9632 → 0.9491 (≈1.5% relative).
- **What I'd change with a real budget**: replace the time-budgeted single-config training (50K rows, 5 epochs) with a multi-fidelity search (ASHA / BOHB) over `n_factors`, regularisation, and epochs. The unobjectively chosen `α=1.0` in the residual correction is the second thing I would tune.
- **Honest limits**: cold-start isn't modelled in the offline lab — the architecture diagram still shows where a vector / content path would sit. No fairness audit was run.
- **What I personally owned vs the team**: segmentation track, plus the production-design layer (README, architecture diagram, MODEL_CARD).

---

## Framing B — AI / ML Product or Strategy

### One-line

> Designed an AWS production architecture for a Netflix-scale recommender (1M MAU) — S3 data lake, SageMaker training and serving, DynamoDB online feature store — sized at **~$9.3K/mo** with line-item costs and a **150 ms p95 latency SLO**.

### Two-line

> Authored a diagram-first AWS reference (ingestion through activation, with MLOps + security + IaC bands) and a Mitchell-template model card for a hybrid recommender.
> Aligned offline metrics (**probe RMSE 0.9491**) with the latency and volume callouts annotated on the same architecture figure (p95 targets, TB/yr bands).

### STAR / full bullet

> **Situation.** Most ML coursework stops at "I trained a model in a notebook." Hiring managers screening for AI-product or ML-strategy roles need to see the next layer: cost, SLA, governance, rollout.
> **Task.** Treat the team's Netflix Prize recommender as a real product and write the artefacts a VP or staff engineer would expect before shipping.
> **Action.** Owned the production-design layer: a **single AWS reference diagram** (ingestion → medallion lake → processing → warehouse/BI → SageMaker MLOps → activation + security / IaC bands) with companion `ARCHITECTURE.md`, plus a Mitchell-style **`MODEL_CARD.md`** for the shipped hybrid recommender.
> **Result.** A reviewer sees one architecture image, one offline pipeline with probe RMSE **0.9491**, and honest scope notes — no stack of overlapping long-form specs.

### Talking points

- **Why diagram-first:** one picture carries the batch + streaming path, lake zones, SageMaker loop, and activation — easier for a cross-functional review than five separate PDFs.
- **Why a model card:** states intended use, limits, and privacy cautions (including re-identification risk on sparse ratings) without overstating deployment status.

---

## Framing C — Generalist / first-job hunt

If you don't yet know which door will open first, lead with breadth:

> Built and shipped a five-person Netflix Prize recommender project as the segmentation lead and production-readiness owner: **probe RMSE 0.9491** (16% better than a global-mean baseline), nine RFM segments cross-referenced with behavioural clusters, plus an AWS **architecture diagram + model card** that tie the offline pipeline to a credible serving story.

---

## Reusable phrases (drop into a cover letter)

- *"…within 1% of Netflix's own Cinematch baseline (0.9525) on the same probe set."*
- *"…sized at $9.3K/mo at 1M monthly active users with a layer-by-layer cost model from the AWS Pricing Calculator."*
- *"…the same numbers — probe RMSE 0.9491, $9.3K/mo, 150 ms p95 — are consistent across the model card, the architecture doc, the API spec, and the MLOps spec."*
- *"…I treated the bridge from notebook to production as a first-class deliverable, not a follow-up."*

---

## What NOT to claim

A few things this portfolio does *not* support — don't put them on a resume:

- ❌ "Deployed to production" — offline code + architecture diagram only; not running in an AWS account for this repo.
- ❌ "A/B tested" — the rollout *policy* is specified; no live experiment was run.
- ❌ "Improved business KPIs" — neither dataset has a live business attached.
- ❌ "Reduced latency by X%" — there is no measured latency, only an SLO.
- ❌ "Built a fairness audit" — not run on this dataset slice.
- ❌ "Deployed this architecture to AWS" — diagram + offline code only; no live account in this repo.

If a recruiter asks "did this go to prod?", the right answer is: *"No — it's reproducible offline work plus an AWS reference diagram and model card. The artefacts are what you'd hand an engineer before a build; they aren't a live service."*

---

*Last reviewed: 2026-05-04. Update when probe RMSE or diagram annotations change.*
