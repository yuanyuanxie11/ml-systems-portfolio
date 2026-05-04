# From Notebook to Production — Two AWS ML Case Studies

> A personal portfolio of two production-grade ML systems built end-to-end on
> AWS: a Netflix-scale **recommender** and a **Customer 360 churn + segmentation
> platform**. Each one ships not just a model, but the full bridge —
> architecture, model card, signed API, MLOps spec, line-item cost model, and
> a drift-to-retrain policy. The point isn't either model on its own; it's
> that the same production discipline is applied twice, to two different
> problem classes, on the same cloud.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Probe RMSE 0.9491](https://img.shields.io/badge/Netflix%20probe%20RMSE-0.9491-brightgreen.svg)
![Customer 360 Architecture](https://img.shields.io/badge/Customer%20360-7%20swim%20lanes-blueviolet.svg)
![AWS](https://img.shields.io/badge/cloud-AWS-orange.svg)
![License MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## Contents

1. [TL;DR](#tldr)
2. [Why this portfolio](#why-this-portfolio)
3. [The two projects at a glance](#the-two-projects-at-a-glance)
4. [Repository layout](#repository-layout)
5. [What's the same across both — the production discipline](#whats-the-same-across-both--the-production-discipline)
6. [Project 1 — Netflix recommender](#project-1--netflix-recommender)
7. [Project 2 — Customer 360 churn + segmentation](#project-2--customer-360-churn--segmentation)
8. [Quickstart](#quickstart)
9. [What I'd do next](#what-id-do-next)
10. [Honest limits](#honest-limits)
11. [Credits and license](#credits-and-license)

---

## TL;DR

| Project | Problem | Headline result | AWS architecture | Cost @ scale | Latency SLO |
|---|---|---|---|---:|---:|
| **Netflix recommender** | Predict user-movie ratings on the Netflix Prize dataset (100M ratings) | **Probe RMSE 0.9491** — within 1% of Cinematch | S3 medallion lake → SageMaker training/serving → DynamoDB online store → API Gateway | **~$9.3K/mo** at 1M MAU | **150 ms p95** |
| **Customer 360** | Real-time churn scoring + segmentation across 11 source systems for an enterprise call-centre | 7-lane AWS reference architecture with closed-loop drift→retrain | S3 4-zone medallion (Raw → Curated → Enriched → Consumption) → Lake Formation → SageMaker Pipelines → Connect screen-pop + Pinpoint | **~$12.5K/mo** | **<150 ms p95**, **<50 ms** agent lookup |

Both systems share the same production muscles: medallion S3 lake, SageMaker
training and serving, DynamoDB online cache, API Gateway with Cognito JWT,
Mitchell-template model card, PSI-driven drift policy, and a line-item AWS
cost model.

---

## Why this portfolio

Most ML coursework ends at *"I trained a model in a notebook."* The next
layer — **what does it cost, what's the SLA, how do you retrain, how do
you roll back** — is where production systems live and die, and it's almost
never demonstrated in a portfolio. So I built it twice, on two different
problem classes, to prove the discipline transfers.

- **Project 1 (Netflix recommender)** stresses *offline ranking quality and
  online candidate generation*: matrix factorisation, item-CF residual
  correction, hybrid scoring, vector store, popularity fallback for
  cold-start.
- **Project 2 (Customer 360)** stresses *multi-source ingestion, real-time
  agent enablement, and closed-loop ML lifecycle*: 11 source systems
  (CRM, Genesys call audio, Salesforce, Zendesk, billing, web), 4-zone S3
  medallion lake, SageMaker Pipelines (Validate → Feature Build → Train →
  Evaluate → Register), DynamoDB online cache for sub-50 ms agent screen
  lookups, drift detection (PSI > 0.2 or AUC < 0.78) wired into a retrain
  trigger, and a multi-channel activation surface (Connect screen-pop,
  Pinpoint, exec dashboards).

The reusable production pattern shows up in both: **S3 + SageMaker +
DynamoDB + API Gateway + 150 ms p95 SLO + Mitchell model card + PSI drift
policy + IaC**. The model on top changes; the system around it doesn't.

---

## The two projects at a glance

### Project 1 — Netflix-scale recommender
A hybrid SVD + Item-based-CF residual recommender on the Netflix Prize
dataset (100,480,507 ratings, 480,189 users, 17,770 movies). Achieves
**probe RMSE 0.9491**, beating SVD-alone (0.9632) and the bias baseline
(0.9965), and lands within 1% of Netflix's own Cinematch (0.9525). Paired
with an AWS production design priced at **~$9.3K/mo at 1M MAU**, a Mitchell
model card, a Cognito-authenticated REST API, and an MLOps runbook.

→ **[`projects/netflix-recommender/`](projects/netflix-recommender/)**

### Project 2 — Customer 360 churn + segmentation platform
An enterprise customer-360 architecture combining churn classification,
RFM-style segmentation, and real-time agent enablement. The deliverable is
a 7-swim-lane AWS reference architecture with: 11 source systems, a
4-zone S3 medallion lake governed by Lake Formation, SageMaker Pipelines
for training, DynamoDB for online lookups (**<50 ms** at the agent screen),
drift-driven retraining (PSI > 0.2 or AUC < 0.78), and a multi-channel
activation surface (Amazon Connect screen-pop, Pinpoint, QuickSight exec
dashboards). Sized at **~$12.5K/mo** with **37.5 TB/yr raw ingestion**
through the medallion lake.

→ **[`projects/customer-360/`](projects/customer-360/)**

---

## Repository layout

This is a monorepo. Each project lives under `projects/` with its own
docs, so a reviewer can read either project independently or both as a
portfolio.

```
.
├── README.md                              <- you are here (portfolio narrative)
├── LICENSE                                <- MIT
├── .gitignore
├── docs/
│   ├── PORTFOLIO_OVERVIEW.md              <- deeper "why these two projects"
│   └── RESUME_BULLETS.md                  <- ready-to-paste resume framings
│
└── projects/
    │
    ├── netflix-recommender/               <- Project 1
    │   ├── README.md                      <- Netflix-specific narrative
    │   ├── ARCHITECTURE.md                <- AWS reference design + cost model
    │   ├── MODEL_CARD.md                  <- Mitchell template
    │   ├── docs/
    │   │   ├── API.md                     <- REST contract, 150 ms p95 SLO
    │   │   └── MLOPS.md                   <- pipeline, drift, rollouts, runbook
    │   ├── notebooks/
    │   │   ├── 01_data_loading.ipynb      <- raw .txt → ratings.parquet
    │   │   ├── 02_eda.ipynb
    │   │   ├── 03_recommendation.ipynb    <- SVD + Item-CF; probe RMSE 0.9491
    │   │   ├── 04_clustering.ipynb        <- KMeans / hierarchical / DBSCAN / t-SNE
    │   │   └── 05_rfm_analysis.ipynb      <- RFM + cross-tab vs clustering
    │   ├── src/                           <- (planned) extracted .py modules
    │   ├── outputs/                       <- figures only; .parquet files git-ignored
    │   ├── requirements.txt
    │   └── LICENSE                        <- MIT
    │
    └── customer-360/                      <- Project 2
        ├── README.md                      <- Customer 360 overview + design rationale
        ├── ARCHITECTURE.md                <- 7-lane AWS architecture, cost, SLOs
        ├── MODEL_CARD.md                  <- planned churn model card
        ├── docs/
        │   ├── API.md                     <- /v1/score, /v1/segment, JWT, <150 ms SLO
        │   ├── MLOPS.md                   <- SageMaker Pipelines + drift→retrain loop
        │   └── SECURITY.md                <- IAM / KMS / Lake Formation / Macie
        ├── architecture/
        │   ├── Customer360_Architecture.drawio       <- editable diagram source
        │   ├── Customer360_Architecture.png          <- exported PNG (for README)
        │   └── Cloud_Engineering.pdf                 <- exported deck (governance, IaC, cost)
        └── REFERENCES.md                  <- citations + course-material lineage
```

The current `Reommendation_Project/` working folder still has the old
flat layout (everything at the root); the **[GitHub upload guide](GITHUB_UPLOAD_GUIDE.md)**
lists exactly what to move where before pushing.

---

## What's the same across both — the production discipline

This is the through-line a reviewer should notice. Both projects ship:

| Pillar | Netflix | Customer 360 |
|---|---|---|
| **Storage** | S3 medallion (Raw → Curated → Feature → Consumption) | S3 medallion (Raw → Curated → Enriched → Consumption) + Lake Formation |
| **Training** | SageMaker training jobs, weekly schedule | SageMaker Pipelines: Validate → Feature Build → Train → Evaluate → Register |
| **Online store** | DynamoDB (precomputed top-N + features) | DynamoDB (real-time score cache, <50 ms read) |
| **Serving** | API Gateway + Lambda, 150 ms p95 | API Gateway + Cognito JWT, <150 ms p95, agent screen <50 ms |
| **Drift** | PSI on user-activity features | PSI > 0.2 OR AUC < 0.78 → auto-retrain trigger |
| **Rollout** | Shadow → canary → ramp → full | Shadow → canary → ramp → full |
| **Governance** | Mitchell model card, ethical considerations, do/don't list | Mitchell model card + IAM, KMS, Lake Formation, Macie, CloudTrail |
| **IaC** | (referenced) | Service Catalog → CodeCommit → CodeBuild → CodePipeline → CDK → CloudFormation |
| **Cost discipline** | Line-item AWS Pricing Calculator estimate | Line-item with cost-per-zone breakdown |

The reusable pattern is the contribution. The model on top is the case study.

---

## Project 1 — Netflix recommender

### Headline numbers

| Rank | Model | Probe RMSE | Δ vs global mean |
|---:|---|---:|---:|
| 1 | **SVD + Item-CF residual correction** | **0.9491** | **−16.0%** |
| 2 | SVD (matrix factorisation) | 0.9632 | −14.7% |
| 3 | User + movie bias | 0.9965 | −11.8% |
| 4 | Global-mean baseline | 1.1296 | — |

Source: `projects/netflix-recommender/notebooks/03_recommendation.ipynb`.
Cinematch reference: 0.9525. BellKor winner: 0.8567 (after 3 years of
ensemble work).

### Why hybrid beat plain SVD

SVD plus a KNN residual correction on the top-1000 popular movies catches
long-tail signal that the latent factors smoothed away. The residual is
weighted with α=1.0 (un-tuned — flagged in the model card as a known
follow-up).

### Production design ([`ARCHITECTURE.md`](projects/netflix-recommender/ARCHITECTURE.md))

S3 medallion lake → Glue / EMR ETL → SageMaker training and serving →
API Gateway + Lambda → DynamoDB online feature/score store → OpenSearch
for vector candidate generation. Sized at **~$9.3K/mo at 1M monthly
active users**, line-item from the AWS Pricing Calculator (us-east-1, 2026).

### Documents

- [`README.md`](projects/netflix-recommender/README.md) — project narrative
- [`ARCHITECTURE.md`](projects/netflix-recommender/ARCHITECTURE.md) — AWS design + cost
- [`MODEL_CARD.md`](projects/netflix-recommender/MODEL_CARD.md) — Mitchell template
- [`docs/API.md`](projects/netflix-recommender/docs/API.md) — REST contract
- [`docs/MLOPS.md`](projects/netflix-recommender/docs/MLOPS.md) — pipeline + drift + rollouts

---

## Project 2 — Customer 360 churn + segmentation

### What it is

A 7-swim-lane AWS reference architecture for an enterprise customer-360
platform: ingest from 11 source systems (CRM, Genesys call streams,
Salesforce, Zendesk, billing, web events, etc.), land in a 4-zone S3
medallion lake (Raw → Curated → Enriched → Consumption) governed by Lake
Formation, train churn + segmentation models through SageMaker Pipelines,
serve real-time scores via DynamoDB cache for **<50 ms agent-screen lookups**,
and activate through Amazon Connect screen-pop, Pinpoint email/SMS, and
QuickSight executive dashboards.

### Architecture diagram

`projects/customer-360/architecture/Customer360_Architecture.drawio` —
editable in [diagrams.net](https://app.diagrams.net/). PNG export embedded
inline in the project README. The deck (`Cloud_Engineering.pdf`) covers
service-by-service justification, security and governance, IaC pipeline,
and the layer-by-layer cost model.

### Headline production numbers

| Dimension | Value |
|---|---|
| Raw ingestion | 37.5 TB / year |
| Consumption | 2 TB / year |
| Prediction latency p95 | <150 ms |
| Agent screen lookup | <50 ms (DynamoDB) |
| Pipeline availability | 99.9% |
| Model availability | 99.95% |
| RPO / RTO | 15 min / 4 hr |
| Drift trigger | PSI > 0.2 OR AUC < 0.78 |
| Cost (priced) | ~$12.5K / month |

### Documents

- [`README.md`](projects/customer-360/README.md) — project narrative
- [`ARCHITECTURE.md`](projects/customer-360/ARCHITECTURE.md) — service-by-service design
- [`docs/API.md`](projects/customer-360/docs/API.md) — `/v1/score`, `/v1/segment`, JWT auth
- [`docs/MLOPS.md`](projects/customer-360/docs/MLOPS.md) — Pipelines + drift→retrain loop
- [`docs/SECURITY.md`](projects/customer-360/docs/SECURITY.md) — IAM, KMS, Lake Formation, Macie, GDPR/CCPA/PCI-DSS
- [`MODEL_CARD.md`](projects/customer-360/MODEL_CARD.md) — planned churn classifier model card
- [`REFERENCES.md`](projects/customer-360/REFERENCES.md) — industry benchmarks, AWS docs, course-material lineage

---

## Quickstart

### Project 1 — Netflix recommender

```bash
cd projects/netflix-recommender

# 1. Get the data (~2 GB unpacked) — Kaggle Netflix Prize, kept out of git
mkdir -p dataset && cd dataset
# download from https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
cd ..

# 2. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Build the parquet dataset (replaces notebook 01)
python run_data_loading.py

# 4. Run the notebooks in order
jupyter lab notebooks/
```

### Project 2 — Customer 360

```bash
cd projects/customer-360
# Open the architecture diagram in your browser:
open architecture/Customer360_Architecture.drawio   # macOS, with diagrams.net desktop
# …or upload to https://app.diagrams.net/ and open the .drawio file
```

This project is a **reference architecture and design portfolio** —
there is no notebook to run. The deliverable is the diagram, the
service justification, the cost model, and the surrounding docs.

---

## What I'd do next

In priority order:

1. **Wire the Netflix MLOps pipeline into a real SageMaker Pipelines DAG**
   and ship a working API behind an API Gateway dev stage. The spec is
   complete; the implementation isn't.
2. **Build a Customer 360 churn classifier on a public benchmark** (e.g.,
   the Telco Customer Churn dataset on Kaggle) so the architecture has a
   working model card with real numbers, not just a planned one.
3. **Replace SVD with an implicit-feedback ALS or two-tower neural
   network** on the Netflix side to handle dwell-time / clicks, not just
   explicit ratings.
4. **Add a fairness audit** (per-genre and per-decade RMSE on Netflix; per
   demographic-segment AUC on Customer 360 once the model exists).
5. **Stand up the IaC pipeline end-to-end** (CodeCommit → CodeBuild →
   CodePipeline → CDK) for one of the two projects, so the whole thing is
   reproducible from git.

---

## Honest limits

What this portfolio does *not* claim:

- ❌ Neither system is **deployed to production**. Both architectures are
  designed and priced; neither is running for real traffic.
- ❌ No **A/B test** has been run. The rollout *policy* is specified; no
  live experiment exists.
- ❌ No **business KPI improvement** is claimed. There is no live business
  attached to either dataset.
- ❌ The **Customer 360 churn model itself is not yet built** — that
  project's deliverable is the architecture and service justification, not
  a trained classifier.
- ❌ No **fairness audit** has been completed; both model cards flag this
  as a gap.

The point of these designs is that an engineer with the same artefacts
could ship them. The point of the Netflix notebooks is to demonstrate the
modelling depth that justifies the production layer above them.

---

## Credits and license

**Project 1 — Netflix recommender** was a five-person MLDS 423 group
project at Northwestern. I owned the segmentation track (RFM + cross-tab
vs clustering) and independently authored the entire production-design
layer (the README, ARCHITECTURE, MODEL_CARD, API, and MLOPS docs); the
modelling work was a team effort. Full credits in
[`projects/netflix-recommender/README.md`](projects/netflix-recommender/README.md).

**Project 2 — Customer 360** is a personal architecture portfolio piece
that builds on the Cloud Engineering / MLDS 423 reference architectures
(Slide 16 — Data & Analytics Pipeline; Slide 17 — ML Pipeline) and the
[423-architecture-lab-activity](https://github.com/sungsoolim90/423-architecture-lab-activity).
The diagram, the service-by-service justification, the cost model, and
the security/IaC narrative are mine.

**Datasets:** [Netflix Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
(rights belong to Netflix Inc.); Customer 360 uses no real customer data —
it is an architecture, not a deployed system.

**License:** Code in this repository is released under the [MIT
License](LICENSE). The Netflix Prize dataset is **not** covered by this
licence and is subject to its own terms.

---

*Yuanyuan Xie · MLDS @ Northwestern · 2026*
