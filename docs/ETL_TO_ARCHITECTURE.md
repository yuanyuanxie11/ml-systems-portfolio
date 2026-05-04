# How ETL, RFM, and enterprise architecture connect in **one** portfolio

This repository is intentionally **one** ML-systems story, not two side-by-side repos. The Netflix Prize corpus is a **bounded, public lab** where we can run real ETL and segmentation end-to-end. The Customer 360 design is the **same ideas at enterprise scale**: more sources, stricter governance, and real-time activation. The link between them is the **data path** and the **production pattern**, not the brand of the dataset.

---

## 1. ETL — single source of truth for everything downstream

| Stage | What happens | Artefacts |
|--------|----------------|-----------|
| Raw ingest | Parse `training_set/`, `movie_titles.txt`, `probe.txt` | `data/ratings.parquet`, `data/probe.parquet` |
| Downstream | EDA, recommender train/probe split, clustering features, RFM recency/frequency | Parquet in `data/`, figures in `outputs/` |

Everything after step 1 reads from the **same** curated tables. That is exactly what a medallion-style lake does in production: raw → curated → consumption features.

---

## 2. Segmentation — RFM + clustering answer “who” before “how we serve”

- **Clustering** (`clustering.py`, notebook 04) builds behavioural cohorts (power users, harsh critics, …).
- **RFM** (`run_rfm.py`, notebook 05) scores recency / frequency / monetary proxy and names segments (Best Customers, At Risk, …).
- **Cross-tab** ties RFM segments to behavioural clusters so a product or retention team can prioritise plays.

In a Customer 360 deployment, these **same outputs** become features and segments in the consumption zone: batch scores in Redshift / SageMaker Feature Store, online segments in DynamoDB for Connect screen-pop and Pinpoint campaigns.

---

## 3. Architecture — two scales of the **same** pattern

| Concern | Offline lab (this repo) | Enterprise Customer 360 ([`CUSTOMER_360_PLATFORM.md`](CUSTOMER_360_PLATFORM.md)) |
|--------|-------------------------|-----------------------------------------------------------------------------------|
| Storage | Local `data/` + Parquet | S3 medallion (Raw → Curated → Enriched → Consumption), Lake Formation |
| Training | Local / notebook / CLI | SageMaker Pipelines (Validate → Feature → Train → Evaluate → Register) |
| Serving (design) | [`ARCHITECTURE.md`](../ARCHITECTURE.md) — API + DynamoDB + latency SLO | API Gateway + Lambda + DynamoDB **<50 ms** agent lookup |
| Segmentation | RFM + KMeans outputs | Same logic as **features** feeding churn + campaign segmentation |
| Governance | [`MODEL_CARD.md`](../MODEL_CARD.md), [`API.md`](API.md), [`MLOPS.md`](MLOPS.md) | Expanded IAM/KMS/Macie/CCPA story in the platform doc |

So: **ETL and RFM are not “Netflix-only”** — they are the *executable slice* of a pipeline whose *full* version is drawn in [`architecture/Customer360_Architecture.drawio`](../architecture/Customer360_Architecture.drawio).

---

## 4. How to read the repo in one pass

1. Skim the root [`README.md`](../README.md) — single through-line.
2. Run or read **01 → 05** (data → EDA → recommend → cluster → RFM).
3. Open **notebook 06** (architecture bridge) or this file, then [`CUSTOMER_360_PLATFORM.md`](CUSTOMER_360_PLATFORM.md) + [`ARCHITECTURE.md`](../ARCHITECTURE.md).

That order is **ETL → analytics → architecture**, one project.
