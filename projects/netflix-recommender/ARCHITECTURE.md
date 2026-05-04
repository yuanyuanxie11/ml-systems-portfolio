# Production Architecture — Netflix-scale Recommender on AWS

This document describes how to take the offline recommender in this repo (notebooks → SVD + Item-CF hybrid, probe RMSE 0.9491) and run it as a production service on AWS for a streaming-video application of ~1M monthly active users and a 17–50K item catalogue.

The design prioritises three things: (1) a clean separation between the **data lake**, the **training/feature** layer, and the **online serving** layer; (2) AWS-native, managed services wherever the trade-off is reasonable (less operational toil for a small team); (3) a defensible cost story — every dollar in the cost table is sized from the AWS Pricing Calculator (us-east-1, 2026 list prices) and pinned to a workload assumption you can challenge.

---

## 1. System context

```
                   ┌──────────────────────────────────────────────────────────┐
                   │                    Streaming-video app                    │
                   │   (web / mobile / TV; emits ratings, plays, dwell time)  │
                   └─────────────────┬────────────────────────────────────────┘
                                     │ events                ▲ recommendations
                                     ▼                       │
              ┌──────────────────────────────────────────────┴───────────────┐
              │                  Recommender Platform (AWS)                   │
              │   data lake  →  features  →  training  →  serving  →  monitor │
              └────────────────────────────────────────────────────────────────┘
```

External systems: an event source (the product) that produces a stream of explicit ratings and implicit signals (plays, dwell, completion); a metadata system for the item catalogue; an identity provider (Cognito) for user authentication.

---

## 2. Layered architecture

The platform follows a six-layer reference pattern, retuned for a recommender workload.

| # | Layer | Primary AWS services | What lives here |
|---:|---|---|---|
| 1 | **Sources** | Kinesis Data Streams, S3 (landing), AppFlow | Raw event stream + nightly catalogue dump |
| 2 | **Ingestion** | Kinesis Data Firehose, AWS Glue Crawlers, AWS DMS | Stream → S3 raw zone, catalogue → S3 |
| 3 | **Storage / data lake** | Amazon S3 (medallion: raw → curated → feature → consumption), AWS Lake Formation, RDS PostgreSQL (metadata) | All durable data; partitioned, governed |
| 4 | **Processing & features** | AWS Glue (Spark ETL), Amazon EMR (transient PySpark for residual computation), AWS Lambda, SageMaker Feature Store | Daily feature builds; online + offline parity |
| 5 | **ML / training & serving** | SageMaker Studio, SageMaker Training Jobs, SageMaker Pipelines, SageMaker Model Monitor, SageMaker real-time + batch endpoints, OpenSearch Serverless (vector index) | Model lifecycle + serving |
| 6 | **Apps & activation** | API Gateway, AWS Lambda, Amazon DynamoDB (online score store), AWS Amplify (web), Amazon Personalize fallback | The HTTP recommendation API |

Cross-cutting: **IAM**, **AWS KMS** (CMK per S3 zone), **VPC + PrivateLink**, **CloudWatch**, **CloudTrail**, **AWS Config**, **CodePipeline + CDK** for IaC.

---

## 3. Data flow (offline + online)

### 3.1 Offline (daily/weekly)

```
Product events ──▶ Kinesis ──▶ Firehose ──▶ S3 raw (Parquet, partitioned by day)
                                              │
                                              ▼
                                       AWS Glue ETL
                                              │
                                              ▼
                                  S3 curated (cleaned, deduped)
                                              │
                                              ▼
                              EMR (PySpark) — residuals + features
                                              │
                                              ▼
                                    SageMaker Feature Store
                                              │           │
                                              ▼           ▼
                                  SageMaker Training Jobs   S3 consumption
                                  (SVD + Item-CF + ALS)     (analyst BI)
                                              │
                                              ▼
                                       Model Registry
                                              │
                                              ▼
                                 SageMaker Endpoint (online)
                                       + Batch transform (precomputed top-N)
```

The offline pipeline runs on a weekly schedule (full retrain) plus a daily incremental re-scoring job (top-N for every active user). Runs are orchestrated by SageMaker Pipelines.

### 3.2 Online (per request)

```
Client ──▶ API Gateway ──▶ Lambda (auth + request validation)
                                  │
                                  ▼
                            DynamoDB (precomputed top-N for user) ─── HIT ───▶ return
                                  │ MISS
                                  ▼
                         SageMaker Endpoint (real-time SVD/ALS) → re-rank
                                  │
                                  ▼
                            OpenSearch (candidate vector search) ─ if cold
                                  │
                                  ▼
                                Lambda re-rank + filter + return
```

Latency budget: **<150 ms p95** end-to-end. The cache-hit path (DynamoDB lookup of precomputed top-N) carries 90%+ of traffic; the SageMaker re-rank path covers cold or stale users; OpenSearch vector candidate generation is the cold-start path.

---

## 4. Components in detail

### 4.1 Data lake — medallion zones in S3

Four logical zones, all under one bucket-per-zone pattern:

| Zone | Contents | Format | Lifecycle |
|---|---|---|---|
| `raw/` | Kinesis Firehose drops; immutable | Parquet (Snappy), partitioned `year/month/day` | 30 d S3 Standard → IA at 90 d → Glacier at 365 d |
| `curated/` | Cleaned, deduped, schema-conforming | Parquet, partitioned `year/month` | 1 y Standard → IA |
| `feature/` | Feature tables for training | Parquet (Iceberg) | 1 y |
| `consumption/` | Top-N per user, item neighbour lists, dashboards | Parquet + DynamoDB exports | 90 d in S3 |

Lake Formation governs column-level access (PII columns isolated). Glue Data Catalog is the single source of schema truth.

### 4.2 Features — offline + online parity via SageMaker Feature Store

User features (per user): rating_count, rating_mean, rating_std, days_since_last_rating, segment_id, recency_band, frequency_band, monetary_band, top_genres (multi-hot).

Item features (per movie): rating_count, rating_mean, year_of_release, popularity_decile, item_neighbours_top50 (precomputed Item-CF neighbour ids).

Pair features (computed at request time): user × item interaction history (Lambda lookup).

Feature Store keeps an **offline** copy in S3 (used by training) and an **online** copy in DynamoDB (used by serving). Same feature definitions, same ingestion path → no train-serve skew.

### 4.3 Training — SageMaker Pipelines

The training pipeline is a directed graph with five stages:

1. **Data check** — Glue job validates incoming `feature/` partition for schema, null rates, row count vs 7-day moving average.
2. **Train** — SageMaker Training Job. Custom container with `scikit-surprise` + a thin wrapper that produces SVD factors + the Item-CF residual model. Reads from `feature/`, writes model artefacts + an evaluation report (probe RMSE, per-segment RMSE, top-K precision/recall).
3. **Evaluate** — A processing job compares the new model to the production model on a frozen evaluation slice. Promotion is gated on probe RMSE not regressing more than 1% AND top-10 precision not regressing more than 2%.
4. **Register** — On pass, the model is registered in SageMaker Model Registry with the evaluation report attached.
5. **Deploy** — A blue/green deployment to a SageMaker real-time endpoint with the new model on 10% shadow traffic for 24 hours; if monitoring is clean, traffic shifts to 100%.

The same DAG is also runnable on demand from a CLI (engineer wants to test a feature change) or triggered by drift events (see §4.6).

### 4.4 Serving — three-tier

The serving layer is intentionally tiered to keep p95 low and cost predictable:

- **Tier 1 — DynamoDB cache (precomputed top-N).** A nightly batch transform produces top-50 recommendations for every user with activity in the last 30 days. Stored in DynamoDB with `user_id` as PK; reads at <10 ms.
- **Tier 2 — SageMaker real-time endpoint.** Hosts the SVD + Item-CF hybrid (ml.m5.large × 2, behind auto-scaling). Used when the cache is empty / stale, or when the request asks for context-specific re-ranking (e.g. "recommendations for tonight, not all-time").
- **Tier 3 — OpenSearch vector index.** Holds 128-dim item embeddings for ~17K items. Used for the cold-start path (new user with no rating history): build an interest vector from a few onboarding answers, kNN-search the item catalogue, return top-N.

Dynamic ranking blends the three tiers using a simple weighted-sum policy maintained as a config artefact in S3 — no code change needed to retune.

### 4.5 API surface

A minimal REST API (full schemas in [`docs/API.md`](docs/API.md)):

- `POST /v1/recommendations` — body `{user_id, n, context?}` → `{items: [{item_id, score, reason_code}], model_version}`
- `GET /v1/users/{id}/segments` — returns RFM + cluster labels for the user
- `POST /v1/feedback` — explicit feedback (rating, like, dismiss); writes to Kinesis for the next training cycle

Stack: API Gateway (regional, with WAF) → Lambda (request validation, segmentation lookup, response shaping) → Tier 1/2/3 as described above. Authentication via Cognito (signed JWT in `Authorization` header). Per-user rate limit at API Gateway (100 RPS).

### 4.6 MLOps and observability

CloudWatch + SageMaker Model Monitor + a small custom dashboard handle the runtime story. The detailed spec lives in [`docs/MLOPS.md`](docs/MLOPS.md); the headlines:

- **Drift**: PSI > 0.2 on the top-10 user features triggers an alert; PSI > 0.3 triggers an automatic retraining run.
- **Performance**: rolling 7-day probe RMSE on a held-out slice. Alert if it crosses 1.00.
- **Latency**: per-endpoint p95 tracked; SLO is 150 ms; auto-scale to 2× capacity if p95 > 200 ms for 10 minutes.
- **Cost**: monthly forecast vs budget on the same dashboard. Alert at 80% of monthly budget.

CI/CD via CodePipeline: feature branch → build container → unit tests → deploy to dev SageMaker endpoint → integration tests → on merge, deploy to staging → manual approval → production blue/green.

---

## 5. Deployment topology

```
┌──────────────────────────────────────────────────────────────────────┐
│                       AWS account: prod                              │
│                                                                      │
│   ┌────────── VPC (10.0.0.0/16) ───────────┐    ┌────────────┐       │
│   │                                          │   │   Public    │       │
│   │  Private subnets (3 AZs)                │   │  edge:      │       │
│   │   ├── SageMaker endpoints               │   │  CloudFront │       │
│   │   ├── EMR transient cluster             │   │  + WAF      │       │
│   │   ├── Lambda (VPC-attached)             │   │             │       │
│   │   └── RDS PostgreSQL (Multi-AZ)         │   └─────────────┘       │
│   │                                          │                        │
│   │  Endpoints (PrivateLink): S3, KMS,      │                        │
│   │   DynamoDB, SageMaker, Secrets Mgr      │                        │
│   └──────────────────────────────────────────┘                        │
│                                                                      │
│   Account-level: KMS, IAM, Config, CloudTrail, CloudWatch,            │
│   GuardDuty, AWS Backup, Cost Explorer + Budgets                      │
└──────────────────────────────────────────────────────────────────────┘

   Separate accounts: dev, staging — same shape, smaller capacity
   Cross-account access via AWS Organizations + Control Tower
```

Three accounts (dev / staging / prod) under an Organizations management account so blast-radius of misconfigurations is bounded. Identity is centralised via IAM Identity Center.

---

## 6. Cost model

Workload assumed: 1M MAU, ~50 events / user / month → 50M events/mo, 5M API requests/day, weekly model retrain, daily batch top-N for active users.

| Layer | Monthly | Drivers |
|---|---:|---|
| **Storage (S3 + Iceberg + DynamoDB + RDS)** | $850 | 30 TB Standard + 40 TB IA/Glacier; DDB on-demand for top-N cache; RDS m5.large RI |
| **Ingestion (Kinesis + Firehose + Glue crawlers)** | $400 | 50M events × $0.029/GB Firehose; 8 KStreams shards |
| **Processing (Glue + EMR transient + Lambda)** | $750 | 400 DPU-hr/mo Glue; 60 hr × 4 m5.xl Spot EMR; 50M Lambda invocations |
| **ML / training (SageMaker Training + Pipelines + Monitor + Feature Store)** | $1,400 | 200 hr/mo training (mixed m5.xl + g4dn for ALS), Studio compute, Model Monitor |
| **ML / serving (SageMaker endpoints + batch transform + OpenSearch)** | $2,100 | 2 × ml.m5.large 24/7 ($165) + auto-scale; nightly batch transform ($300); OpenSearch Serverless 2 OCU ($300); rest is data transfer + endpoint scale-out |
| **Data warehouse / BI (Redshift Serverless + Athena + QuickSight)** | $750 | 1,000 RPU-hr Redshift; Athena queries + QuickSight 5 authors |
| **API surface (API Gateway + Amplify + WAF)** | $400 | 5M requests/day → ~150M/mo at $3.50/M = $525, plus Amplify hosting and WAF |
| **Security + monitoring (KMS, GuardDuty, CloudWatch, Config, Macie)** | $900 | KMS keys, GuardDuty findings, CloudWatch metrics + logs, Config rules |
| **DR + networking (S3 CRR, NAT, transfer)** | $500 | Cross-region replication for `curated/` and `feature/`, NAT GW data transfer |
| **Reserve / contingency (15%)** | $1,250 | Cushion for spikes; reduce as usage stabilises |
| **Total** | **≈ $9,300 / mo** | **≈ $112K / yr** |

Sourcing: AWS Pricing Calculator, us-east-1, 2026 list prices. Reserved-instance and Spot discounts applied where indicated. Numbers conservative — most realistic operating point is $7–8K/mo once Reserved-instance commitments mature.

### Levers if budget is tight

- S3 lifecycle to Glacier for cold raw events older than 180 days (−$200/mo).
- Drop SageMaker Feature Store online tier in favour of a thinner Lambda → DynamoDB cache (−$300/mo, but adds train-serve skew risk).
- Use a single ml.m5.large endpoint instead of 2 (−$80/mo, but loses HA). Don't recommend this for prod.
- Move Redshift Serverless work to Athena-only (−$500/mo, but slower analyst queries).

---

## 7. Security and governance

| Control | Implementation |
|---|---|
| Identity | Cognito user pools for end users; IAM Identity Center + permission sets for engineers |
| Encryption at rest | KMS CMK per S3 zone; per-table encryption for DynamoDB and RDS |
| Encryption in transit | TLS 1.3 enforced on all endpoints; ACM-managed certs |
| Network | All compute in private subnets; egress through NAT GW; PrivateLink to AWS APIs |
| Secrets | AWS Secrets Manager; rotation lambdas where applicable |
| Auditability | CloudTrail org-wide → centralised log archive bucket; AWS Config rules for drift; access logs on every endpoint |
| Privacy | Macie scans S3 for unintended PII; content-based redaction in feedback events; GDPR right-to-erasure runbook (per-user delete pipeline across S3 + DDB + RDS + Feature Store) |
| Threat detection | GuardDuty; AWS WAF rate limits + managed rule groups in front of API Gateway |

---

## 8. Reliability & operations

- **Availability target**: 99.9% on the recommendation API (≈ 43 min/mo downtime allowed).
- **Multi-AZ**: SageMaker endpoints, RDS, OpenSearch Serverless, Redshift Serverless all multi-AZ by default.
- **DR**: cross-region replication of `curated/` and `feature/` zones; an annual game-day exercise restores the platform in `us-west-2` from S3 CRR + last good model artefact. RTO 4h, RPO 1h.
- **Backups**: AWS Backup covers RDS (daily, 7-day retention) and DynamoDB (PITR on).
- **Runbooks**: live in `docs/runbooks/` (out of scope for this repo's MVP). One-pagers for: endpoint p95 spike, feature-store ingestion failure, drift alert, retraining job failure, S3 lifecycle misfire.

---

## 9. Roadmap (12-month build)

| Phase | Months | Deliverables |
|---|---|---|
| **1 — Foundation** | 1–3 | VPC + IAM + KMS + S3 zones; Lake Formation; Glue catalog; CodePipeline; first DMS pilot; offline notebook ported into a SageMaker container |
| **2 — Data integration** | 4–6 | Kinesis → Firehose → S3 raw; Glue ETL to curated; first Feature Store group; weekly retraining cron live; Athena + QuickSight v1 dashboards |
| **3 — Online serving** | 7–9 | SageMaker endpoint live behind API Gateway + Lambda; DynamoDB top-N cache populated nightly; Cognito auth; A/B framework; first end-to-end product integration |
| **4 — Scale & polish** | 10–12 | OpenSearch vector index + cold-start path; per-segment evaluation in Pipelines; cost optimisation pass (RIs, Spot, lifecycle); fairness audit and slice metrics |

---

## 10. Open questions and trade-offs

A few live trade-offs the next reviewer should weigh in on:

- **Surprise vs implicit-feedback ALS vs neural two-tower.** Surprise is excellent for rated data but the production world is mostly implicit (plays, dwell). I'd plan to swap to ALS or a two-tower model in Phase 4; the architecture is agnostic.
- **OpenSearch Serverless vs FAISS-on-EFS for the vector index.** OpenSearch is operationally easier; FAISS is cheaper at low QPS but adds an operational footprint.
- **Personalize as a fallback.** Amazon Personalize would simplify Phases 1–3 dramatically at the cost of less control and a higher per-recommendation price. Worth considering if the team is small.
- **Top-N cache freshness.** Daily refresh is a defensible default; some product teams will want hourly. Hourly costs ~5× more in batch-transform compute and adds complexity.

---

## References

- AWS Pricing Calculator — <https://calculator.aws/>
- AWS Well-Architected Framework — <https://aws.amazon.com/architecture/well-architected/>
- AWS Analytics Lens — <https://docs.aws.amazon.com/wellarchitected/latest/analytics-lens/>
- SageMaker Feature Store concepts — <https://docs.aws.amazon.com/sagemaker/latest/dg/feature-store.html>
- Mitchell et al. 2019, "Model Cards for Model Reporting" — <https://arxiv.org/abs/1810.03993>
- Netflix Prize — <https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data>
