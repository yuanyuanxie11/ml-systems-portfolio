# Customer 360 — Churn + Segmentation Platform on AWS

> A 7-swim-lane AWS reference architecture for an enterprise customer-360
> platform: ingest from 11 source systems, land in a 4-zone S3 medallion
> lake governed by Lake Formation, train churn + segmentation models
> through SageMaker Pipelines, serve real-time scores via DynamoDB cache
> for **<50 ms agent-screen lookups**, and activate through Amazon
> Connect screen-pop, Pinpoint, and QuickSight executive dashboards.
>
> This project is the **architecture, service-by-service justification,
> security/governance design, IaC pipeline, and line-item cost model.**
> The churn model itself is planned, not yet built — see
> [§ Honest scope](#honest-scope) below.

![Architecture](../architecture/Customer360_Architecture.png)

---

## Contents

1. [TL;DR](#tldr)
2. [What this project is — and isn't](#what-this-project-is--and-isnt)
3. [Architecture overview](#architecture-overview)
4. [The seven swim lanes](#the-seven-swim-lanes)
5. [Cross-cutting concerns](#cross-cutting-concerns)
6. [Headline numbers](#headline-numbers)
7. [Documents](#documents)
8. [How to read the deliverables](#how-to-read-the-deliverables)
9. [Honest scope](#honest-scope)
10. [What I'd build next](#what-id-build-next)
11. [References + course-material lineage](#references--course-material-lineage)

---

## TL;DR

| Dimension | Value |
|---|---|
| Source systems | 11 (CRM, Genesys call audio, Salesforce, Zendesk, billing, web events, mobile, IVR, social, marketing automation, NPS) |
| Storage | S3 4-zone medallion (Raw → Curated → Enriched → Consumption), Lake Formation governance |
| Training | SageMaker Pipelines: Validate → Feature Build → Train → Evaluate → Register |
| Serving | API Gateway + Cognito JWT → Lambda + SageMaker Endpoint → DynamoDB online cache |
| Activation | Amazon Connect screen-pop, Pinpoint email/SMS, QuickSight dashboards |
| Drift policy | PSI > 0.2 OR AUC < 0.78 → auto-trigger SageMaker retrain pipeline |
| Prediction latency p95 | <150 ms |
| Agent screen lookup | <50 ms (DynamoDB) |
| Pipeline / model availability | 99.9% / 99.95% |
| RPO / RTO | 15 min / 4 hr |
| Raw ingestion | ~37.5 TB / year |
| Cost (priced) | ~$12.5K / month |

The deliverable is the design — not a deployed system. Source: the AWS
Pricing Calculator (us-east-1, 2026) plus the architecture diagram and
deck (`architecture/Cloud_Engineering.pdf`).

---

## What this project is — and isn't

**It is** a portfolio piece showing that I can take a vague enterprise
problem (*"build a customer 360 platform that helps the call-centre
reduce churn"*) and translate it into a defensible AWS architecture with
service-by-service justification, cost, SLOs, security, governance, and
IaC.

**It isn't** a deployed system. There is no live AWS account behind this
diagram. The drift trigger isn't pulling real PSI numbers. The agent
screen-pop isn't running. Treating it like one would be dishonest —
hiring managers spot that immediately, and so does the code review when
they ask "show me the deployment."

**The work is in the design.** The diagram answers questions like:

- Why MSK and not third-party Kafka? *(In-VPC, IAM-integrated, no extra
  vendor.)*
- Why Redshift RA3 and not Serverless? *(Sustained dashboard workload
  makes RA3 cheaper at full duty cycle.)*
- Why Pinpoint and not just SNS+SES? *(Pinpoint adds segmentation, A/B
  testing, channel orchestration — SNS+SES alone covers transport, not
  campaigns.)*
- Why Cognito JWT and not API keys? *(Per-user identity, MFA, refresh
  tokens — table stakes for an agent-facing API.)*
- Why a 4-zone medallion and not 3? *(Consumption zone is materialised
  for QuickSight + DynamoDB; keeping it separate from "Enriched" is what
  makes the 99.9% pipeline SLO independent of model retraining.)*

The deck (`architecture/Cloud_Engineering.pdf`) walks through every one
of those choices with a "considered alternatives — rejected" frame.

---

## Architecture overview

The diagram has **7 vertical swim lanes** plus three horizontal bands
(Sources at the top, Security & Governance and IaC at the bottom):

```
SOURCES   →   INGESTION   →   STORAGE   →   PROCESSING   →   WAREHOUSE/BI   →   ML/AI   →   APPS
   (11)        Kinesis,        S3 4-zone     Glue,            Redshift RA3,      SageMaker     Connect,
               MSK, Glue,      medallion +   DataBrew,        Athena,            Pipelines     Amplify,
               Lambda,         Lake          Lambda,          QuickSight,        + Feature     Pinpoint,
               AppFlow,        Formation     Transcribe,      QuickSight Q       Store +       API GW,
               DMS                           Comprehend,                         Model         dashboards
                                             Step Functions                      Registry +
                                                                                 Endpoint +
                                                                                 DynamoDB
```

Below the lanes:

- **Security & Governance band** — IAM, KMS, Lake Formation, Macie,
  CloudTrail, CloudWatch, AWS Backup. Cross-cuts every lane.
- **IaC band** — Service Catalog → CodeCommit → CodeBuild → CodePipeline
  → CDK → CloudFormation. Everything in the architecture is provisioned
  this way.

**Three flow types** in the diagram:

- **Black solid** — primary data flow (batch + streaming).
- **Red dashed** — drift / retrain loop (Model Monitor → SageMaker
  Pipelines → new model registered → endpoint deployed).
- **Blue solid** — real-time read path (API Gateway → Lambda → DynamoDB
  cache, <50 ms p95).

Open `architecture/Customer360_Architecture.drawio` in
[diagrams.net](https://app.diagrams.net/) for the editable source.

---

## The seven swim lanes

### 1. Sources (11)
CRM, Genesys (call audio + transcripts), Salesforce (sales pipeline),
Zendesk (support tickets), billing, web events (clickstream), mobile
events, IVR logs, social listening, marketing automation, NPS surveys.
Each source has a contract — schema, frequency, PII classification, owner.

### 2. Ingestion
- **Kinesis Firehose** for high-volume streaming events (~$0.029/GB).
- **MSK (managed Kafka)** for Genesys call streams — keeps streaming
  in-VPC and IAM-integrated. Considered third-party Confluent Cloud;
  rejected (extra vendor, no IAM).
- **Glue (jobs + connectors)** for batch / CDC ingestion.
- **AppFlow** for zero-code Salesforce / Zendesk pulls.
- **DMS** for full + ongoing CDC from on-prem databases.

### 3. Storage — S3 medallion + Lake Formation
- **Raw** (immutable, source-faithful, retention per system).
- **Curated** (typed, deduped, partitioned by event date).
- **Enriched** (joined across sources, ML-ready).
- **Consumption** (materialised views for BI + DynamoDB hydration).
- **Lake Formation** governs column-level access; the Data Catalog is
  the source of truth for schemas.
- **Lifecycle**: hot in S3 Standard, cold to S3 Glacier after 90 days
  (saves ~60% on long-tail storage).

### 4. Processing
- **Glue + DataBrew** for serverless ETL and data-quality rules.
- **EMR (transient, Spot)** for heavy ad-hoc transforms.
- **Lambda** for lightweight transforms.
- **Transcribe** turns Genesys call audio into text; **Comprehend** runs
  sentiment + entity extraction on the text.
- **Step Functions** orchestrates multi-step jobs.

### 5. Warehouse + BI
- **Redshift RA3 + concurrency scaling** for the BI / dashboard layer
  (predictable workload; RA3 separates compute from S3-managed storage).
- **Redshift Spectrum** queries S3 directly so we don't duplicate data
  into the warehouse.
- **Athena** for ad-hoc SQL.
- **QuickSight** + **QuickSight Q** for executive dashboards and
  natural-language BI.

### 6. ML / AI
- **SageMaker Pipelines**: Validate & QA → Feature Build → Train →
  Evaluate → Register. Triggered on a schedule (weekly) or by drift.
- **Feature Store** with online (DynamoDB-backed) and offline (S3)
  parity.
- **Model Registry** holds versioned, approved models.
- **Endpoint** (real-time inference) behind autoscaling.
- **Model Monitor** computes PSI on input features and AUC on a held-out
  slice; emits CloudWatch metrics that fire the drift trigger.
- **DynamoDB** holds the latest score per customer for the agent-screen
  read path.

### 7. Applications + activation
- **API Gateway + Cognito JWT** for agent-facing REST APIs.
- **Lambda** handles request routing and personalisation.
- **Amplify** hosts internal dashboards.
- **Amazon Connect** integrates the agent screen — when a call lands,
  the screen pops the customer's churn risk + recommended next-best
  action.
- **Pinpoint** runs the multichannel activation (email / SMS / push
  campaigns segmented by the model output).
- **QuickSight** + custom React dashboards for the leadership layer.

---

## Cross-cutting concerns

### Security & governance ([`docs/SECURITY.md`](docs/SECURITY.md))

- **Identity & access**: IAM least-privilege per service. SSO via
  Okta / Cognito. MFA for production deploys.
- **Encryption**: KMS customer-managed keys per S3 zone; TLS 1.2+ in
  transit; client-side encryption for PII columns in Parquet.
- **Compliance**: GDPR (right-to-erasure pipeline), CCPA opt-out,
  PCI-DSS scope isolation for payment data, SOC 2 controls inherited
  from AWS.
- **PII detection**: Macie scans S3 for unintended PII leakage;
  auto-redaction in chat / email pipelines.

### IaC

The whole stack is provisioned via Service Catalog → CodeCommit →
CodeBuild → CodePipeline → CDK → CloudFormation. No console click-ops
in production accounts.

### Drift → retrain loop

- **Trigger**: Model Monitor publishes PSI on top features daily; the
  alarm fires when **PSI > 0.2** or **rolling AUC < 0.78**.
- **Action**: SageMaker Pipelines auto-runs the retrain pipeline.
- **Approval gate**: a human approves the new model in the Registry
  before it's deployed.
- **Rollout**: shadow → canary 5% → ramp 25%/50%/100%. Rollback is one
  command (`sagemaker-cli endpoint-config update --reset-prev`).

---

## Headline numbers

(Same as the TL;DR table — repeated here for the people who skip
straight to the documents.)

- **Throughput**: 37.5 TB/yr raw, 15 TB/yr curated, 5 TB/yr enriched, 2 TB/yr consumption.
- **Latency**: <150 ms p95 prediction; <50 ms agent-screen DynamoDB lookup.
- **Availability**: 99.9% pipeline, 99.95% model serving.
- **Recovery**: RPO 15 min (S3 + DDB versioning); RTO 4 hr (region failover).
- **Cost**: ~$12.5K/month at the modelled traffic, dominated by Redshift RA3
  and SageMaker Endpoint hours.

---

## Documents

| Document | What it covers | Status |
|---|---|---|
| `README.md` | This file — project overview | ✅ |
| `architecture/Customer360_Architecture.drawio` | Editable diagram source | ✅ |
| `architecture/Customer360_Architecture.png` | PNG export for GitHub rendering | ✅ |
| `architecture/Cloud_Engineering.pdf` | Slide deck — service justification, cost, governance, IaC | ✅ |
| `ARCHITECTURE.md` | Service-by-service narrative + cost model | 📝 planned |
| `docs/API.md` | `/v1/score`, `/v1/segment`, JWT auth, <150 ms SLO | 📝 planned |
| `docs/MLOPS.md` | SageMaker Pipelines DAG, drift policy, rollouts | 📝 planned |
| `docs/SECURITY.md` | IAM / KMS / Lake Formation / Macie / GDPR / CCPA | 📝 planned |
| `MODEL_CARD.md` | Mitchell template — planned churn classifier | 📝 planned |
| `REFERENCES.md` | Industry benchmarks + AWS docs + course-material lineage | 📝 planned |

The "planned" docs are intentionally not pushed in the first cut. The
diagram, the deck PDF, and this README alone are enough to evaluate the
project; the supporting docs add depth and will land incrementally.

---

## How to read the deliverables

For a hiring manager / staff engineer reviewing this in 5 minutes:

1. **Start with the diagram** — `architecture/Customer360_Architecture.png`.
2. **Read this README's [Seven swim lanes](#the-seven-swim-lanes)** for the
   service rationale.
3. **Skim the deck PDF** (`architecture/Cloud_Engineering.pdf`) for the
   "considered alternatives — rejected" frame on each major service choice.

For a peer engineer wanting to understand the operational model:

1. **Open `architecture/Customer360_Architecture.drawio`** in diagrams.net.
2. **Trace the three flow types**: black (data), red dashed (drift loop),
   blue (real-time read path).
3. **When `docs/MLOPS.md` lands**, read it for the pipeline DAG + drift
   policy + rollout policy.

---

## Honest scope

What this project does **not** claim:

- ❌ **Not deployed.** No live AWS account, no real traffic, no real users.
- ❌ **Churn model not yet built.** The architecture is designed to host
  one; the model itself is a follow-up workstream (planned: train on the
  Telco Customer Churn public dataset on Kaggle so the architecture has
  measured numbers).
- ❌ **Cost is modelled, not measured.** ~$12.5K/mo comes from the AWS
  Pricing Calculator at the volumes listed, not from a real billing
  console.
- ❌ **Compliance posture is design-only.** GDPR / CCPA / PCI-DSS / SOC 2
  controls are described, not audited.
- ❌ **No fairness audit.** The planned model card flags this as a known
  gap.

If a recruiter asks "did this go to prod?", the right answer is: *"No —
this is an architecture and design portfolio. The point is that an
engineer with the same artefacts could provision and operate it. The
churn-model layer is a follow-up workstream on a public benchmark
dataset."*

---

## What I'd build next

In priority order:

1. **Train a churn classifier on the Telco Customer Churn public
   dataset** (Kaggle) so the `MODEL_CARD.md` has measured numbers, not
   planned ones.
2. **Stand up the IaC pipeline** for one corner of the architecture
   (S3 medallion + a single SageMaker training job) to prove the design
   is buildable end-to-end.
3. **Wire the drift→retrain loop** with synthetic PSI on a public
   dataset, so the closed-loop ML lifecycle is demonstrable.
4. **Write the four planned sub-docs** (`ARCHITECTURE.md`,
   `docs/MLOPS.md`, `docs/API.md`, `docs/SECURITY.md`).
5. **Add a fairness slice** to the planned model-card evaluation
   (per-tenure, per-region AUC).

---

## References + course-material lineage

This project builds on:

**Industry benchmarks**
- CustomerGauge (2025) — *Telecom Churn & NPS Benchmark Report*.
- Reichheld, F. (1990) — *Zero Defections: Quality Comes to Services*,
  Harvard Business Review.
- Statista — *Customer Churn Rate by Industry (U.S.)*.

**ML model benchmarks**
- *Springer J. Big Data* (2019) — Telecom Churn (XGBoost AUC 0.84–0.93).
- *Nature Sci. Reports* (2025) — GA-XGBoost (AUC 0.95).

**AWS pricing & architecture**
- AWS Pricing Calculator (`calculator.aws`), us-east-1, 2026.
- AWS Well-Architected Framework — Reliability + Cost pillars.
- AWS Analytics Lens reference architectures.

**Course material lineage**
- MLDS 423 (Cloud Engineering) — *Slide 16: Data & Analytics Pipeline*
  and *Slide 17: ML Pipeline* are the reference architectures this
  Customer 360 design extends.
- [`sungsoolim90/423-architecture-lab-activity`](https://github.com/sungsoolim90/423-architecture-lab-activity)
  — the lab framing that informed the swim-lane structure.

The full citation list lives in
[`REFERENCES.md`](REFERENCES.md) (planned).

---

*Yuanyuan Xie · MLDS @ Northwestern · 2026*
