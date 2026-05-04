# AWS architecture — companion to the diagram

The **source of truth** for this portfolio is the diagram committed as:

[`architecture/AWS_Architecture_Overview.png`](architecture/AWS_Architecture_Overview.png)

This file only restates the same story in words so search and diffs stay readable.

---

## End-to-end flow (matches the diagram)

1. **Sources** — CRM, call logs, billing, Zendesk, web/mobile, plus external feeds (social, credit/risk, benchmarks).
2. **Ingestion** — Real-time: Kinesis, MSK. Batch: Glue connectors, AppFlow, S3 transfer.
3. **Storage** — S3 **four-zone medallion**: Raw (immutable) → Curated (conformed) → Enriched (joined / ML-ready) → Consumption (features & aggregates).
4. **Processing & enrichment** — Glue Spark, Lambda micro-transforms, Transcribe, Comprehend; Step Functions orchestration.
5. **Warehouse & BI** — Redshift, Athena (federated), QuickSight.
6. **ML / AI** — SageMaker Pipelines (feature build → train → evaluate → register), Model Monitor (drift → retrain loop), endpoints, DynamoDB cache for low-latency reads.
7. **Apps & activation** — Amazon Connect, Amplify, Pinpoint, API Gateway + Cognito.

**Cross-cutting:** IAM, KMS, Lake Formation, Prometheus/Grafana, CloudTrail, CloudWatch, Backup.

**Foundation:** Service Catalog → CodeCommit → CodeBuild → CodePipeline → CDK → CloudFormation.

---

## How the offline code in this repo maps (intentionally narrow)

The Python pipeline here exercises **ETL + behavioural analytics** on a **single public ratings table** (Netflix Prize). That is one row in the diagram: land curated tables, engineer features, segment users, score a model. The **same diagram** describes how that pattern extends when many sources, real-time ingest, and activation channels are in scope.

---

## Honest scope

- Nothing in this repository is **deployed** to AWS; the diagram is the design reference.
- The recommender **model card** is grounded in offline probe RMSE; see `MODEL_CARD.md`.
