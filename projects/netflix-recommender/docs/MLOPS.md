# MLOps for the Recommender Service

How the offline notebooks in this repo become a production system that
trains on a schedule, ships behind an A/B harness, monitors itself, and
gets a human paged when it shouldn't. Companion to
[`ARCHITECTURE.md`](../ARCHITECTURE.md) (the static design),
[`MODEL_CARD.md`](../MODEL_CARD.md) (what the model is), and
[`API.md`](API.md) (how callers see it).

The reader of this doc should be able to (a) trace any production
prediction back to a specific training run, (b) understand which signals
trigger a retrain, and (c) handle a SEV-2 page at 3 AM without paging
their tech lead.

---

## 1. Pipeline overview

Five stages, each owned by a SageMaker Pipelines step. Stages run in a
DAG; the whole pipeline is one CloudFormation-managed resource.

```
[ data validation ] → [ feature build ] → [ training ]
                                              ↓
                                        [ evaluation + gate ]
                                              ↓
                                  [ shadow deploy → A/B promote ]
```

| Stage | Compute | Typical wall time | Retry policy |
|---|---|---|---|
| Data validation | Glue Spark job, 10 G.1X workers | 8 min | 1 retry on transient S3 errors; alert on data contract failure |
| Feature build | EMR Serverless, 8 vCPU × 4 | 22 min | 1 retry; alert on feature-store write error |
| Training | SageMaker Training, `ml.m5.4xlarge` × 1 (CPU-only for SVD) | 35 min | No automatic retry; manual rerun on failure |
| Evaluation + gate | SageMaker Processing, `ml.m5.xlarge` | 6 min | No retry; gate failure aborts the pipeline |
| Shadow deploy | Lambda + SageMaker UpdateEndpoint | 4 min | Idempotent retries up to 3 times |

Total wall time on a clean run: ~75 minutes. The pipeline is
infrastructure-as-code: changes to any stage go through a PR and CI
re-runs the full pipeline against a sample dataset before merge.

A walk-through of each stage:

### 1.1 Data validation

Pulls the latest partition from `s3://reco-data-curated/ratings/dt=YYYY-MM-DD/`
and checks the contract with [Great Expectations](https://greatexpectations.io):

* row count within ±20% of the 30-day rolling median;
* `rating ∈ [1, 5]`, no nulls;
* `user_id` and `item_id` foreign-key coverage ≥ 99.9% against
  `dim_user` and `dim_item`;
* `event_date` not in the future, not older than the partition date.

Failures emit a `data_contract_violation` event to EventBridge and abort
the pipeline. The on-call gets paged with a link to the validation report
in S3.

### 1.2 Feature build

* Reads ratings and interaction events for the last 90 days.
* Joins to user/movie dimension tables.
* Recomputes the eight features used by the production model: user mean
  rating, user count, user variance, movie mean, movie count, movie
  release-decade one-hot, item-CF top-1000 popularity flag, time-since-
  last-interaction.
* Writes to two locations:
  1. **Offline feature store** (S3 + Glue catalog) for training.
  2. **Online feature store** (DynamoDB) for serving — only the features
     needed at request time, namespaced by `feature_view_version`.

The dual write is wrapped in a small library (`feature_store/io.py`) so
training and serving use the same column names, types, and defaults.
Schema drift between the two is the single most common cause of online/
offline skew in this kind of system; the library exists to prevent it.

### 1.3 Training

`scripts/train_svd_hybrid.py`, vendored from `notebooks/03_recommendation.ipynb`.
Inputs: training Parquet from §1.2, hyperparameters from
`config/hparams.yaml`. Outputs:

* `model.pkl` — the SVD factor matrices and item-CF residual table;
* `training_metrics.json` — losses, RMSE on validation, training time;
* `feature_view.json` — pinned feature view used for this run;
* `data_lineage.json` — input partitions and their checksums.

All artefacts are written to a versioned S3 prefix
`s3://reco-models/<model_id>/<git_sha>-<run_id>/` and registered in the
SageMaker Model Registry under the `netflix-svd-hybrid` model package
group.

The training job is deterministic: same seed, same hyperparameters, same
input partitions ⇒ identical artefacts. The CI smoke test verifies this
on a 10-row fixture.

### 1.4 Evaluation + gate

* Re-evaluates the candidate model on the held-out probe set and a
  fairness slice (per-genre, per-decade, per-popularity-decile).
* Compares against the current production model on the same probe.
* Promotes only if **all** of the following hold:

| Gate | Threshold |
|---|---|
| Probe RMSE | ≤ current production RMSE × 1.005 (i.e. no worse than +0.5%) |
| Per-decile RMSE worst slice | ≤ current production worst slice × 1.05 |
| Calibration ECE | ≤ 0.04 |
| Cold-start fallback fire rate (synthetic test) | ≤ 5% |

If any gate fails, the pipeline aborts and emits a `model_gate_failed`
event. The on-call is *not* paged automatically — failures here are
expected and the offline modelling team triages them in business hours.

### 1.5 Shadow deploy → A/B promote

Promotion is staged:

1. **Shadow** for 24 hours on 100% of traffic, predictions logged but
   not returned to clients. Verifies online/offline parity (predicted
   score difference < 1e-3 on 99.9% of requests).
2. **Canary** at 5% of traffic for 48 hours. Monitors live metrics.
3. **Ramp** to 50% over the next 72 hours, with `pause-and-hold` gates
   at 10%, 25%, 50%.
4. **Full rollout** to 100% only after a human (`reco:owner`) approves
   in the deploy console.

A rollback is one Lambda call; it flips the SageMaker endpoint variant
weights back to 100%/0% in under a minute.

---

## 2. Feature store

A single store keeps offline training and online serving on the same
schema.

| Surface | Backed by | Latency target | Retention |
|---|---|---:|---|
| Offline | S3 (Parquet) + Glue catalog | minutes (batch) | 18 months |
| Online | DynamoDB | p95 < 5 ms | rolling 30 days |
| Streaming | Kinesis Data Streams → Lambda → DynamoDB | event-time + 60 s | 7 days |

**Online/offline parity** is enforced by:

* one schema definition (`feature_views/<name>.yaml`) generating both
  read paths;
* a daily reconciliation job that samples 10K user IDs, reads features
  from both stores, and asserts equality up to a tolerance. Drift > 0.1%
  pages the on-call.

**Point-in-time correctness**: training queries always include an
`as_of_ts`. The offline store joins are time-travel safe, so a feature
value that was recomputed yesterday does not leak into a training row
with `event_date` from last week.

---

## 3. Monitoring

Three layers, each with its own dashboard.

### 3.1 Service-level (API health)

Latency, error rate, traffic, saturation — the four golden signals.
Source: API Gateway + Lambda + SageMaker endpoint metrics. Alerts:

| Signal | Warning | Page |
|---|---|---|
| 5xx rate | > 0.5% over 5 min | > 2% over 5 min |
| p95 latency | > 200 ms | > 350 ms |
| SageMaker endpoint invocation errors | > 0.1% | > 1% |
| Lambda concurrency saturation | > 70% of reserved | > 90% |

### 3.2 Model-level (prediction quality)

Daily Glue job materialises a "served-vs-acted" table that joins
recommendations to feedback events using `request_id`. Metrics:

| Metric | Computed how | Alert |
|---|---|---|
| **Online RMSE proxy** | Mean squared error between predicted score and observed `value.rating` for users who rated within 7 days | Drop > 5% vs 28-day baseline |
| **Online MAP@10** | Mean average precision over recommended items the user interacted with positively | Drop > 5% vs 28-day baseline |
| **Click-through rate (CTR)** | Clicks / impressions per surface | Drop > 10% vs 28-day baseline (per-surface) |
| **Cold-start fallback rate** | Fraction of requests where the cold-start path fired | Increase > 30% week-over-week |
| **Prediction histogram** | Distribution of `score` and `score_calibrated` | KL-divergence > 0.1 vs reference |

### 3.3 Feature-level (drift)

Population Stability Index (PSI) computed nightly per feature, comparing
the last 24 hours of online feature values against the training-time
distribution.

| PSI | Interpretation | Action |
|---|---|---|
| < 0.10 | No meaningful drift | None |
| 0.10–0.25 | Moderate drift | Slack notice; investigate within 1 business day |
| > 0.25 | Severe drift | Auto-trigger a retrain; page on-call if `> 0.4` |

Why PSI: cheap, easy to interpret, well understood by stakeholders. KS
or Wasserstein would be slightly more sensitive but harder to threshold
and harder to communicate; we pick the boring tool.

---

## 4. Retraining

Two triggers — scheduled and event-driven.

### 4.1 Scheduled

Every Sunday 02:00 UTC. Runs the full pipeline against the previous
week's data. Cost ~$310/run × 52 = ~$16K/year, comfortably inside the
ML-training budget line in [`ARCHITECTURE.md` §6](../ARCHITECTURE.md#6-cost-model).

### 4.2 Event-driven

EventBridge rules wire the following triggers to a `start-pipeline`
Lambda:

| Trigger | Source | Cooldown |
|---|---|---|
| Feature PSI > 0.25 sustained for 24 h | Drift dashboard | 7 days |
| Online MAP@10 drop > 5% sustained for 7 days | Quality dashboard | 14 days |
| Catalog change > 5% (new movies added or pulled) | Content service event | 7 days |
| Manual trigger (button in deploy console) | Operator | None |

Cooldowns prevent retrain storms — if the model is bad enough to fire
two triggers in one day, a human should look at it before the next
training kicks off.

### 4.3 Hyperparameter search

Standard runs use the pinned hyperparameters in `config/hparams.yaml`
(`n_factors=50`, `n_epochs=5`, `reg_all=0.05`, `α=1.0`). A monthly
"deeper search" runs ASHA over 32 trials with a 5x larger budget; if
the best trial beats the production gate, the new hyperparameters are
promoted to `hparams.yaml` via PR.

---

## 5. On-call runbook

### 5.1 Severities

| SEV | Signal | Initial response | Pager |
|---|---|---|---|
| SEV-1 | Service down (5xx > 50% for > 2 min) | Roll back endpoint to last-known-good variant via runbook §5.3 | Yes (24/7) |
| SEV-2 | p95 latency > 350 ms or 5xx > 2% sustained | Investigate; consider scaling up endpoint | Yes (business hours) |
| SEV-3 | Quality regression alert | Triage in business hours; freeze rollouts | Slack only |
| SEV-4 | Drift alert (single feature, PSI 0.25–0.4) | Triage in business hours | Slack only |

### 5.2 First-five-minutes checklist

1. Check the [service dashboard](https://dashboard.reco.example.com/api).
   Is the issue real or a false positive?
2. Confirm the affected variant: open `GET /v1/models` and the deploy
   console. If a recent rollout correlates, prepare to roll back.
3. Check upstream dependencies: DynamoDB throttles, Kinesis age,
   SageMaker endpoint health.
4. If unresolved in 15 min, escalate to SEV-1 and page the second on-
   call.

### 5.3 Rollback

```bash
aws sagemaker update-endpoint \
  --endpoint-name reco-endpoint-prod \
  --endpoint-config-name reco-endpoint-config-prod-lkg
```

`reco-endpoint-config-prod-lkg` ("last-known-good") is updated by the
deploy pipeline whenever a 100% rollout has been stable for 7 days. The
rollback completes in ~60 seconds; the endpoint stays warm.

If the issue is upstream of the model (e.g. corrupt feature partition),
the rollback above will not help; the runbook then routes to the
data-engineering on-call.

---

## 6. Reproducibility

Every served prediction is, in principle, reproducible from artefacts in
S3. The chain is:

```
request_id → served-vs-acted table → model_id, model_version
           → SageMaker Model Registry → s3://reco-models/.../model.pkl
                                       → data_lineage.json
                                       → s3 input partitions
                                       → feature_view.json
```

Concretely, a reviewer can:

1. Pull `model.pkl` and `feature_view.json` for the version that
   served a given prediction.
2. Recompute the user's feature vector from the offline store at the
   serving timestamp.
3. Re-score in a notebook and confirm the score matches the served
   score within 1e-6.

Step 3 is run weekly on a 1,000-row sample as a regression test; a
mismatch fires an `online_offline_skew` alert.

---

## 7. Security and compliance

Inherits everything in [`ARCHITECTURE.md` §7](../ARCHITECTURE.md#7-security-and-governance).
MLOps-specific items:

* **Model artefact encryption**: KMS CMK; access scoped to the
  SageMaker execution role.
* **PII in training data**: hashed user IDs only; no raw email,
  device ID, or IP. The hashing salt is rotated yearly.
* **Audit log**: every model promotion writes a row to a CloudTrail-
  backed S3 prefix that is WORM-locked for 7 years.
* **Right to be forgotten**: on a deletion request the user is removed
  from the offline store and a flag is set in DynamoDB blocking
  serving. The next scheduled retrain produces a model trained without
  their data; until then the served scores are non-personalised
  fallbacks.

---

## 8. Cost guardrails

Each pipeline stage has a budget alarm in CloudWatch:

| Stage | Monthly budget | Alarm at |
|---|---:|---:|
| Training (compute + storage) | $1,400 | $1,200 |
| Serving (endpoint) | $2,100 | $1,800 |
| Offline feature build | $750 | $650 |
| Drift + quality jobs | $400 | $350 |

Breaches notify `#reco-finops` via Slack. A breach for 2 consecutive
weeks triggers a cost review.

---

## 9. What this design deliberately avoids

* **No bespoke ML platform.** SageMaker Pipelines + the Model Registry
  + EventBridge cover every stage we need. Building a custom
  orchestrator would multiply on-call burden for no gain at this scale.
* **No bespoke feature store.** The reference DynamoDB + S3 setup is
  cheaper and simpler than running Feast or Tecton at this volume; the
  abstraction layer (`feature_store/io.py`) leaves us room to swap in a
  real store later.
* **No GPU training for v1.** SVD on ~100M ratings runs comfortably on
  a CPU instance; GPU is only justified once we move to a two-tower or
  transformer model.
* **No live human-in-the-loop labelling.** Ratings are explicit; the
  feedback signal is rich enough that we don't need RLHF-style label
  collection in the v1 pipeline.

---

## 10. Open items

* Wire SageMaker Clarify into the evaluation step for first-class bias
  reports.
* Replace the bespoke daily reconciliation with `feast validate`-style
  contracts.
* Move the deploy console from a Streamlit prototype to an internal
  app under `admin.reco.example.com`.
* A real ground-truth labelling loop for cold-start segments — today
  the cold-start gate uses a synthetic test rather than held-out new
  users.

---

*Last reviewed: 2026-04-29. Owner: Yuanyuan Xie. Review cadence: every
quarter, or after any SEV-1.*
