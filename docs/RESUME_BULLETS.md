# Resume bullets for this project

Ready-to-paste resume framings, grouped by the kind of role you're
applying to. Every number is grounded in something that exists in this
repo or in the Customer 360 deck:

- RMSE figures: `notebooks/03_recommendation.ipynb`, "Result Summary" cell
- Netflix cost figures: [`ARCHITECTURE.md` §6](ARCHITECTURE.md#6-cost-model)
- Netflix latency / SLOs: [`docs/API.md` §1.6](docs/API.md)
- Netflix dataset scale: [Netflix Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) — 100,480,507 ratings / 480,189 users / 17,770 movies
- Customer 360 architecture: `Customer360_Architecture.drawio` (7 swim lanes, security band, IaC band)
- Customer 360 cost / SLOs / volumes: `Cloud Engineering.pptx` slide 7 callouts + slide 8 service justification

**Pick the framing that matches the role you're applying to.** Use the
**Combined Portfolio** framing when you want the recruiter to see breadth
(two complete AWS systems, two model families). Use Framing A/B/C when
you only have room for one project or the role is squarely in one lane.

---

## Combined Portfolio framing — both projects as one narrative

> Use this when you want the recruiter to read your portfolio as a coherent
> AI/ML Systems body of work rather than as two unrelated course projects.
> The through-line is **AWS production architecture, model lifecycle, and
> the bridge from notebook to production** — applied twice, to two different
> problem classes (recommendation, customer churn + segmentation).

### Candidate project names

> Pick one for the resume header. The first is more conservative, the
> second is more memorable.
>
> 1. **AWS Production ML Portfolio — Recommendation + Customer 360**
> 2. **From Notebook to Production: Two AWS ML Case Studies**

### One-line (best for tight resume real estate)

> Designed and documented two end-to-end AWS ML systems — a hybrid SVD + Item-CF **recommender** on Netflix Prize (100M ratings, **probe RMSE 0.9491**, ~$9.3K/mo at 1M MAU) and a **Customer 360 churn + segmentation platform** (7-lane AWS architecture, real-time scoring at <150 ms p95, ~$12.5K/mo) — each shipped with a Mitchell model card, SLA-grade API, and MLOps spec.

### Two-line

> Built two production-grade AWS ML systems from notebook through architecture: a **hybrid recommender** (SVD + Item-CF residual correction) on Netflix Prize achieving **probe RMSE 0.9491** (within 1% of Cinematch), and a **Customer 360 churn + segmentation platform** with an S3 medallion lake, SageMaker Pipelines, real-time DynamoDB cache (<50 ms agent lookup), and a PSI-driven drift→retrain loop.
> Each system ships the full bridge — Mitchell-template model card, REST + Cognito JWT API, MLOps runbook, line-item AWS cost model — so a reviewer can answer **"what does it cost, what's the SLA, how do you retrain, how do you roll back"** for both before reading any code.

### STAR / full bullet (cover letter, interview prep)

> **Situation.** Two MLDS 423 projects covered different problem classes (consumer-scale recommendation, enterprise churn + segmentation), but both live or die on the same question: *can you take an offline model and ship it?* That question is rarely answered in coursework.
> **Task.** Treat both as real products. Own the segmentation track and the production-design layer for the Netflix project; own the architecture and AWS-services justification for Customer 360.
> **Action.** Built two AWS reference architectures with line-item cost models — Netflix at **~$9.3K/mo at 1M MAU**, Customer 360 at **~$12.5K/mo** with **37.5 TB/yr raw ingestion** through a 4-zone S3 medallion (Raw → Curated → Enriched → Consumption). Wrote a Mitchell-template model card for the recommender, a REST contract with **150 ms p95 SLO** and Cognito JWT auth, and an MLOps spec with PSI drift detection, shadow/canary rollouts, and an on-call runbook. For Customer 360, designed the SageMaker Pipelines chain (Validate → Feature Build → Train → Evaluate → Register) feeding a DynamoDB online cache for **<50 ms agent-screen lookups**, with a closed-loop drift→retrain trigger (PSI > 0.2 or AUC < 0.78), and justified every AWS service choice against rejected alternatives (Redshift Serverless vs RA3, MSK vs third-party Kafka, Pinpoint vs SNS+SES).
> **Result.** Two repos a hiring manager can read and answer *what does it cost, what's the SLA, how do you retrain, how do you roll back* before reading a line of model code. Probe RMSE **0.9491** on Netflix; AWS architecture priced and SLA'd for both. The same discipline (cost model, model card, drift policy, IaC) appears in both — making the portfolio a body of work, not two unrelated assignments.

### Talking points (for the interview)

- **Why two projects, not one?** Recommendation and churn classification stress different parts of the stack. Recommendation pushes you on **offline ranking quality and online candidate generation** (vector store, item-CF residual). Churn pushes you on **multi-source ingestion, segmentation, and real-time agent enablement** (11 source systems, Genesys call audio through Transcribe + Comprehend, <50 ms DynamoDB cache). Doing both proves the bridge-to-production muscle is repeatable, not a one-off.
- **What's the same across both?** S3 medallion + SageMaker training/serving + DynamoDB online cache + API Gateway + 150 ms p95 SLO + PSI drift detection + Mitchell-template model card. That's the reusable production pattern; the model on top changes.
- **What's different?** Netflix is **batch-trained, online-ranked** with a popularity fallback for cold-start. Customer 360 is **near-real-time scored** at the agent screen with a closed-loop drift→retrain trigger and a multi-channel activation surface (Connect screen-pop, Pinpoint email/SMS, exec dashboards in QuickSight).
- **Honest limits**: neither is deployed; both are designed and priced. The point is the design discipline, not a fictional production claim.

### Reusable phrases (drop into a cover letter)

- *"…two AWS ML systems designed end-to-end — recommendation (Netflix Prize, 100M ratings) and Customer 360 (churn + segmentation, 11 source systems)."*
- *"…the same production discipline applied twice: model card, SLA-grade API, line-item cost model, drift→retrain policy."*
- *"…probe RMSE 0.9491 within 1% of Cinematch, plus a Customer 360 architecture sized at ~$12.5K/mo with sub-50 ms agent-screen lookups."*
- *"…I treat the bridge from notebook to production as a first-class deliverable, not a follow-up — and I've now done it twice on AWS."*

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
- **Honest limits**: cold-start isn't modelled in the notebooks — that's why the architecture proposes a content-based candidate generator and a popularity fallback. No fairness audit was run.
- **What I personally owned vs the team**: segmentation track, plus the production-design layer (README, ARCHITECTURE, MODEL_CARD, API, MLOPS).

---

## Framing B — AI / ML Product or Strategy

### One-line

> Designed an AWS production architecture for a Netflix-scale recommender (1M MAU) — S3 data lake, SageMaker training and serving, DynamoDB online feature store — sized at **~$9.3K/mo** with line-item costs and a **150 ms p95 latency SLO**.

### Two-line

> Authored an end-to-end AI-product spec for a recommender service: AWS reference architecture (S3 → Glue → SageMaker → API Gateway → DynamoDB), a Mitchell-template model card, a signed REST contract, and an MLOps pipeline with PSI drift detection and shadow/canary rollouts.
> Sized for **1M monthly active users at ~$9.3K/mo** with a layer-by-layer cost model from the AWS Pricing Calculator; latency budgeted at **150 ms p95** and decomposed across API Gateway, DynamoDB cache, and SageMaker reranking.

### STAR / full bullet

> **Situation.** Most ML coursework stops at "I trained a model in a notebook." Hiring managers screening for AI-product or ML-strategy roles need to see the next layer: cost, SLA, governance, rollout.
> **Task.** Treat the team's Netflix Prize recommender as a real product and write the artefacts a VP or staff engineer would expect before shipping.
> **Action.** Independently authored four production documents: (1) **`ARCHITECTURE.md`** — six-layer AWS design with a $9,300/mo cost model at 1M MAU; (2) **`MODEL_CARD.md`** — Mitchell et al. (2019) template covering intended use, factors, ethical considerations and failure modes; (3) **`docs/API.md`** — REST contract with auth (Cognito JWT), latency SLOs, error codes, idempotency, versioning; (4) **`docs/MLOPS.md`** — pipeline DAG, feature store with online/offline parity, PSI drift detection, retraining triggers, A/B + shadow rollout, on-call runbook.
> **Result.** A reviewer can read this repo and answer "what does it cost?", "what's the SLA?", "how do you retrain?", "what happens when it drifts?", "how do you roll back?" — every question that gates a real launch. The same numbers (probe RMSE 0.9491, $9.3K/mo, 150 ms p95) are consistent across docs.

### Talking points

- **Why $9.3K/mo and not $50K/mo or $1K/mo**: cost is dominated by ML serving (~$2.1K/mo — two `ml.m5.large` SageMaker endpoints behind auto-scaling, plus a nightly batch-transform job and OpenSearch Serverless for vector candidate generation) and ML training (~$1.4K/mo for weekly retraining + monthly hyperparameter sweep). Cheaper alternatives — Lambda-only inference, batch precomputation only — sacrifice tail latency or freshness; more expensive ones (multi-region active-active, GPU inference) are not justified at 1M MAU and a 150 ms SLA.
- **Why PSI for drift instead of KS or Wasserstein**: cheaper to compute, easier to threshold (the 0.10 / 0.25 bands are an industry convention), and easier to communicate to stakeholders. A more sensitive metric isn't useful if the on-call can't act on it.
- **Why a model card at all**: forces the team to write down what the model is *not* for. The Netflix Prize re-identification work (Narayanan & Shmatikov 2008) is a real risk class — the model card is where you say "do not join this output to externally-purchased datasets."
- **Where the spec is honest about gaps**: shadow→canary→ramp→full rollout is documented but not yet implemented; the daily online/offline reconciliation is described but the CI job is on the open-items list. I'd rather flag the gap than pretend it's done.

---

## Framing C — Generalist / first-job hunt

If you don't yet know which door will open first, lead with breadth:

> Built and shipped a five-person Netflix Prize recommender project as the segmentation lead and production-readiness owner: **probe RMSE 0.9491** (16% better than a global-mean baseline), nine RFM segments cross-referenced with six behavioural clusters, plus a complete AWS production design (architecture, model card, REST API, MLOps pipeline, ~$9.3K/mo at 1M MAU). Treats the bridge from notebook to production as a first-class deliverable.

---

## Reusable phrases (drop into a cover letter)

- *"…within 1% of Netflix's own Cinematch baseline (0.9525) on the same probe set."*
- *"…sized at $9.3K/mo at 1M monthly active users with a layer-by-layer cost model from the AWS Pricing Calculator."*
- *"…the same numbers — probe RMSE 0.9491, $9.3K/mo, 150 ms p95 — are consistent across the model card, the architecture doc, the API spec, and the MLOps spec."*
- *"…I treated the bridge from notebook to production as a first-class deliverable, not a follow-up."*

---

## What NOT to claim

A few things this portfolio does *not* support — don't put them on a resume:

- ❌ "Deployed to production" — both architectures are designed and priced, not deployed.
- ❌ "A/B tested" — the rollout *policy* is specified; no live experiment was run.
- ❌ "Improved business KPIs" — neither dataset has a live business attached.
- ❌ "Reduced latency by X%" — there is no measured latency, only an SLO.
- ❌ "Built a fairness audit" — flagged as a known gap on Netflix; not in scope for Customer 360.
- ❌ "Trained a churn model on real customer data" — Customer 360 is an architecture + service-justification deliverable; the churn model itself is course-scoped on synthetic / public benchmarks.

If a recruiter asks "did this go to prod?", the right answer is: *"No — these are course projects paired with production designs. The point of the design is that an engineer with the same artefacts could ship it; the point of the notebooks and the architecture diagram is to demonstrate the modelling and systems depth that justifies it."*

---

*Last reviewed: 2026-05-04. Update when probe RMSE, cost, or latency numbers change in the underlying docs.*
