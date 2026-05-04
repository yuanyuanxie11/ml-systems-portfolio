# Netflix-Scale Recommender + Customer Segmentation

> An end-to-end recommender system and customer-segmentation pipeline on the
> Netflix Prize dataset, paired with a complete AWS production design.
> Built to demonstrate **modelling depth** *and* **AI-product thinking** —
> from a first-principles RMSE story all the way to a sized, priced, and
> operable serving stack.

![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)
![Probe RMSE 0.9491](https://img.shields.io/badge/probe%20RMSE-0.9491-brightgreen.svg)
![Cost ~$9.3K/mo @ 1M MAU](https://img.shields.io/badge/AWS%20cost-~%249.3K%2Fmo-orange.svg)
![License MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)

---

## Contents

1. [TL;DR](#tldr)
2. [What this repo demonstrates](#what-this-repo-demonstrates)
3. [Repository layout](#repository-layout)
4. [Results](#results)
5. [Quickstart](#quickstart)
6. [From notebook to production](#from-notebook-to-production-the-bridge)
7. [Limitations and roadmap](#limitations-and-roadmap)
8. [Team and credits](#team-and-credits)
9. [License](#license)

---

## TL;DR

| What | Result |
|---|---|
| Best model on the Netflix Prize probe set | **SVD + Item-based CF residual correction**, **probe RMSE 0.9491** |
| SVD alone | RMSE 0.9632 |
| User + movie bias only | RMSE 0.9965 |
| Naïve global-mean baseline | RMSE 1.1296 |
| Customer segmentation | RFM + K-Means cross-tab, 9 named segments |
| Dataset | 100,480,507 ratings, 480,189 users, 17,770 movies (Netflix Prize) |
| Stack | Python, pandas, scikit-learn, scikit-surprise, scipy, plotly |
| Production target | AWS (S3 + Glue + EMR + SageMaker + API Gateway), see [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Sized & priced | ~$9.3K/mo at 1M monthly active users (line-item cost model) |

The hybrid model beats the bias baseline by **4.7% RMSE** and the global mean
by **16.0%** — close to the regime where matrix-factorisation entrants landed
in the original $1M Netflix Prize, on far less compute. The point of this
repo, though, is not the RMSE — it is what the offline result becomes once
you put it behind an SLA.

---

## What this repo demonstrates

Most "recommender system" portfolio projects stop at "I trained a model in
a notebook." This repo deliberately overshoots that bar in two directions:

**ML / Data Science track**

- A **disciplined modelling pipeline** — reproducible Python under
  `src/netflix_recommender/` plus thin Jupyter notebooks call the same
  functions used by the CLIs: raw per-movie text files → Parquet,
  probe-set evaluation for four ranking models, and documentation of
  *why* each model behaves as it does.
- **Two segmentation methods cross-validated against each other** —
  K-Means clusters on user/movie behaviour features (notebook 04) and
  RFM quintile scoring on activity (notebook 05), reconciled in a
  cross-tab heatmap so a downstream marketer can pick which behavioural
  cohort to target with which retention play.
- **Honest, defensible numbers** — every metric in this README appears
  verbatim in a notebook output cell. The model card calls out the
  un-tuned hyperparameter (α=1.0 in the residual correction) and the
  evaluation slices that were not run.

**AI / Product strategy track**

- **A complete AWS production design** ([`ARCHITECTURE.md`](ARCHITECTURE.md))
  — medallion S3 data lake, Glue / EMR ETL, SageMaker training + serving,
  API Gateway + Lambda for the recommendation API, DynamoDB as the
  online feature/score store, OpenSearch for vector candidate
  generation. Sized for 1M monthly active users with a layer-by-layer
  cost model (~$9.3K/mo) sourced from the AWS Pricing Calculator.
- **A signed REST contract** ([`docs/API.md`](docs/API.md)) — endpoints,
  JSON schemas, auth model, rate limits, latency SLOs (150 ms p95),
  error envelope, idempotency, versioning policy. The kind of doc a
  client team would integrate against without asking follow-up
  questions.
- **An MLOps spec** ([`docs/MLOPS.md`](docs/MLOPS.md)) — pipeline DAG,
  feature store with online/offline parity, drift detection (PSI),
  retraining triggers, A/B + shadow rollout policy, on-call runbook
  with rollback command.
- **A model card** ([`MODEL_CARD.md`](MODEL_CARD.md)) — Mitchell et al.
  (2019) template with intended use, factors, ethical considerations
  (popularity amplification, re-identification risk per Narayanan &
  Shmatikov 2008), and a do/don't list for downstream consumers.

The result is a single repo a hiring manager can use to assess (a) whether
you can train a model and (b) whether you can ship one. That dual lens is
exactly the gap that closes the bridge from MLDS coursework to either a
production ML role or an AI-product / strategy role.

---

## Repository layout

```
.
├── README.md                       <- portfolio entry (repo root)
├── README_academic.md              <- original course-project README, preserved
├── ARCHITECTURE.md                 <- production AWS architecture + cost model
├── MODEL_CARD.md                   <- Mitchell-template model card
├── LICENSE                         <- MIT
├── requirements.txt                <- pinned dependencies (Python 3.11)
│
├── src/netflix_recommender/        <- importable pipeline (data, EDA, recommend, cluster, RFM)
│
├── architecture/                   <- Customer 360 diagram source (.drawio)
├── docs/
│   ├── ETL_TO_ARCHITECTURE.md      <- glue: ETL → RFM → enterprise design
│   ├── NETFLIX_PIPELINE.md         <- this document (modelling deep dive)
│   ├── CUSTOMER_360_PLATFORM.md
│   ├── RESUME_BULLETS.md
│   ├── API.md                      <- REST API contract for the serving layer
│   ├── MLOPS.md                    <- training pipeline, monitoring, retraining
│   └── assignment/                 <- original course briefs (Chinese + English)
│
├── notebooks/                      <- thin wrappers; same code as the CLIs below
│   ├── 01_data_loading.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_recommendation.ipynb
│   ├── 04_clustering.ipynb
│   ├── 05_rfm_analysis.ipynb
│   └── 06_architecture_bridge.ipynb
│
├── run_data_loading.py             <- CLI: raw .txt → Parquet
├── run_eda.py                      <- CLI: summary tables
├── run_recommendation.py           <- CLI: Surprise + hybrid (long run)
├── clustering.py                   <- CLI: clustering figures + CSV
├── run_rfm.py                      <- CLI: RFM + figures
│
└── outputs/
    ├── 04_clustering/              <- cluster figures + cluster-membership parquet
    └── 05_rfm_analysis/            <- treemap, heatmap, scatter, segments parquet
```

`README_academic.md` preserves the original course deliverable's framing.
The repo root **`README.md`** is the unified portfolio narrative; **this
file** is the modelling + CLI deep dive for the Netflix Prize track.

---

## Results

### Recommendation models — probe RMSE (lower is better)

| Rank | Model | Probe RMSE | Δ vs global mean | Notes |
|---:|---|---:|---:|---|
| 1 | **SVD + Item-CF residual correction** | **0.9491** | **−16.0%** | Hybrid: SVD prediction + α·KNN residual on top-1000 popular movies, clipped to [1, 5] |
| 2 | SVD (matrix factorisation) | 0.9632 | −14.7% | `scikit-surprise` SVD, 50 factors, 5 epochs, λ=0.05 |
| 3 | User + movie bias | 0.9965 | −11.8% | `pred = global_mean + user_bias + movie_bias` |
| 4 | Global-mean baseline | 1.1296 | — | `pred = train.mean()` |
| 5 | NMF | 1.4856 | +31.5% | Default settings; included for completeness |

Source: `run_recommendation.py` / `netflix_recommender.recommendation` (same numbers as the original notebook). NMF
underperforms here because it constrains factors to be non-negative and
was not extensively tuned; it is included to show the modelling space, not
because we recommend it.

For context: the original Netflix Cinematch baseline scored RMSE 0.9525 on
the probe; the BellKor's Pragmatic Chaos winner scored 0.8567 after three
years of ensemble work. This hybrid sits between Cinematch and a single
strong matrix-factorisation entrant — a respectable place for a
five-week course project, and a reasonable starting-point model for the
production system.

### Customer segmentation — RFM × behavioural clusters

The RFM analysis produces nine named segments (*Best Customers, Loyal
Customers, Big Spenders, Potential Loyal, At Risk, Need Attention,
Hibernating, Lost, Lost Cheap*) and cross-references them against the
K-Means behavioural clusters from notebook 04 (*Casual Users, Power Users,
Harsh Critics, Generous Raters, Selective Viewers, Active Explorers*). The
cross-tab heatmap in `outputs/05_rfm_analysis/rfm_vs_clustering_heatmap.png`
identifies which behavioural cohort dominates each value segment — useful
for ranking which kind of user to target with which retention play.

See `outputs/05_rfm_analysis/` for the figures and
[`MODEL_CARD.md`](MODEL_CARD.md) for full evaluation details.

---

## Quickstart

### 1. Get the data

The Netflix Prize dataset is publicly available on Kaggle:
<https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data>. Unpack
into the **repository root** (same folder as `README.md` and `requirements.txt`):

```
./
├── dataset/
│   ├── training_set/        # 17,770 files mv_*.txt
│   ├── movie_titles.txt
│   └── probe.txt
```

The `dataset/` folder is git-ignored.

### 2. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.11. Main dependencies: `pandas`, `numpy`, `scipy`,
`scikit-learn`, `scikit-surprise`, `pyarrow`, `matplotlib`, `seaborn`,
`plotly`, `tqdm`.

### 3. Build the dataset

```bash
python run_data_loading.py            # produces data/ratings.parquet, data/probe.parquet
```

Use `python run_data_loading.py --sample-fraction 0.1` for a 10%-of-movies
smoke test. You can also open `notebooks/01_data_loading.ipynb` (wrapper).

### 4. Train and evaluate

Prefer the CLIs (same code the notebooks import):

```bash
python run_eda.py
python run_recommendation.py          # add --skip-hybrid for a shorter dev run
python clustering.py
python run_rfm.py
```

Or run `notebooks/02_*.ipynb` … `05_*.ipynb` in order from this directory.

Each step depends on `data/ratings.parquet`; recommendation also writes
`data/train.parquet` and `data/probe_with_ratings.parquet`.

---

## From notebook to production: the bridge

The CLIs and notebooks above are the *offline* path. The production design lives
in four cross-linked documents:

- **[`ARCHITECTURE.md`](ARCHITECTURE.md)** — AWS reference architecture.
  Medallion S3 data lake (raw → curated → feature → consumption),
  Glue / EMR for batch ETL, SageMaker for training and serving, API
  Gateway + Lambda for the recommendation API, DynamoDB as the online
  feature/score store, OpenSearch for the candidate-generation tier.
  Layer-by-layer cost estimate (~$9.3K/mo at 1M MAU) sourced from the
  AWS Pricing Calculator (us-east-1, 2026 list prices).
- **[`MODEL_CARD.md`](MODEL_CARD.md)** — model card per Mitchell et al.
  (2019): intended use, training data, evaluation, ethical considerations,
  known failure modes, and a do/don't list for downstream consumers.
- **[`docs/API.md`](docs/API.md)** — REST API contract:
  `POST /v1/recommendations`, `GET /v1/users/{id}/segments`,
  `POST /v1/feedback`, plus request/response JSON schemas, error
  envelope, latency SLOs (150 ms p95), Cognito auth model, rate limits.
- **[`docs/MLOPS.md`](docs/MLOPS.md)** — training pipeline (SageMaker
  Pipelines), feature store with online + offline parity, drift detection
  (PSI on user-activity features), retraining triggers (scheduled weekly +
  drift-triggered), shadow / canary / A/B rollout policy, and on-call
  runbook with the rollback command.

The intent is that an engineer reading these documents can size, price,
build, and operate this system — not just reproduce the notebooks.

---

## Limitations and roadmap

The model card lists the full set; the headline limitations are:

- **The Netflix Prize data stops in 2005.** Recency-based features have no
  contemporary signal. The production architecture compensates by
  ingesting streaming events from a live product instead.
- **Hyperparameter search was time-budgeted** — 50K-row tuning subset, 5
  epochs. On a real budget I would run a multi-fidelity search (BOHB /
  ASHA) over `n_factors`, regularisation, and epochs.
- **Cold-start is not modelled in the offline notebooks.** The hybrid
  uses item-based CF on popular movies; for new users / new items the
  architecture proposes a content-based candidate generator (movie
  metadata embeddings) and a popularity-based fallback.
- **No fairness audit.** The model card flags this and proposes per-genre
  and per-decade RMSE breakdowns as a next step; the production design
  includes a fairness slice in the evaluation report.

What I would build next, in priority order:

1. Replace SVD with an implicit-feedback ALS or two-tower neural network
   to handle dwell-time / clicks, not just explicit ratings.
2. Add the candidate-generation tier (vector index in OpenSearch / FAISS)
   so we don't score the full 17K-movie catalogue per request.
3. Wire the `MLOPS.md` retraining pipeline into a real SageMaker
   Pipelines DAG and ship a working API behind an API Gateway dev stage.

---

## Team and credits

Original course project (MLDS @ Northwestern, MLDS 423). All five team
members contributed:

| Member | Phase |
|---|---|
| Yu-Sen Wu | Data loading + initial EDA |
| Eric Wu | EDA: user / movie behaviour, correlations |
| Xinqi Huang | Recommendation engine: SVD / NMF / hybrid |
| Kun-Yu Lee | Clustering: K-Means, hierarchical, DBSCAN, t-SNE |
| Yuanyuan Xie | RFM segmentation, cross-reference with clustering, business insights — **plus the entire production-design layer (this README, [`ARCHITECTURE.md`](ARCHITECTURE.md), [`MODEL_CARD.md`](MODEL_CARD.md), [`docs/API.md`](docs/API.md), [`docs/MLOPS.md`](docs/MLOPS.md)).** |

Dataset: [Netflix Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
(released by Netflix; rights belong to Netflix Inc.). Model code uses
[`scikit-surprise`](https://surprise.readthedocs.io/) (BSD-3) and standard
SciPy / scikit-learn primitives.

---

## License

Code in this repository is released under the [MIT License](LICENSE).
The Netflix Prize dataset is **not** covered by this licence and is
subject to its [own terms](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data);
review before redistributing.
