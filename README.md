# Personal project — ML on large-scale ratings + AWS systems design

**Yuanyuan Xie** — end-to-end work: **ETL and modelling** on the Netflix Prize dataset (100M+ ratings), **customer segmentation** (clustering + RFM), and a **reference AWS architecture** for how the same class of workloads runs in production (ingest → lake → ML → activation).

---

## Tools & stack

### Python & data (this repo)

| Area | Libraries / tools |
|------|-------------------|
| Language | Python **3.11** |
| Data & ETL | **pandas**, **NumPy**, **PyArrow** (Parquet), **tqdm** |
| ML / stats | **scikit-learn** (clustering, PCA, scaling, metrics), **SciPy** (sparse linear algebra), **scikit-surprise** (SVD, NMF, tuning) |
| Visualisation | **Matplotlib**, **Seaborn**, **Plotly** |
| Notebooks | **Jupyter**, **ipykernel** |

Pinned versions: [`requirements.txt`](requirements.txt).

### AWS (reference design — diagram below)

End-to-end platform: **Kinesis / MSK**, **Glue**, **S3** (medallion zones), **Lambda**, **Step Functions**, **Transcribe**, **Comprehend**, **Redshift**, **Athena**, **QuickSight**, **SageMaker** (Pipelines, endpoints, Model Monitor), **DynamoDB**, **API Gateway + Cognito**, **Connect**, **Amplify**, **Pinpoint**; cross-cutting **IAM**, **KMS**, **Lake Formation**, observability, **CodePipeline / CDK / CloudFormation**.

![AWS architecture — sources through activation](architecture/AWS_Architecture_Overview.png)

Column-by-column walkthrough: [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Results (what we measured)

### Recommender — Netflix Prize **probe** set

| Model | Probe RMSE | vs global mean |
|--------|------------|----------------|
| **Hybrid (SVD + item–item residual)** | **0.9491** | **−16.0%** |
| SVD alone | 0.9632 | −14.7% |
| User + movie biases | 0.9965 | −11.8% |
| Global mean baseline | 1.1296 | — |

**Context:** Netflix’s **Cinematch** benchmark was **~0.9525** on the same era probe; this hybrid is in that band on commodity hardware with a constrained hyperparameter search. Implementation: `run_recommendation.py` / `src/netflix_recommender/recommendation.py`. Model scope and limits: [`MODEL_CARD.md`](MODEL_CARD.md).

### Data scale (after ETL)

| | Count |
|---|--------|
| Ratings | 100,480,507 |
| Users | 480,189 |
| Titles | 17,770 |

Source: Netflix Prize via Kaggle; not redistributed in this repo.

### Segmentation

- **Clustering:** user and movie **K-Means** on engineered behaviour features; method comparison (silhouette on movies): see [`outputs/04_clustering/algorithm_comparison.csv`](outputs/04_clustering/algorithm_comparison.csv) and figures under `outputs/04_clustering/`.
- **RFM-style segments:** nine value / lifecycle segments with **cross-tab vs behavioural clusters** — supports “who to treat differently” for CRM-style use cases (`run_rfm.py`, `outputs/05_rfm_analysis/`).

---

## Business impact (how this maps to real decisions)

This is **not** a deployed product; impact is stated as **what the artefacts enable** for a subscription or retail-style business with similar data:

| Outcome area | How the work supports it |
|--------------|---------------------------|
| **Personalisation quality** | A **~16% RMSE reduction** vs a naive global baseline means materially better ranking for the same catalogue size — fewer irrelevant recommendations, better use of catalogue inventory. |
| **Retention & CRM** | **RFM + cluster cross-tab** gives segments you can attach to **different plays** (e.g. high-value / at-risk / dormant) instead of one-size-fits-all campaigns. |
| **Operations at scale** | The **architecture** encodes **latency and freshness** (e.g. sub–5 min ingest, p95 scoring targets on the diagram), **drift → retrain** loops, and **governance** bands — the same decisions a team would need before spending on infra. |

---

## How to run

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the [Netflix Prize](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data) files under `./dataset/`, then:

```bash
python run_data_loading.py
python run_eda.py
python run_recommendation.py    # use --skip-hybrid while iterating
python clustering.py
python run_rfm.py
```

Or step through [`notebooks/01_offline_pipeline.ipynb`](notebooks/01_offline_pipeline.ipynb) from the repo root.

`dataset/` and `data/` are not committed (size + licence).

---

## Licence

Code: [MIT](LICENSE). **Dataset:** use under Kaggle / Netflix terms; not included here.
