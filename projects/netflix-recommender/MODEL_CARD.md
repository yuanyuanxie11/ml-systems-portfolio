# Model Card — Netflix Prize Hybrid Recommender

A model card following the Mitchell et al. (2019) template. Describes the *current best* model in this repository: **SVD with Item-based collaborative-filtering residual correction**, trained on the Netflix Prize dataset and evaluated on the official probe set.

If you are deciding whether to use this model — or to copy its design as a baseline — please read this end-to-end. The "Limitations" and "Out-of-scope uses" sections are not boilerplate.

---

## 1. Model details

| Field | Value |
|---|---|
| Model name | `netflix-svd-hybrid` |
| Version | 1.0.0 |
| Date | 2026-04-29 |
| Owners | Yuanyuan Xie + course team (see README, "Team and credits") |
| Model type | Hybrid: matrix factorisation (SVD) + item-based collaborative-filtering residual correction |
| Algorithm class | Explicit-feedback collaborative filtering |
| Underlying libraries | [`scikit-surprise`](https://surprise.readthedocs.io/) 1.1+ for SVD; SciPy sparse matrices and `sklearn.metrics.pairwise.cosine_similarity` for Item-CF |
| Training framework | Python 3.8+, scikit-surprise, NumPy, SciPy, scikit-learn |
| Output | Predicted rating ∈ [1, 5] for a (user_id, item_id) pair |
| Code | `notebooks/03_recommendation.ipynb`, "Result Summary" cell |

### 1.1 Inputs

A request comes in as `(user_id, item_id)` (or a user_id and a list of candidate item_ids).
For each pair the model needs:

- The user's training-set rating history (for SVD's user-factor lookup)
- The item's training-set rating history (for the SVD item-factor lookup and for the Item-CF neighbourhood)
- Whether the item is in the "popular" subset (top-1000 by rating count) — the residual correction only fires for popular items

### 1.2 Outputs

A scalar prediction in [1.0, 5.0]. Ranking lists are produced by sorting predictions for a candidate set of items.

The residual model outputs a small additive correction: `final = SVD(u,i) + α · KNN_residual(u,i)`, clipped to [1, 5]. α was set to 1.0 in the reference implementation (no further blending tuning) — this is a known tuning lever.

### 1.3 Architecture

```
              ┌─────────────────────────────────────┐
   (u, i) ──▶ │ SVD                                 │ ──▶ ŷ_svd ∈ ℝ
              │  rank=50, lr=0.005, reg=0.05,        │
              │  n_epochs=5                          │
              └─────────────────────────────────────┘
                         │
                         ▼
              ┌─────────────────────────────────────┐
              │ Item-CF on residuals                │
              │  cosine sim, k=20, popular items    │
              │  ε = ŷ_svd − ŷ_true on train_knn     │
              │  KNN_residual(u,i) = Σ sim·ε         │
              └─────────────────────────────────────┘
                         │
                         ▼
              ŷ = clip(ŷ_svd + α·KNN_residual, 1, 5)
```

Total parameter count: ~25M (SVD user + item factors at rank 50 over 480K users and 17.7K items). The Item-CF component is non-parametric (lookup over precomputed similarities and residuals).

---

## 2. Intended use

### 2.1 Primary intended uses

- Predicting how a user would rate a movie they haven't yet rated, on a 1–5 scale.
- Producing a top-N list of candidate movies for a user, ranked by predicted rating.
- A baseline / reference point for evaluating richer recommender-system architectures (implicit-feedback ALS, two-tower neural nets, etc.).

### 2.2 Primary intended users

- Engineers building a movie / video recommendation experience and evaluating CF baselines.
- Reviewers / hiring managers reading this repo to assess production-readiness.
- Students learning matrix-factorisation and hybrid CF techniques.

### 2.3 Out-of-scope uses

- **Real-time, implicit-feedback recommendation.** This model is trained on explicit 1–5 star ratings; in production, most signal is implicit (plays, dwell, completion). The architecture in [ARCHITECTURE.md](ARCHITECTURE.md) anticipates a swap to an implicit-feedback model in Phase 4.
- **Cold-start recommendations** (new user with no rating history, or new item not seen in training). The Item-CF residual correction only covers items in the popular-items list; brand-new items get the SVD output, which falls back to the popularity-weighted bias. The architecture's vector-index tier is the proper cold-start path; this model is not.
- **Sensitive-content filtering or moderation.** This model has no awareness of content-safety attributes.
- **High-stakes individual decisions.** The model is appropriate for entertainment content suggestions and analogous low-stakes ranking, not for hiring, lending, healthcare, or any decision with material impact on a user.

---

## 3. Factors

### 3.1 Relevant factors

The model's behaviour and quality vary along several factors that downstream consumers should track:

- **Item popularity decile.** Performance is best on heavily-rated items (more support for both SVD factors and Item-CF neighbourhoods). On items with <50 ratings the SVD prediction collapses toward the mean.
- **User activity decile.** Users with very few ratings get less informative factors. The bias model's per-user term partly compensates, but RMSE on low-activity users is materially worse than on high-activity ones.
- **Year of release.** Older movies in the training set have more lifetime ratings and tend to be predicted more accurately. New releases (in the production-deployment scenario) need a content-based or popularity fallback.
- **Genre / language.** Not modelled here. A fairness audit would slice RMSE by these to look for disparate quality.

### 3.2 Evaluation factors

The evaluation in §4 reports overall probe RMSE only — that is the standard Netflix Prize metric. A production deployment of this model should additionally report:

- RMSE per item-popularity decile
- RMSE per user-activity decile
- Top-K precision / recall / NDCG (the rating-prediction metric is only weakly correlated with ranking quality)

These are listed under "Limitations" as work I would do next, not as work that has been done.

---

## 4. Metrics

### 4.1 Model performance — probe RMSE

| Model | Probe RMSE | Δ vs global mean |
|---|---:|---:|
| **SVD + Item-CF residual correction** (this model) | **0.9491** | **−16.0%** |
| SVD alone | 0.9632 | −14.7% |
| User + movie bias | 0.9965 | −11.8% |
| Global-mean baseline | 1.1296 | — |
| NMF | 1.4856 | +31.5% |

**Headline result**: probe RMSE 0.9491. For context, the original 2009 Netflix Prize was won at probe RMSE ≈ 0.8567 (BellKor's Pragmatic Chaos), and Netflix's own Cinematch system at the time of the contest sat at ~0.9525. This single-laptop, lightly-tuned hybrid lands within a couple of percent of Cinematch — useful as a baseline; not state-of-the-art.

### 4.2 Decision threshold

For ranking use cases, no decision threshold is needed (sort by predicted rating). For "should I recommend this?" use cases (binary), a threshold of 4.0 on the predicted rating is a sensible default; this should be tuned for the specific surface.

### 4.3 Variation approach

Probe RMSE is reported on the official Netflix probe set, evaluated once on a single seed. The original study did not perform a multi-seed variance analysis. A next-step improvement is k-fold or seed-averaged evaluation; expected variance based on data-set scale is small (<0.005 RMSE).

---

## 5. Evaluation data

| Field | Value |
|---|---|
| Source | Netflix Prize probe set (`probe.txt`) |
| Size | ~1.4M (user, movie) pairs |
| Procedure | Probe pairs are removed from the training set; ground-truth ratings are joined back from the original ratings to form `probe_with_ratings.parquet`; the model is scored on this set |
| Distribution | Held-out by Netflix's original construction; not a uniform random sample (probe overweights movies with more ratings) |

The probe is a reasonable test set for *the original problem statement*, but its distribution does not match a 2026 production stream (no clickstream, no implicit feedback, dataset stops in 2005). Use evaluation results here as a calibrated baseline, not as a production-readiness signal on its own.

---

## 6. Training data

| Field | Value |
|---|---|
| Source | Netflix Prize training set (`training_set/mv_*.txt`, 17,770 files, one per movie) |
| Size | 100,480,507 ratings, 480,189 users, 17,770 movies |
| Date range | 1999-12-31 to 2005-12-31 |
| Schema | (CustomerID, Rating ∈ {1..5}, Date) per movie file |
| Source license | Netflix Prize dataset; subject to the original Netflix terms |
| Class balance | Imbalanced toward 4-star ratings (mean ≈ 3.6); long-tail in both users and items |

Pre-processing happens in `notebooks/01_data_loading.ipynb`:

1. Parse each `mv_*.txt`, attach `MovieID` from the filename header.
2. Join with `movie_titles.txt` for `YearOfRelease` and `Title`.
3. Cast `Date` to datetime; derive `Year`, `Month`, `DayOfWeek`.
4. Save as `data/ratings.parquet` (Snappy-compressed).

For training (`03_recommendation.ipynb`):

5. Build `train.parquet` = `ratings.parquet` minus probe pairs.
6. For tuning, sample 50,000 ratings (a 0.05% subsample of train) with seed 42; fit GridSearchCV for SVD over `n_factors ∈ {20, 50}`, `reg_all ∈ {0.02, 0.05}`, `lr_all = 0.005`, `n_epochs = 5`. Pick best-RMSE setting.
7. Refit best SVD on the full training set.
8. Compute SVD residuals on the top-1000 popular movies' rating slice; build a Customer × Movie sparse residual matrix; compute item-item cosine similarity; precompute neighbourhoods (k=20).

The 50K-row tuning subset is a deliberate compute trade-off (full-train CV is ~2 hours per cell). It is documented in the model card as a known limitation.

---

## 7. Quantitative analyses

### 7.1 Unitary results

Probe RMSE 0.9491 (overall). See §4.1.

### 7.2 Intersectional results

**Not yet performed** in this version of the work. See "Limitations" — slicing RMSE by popularity decile, user-activity decile, year-of-release, and genre is the highest-priority next analytic step for a fairness-aware deployment.

---

## 8. Ethical considerations

- **Concentration of attention.** Recommendation systems can amplify popularity (the rich-get-richer effect), narrowing what users discover. The Item-CF residual correction only fires on popular items, which biases this design toward popular content. Mitigation: a "diversity" knob in the API that enforces a maximum share of any one cluster / genre in the returned top-N.
- **Sensitive attributes.** The Netflix Prize dataset famously triggered a re-identification result (Narayanan & Shmatikov, 2008); the dataset itself is anonymised but pseudonymous IDs can be re-identified by joining with public ratings on other platforms. This is one reason any production system using a similar data shape must encrypt at rest, restrict column-level access (Lake Formation), and treat `user_id` as PII.
- **Cold-start fairness.** Models trained on past behaviour can fail fairly for new users / new items; in a production setting with a cold-start vector tier, evaluate the cold-start path's quality separately and document it.
- **Feedback loops.** Once recommendations are shown, future training data is biased toward what was recommended. The architecture's A/B framework supports holdouts (a small slice of users gets random recommendations) to break the loop and gather counterfactual data.

This model has not been audited for fairness across protected attributes; the dataset does not include such attributes, but a production deployment that does should run a fairness slice in the evaluation report (per [`docs/MLOPS.md`](docs/MLOPS.md)).

---

## 9. Caveats and recommendations

### 9.1 Limitations (most important first)

1. **Hyperparameter search was time-budgeted.** The 50K-row tuning subset and 5-epoch search ceiling almost certainly leave headroom. A multi-fidelity search (BOHB / ASHA) over n_factors, regularisation, learning rate, and epochs would be expected to reduce RMSE by another 0.005–0.015 based on Surprise community benchmarks.
2. **NMF was included for completeness but not tuned.** Its 1.49 RMSE is not representative of NMF's real ceiling on this dataset; do not interpret it as a quality verdict on NMF generally.
3. **Dataset stops in 2005.** Recency-based features (last-7-day frequency, momentum) have no contemporary signal; the production architecture assumes streaming events.
4. **No implicit-feedback signals.** Plays, dwell time, completion fraction are absent here; an implicit-feedback ALS or two-tower model is the right next architecture for a real video product.
5. **Residual correction's α was not tuned.** The reference implementation uses α = 1.0 (full residual blend). A small grid over α ∈ {0.3, 0.5, 0.7, 1.0} could yield further RMSE gains — recommended as a quick win.

### 9.2 Recommendations to downstream consumers

- **Use as a baseline.** This is a calibrated reference point for evaluating richer architectures, not a production model.
- **Treat top-N rankings, not single ratings, as the consumer signal.** Always evaluate ranking metrics (NDCG@10, Precision@10) before deciding the model is "good enough" for a particular surface.
- **Pair the model with a cold-start path.** SVD + Item-CF degrades for new users / items; the architecture's vector-index tier is the proper cold-start path.
- **Monitor drift, not just performance.** PSI on the top user features detects distributional shift earlier than RMSE crosses threshold.
- **Plan for a swap.** The serving abstraction in [ARCHITECTURE.md](ARCHITECTURE.md) is intentionally model-agnostic so a future implicit-feedback model can drop in without API changes.

---

## References

- Mitchell, M., Wu, S., Zaldivar, A., et al. (2019). "Model Cards for Model Reporting." FAT* '19. <https://arxiv.org/abs/1810.03993>
- Koren, Y. (2009). "The BellKor solution to the Netflix Grand Prize." <https://www.netflixprize.com/assets/GrandPrize2009_BPC_BellKor.pdf>
- Hug, N. (2020). "Surprise: A Python library for recommender systems." JOSS. <https://surpriselib.com/>
- Narayanan, A., & Shmatikov, V. (2008). "Robust De-anonymization of Large Sparse Datasets." IEEE S&P. <https://www.cs.utexas.edu/~shmat/shmat_oak08netflix.pdf>
