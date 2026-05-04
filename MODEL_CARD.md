# Model card — hybrid recommender (Netflix Prize probe)

Mitchell et al. (2019) style summary for the **best offline model in this repo**: SVD (scikit-surprise) plus an **item–item residual** correction on popular movies. Training and evaluation code: `run_recommendation.py` and `src/netflix_recommender/recommendation.py`.

---

## 1. Model summary

| Field | Value |
|--------|--------|
| Name | `svd-item-residual-hybrid` |
| Type | Explicit-feedback collaborative filtering |
| Libraries | scikit-surprise, SciPy sparse, scikit-learn cosine similarity |
| Output | Predicted rating in \([1,5]\) for \((user, item)\) |
| Headline metric | **Probe RMSE 0.9491** (vs Cinematch ~0.9525 on same contest era) |

**Formula (conceptual):** \(\hat{y} = \mathrm{clip}(\hat{y}_{SVD} + \alpha \cdot r_{KNN}, 1, 5)\) with \(\alpha = 0.3\) in the saved implementation; residual neighbourhood on top-1000 movies by volume.

---

## 2. Intended use

- Offline baseline for explicit-feedback ranking on the Netflix Prize probe split.
- Reference hybrid: matrix factorisation plus item–item residual correction.

**Not for:** implicit-only streams, cold-start without a separate policy, credit/health/hiring decisions, or production deployment without a new evaluation on your own data.

---

## 3. Metrics (probe)

| Model | Probe RMSE |
|--------|------------|
| **Hybrid (this card)** | **0.9491** |
| SVD alone | 0.9632 |
| User + item bias | 0.9965 |
| Global mean | 1.1296 |

---

## 4. Data

- **Train:** Netflix Prize training files minus probe rows → `data/train.parquet` (produced by the recommender step).
- **Evaluate:** Official probe pairs with ratings → `data/probe_with_ratings.parquet`.
- **Licence:** Dataset terms via Kaggle / Netflix; **not** covered by the repo MIT licence.

---

## 5. Ethical & safety notes

- **Re-identification risk** on sparse rating data (Narayanan & Shmatikov, 2008) — treat IDs as sensitive; encrypt and restrict access in any real deployment.
- **Popularity bias** — residual path emphasises popular titles; product policy should add diversity / freshness controls.
- **No fairness slices** on protected attributes (not present in data) — required if you join external attributes in production.

---

## 6. Limitations

1. Tuning used a **50k-row subset** for speed; full-grid search would move RMSE slightly.
2. **NMF** in the script is for comparison, not the shipping candidate.
3. **Dataset ends in 2005** — not a model of modern streaming behaviour without new features.

---

## References

- Mitchell et al. (2019), Model Cards for Model Reporting — https://arxiv.org/abs/1810.03993  
- scikit-surprise — https://surpriselib.com/  
- Netflix Prize rules and data — via Kaggle dataset page.
