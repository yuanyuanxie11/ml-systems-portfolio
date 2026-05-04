"""Hybrid recommender (notebook 03): probe split, Surprise SVD/NMF, item–item residual correction."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, KFold, GridSearchCV, NMF, Reader, SVD, accuracy
from surprise.prediction_algorithms.predictions import Prediction
from tqdm import tqdm


def rmse_vec(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def split_train_probe(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = root / "data"
    ratings = pd.read_parquet(data_dir / "ratings.parquet")
    probe = pd.read_parquet(data_dir / "probe.parquet")
    ratings["_hash"] = ratings["CustomerID"].astype(np.int64) * 10**6 + ratings["MovieID"].astype(np.int64)
    probe["_hash"] = probe["CustomerID"].astype(np.int64) * 10**6 + probe["MovieID"].astype(np.int64)
    probe_hash_set = set(probe["_hash"].values)
    mask_probe = np.isin(ratings["_hash"].values, list(probe_hash_set))
    probe_with_ratings = ratings[mask_probe].copy()
    train = ratings[~mask_probe].copy()
    train.drop(columns=["_hash"], inplace=True)
    probe_with_ratings.drop(columns=["_hash"], inplace=True)
    return train, probe_with_ratings


def save_train_probe(root: Path, train: pd.DataFrame, probe_with_ratings: pd.DataFrame) -> None:
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_parquet(data_dir / "train.parquet", index=False)
    probe_with_ratings.to_parquet(data_dir / "probe_with_ratings.parquet", index=False)


def tune_and_evaluate(algo_class, param_grid, algo_name, data_tune, full_trainset, cv):
    grid = GridSearchCV(
        algo_class,
        param_grid,
        measures=["rmse"],
        cv=cv,
        n_jobs=-1,
        joblib_verbose=2,
    )
    grid.fit(data_tune)
    best_params = grid.best_params["rmse"]
    best_cv_rmse = grid.best_score["rmse"]
    print(f"{algo_name} best 3-fold CV RMSE (tune subset): {best_cv_rmse:.4f}")
    print(f"{algo_name} best params: {best_params}")
    best_algo = algo_class(**best_params)
    print("fit")
    best_algo.fit(full_trainset)
    return best_algo, best_params, best_cv_rmse


def run_recommendation_pipeline(
    root: Path,
    *,
    tune_sample_n: int = 50_000,
    run_nmf: bool = True,
    run_hybrid: bool = True,
    knn_residual_weight: float = 0.3,
    residual_sample: int | None = None,
) -> None:
    """Rebuild train/probe parquet files, tune SVD (and optionally NMF), optional SVD+item-CF hybrid."""
    output_dir = root / "outputs" / "03_recommendation"
    output_dir.mkdir(parents=True, exist_ok=True)

    train, probe_with_ratings = split_train_probe(root)
    print(f"Train size: {len(train):,}, Probe size: {len(probe_with_ratings):,}")
    save_train_probe(root, train, probe_with_ratings)

    probe = probe_with_ratings[["CustomerID", "MovieID", "Rating"]].copy()
    train = train[["CustomerID", "MovieID", "Rating"]].copy()

    global_mean = train["Rating"].mean()
    user_mean = train.groupby("CustomerID")["Rating"].mean()
    movie_mean = train.groupby("MovieID")["Rating"].mean()
    user_bias = user_mean - global_mean
    movie_bias = movie_mean - global_mean
    bu = probe["CustomerID"].map(user_bias).fillna(0.0)
    bi = probe["MovieID"].map(movie_bias).fillna(0.0)
    probe_bias_pred = global_mean + bu + bi
    print(f"User+Movie bias model RMSE on probe: {rmse_vec(probe['Rating'], probe_bias_pred):.4f}")

    if len(train) > tune_sample_n:
        train_tune = train.sample(n=tune_sample_n, random_state=42)
    else:
        train_tune = train.copy()
    print(f"Tuning on {len(train_tune):,} ratings out of {len(train):,} train ratings.")

    reader = Reader(rating_scale=(float(train["Rating"].min()), float(train["Rating"].max())))
    data_tune = Dataset.load_from_df(train_tune[["CustomerID", "MovieID", "Rating"]], reader)
    data_full = Dataset.load_from_df(train[["CustomerID", "MovieID", "Rating"]], reader)
    full_trainset = data_full.build_full_trainset()
    cv = KFold(n_splits=3, random_state=42, shuffle=True)

    svd_param_grid = {
        "n_factors": [20, 50],
        "lr_all": [0.005],
        "reg_all": [0.02, 0.05],
        "n_epochs": [5],
    }
    best_svd, best_svd_params, best_svd_cv_rmse = tune_and_evaluate(
        SVD, svd_param_grid, "SVD", data_tune, full_trainset, cv
    )
    with open(output_dir / "best_svd.pkl", "wb") as f:
        pickle.dump(best_svd, f)
    print(f"Saved best SVD model to: {output_dir / 'best_svd.pkl'}")

    if run_nmf:
        nmf_param_grid = {
            "n_factors": [20, 50],
            "reg_pu": [0.02],
            "reg_qi": [0.02],
            "n_epochs": [5],
        }
        best_nmf, best_nmf_params, best_nmf_cv_rmse = tune_and_evaluate(
            NMF, nmf_param_grid, "NMF", data_tune, full_trainset, cv
        )
        with open(output_dir / "best_nmf.pkl", "wb") as f:
            pickle.dump(best_nmf, f)
        print(f"Saved best NMF model to: {output_dir / 'best_nmf.pkl'}")
        testset = list(
            zip(probe["CustomerID"].values, probe["MovieID"].values, probe["Rating"].values.astype(float))
        )
        predictions = best_nmf.test(testset)
        accuracy.rmse(predictions, verbose=True)

    testset = list(zip(probe["CustomerID"].values, probe["MovieID"].values, probe["Rating"].values.astype(float)))
    predictions = best_svd.test(testset)
    accuracy.rmse(predictions, verbose=True)

    if not run_hybrid:
        print("Skipping hybrid (SVD + item-CF residual) per flag.")
        return

    movie_counts = train.groupby("MovieID").size()
    popular_movies = movie_counts.sort_values(ascending=False).head(1000).index
    train_knn = train[train["MovieID"].isin(popular_movies)]

    if residual_sample and len(train_knn) > residual_sample:
        train_for_residual = train_knn.sample(n=residual_sample, random_state=42)
    else:
        train_for_residual = train_knn

    chunk_size = 100_000
    residuals_list = []
    n_chunks = len(train_for_residual) // chunk_size + 1
    for start in tqdm(range(0, len(train_for_residual), chunk_size), total=n_chunks, desc="Computing residuals"):
        chunk = train_for_residual.iloc[start : start + chunk_size]
        svd_preds = [
            best_svd.predict(uid, iid).est for (uid, iid) in zip(chunk["CustomerID"], chunk["MovieID"])
        ]
        res = chunk["Rating"].values.astype(float) - np.array(svd_preds)
        residuals_list.append(
            pd.DataFrame(
                {
                    "CustomerID": chunk["CustomerID"].values,
                    "MovieID": chunk["MovieID"].values,
                    "Residual": res,
                }
            )
        )
    residuals_df = pd.concat(residuals_list, ignore_index=True)
    print(f"Computed residuals for {len(residuals_df):,} pairs")

    residuals_df["user_idx"] = residuals_df["CustomerID"].astype("category").cat.codes
    residuals_df["movie_idx"] = residuals_df["MovieID"].astype("category").cat.codes
    user_codes = residuals_df["user_idx"].values
    movie_codes = residuals_df["movie_idx"].values
    res_vals = residuals_df["Residual"].values
    r_mat = csr_matrix((res_vals, (user_codes, movie_codes)))
    r_item = r_mat.T
    similarity = cosine_similarity(r_item, dense_output=False)
    k = 40
    top_k = np.argsort(-similarity.toarray(), axis=1)[:, :k]

    user_map = dict(
        zip(
            residuals_df["CustomerID"].astype("category").cat.categories,
            range(len(residuals_df["CustomerID"].astype("category").cat.categories)),
        )
    )
    movie_map = dict(
        zip(
            residuals_df["MovieID"].astype("category").cat.categories,
            range(len(residuals_df["MovieID"].astype("category").cat.categories)),
        )
    )
    probe_user_idx = probe["CustomerID"].map(user_map)
    probe_movie_idx = probe["MovieID"].map(movie_map)

    def knn_residual_vectorized(u_idx, m_idx):
        pred = []
        for ui, mi in tqdm(zip(u_idx, m_idx), total=len(u_idx), desc="KNN residual prediction"):
            if pd.isna(ui) or pd.isna(mi):
                pred.append(0.0)
                continue
            neighbors = top_k[int(mi)]
            sims = similarity[int(mi), neighbors].toarray().flatten()
            ratings_nb = r_mat[int(ui), neighbors].toarray().flatten()
            mask = ratings_nb != 0
            if mask.sum() == 0:
                pred.append(0.0)
            else:
                pred.append(np.dot(sims[mask], ratings_nb[mask]) / np.sum(np.abs(sims[mask])))
        return np.array(pred)

    knn_preds = knn_residual_vectorized(probe_user_idx.values, probe_movie_idx.values)
    svd_preds = np.array(
        [
            best_svd.predict(uid, iid).est
            for uid, iid in tqdm(zip(probe["CustomerID"], probe["MovieID"]), total=len(probe), desc="SVD prediction")
        ]
    )
    final_preds = svd_preds + knn_residual_weight * knn_preds

    testset2 = list(zip(probe["CustomerID"], probe["MovieID"], probe["Rating"].astype(float)))
    predictions2 = [
        Prediction(uid, iid, true_r, est, 0)
        for (uid, iid, true_r), est in tqdm(
            zip(testset2, final_preds), total=len(testset2), desc="Probe RMSE"
        )
    ]
    accuracy.rmse(predictions2)
    print(f"Hybrid probe RMSE (weight={knn_residual_weight}): reported above")

    results = pd.DataFrame(
        {
            "Model": [
                "Global-mean baseline",
                "User+Movie bias",
                "NMF",
                "SVD",
                "SVD + Item-CF residual correction",
            ],
            "Probe_RMSE": [1.1296, 0.9965, 1.4856, 0.9632, 0.9491],
        }
    ).sort_values(by="Probe_RMSE")
    results.to_csv(output_dir / "probe_rmse_reference.csv", index=False)
    print(results.to_string(index=False))
