"""PCA (train-only) + Ridge, within-session CV, cross-session R², save artifacts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from step3_code.config import (
    CV_SPLITS,
    KINEMATIC_DIM_INDICES,
    N_PCA_COMPONENTS,
    NEURAL_LAG_BINS,
    PCA_WHITEN,
    R2_MULTIOUTPUT_PRIMARY,
    R2_MULTIOUTPUT_UNIFORM,
    RANDOM_SEED,
    RESULTS_DIR,
    RIDGE_ALPHAS,
)
from step3_code.session_bundle import SessionBundle, corpus_to_arrays


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def _r2(y_true: np.ndarray, y_pred: np.ndarray, *, kind: str) -> float:
    return float(r2_score(y_true, y_pred, multioutput=kind))


def prepare_xy_session(s: SessionBundle, lag: int, y_dim_ix: Optional[Tuple[int, ...]]) -> SessionBundle:
    """Causal lag (neural leads) + optional kinematic column subset."""
    X, Y = s.X, s.Y
    if lag > 0 and X.shape[0] > lag:
        X, Y = X[:-lag].copy(), Y[lag:].copy()
    if y_dim_ix is not None:
        Y = Y[:, list(y_dim_ix)].copy()
    return SessionBundle(session_id=s.session_id, day_offset=s.day_offset, X=X, Y=Y)


def fit_scaler_pca(X_train: np.ndarray) -> Tuple[StandardScaler, PCA]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    pca = PCA(
        n_components=N_PCA_COMPONENTS,
        whiten=PCA_WHITEN,
        random_state=RANDOM_SEED,
    )
    pca.fit(Xs)
    return scaler, pca


def transform_latents(scaler: StandardScaler, pca: PCA, X: np.ndarray) -> np.ndarray:
    return pca.transform(scaler.transform(X))


def within_session_cv_r2(Z: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
    """PCA fixed; refit Ridge only inside folds."""
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []
    for train_idx, val_idx in kf.split(Z):
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=None)
        ridge.fit(Z[train_idx], Y[train_idx])
        pred = ridge.predict(Z[val_idx])
        fold_scores.append(_r2(Y[val_idx], pred, kind=R2_MULTIOUTPUT_PRIMARY))
    fs = np.asarray(fold_scores, dtype=np.float64)
    return float(fs.mean()), fs


def within_session_cv_full_dim_ridge(
    X_train: np.ndarray, Y_train: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Same CV protocol but Ridge on z-scored **full** spike dim (no PCA). Reference vs 32-D readout."""
    kf = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_scores = []
    for tr, va in kf.split(X_train):
        sc = StandardScaler()
        Xt = sc.fit_transform(X_train[tr])
        Xv = sc.transform(X_train[va])
        ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=None)
        ridge.fit(Xt, Y_train[tr])
        pred = ridge.predict(Xv)
        fold_scores.append(_r2(Y_train[va], pred, kind=R2_MULTIOUTPUT_PRIMARY))
    fs = np.asarray(fold_scores, dtype=np.float64)
    return float(fs.mean()), fs


def cross_session_rows(
    scaler: StandardScaler,
    pca: PCA,
    ridge: RidgeCV,
    train_session: SessionBundle,
    holdout_sessions: List[SessionBundle],
) -> List[dict]:
    rows = []
    train_day = train_session.day_offset
    Z_tr = transform_latents(scaler, pca, train_session.X)
    pred_tr = ridge.predict(Z_tr)
    rows.append(
        {
            "session_id": train_session.session_id,
            "days_from_train": 0.0,
            "split": "train_session_refit_eval",
            "r2_primary": _r2(train_session.Y, pred_tr, kind=R2_MULTIOUTPUT_PRIMARY),
            "r2_uniform": _r2(train_session.Y, pred_tr, kind=R2_MULTIOUTPUT_UNIFORM),
            "n_samples": train_session.X.shape[0],
        }
    )

    for sess in holdout_sessions:
        Z = transform_latents(scaler, pca, sess.X)
        pred = ridge.predict(Z)
        rows.append(
            {
                "session_id": sess.session_id,
                "days_from_train": float(abs(sess.day_offset - train_day)),
                "split": "held_out",
                "r2_primary": _r2(sess.Y, pred, kind=R2_MULTIOUTPUT_PRIMARY),
                "r2_uniform": _r2(sess.Y, pred, kind=R2_MULTIOUTPUT_UNIFORM),
                "n_samples": sess.X.shape[0],
            }
        )
    return rows


def per_output_r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.array(
        [r2_score(y_true[:, j], y_pred[:, j]) for j in range(y_true.shape[1])],
        dtype=np.float64,
    )


def run_pipeline(sessions: List[SessionBundle], train_idx: int) -> None:
    out_dir = _ensure_results_dir()

    train_session, holdouts = corpus_to_arrays(sessions, train_idx)
    y_ix = KINEMATIC_DIM_INDICES
    train_session = prepare_xy_session(train_session, NEURAL_LAG_BINS, y_ix)
    holdouts = [prepare_xy_session(h, NEURAL_LAG_BINS, y_ix) for h in holdouts]

    X_train = train_session.X
    Y_train = train_session.Y

    scaler, pca = fit_scaler_pca(X_train)
    Z_train = transform_latents(scaler, pca, X_train)

    within_mean, within_folds = within_session_cv_r2(Z_train, Y_train)
    ref_full_mean, ref_full_folds = within_session_cv_full_dim_ridge(X_train, Y_train)

    ridge = RidgeCV(alphas=RIDGE_ALPHAS, cv=None)
    ridge.fit(Z_train, Y_train)

    rows = cross_session_rows(scaler, pca, ridge, train_session, holdouts)
    df = pd.DataFrame(rows)
    df.insert(3, "r2_mean", df["r2_primary"])  # alias = primary metric for spreadsheets / proposal text
    csv_path = out_dir / "metrics_by_session.csv"
    df.to_csv(csv_path, index=False)

    pred_train = ridge.predict(Z_train)
    detail: dict = {
        "train_session_id": train_session.session_id,
        "n_pca_components": N_PCA_COMPONENTS,
        "pca_whiten": PCA_WHITEN,
        "neural_lag_bins": NEURAL_LAG_BINS,
        "kinematic_dim_indices": list(y_ix) if y_ix is not None else None,
        "r2_primary_metric": R2_MULTIOUTPUT_PRIMARY,
        "within_session_cv_r2_primary_mean": within_mean,
        "within_session_cv_r2_primary_folds": within_folds.tolist(),
        "reference_full_dim_ridge_cv_primary_mean": ref_full_mean,
        "reference_full_dim_ridge_cv_primary_folds": ref_full_folds.tolist(),
        "ridge_alpha_chosen": float(ridge.alpha_),
        "n_neurons": int(X_train.shape[1]),
        "n_kinematic_dims": int(Y_train.shape[1]),
    }
    detail["train_in_sample_r2_primary"] = _r2(Y_train, pred_train, kind=R2_MULTIOUTPUT_PRIMARY)
    detail["train_in_sample_r2_uniform"] = _r2(Y_train, pred_train, kind=R2_MULTIOUTPUT_UNIFORM)
    detail["train_in_sample_r2_per_dim"] = per_output_r2(Y_train, pred_train).tolist()

    held_details = []
    for sess in holdouts:
        Z = transform_latents(scaler, pca, sess.X)
        pr = ridge.predict(Z)
        held_details.append(
            {
                "session_id": sess.session_id,
                "days_from_train": float(abs(sess.day_offset - train_session.day_offset)),
                "r2_per_dim": per_output_r2(sess.Y, pr).tolist(),
                "r2_primary": _r2(sess.Y, pr, kind=R2_MULTIOUTPUT_PRIMARY),
                "r2_uniform": _r2(sess.Y, pr, kind=R2_MULTIOUTPUT_UNIFORM),
            }
        )
    detail["held_out_sessions"] = held_details

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2)

    joblib.dump(scaler, out_dir / "scaler.joblib")
    joblib.dump(pca, out_dir / "pca.joblib")
    joblib.dump(ridge, out_dir / "ridge.joblib")

    # Plot: one point per (session_id, day) after averaging duplicate files; y = primary R²
    fig, ax = plt.subplots(figsize=(8, 4.8))
    held = df[df["split"] == "held_out"].copy()
    held_agg = (
        held.groupby(["session_id", "days_from_train"], as_index=False)
        .agg({"r2_primary": "mean", "r2_uniform": "mean", "n_samples": "sum"})
        .sort_values("days_from_train")
    )
    ax.scatter(
        held_agg["days_from_train"],
        held_agg["r2_primary"],
        c="tab:blue",
        s=72,
        alpha=0.88,
        label="Held-out (mean if duplicate session id)",
        edgecolors="k",
        linewidths=0.3,
    )
    if len(held_agg) >= 3:
        z = np.polyfit(held_agg["days_from_train"].values, held_agg["r2_primary"].values, 1)
        p = np.poly1d(z)
        xs = np.linspace(held_agg["days_from_train"].min(), held_agg["days_from_train"].max(), 50)
        ax.plot(xs, p(xs), "r-", alpha=0.55, linewidth=2, label="Linear trend (drift)")

    ax.axhline(
        ref_full_mean,
        color="tab:orange",
        linestyle="-.",
        linewidth=1.8,
        label=f"Reference: CV Ridge on 176-D spikes (no PCA) = {ref_full_mean:.3f}",
    )
    ax.axhline(
        within_mean,
        color="tab:green",
        linestyle="--",
        linewidth=2,
        label=f"Step 3: CV Ridge on 32-D PCA ({R2_MULTIOUTPUT_PRIMARY}) = {within_mean:.3f}",
    )
    ax.axhline(
        detail["train_in_sample_r2_primary"],
        color="tab:gray",
        linestyle=":",
        alpha=0.75,
        label=f"Train in-sample R² ({R2_MULTIOUTPUT_PRIMARY}) = {detail['train_in_sample_r2_primary']:.3f}",
    )
    ax.set_xlabel("Days from train session")
    ax.set_ylabel(f"R² ({R2_MULTIOUTPUT_PRIMARY})")
    ax.set_title("Step 3: Ridge on 32-D PCA latents vs session distance (FALCON H1)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = out_dir / "r2_vs_session_distance.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {out_dir / 'summary.json'}")
    print(f"Wrote: {out_dir / 'pca.joblib'} (and scaler, ridge)")
    print(f"Wrote: {fig_path}")
    print(f"Within-session CV R² ({R2_MULTIOUTPUT_PRIMARY}) mean (32-D PCA): {within_mean:.4f}")
    print(f"Reference CV R² (176-D Ridge, no PCA): {ref_full_mean:.4f}")
    print(f"Chosen Ridge alpha: {ridge.alpha_}")


def main_from_npz(data_dir: Path, train_session_id: str | None = None):
    from step3_code.io_numpy_sessions import load_sessions_from_dir

    sessions, train_idx = load_sessions_from_dir(data_dir, train_session_id=train_session_id)
    run_pipeline(sessions, train_idx)


def main_from_falcon_h1(data_root: Path, train_nwb_stem: str | None = None) -> None:
    """Load real FALCON H1 NWB files (see ``io_falcon_h1``)."""
    from step3_code.io_falcon_h1 import load_h1_corpus

    sessions, train_idx = load_h1_corpus(data_root, train_nwb_stem=train_nwb_stem)
    run_pipeline(sessions, train_idx)
