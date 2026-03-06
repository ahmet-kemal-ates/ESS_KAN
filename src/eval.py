import argparse
import json
import os

import numpy as np
import torch
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data import load_all_data
from src.model import make_kan_model


def load_norm(path: str):
    n = np.load(path)
    return {
        "x_mean": n["x_mean"].astype(np.float32),
        "x_std": n["x_std"].astype(np.float32),
        "y_mean": float(np.asarray(n["y_mean"]).squeeze()),
        "y_std": float(np.asarray(n["y_std"]).squeeze()),
    }


def normalize_x(X: np.ndarray, norm):
    return ((X - norm["x_mean"]) / norm["x_std"]).astype(np.float32)


def denorm_y(y_norm: np.ndarray, norm):
    return y_norm * norm["y_std"] + norm["y_mean"]


def residual_bins(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    err = y_pred - y_true
    abs_err = np.abs(err)
    out = []

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            m = (y_true >= lo) & (y_true < hi)
            label = f"[{lo:.1f},{hi:.1f})"
        else:
            m = (y_true >= lo) & (y_true <= hi)
            label = f"[{lo:.1f},{hi:.1f}]"

        if np.any(m):
            out.append(
                {
                    "bin": label,
                    "count": int(np.sum(m)),
                    "mae": float(abs_err[m].mean()),
                    "rmse": float(np.sqrt((err[m] ** 2).mean())),
                    "bias": float(err[m].mean()),
                }
            )

    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate ESS KAN model on unseen test set")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--artifact_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = os.path.dirname(cfg["data"]["train_mat"]) or "data"
    d = load_all_data(
        data_dir=data_dir,
        val_ratio=float(cfg["data"].get("val_ratio", 0.3)),
        seed=int(cfg.get("seed", 42)),
    )

    ckpt_path = os.path.join(args.artifact_dir, "model_best.pt")
    norm_path = os.path.join(args.artifact_dir, "norm.npz")
    out_path = os.path.join(args.artifact_dir, "test_metrics.json")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing norm file: {norm_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    norm = load_norm(norm_path)

    model = make_kan_model(
        in_dim=d["X_test"].shape[1],
        hidden_dims=cfg["model"]["hidden_dims"],
        out_dim=1,
        grid_size=int(cfg["model"]["grid_size"]),
        spline_order=int(cfg["model"]["spline_order"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_test_n = normalize_x(d["X_test"], norm)
    y_test = d["y_test"].astype(np.float32)

    with torch.no_grad():
        y_pred_n = model(torch.from_numpy(X_test_n)).cpu().numpy().reshape(-1)

    y_pred = denorm_y(y_pred_n, norm).astype(np.float32)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_test": int(len(y_test)),
        "residual_by_soc_bin": residual_bins(y_test, y_pred, n_bins=10),
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({k: metrics[k] for k in ["rmse", "mae", "r2", "n_test"]}, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
