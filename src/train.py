import argparse
import json
import os
import random
from copy import deepcopy
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import apply_norm, fit_norm, load_all_data
from src.model import count_params, make_kan_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_dir(cfg: Dict) -> str:
    data_cfg = cfg.get("data", {})
    train_mat = data_cfg.get("train_mat", "data")
    if os.path.isdir(train_mat):
        return train_mat
    parent = os.path.dirname(train_mat)
    return parent if parent else "data"


def make_dataloaders(data: Dict[str, np.ndarray], batch_size: int) -> Tuple[DataLoader, DataLoader]:
    x_train = torch.from_numpy(data["X_train"])
    y_train = torch.from_numpy(data["y_train"]).unsqueeze(1)
    x_val = torch.from_numpy(data["X_val"])
    y_val = torch.from_numpy(data["y_val"]).unsqueeze(1)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_items = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        if is_train:
            optimizer.zero_grad()

        pred = model(xb)
        loss = criterion(pred, yb)

        if is_train:
            loss.backward()
            optimizer.step()

        bsz = xb.size(0)
        total_loss += loss.item() * bsz
        total_items += bsz

    return total_loss / max(total_items, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone ESS KAN model")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Optional override")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device_name = cfg.get("runtime", {}).get("device", "cpu")
    device = torch.device(device_name if device_name == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = resolve_data_dir(cfg)
    val_ratio = float(cfg.get("data", {}).get("val_ratio", 0.3))
    data = load_all_data(data_dir=data_dir, val_ratio=val_ratio, seed=seed)

    norm = fit_norm(data["X_train"], data["y_train"])
    for split in ("train", "val", "test"):
        xs = f"X_{split}"
        ys = f"y_{split}"
        data[xs], data[ys] = apply_norm(data[xs], data[ys], norm)

    train_cfg = cfg.get("train", {})
    batch_size = int(train_cfg.get("batch_size", 256))
    lr = float(train_cfg.get("lr", 1e-3))
    wd = float(train_cfg.get("weight_decay", 1e-5))
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 200))
    patience = int(train_cfg.get("early_stop_patience", 20))

    model_cfg = cfg.get("model", {})
    model = make_kan_model(
        in_dim=int(data["X_train"].shape[1]),
        hidden_dims=model_cfg.get("hidden_dims", [32, 32]),
        out_dim=1,
        grid_size=int(model_cfg.get("grid_size", 5)),
        spline_order=int(model_cfg.get("spline_order", 3)),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    train_loader, val_loader = make_dataloaders(data, batch_size)

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, device, optimizer=optimizer)
        val_loss = run_epoch(model, val_loader, criterion, device, optimizer=None)

        history.append({"epoch": epoch, "train_mse": train_loss, "val_mse": val_loss})
        print(f"epoch={epoch:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (patience={patience}).")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state")

    model.load_state_dict(best_state)

    if args.out_dir:
        out_dir = args.out_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("artifacts", f"run_{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, "model_best.pt")
    norm_path = os.path.join(out_dir, "norm.npz")
    hist_path = os.path.join(out_dir, "train_history.json")
    summary_path = os.path.join(out_dir, "train_summary.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg,
            "best_epoch": best_epoch,
            "best_val_mse": best_val,
            "param_count": count_params(model),
        },
        ckpt_path,
    )

    np.savez(
        norm_path,
        x_mean=norm["x_mean"],
        x_std=norm["x_std"],
        y_mean=np.array(norm["y_mean"], dtype=np.float32),
        y_std=np.array(norm["y_std"], dtype=np.float32),
    )

    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "data_dir": data_dir,
        "seed": seed,
        "device": str(device),
        "num_train": int(data["X_train"].shape[0]),
        "num_val": int(data["X_val"].shape[0]),
        "num_test": int(data["X_test"].shape[0]),
        "feature_dim": int(data["X_train"].shape[1]),
        "param_count": int(count_params(model)),
        "best_epoch": int(best_epoch),
        "best_val_mse": float(best_val),
        "checkpoint": ckpt_path,
        "norm_file": norm_path,
        "history_file": hist_path,
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Training complete")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
