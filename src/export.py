import argparse
import json
import os
import time
from typing import Dict

import numpy as np
import torch
import yaml

from src.model import count_params, make_kan_model


def load_norm(path: str) -> Dict:
    n = np.load(path)
    return {
        "x_mean": n["x_mean"].astype(np.float32),
        "x_std": n["x_std"].astype(np.float32),
        "y_mean": float(np.asarray(n["y_mean"]).squeeze()),
        "y_std": float(np.asarray(n["y_std"]).squeeze()),
    }


def benchmark_cpu(model: torch.nn.Module, in_dim: int, warmup: int, steps: int, batch_size: int) -> Dict:
    model.eval()
    torch.set_grad_enabled(False)

    x_single = torch.randn(1, in_dim, dtype=torch.float32)
    for _ in range(max(warmup, 1)):
        _ = model(x_single)

    t0 = time.perf_counter()
    for _ in range(max(steps, 1)):
        _ = model(x_single)
    t1 = time.perf_counter()

    single_total_s = t1 - t0
    single_step_ms = (single_total_s / max(steps, 1)) * 1000.0

    x_batch = torch.randn(batch_size, in_dim, dtype=torch.float32)
    for _ in range(max(warmup // 10, 1)):
        _ = model(x_batch)

    tb0 = time.perf_counter()
    for _ in range(max(steps // 10, 1)):
        _ = model(x_batch)
    tb1 = time.perf_counter()

    batch_iters = max(steps // 10, 1)
    batch_total_s = tb1 - tb0
    samples = batch_iters * batch_size

    return {
        "single_step_ms": float(single_step_ms),
        "single_steps_per_sec": float(1000.0 / single_step_ms if single_step_ms > 0 else 0.0),
        "batch_size": int(batch_size),
        "batch_samples_per_sec": float(samples / batch_total_s if batch_total_s > 0 else 0.0),
        "warmup_steps": int(warmup),
        "measure_steps": int(steps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ESS KAN runtime artifact and benchmark CPU latency")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--artifact_dir", type=str, required=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=1024)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = os.path.join(args.artifact_dir, "model_best.pt")
    norm_path = os.path.join(args.artifact_dir, "norm.npz")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Missing norm file: {norm_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    norm = load_norm(norm_path)

    in_dim = int(len(norm["x_mean"]))
    model = make_kan_model(
        in_dim=in_dim,
        hidden_dims=cfg["model"]["hidden_dims"],
        out_dim=1,
        grid_size=int(cfg["model"]["grid_size"]),
        spline_order=int(cfg["model"]["spline_order"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    export_dir = os.path.join(args.artifact_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    # TorchScript export for deployment in Python runtimes without training code.
    example = torch.randn(1, in_dim, dtype=torch.float32)
    ts_model = torch.jit.trace(model, example)
    ts_path = os.path.join(export_dir, "model_ts.pt")
    ts_model.save(ts_path)

    norm_json_path = os.path.join(export_dir, "norm.json")
    norm_json = {
        "x_mean": norm["x_mean"].tolist(),
        "x_std": norm["x_std"].tolist(),
        "y_mean": norm["y_mean"],
        "y_std": norm["y_std"],
    }
    with open(norm_json_path, "w", encoding="utf-8") as f:
        json.dump(norm_json, f, indent=2)

    meta_path = os.path.join(export_dir, "model_meta.json")
    meta = {
        "framework": "efficient-kan",
        "param_count": int(count_params(model)),
        "input_dim": in_dim,
        "output_dim": 1,
        "features": cfg.get("data", {}).get("input_features", ["Iin_A", "Temp_C", "SoC_prev"]),
        "target": cfg.get("data", {}).get("target", "SoC_next"),
        "hidden_dims": cfg["model"]["hidden_dims"],
        "grid_size": int(cfg["model"]["grid_size"]),
        "spline_order": int(cfg["model"]["spline_order"]),
        "source_artifact_dir": args.artifact_dir,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    bench = benchmark_cpu(
        model=model,
        in_dim=in_dim,
        warmup=args.warmup,
        steps=args.steps,
        batch_size=args.batch_size,
    )
    bench["artifact_dir"] = args.artifact_dir
    bench["export_dir"] = export_dir

    bench_path = os.path.join(args.artifact_dir, "runtime_benchmark.json")
    with open(bench_path, "w", encoding="utf-8") as f:
        json.dump(bench, f, indent=2)

    print(f"Saved: {ts_path}")
    print(f"Saved: {norm_json_path}")
    print(f"Saved: {meta_path}")
    print(f"Saved: {bench_path}")
    print(json.dumps(bench, indent=2))


if __name__ == "__main__":
    main()
