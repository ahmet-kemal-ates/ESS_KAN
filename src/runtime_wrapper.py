import json
import os
from typing import Iterable, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, Iterable[float]]


class ESSKANRuntime:
    """Runtime wrapper for integration-style SoC next-step inference."""

    def __init__(self, model_ts_path: str, norm_json_path: str) -> None:
        if not os.path.exists(model_ts_path):
            raise FileNotFoundError(f"Missing model_ts.pt: {model_ts_path}")
        if not os.path.exists(norm_json_path):
            raise FileNotFoundError(f"Missing norm.json: {norm_json_path}")

        self.model = torch.jit.load(model_ts_path, map_location="cpu")
        self.model.eval()

        with open(norm_json_path, "r", encoding="utf-8") as f:
            norm = json.load(f)

        self.x_mean = np.asarray(norm["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(norm["x_std"], dtype=np.float32)
        self.y_mean = float(norm["y_mean"])
        self.y_std = float(norm["y_std"])

        if self.x_mean.shape != (3,) or self.x_std.shape != (3,):
            raise ValueError("Expected 3 input features: [Iin_A, Temp_C, SoC_prev]")

    @classmethod
    def from_artifact_dir(cls, artifact_dir: str) -> "ESSKANRuntime":
        export_dir = os.path.join(artifact_dir, "export")
        return cls(
            model_ts_path=os.path.join(export_dir, "model_ts.pt"),
            norm_json_path=os.path.join(export_dir, "norm.json"),
        )

    def _normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def _denorm_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean

    def predict_next_soc(self, iin_a: float, temp_c: float, soc_prev: float) -> float:
        x = np.array([[iin_a, temp_c, soc_prev]], dtype=np.float32)
        xn = self._normalize_x(x)

        with torch.no_grad():
            yn = self.model(torch.from_numpy(xn)).cpu().numpy().reshape(-1)

        y = self._denorm_y(yn)
        return float(np.clip(y[0], 0.0, 1.0))

    def predict_batch(self, x_raw: ArrayLike) -> np.ndarray:
        x = np.asarray(x_raw, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("x_raw must have shape [N, 3] with columns [Iin_A, Temp_C, SoC_prev]")

        xn = self._normalize_x(x)
        with torch.no_grad():
            yn = self.model(torch.from_numpy(xn)).cpu().numpy().reshape(-1)

        y = self._denorm_y(yn)
        return np.clip(y, 0.0, 1.0).astype(np.float32)
