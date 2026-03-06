import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split


def _to_1d(x) -> np.ndarray:
    arr = np.asarray(x).squeeze()
    if arr.ndim == 0:
        arr = np.array([arr], dtype=np.float32)
    return arr.astype(np.float32)


def _profile_id_from_name(path: str) -> str:
    name = os.path.basename(path)
    m = re.search(r"RW(\d+)", name, re.IGNORECASE)
    return m.group(1) if m else name


def _load_single_mat(path: str, is_train: bool) -> Dict[str, np.ndarray]:
    d = sio.loadmat(path)
    p = "tr" if is_train else "ts"

    iin = _to_1d(d[f"{p}_Iin"])
    soc = _to_1d(d[f"{p}_SoC"])
    temp = _to_1d(d["Temp"])
    vout = _to_1d(d.get(f"{p}_Vout", np.zeros_like(iin)))
    time = _to_1d(d.get(f"{p}_Time", np.arange(len(iin), dtype=np.float32)))

    n = min(len(iin), len(soc), len(vout), len(time))
    if len(temp) == 1:
        temp = np.full(n, temp.item(), dtype=np.float32)
    else:
        n = min(n, len(temp))
        temp = temp[:n]

    return {
        "Iin": iin[:n],
        "SoC": soc[:n],
        "Temp": temp[:n],
        "Vout": vout[:n],
        "Time": time[:n],
        "profile_id": _profile_id_from_name(path),
    }


def _series_to_supervised(series: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    iin = series["Iin"]
    temp = series["Temp"]
    soc = series["SoC"]

    n = min(len(iin), len(temp), len(soc))
    if n < 2:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

    x = np.stack([iin[: n - 1], temp[: n - 1], soc[: n - 1]], axis=1).astype(np.float32)
    y = soc[1:n].astype(np.float32)
    return x, y


def load_all_data(data_dir: str, val_ratio: float = 0.3, seed: int = 42) -> Dict[str, np.ndarray]:
    train_files = sorted(glob.glob(os.path.join(data_dir, "trainRW*.mat")))
    test_files = sorted(glob.glob(os.path.join(data_dir, "testRW*.mat")))

    if not train_files or not test_files:
        raise FileNotFoundError("Expected trainRW*.mat and testRW*.mat under data_dir")

    x_train_all: List[np.ndarray] = []
    y_train_all: List[np.ndarray] = []
    for f in train_files:
        s = _load_single_mat(f, is_train=True)
        x, y = _series_to_supervised(s)
        if len(y) > 0:
            x_train_all.append(x)
            y_train_all.append(y)

    x_test_all: List[np.ndarray] = []
    y_test_all: List[np.ndarray] = []
    for f in test_files:
        s = _load_single_mat(f, is_train=False)
        x, y = _series_to_supervised(s)
        if len(y) > 0:
            x_test_all.append(x)
            y_test_all.append(y)

    x_train_full = np.concatenate(x_train_all, axis=0)
    y_train_full = np.concatenate(y_train_all, axis=0)
    x_test = np.concatenate(x_test_all, axis=0)
    y_test = np.concatenate(y_test_all, axis=0)

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full,
        y_train_full,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    return {
        "X_train": x_train.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "X_val": x_val.astype(np.float32),
        "y_val": y_val.astype(np.float32),
        "X_test": x_test.astype(np.float32),
        "y_test": y_test.astype(np.float32),
    }


def fit_norm(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, np.ndarray]:
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0) + 1e-8
    y_mean = y_train.mean()
    y_std = y_train.std() + 1e-8
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std.astype(np.float32),
        "y_mean": np.float32(y_mean),
        "y_std": np.float32(y_std),
    }


def apply_norm(X: np.ndarray, y: np.ndarray, norm: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    Xn = (X - norm["x_mean"]) / norm["x_std"]
    yn = (y - norm["y_mean"]) / norm["y_std"]
    return Xn.astype(np.float32), yn.astype(np.float32)
