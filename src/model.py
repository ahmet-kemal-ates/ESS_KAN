from typing import List

import torch
import torch.nn as nn
from efficient_kan import KAN


def make_kan_model(
    in_dim: int = 3,
    hidden_dims: List[int] = None,
    out_dim: int = 1,
    grid_size: int = 5,
    spline_order: int = 3,
) -> nn.Module:
    if hidden_dims is None:
        hidden_dims = [32, 32]

    layers = [in_dim] + list(hidden_dims) + [out_dim]
    model = KAN(
        layers_hidden=layers,
        grid_size=grid_size,
        spline_order=spline_order,
    )
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
