from __future__ import annotations

import math

import torch
from torch import nn


class SpatioTemporalEncoder(nn.Module):
    """Decoupled Fourier-feature encoder for 3D space and 1D time."""

    def __init__(self, params: dict):
        super().__init__()
        spatial_size = int(params["embedding_size_spatial"])
        temporal_size = int(params["embedding_size_temporal"])
        spatial_dim = int(params.get("coordinates_size_spatial", 3))
        temporal_dim = int(params.get("coordinates_size_temporal", 1))

        spatial_scale = float(params.get("scale_spatial", 5.0))
        temporal_scale = float(params.get("scale_temporal", 2.0))

        b_spatial = self._make_matrix(
            params.get("spatial_embedding", "randn"),
            spatial_size,
            spatial_dim,
            spatial_scale,
        )
        b_temporal = self._make_matrix(
            params.get("temporal_embedding", "gauss"),
            temporal_size,
            temporal_dim,
            temporal_scale,
        )

        self.register_buffer("B_spatial", b_spatial)
        if bool(params.get("learnable_temporal", True)):
            self.B_temporal = nn.Parameter(b_temporal)
        else:
            self.register_buffer("B_temporal", b_temporal)

    @staticmethod
    def _make_matrix(kind: str, rows: int, cols: int, scale: float) -> torch.Tensor:
        if kind not in {"randn", "gauss"}:
            raise ValueError(f"Unknown Fourier embedding kind: {kind}")
        return (torch.randn(rows, cols) * scale).float()

    @staticmethod
    def _encode(coords: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
        original_shape = coords.shape[:-1]
        flat = coords.reshape(-1, coords.shape[-1]).to(dtype=basis.dtype)
        projected = (2.0 * math.pi) * (flat @ basis.T)
        encoded = torch.cat([torch.sin(projected), torch.cos(projected)], dim=-1)
        return encoded.reshape(*original_shape, encoded.shape[-1])

    def spatial_encode(self, coords_spatial: torch.Tensor) -> torch.Tensor:
        return self._encode(coords_spatial, self.B_spatial)

    def temporal_encode(self, coords_temporal: torch.Tensor) -> torch.Tensor:
        return self._encode(coords_temporal, self.B_temporal)

    def forward(
        self,
        coords_spatial: torch.Tensor,
        coords_temporal: torch.Tensor,
        concat: bool = False,
    ):
        spatial = self.spatial_encode(coords_spatial)
        temporal = self.temporal_encode(coords_temporal)
        if concat:
            return torch.cat([spatial, temporal], dim=-1)
        return spatial, temporal


class SirenLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, w0: float = 30.0, is_first: bool = False, is_last: bool = False):
        super().__init__()
        self.in_features = int(in_features)
        self.w0 = float(w0)
        self.is_first = bool(is_first)
        self.is_last = bool(is_last)
        self.linear = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / self.in_features if self.is_first else math.sqrt(6.0 / self.in_features) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        return y if self.is_last else torch.sin(self.w0 * y)


class SIREN(nn.Module):
    """SIREN MLP producing real/imaginary dynamic residual channels."""

    def __init__(self, params: dict):
        super().__init__()
        depth = int(params["network_depth"])
        width = int(params["network_width"])
        input_size = int(params["network_input_size"])
        output_size = int(params.get("network_output_size", 2))

        if depth < 2:
            raise ValueError("network_depth must be at least 2")

        layers: list[nn.Module] = [SirenLayer(input_size, width, is_first=True)]
        for _ in range(depth - 2):
            layers.append(SirenLayer(width, width))
        layers.append(SirenLayer(width, output_size, is_last=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

