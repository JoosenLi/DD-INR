from __future__ import annotations

import torch


class NufftAutograd(torch.autograd.Function):
    """Autograd bridge for MRI-NUFFT operators."""

    @staticmethod
    def forward(ctx, image: torch.Tensor, operator):
        ctx.operator = operator
        return operator.op(image)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return ctx.operator.adj_op(grad_output), None


def apply_nufft(image: torch.Tensor, operator):
    return NufftAutograd.apply(image, operator)


def build_frame_operators(trajs, shape: tuple[int, int, int], n_coils: int, csm=None, backend: str = "cufinufft", density: bool = True):
    """Build one MRI-NUFFT operator per frame for time-varying sampling."""
    try:
        from mrinufft import get_operator
    except ImportError as exc:
        raise ImportError("Full reconstruction requires mrinufft. Install with `pip install mrinufft`.") from exc

    operator_factory = get_operator(backend)
    return [
        operator_factory(samples=trajs[t], shape=list(shape), n_coils=int(n_coils), density=density, smaps=csm)
        for t in range(len(trajs))
    ]

