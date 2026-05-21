import torch

from dd_inr.data import FrameNufftDataset, create_spatiotemporal_grid_3dt, make_frame_loader
from dd_inr.losses import l2_grad3d, ramp_factor, tv3d
from dd_inr.models import SIREN, SpatioTemporalEncoder


def test_encoder_and_siren_shapes():
    enc = SpatioTemporalEncoder(
        {
            "embedding_size_spatial": 4,
            "embedding_size_temporal": 2,
            "coordinates_size_spatial": 3,
            "coordinates_size_temporal": 1,
            "scale_spatial": 1.0,
            "scale_temporal": 1.0,
            "learnable_temporal": True,
        }
    )
    xyz, t = create_spatiotemporal_grid_3dt(3, 4, 5, 2)
    s = enc.spatial_encode(xyz)
    tt = enc.temporal_encode(t)
    assert s.shape == (4, 5, 2, 8)
    assert tt.shape == (3, 4)

    model = SIREN({"network_depth": 3, "network_width": 16, "network_input_size": 12, "network_output_size": 2})
    out = model(torch.randn(2, 4, 5, 2, 12))
    assert out.shape == (2, 4, 5, 2, 2)


def test_losses_are_finite():
    x = torch.randn(2, 4, 5, 6)
    assert torch.isfinite(tv3d(x))
    assert torch.isfinite(l2_grad3d(x))
    assert ramp_factor(0, 10, 5) == 0.0
    assert ramp_factor(20, 10, 5) == 1.0


def test_frame_loader_keeps_operator_objects():
    ops = [object(), object(), object()]
    dataset = FrameNufftDataset(torch.randn(3, 1, 8, dtype=torch.complex64), ops, torch.linspace(0, 1, 3))
    batch = next(iter(make_frame_loader(dataset, batch_size=2, pin_memory=False)))
    assert batch[0].shape[0] == 2
    assert isinstance(batch[1], list)
    assert batch[2].ndim == 1

