import torch

from mapdet3d.heads.box3d_head import Box3DHead


def _is_rotation_matrix(R: torch.Tensor, atol: float = 1e-4) -> bool:
    eye = torch.eye(3, dtype=R.dtype).expand_as(R)
    orthonormal = torch.allclose(R @ R.transpose(-1, -2), eye, atol=atol)
    det = torch.linalg.det(R)
    proper = torch.allclose(det, torch.ones_like(det), atol=atol)
    return orthonormal and proper


def test_output_shapes():
    torch.manual_seed(0)
    head = Box3DHead(in_dim=32, hidden_dim=16)
    query = torch.randn(5, 32)
    rho = torch.full((5,), 2.0)

    out = head(query, rho)
    assert out["center"].shape == (5, 3)
    assert out["dims"].shape == (5, 3)
    assert out["rot"].shape == (5, 3, 3)
    assert out["conf"].shape == (5, 1)


def test_rotation_output_is_a_valid_rotation_matrix():
    torch.manual_seed(1)
    head = Box3DHead(in_dim=16, hidden_dim=16)
    query = torch.randn(8, 16)
    rho = torch.rand(8) + 0.5

    out = head(query, rho)
    assert _is_rotation_matrix(out["rot"])


def test_dimensions_are_always_positive():
    torch.manual_seed(2)
    head = Box3DHead(in_dim=16, hidden_dim=16)
    query = torch.randn(20, 16) * 5.0  # wide range of MLP outputs
    rho = torch.rand(20) + 0.1

    out = head(query, rho)
    assert (out["dims"] > 0).all()


def test_center_and_dims_scale_linearly_with_rho():
    # x = rho * x~, y = rho * y~, z = rho * exp(d~), dims = rho * exp(log-size):
    # holding the query fixed (so x~, y~, d~, log-size are fixed), doubling
    # rho must exactly double both center and dims. This guards against bugs
    # like applying rho twice or forgetting it on one attribute.
    torch.manual_seed(3)
    head = Box3DHead(in_dim=16, hidden_dim=16)
    query = torch.randn(4, 16)
    rho1 = torch.full((4,), 1.0)
    rho2 = torch.full((4,), 2.0)

    with torch.no_grad():
        out1 = head(query, rho1)
        out2 = head(query, rho2)

    assert torch.allclose(out2["center"], 2.0 * out1["center"], atol=1e-5)
    assert torch.allclose(out2["dims"], 2.0 * out1["dims"], atol=1e-5)
    # Rotation and confidence must not depend on rho at all.
    assert torch.allclose(out2["rot"], out1["rot"], atol=1e-5)
    assert torch.allclose(out2["conf"], out1["conf"], atol=1e-5)


def test_gradients_flow_to_all_submodules():
    head = Box3DHead(in_dim=16, hidden_dim=16)
    query = torch.randn(3, 16, requires_grad=True)
    rho = torch.rand(3) + 0.5

    out = head(query, rho)
    loss = out["center"].sum() + out["dims"].sum() + out["rot"].sum() + out["conf"].sum()
    loss.backward()

    assert query.grad is not None and torch.isfinite(query.grad).all()
    for name, p in head.named_parameters():
        assert p.grad is not None, f"no gradient reached {name}"
