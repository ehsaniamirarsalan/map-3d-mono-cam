import torch

from mapdet3d.heads.rotation import (
    _ray_alignment_rotation,
    allocentric_to_egocentric,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def _is_rotation_matrix(R: torch.Tensor, atol: float = 1e-5) -> bool:
    eye = torch.eye(3, dtype=R.dtype).expand_as(R)
    orthonormal = torch.allclose(R @ R.transpose(-1, -2), eye, atol=atol)
    det = torch.linalg.det(R)
    proper = torch.allclose(det, torch.ones_like(det), atol=atol)
    return orthonormal and proper


def test_6d_to_matrix_roundtrip_is_a_valid_rotation():
    torch.manual_seed(0)
    d6 = torch.randn(5, 6)
    R = rotation_6d_to_matrix(d6)
    assert _is_rotation_matrix(R)


def test_6d_matrix_roundtrip_recovers_same_rotation():
    torch.manual_seed(1)
    d6 = torch.randn(4, 6)
    R1 = rotation_6d_to_matrix(d6)
    d6_recovered = matrix_to_rotation_6d(R1)
    R2 = rotation_6d_to_matrix(d6_recovered)
    assert torch.allclose(R1, R2, atol=1e-5)


def test_ray_alignment_identity_when_ray_is_plus_z():
    ray_dir = torch.tensor([[0.0, 0.0, 1.0]])
    R = _ray_alignment_rotation(ray_dir)
    assert torch.allclose(R[0], torch.eye(3), atol=1e-5)


def test_ray_alignment_maps_plus_z_onto_ray_dir():
    torch.manual_seed(2)
    ray_dir = torch.nn.functional.normalize(torch.randn(6, 3), dim=-1)
    R = _ray_alignment_rotation(ray_dir)
    ez = torch.zeros(6, 3)
    ez[:, 2] = 1.0
    mapped = torch.einsum("nij,nj->ni", R, ez)
    assert torch.allclose(mapped, ray_dir, atol=1e-4)
    assert _is_rotation_matrix(R)


def test_ray_alignment_handles_antiparallel_ray():
    ray_dir = torch.tensor([[0.0, 0.0, -1.0]])
    R = _ray_alignment_rotation(ray_dir)
    assert _is_rotation_matrix(R)
    ez = torch.tensor([[0.0, 0.0, 1.0]])
    mapped = torch.einsum("nij,nj->ni", R, ez)
    assert torch.allclose(mapped, ray_dir, atol=1e-4)


def test_allocentric_to_egocentric_identity_when_center_on_optical_axis():
    # When the object lies exactly along the camera's forward axis, the
    # allocentric and egocentric rotations coincide (no ray-alignment needed).
    rot_allo = rotation_6d_to_matrix(torch.randn(3, 6))
    center = torch.tensor([[0.0, 0.0, 5.0]]).expand(3, 3)
    rot_ego = allocentric_to_egocentric(rot_allo, center)
    assert torch.allclose(rot_ego, rot_allo, atol=1e-5)


def test_allocentric_to_egocentric_output_is_valid_rotation():
    torch.manual_seed(3)
    rot_allo = rotation_6d_to_matrix(torch.randn(8, 6))
    center = torch.randn(8, 3) + torch.tensor([0.0, 0.0, 3.0])
    rot_ego = allocentric_to_egocentric(rot_allo, center)
    assert _is_rotation_matrix(rot_ego)
