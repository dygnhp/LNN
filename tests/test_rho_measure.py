"""§5.3 ρ 측정 정확도 — 스펙트럼 반경을 아는 고정 행렬에서 검증."""

import numpy as np

from lnn_block2 import spectral
from lnn_block2.spectral import spectral_radius


def test_diagonal():
    A = np.diag([0.5, 0.9, 0.3, -0.7])
    assert abs(spectral_radius(A) - 0.9) < 1e-6


def test_rotation_radius_one():
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert abs(spectral_radius(R) - 1.0) < 1e-6


def test_scaled_rotation():
    th, s = 1.1, 0.73
    R = s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert abs(spectral_radius(R) - s) < 1e-6


def test_nilpotent_zero():
    A = np.array([[0.0, 1.0], [0.0, 0.0]])  # nilpotent → ρ=0
    assert spectral_radius(A) < 1e-9


def test_rho_regularization():
    import jax.numpy as jnp
    assert float(spectral.rho_regularization_loss(jnp.array(0.9), 0.95)) == 0.0
    assert float(spectral.rho_regularization_loss(jnp.array(1.05), 0.95)) > 0.0


def test_project_rho():
    A = np.diag([1.5, 0.2])
    Ap = spectral.project_rho(A, rho_target=0.95)
    assert spectral_radius(Ap) <= 0.95 + 1e-6
