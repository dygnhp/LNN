"""§2.1 주파수 채널 직교성 — 선형 영역에서 OFDM 채널 간 최대 교차상관 < 0.1.

Block I 경로 B 의 코드 직교성은 학습 후 0.78~0.99 로 붕괴(§11★). 경로 A 의 주파수 기저는
DFT sin 기저라 윈도 W 에서 **엄밀히 직교** → 교차상관 ≈ 0. 이것이 경로 A 의 핵심 자산.
"""

import jax.numpy as jnp

from lnn.encodings import channel_orthogonality
from lnn_block2.freq_encoding import ofdm_basis


def test_ofdm_basis_orthogonal():
    # 직교 조건: D < W/2 (Nyquist 주파수 W/2 는 정수 t 에서 0 → 제외). §2.1.
    for D, W in [(4, 16), (8, 20), (8, 24), (12, 32)]:
        basis = ofdm_basis(D, W)            # [D, W]
        assert basis.shape == (D, W)
        # 채널 간 최대 비대각 절대 코사인 유사도
        assert channel_orthogonality(basis) < 0.1


def test_ofdm_gram_near_diagonal():
    D, W = 8, 20
    basis = ofdm_basis(D, W)
    g = basis @ basis.T
    off = g - jnp.diag(jnp.diag(g))
    # 비대각 / 대각 평균 비율이 작아야(직교)
    assert float(jnp.max(jnp.abs(off))) / float(jnp.mean(jnp.diag(g))) < 0.1
