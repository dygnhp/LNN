"""Phase 3 §4 — 차원 분할이 D 를 Area별로 올바로 쪼개고 각 Area 채널이 직교인지."""

import jax
import jax.numpy as jnp

from lnn.encodings import channel_orthogonality
from lnn_block2.dim_scaling import DimSplitEncoder, n_areas_for
from lnn_block2.freq_encoding import ofdm_basis


def test_n_areas_rule():
    assert n_areas_for(8, 8) == 1
    assert n_areas_for(16, 8) == 2
    assert n_areas_for(32, 8) == 4
    assert n_areas_for(8, 4) == 2


def test_dim_split_partitions_D():
    for D in (8, 16, 32):
        M = n_areas_for(D, 8)
        fpa = D // M
        emb = jnp.zeros((20, D))
        enc = DimSplitEncoder(embedding=emb, n_areas=M, freq_per_area=fpa,
                              carriers=tuple((m,) for m in range(M)),
                              window=20, stride=20, P=8, n_steps=90, n_cells=91)
        injs = enc.encode(jnp.zeros((2, 4), jnp.int32))
        assert len(injs) == M                       # Area마다 하나
        assert M * fpa == D                          # 채널 합 = D


def test_per_area_channels_orthogonal():
    # Area당 채널(freq_per_area)은 윈도에서 직교(<0.1) — D 키워도 Area 내부 유지.
    for fpa, W in [(8, 20), (8, 24)]:
        basis = ofdm_basis(fpa, W)
        assert channel_orthogonality(basis) < 0.1
