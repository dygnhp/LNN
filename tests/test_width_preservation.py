"""Phase 7 수선 2 — 뱅크 전폭 readout 이 경계에서 폭을 안 깎는지 + 전폭 φ shape 보존."""

import jax.numpy as jnp
import numpy as np

from lnn_block2.boundary_split import full_width_regen, project_u, u_outside_energy
from lnn_block2.kei_extract import baseline_u
from lnn_block2.readout_bank import make_bank, MatchedFilterBank


def _o_mask(T=40, S=5, P=8, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(T, S)).astype("float32")), jnp.asarray(np.ones(T - P + 1, bool))


def test_bank_width_exceeds_scalar_readout():
    o, mask = _o_mask()
    base = baseline_u(o, mask, 8)              # [S]  (셀당 스칼라 1개)
    import jax
    bank = make_bank(M=6, P=8, key=jax.random.PRNGKey(0))
    r = bank.read(o, mask)                     # [S, 6]
    assert r.shape[0] == base.shape[0]
    assert r.size == base.size * 6             # 전폭(셀×M) > 스칼라(셀) — 압축 없음


def test_full_width_phi_preserves_shape():
    o, mask = _o_mask()
    import jax
    r = make_bank(M=6, P=8, key=jax.random.PRNGKey(0)).read(o, mask)
    assert full_width_regen(r).shape == r.shape   # φ 가 폭을 안 줄임


def test_u_outside_energy_in_unit_range():
    o, mask = _o_mask()
    import jax
    r = make_bank(M=6, P=8, key=jax.random.PRNGKey(0)).read(o, mask)  # [S,6]
    R = r.shape[1]
    # u_dim=2 정규직교 사영
    proj = jnp.asarray(np.linalg.qr(np.random.default_rng(0).normal(size=(R, 2)))[0].T)  # [2,R]
    ratio = u_outside_energy(r, proj)
    assert 0.0 <= ratio <= 1.0
