"""Phase 7 — 수선 모두 off 면 KEI Phase5/6 추출과 비트 동일(생성 코어 불변 보증)."""

import jax.numpy as jnp
import numpy as np

from lnn.readout import wavelet
from lnn_block2.kei_extract import baseline_u, static_query_u, bank_r
from lnn_block2.readout_bank import make_bank


def _o_mask(seed=0, T=40, S=5, P=8):
    rng = np.random.default_rng(seed)
    o = jnp.asarray(rng.normal(size=(T, S)).astype("float32"))
    mask = jnp.asarray(np.ones(T - P + 1, dtype=bool))
    return o, mask


def test_static_query_equals_baseline():
    o, mask = _o_mask()
    assert jnp.allclose(static_query_u(o, mask, 8), baseline_u(o, mask, 8), atol=1e-6)


def test_bank_M1_equals_baseline():
    # 명명 템플릿 m=0 = sin(2π·1·t/P) = wavelet → 뱅크 M=1 첫 채널 = baseline.
    o, mask = _o_mask()
    bank = make_bank(M=1, P=8, key=None if False else __import__("jax").random.PRNGKey(0))
    r = bank_r(o, mask, bank)                 # [S, 1]
    assert jnp.allclose(r[:, 0], baseline_u(o, mask, 8), atol=1e-5)
