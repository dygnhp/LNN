"""§3.6 Readout 검증 — 정합필터·logsumexp 안정성·부호 복원."""

import jax.numpy as jnp
import numpy as np

from lnn import readout


def test_matched_filter_recovers_amplitude():
    P = 8
    w = readout.wavelet(P)
    A = 1.7
    T = 30
    o = np.zeros((T, 1), np.float32)
    o[10:10 + P, 0] = A * np.asarray(w)        # 진폭 A 의 웨이블릿을 t=10 에 심음
    c = readout.matched_filter(jnp.asarray(o), w)
    assert np.isclose(float(jnp.max(c)), A, atol=1e-3)  # 에너지 정규화 → 피크 ≈ A


def test_matched_filter_sign():
    P = 8
    w = readout.wavelet(P)
    T = 30
    o = np.zeros((T, 1), np.float32)
    o[5:5 + P, 0] = -np.asarray(w)             # 음극성
    c = readout.matched_filter(jnp.asarray(o), w)
    mask = jnp.ones(c.shape[0], bool)
    _, sgn = readout.peak_and_sign(c, mask)
    assert float(sgn[0]) < 0


def test_logsumexp_no_overflow():
    """큰 ĉ 에서도 max-subtraction 으로 NaN/inf 가 나지 않아야 한다(§3.6 ※2)."""
    c = jnp.asarray(np.full((20, 3), 5.0, np.float32))  # ĉ²=25, λ·ĉ²=200 → 미안정시 overflow
    mask = jnp.ones(20, bool)
    score = readout.score_from_c(c, mask, lam=8.0)
    assert bool(jnp.all(jnp.isfinite(score)))


def test_window_mask():
    m = readout.window_mask(40, t_min=20.0, P=8, pre=2)
    m = np.asarray(m)
    assert m[39] and not m[0]
    assert m.sum() > 0


def test_coherence_ratio():
    P = 8
    w = readout.wavelet(P)
    T = 40
    o = np.zeros((T, 1), np.float32)
    o[15:15 + P, 0] = np.asarray(w)
    c = readout.matched_filter(jnp.asarray(o), w)
    mask = jnp.ones(c.shape[0], bool)
    coh = readout.coherence_ratio(c, mask)
    assert float(coh[0]) > 4.0   # 또렷한 단일 피크 → 높은 일관성비
