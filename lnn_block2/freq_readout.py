"""§2.2 경로 A 출구 — 주파수 정합필터 뱅크 readout.

Block I readout 은 출력 Cell 당 정합필터 1개라 주파수 성분이 섞여 복원된다(Exp2 H4 의 원인).
뱅크는 출력 Cell 당 **D개 정합필터**(주파수 채널별 1개)를 두어 채널을 분리 복원한다 —
인코더의 ``ofdm_basis`` 와 동일한 기저를 쓴다.

무조건 준수 계승(§2.2): Dijkstra t_min stop-gradient + logsumexp max-subtraction.
t_min 은 변지연 τ 가 주파수 무관이라 채널 공유(물리적으로 동일) — 보수적으로 채널별
독립 적용해도 같은 창이 나오므로 공유 창을 쓴다(주석).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import logsumexp

from lnn.readout import DEFAULT_LAMBDA, dijkstra_tmin, window_mask  # Block I 재사용


def bank_matched_filter(o_TS, basis):
    """o:[T, S] (출력 셀 시계열), basis:[D, W] → c:[valid, S, D] 채널별 정합필터 응답."""
    T = o_TS.shape[0]
    D, W = basis.shape
    valid = T - W + 1
    idx = jnp.arange(valid)[:, None] + jnp.arange(W)[None, :]   # [valid, W]
    o_win = o_TS[idx]                                           # [valid, W, S]
    norm = jnp.sum(basis * basis, axis=1) + 1e-12              # [D]
    c = jnp.einsum("vws,dw->vsd", o_win, basis) / norm[None, None, :]
    return c                                                    # [valid, S, D]


def signed_bank_readout(o_TS, basis, mask):
    """채널 분리 부호 있는 정합 피크 u:[S, D] (출력 셀 × 주파수 채널)."""
    c = bank_matched_filter(o_TS, basis)                        # [valid, S, D]
    masked = jnp.where(mask[:, None, None], c, 0.0)
    t_star = jnp.argmax(jnp.abs(masked), axis=0)                # [S, D]
    peak = jnp.take_along_axis(masked, t_star[None], axis=0)[0]  # [S, D] 부호 있음
    return peak


def bank_scores(o_TS, basis, mask, lam=DEFAULT_LAMBDA):
    """채널별 logsumexp 점수 [S, D] (max-subtraction 안정화)."""
    c = bank_matched_filter(o_TS, basis)                        # [valid, S, D]
    neg = jnp.where(mask[:, None, None], 0.0, -jnp.inf)
    return logsumexp(lam * c**2 + neg, axis=0) / lam            # [S, D]


def bank_window(geo, tau, sources, targets, window, P, pre=2):
    """뱅크 적분 창 마스크. valid = n_steps − W + 1. t_min 은 채널 공유(τ 주파수 무관)."""
    tmins = dijkstra_tmin(geo, tau, list(sources), list(targets))
    valid = window  # placeholder; 실제 valid 는 호출부 n_steps-W+1 로 마스크 생성
    return tmins, valid


def make_bank_mask(n_steps, W, t_min, P, pre=2):
    """[valid=n_steps-W+1] bool 창 마스크 (Block I window_mask 재사용, 커널 길이 W 반영)."""
    valid = n_steps - W + 1
    return window_mask(valid, float(t_min), P, pre)
