"""§3.3 변 지연 — Tobler형 비선형 법칙 (텔레스코핑 회피).

변 e=i→j, 중점 m, 단위방향 ê 에 대해:

    s      = ∇T(m) · ê                                  # 부호 있는 중점 경사
    f(s)   = τ_base · exp( γ·(|s − s*| − |s*|) )        # 볼록·비대칭, 최솟값 s*<0
    τ      = clamp_τ( f(s) )

**선형 법칙(f=a+b·s) 금지 이유** — 경로 총지연이
    Σ s ≈ ∫∇T·dℓ = T(끝) − T(시작)
로 텔레스코핑되어 끝점에만 의존 → 지형이 경로 간 시차를 못 만들고 학습 붕괴.
볼록 f는 Jensen 항으로 경사 분산이 경로를 구분하게 해 이를 회피한다.
(정량 검증: ``tests/test_telescoping.py``, §6.)

기호 규약(§3): 변 지연 하한은 ``clamp_tau``(구 ρ(u)). ``rho`` 심볼 사용 금지.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def clamp_tau(u, tau_min, tau_max, k=4.0):
    """clamp_τ(u) = τ_min + softplus(k·(u − τ_min))/k, 상한 τ_max로 클립 (§3.3).

    인과율 하한 τ_min (지연은 한 스텝 이상)을 매끄럽게 강제하고 상한으로 클립.
    """
    v = tau_min + jax.nn.softplus(k * (u - tau_min)) / k
    return jnp.minimum(v, tau_max)


def tobler_f(s, tau_base, gamma, s_star):
    """f(s) = τ_base · exp(γ·(|s − s*| − |s*|)). 볼록·비대칭, 최솟값 s* (<0)."""
    return tau_base * jnp.exp(gamma * (jnp.abs(s - s_star) - jnp.abs(s_star)))


def linear_f(s, a, b):
    """f(s) = a + b·s. **모델 동역학 금지** — 텔레스코핑 대조군(테스트) 전용."""
    return a + b * s


def edge_delay_from_slope(s, params, law="tobler", clamp=True):
    """변 중점 경사 s → 지연 τ.

    law="tobler": Tobler형(모델 기본). law="linear": 텔레스코핑 대조군.
    clamp=True 이면 clamp_τ 적용(모델). 텔레스코핑 테스트에서는 clamp=False —
    softplus 자체가 볼록이라 선형 대조군의 텔레스코핑을 가려버리기 때문(§6 주석).
    """
    if law == "tobler":
        f = tobler_f(s, params["tau_base"], params["gamma"], params["s_star"])
    elif law == "linear":
        f = linear_f(s, params["a"], params["b"])
    else:
        raise ValueError(f"unknown law: {law}")
    if clamp:
        f = clamp_tau(f, params["tau_min"], params["tau_max"])
    return f
