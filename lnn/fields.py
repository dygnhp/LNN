"""§3.2 지형 T·기울기 ∇T,  §3.4 이득장 G.

해석적 RBF 장. **값 성분은 지연에 쓰지 않는다 — 순수 기울기 지연**(§3.2).
지형 기울기에는 선택적 나사 전위(screw potential)를 더할 수 있다(순환/메모리용,
Block I 기본 미사용): 지형은 다가(multivalued)지만 **기울기는 단가**다.

기호 규약(§3 필수):
- 지형 RBF 폭 = ``sigma_k`` (코드에서 ``sigma_t``).
- 이득장 RBF 폭 = ``sigma_g``.
- ``rho`` 라는 심볼은 이 파일에서 쓰지 않는다(스펙트럼 반경 전용).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# softplus(GAIN_BIAS) = 1 이 되도록 하는 오프셋: a_k=0 초기화에서 이득 G≈1 (§3.4).
GAIN_BIAS = float(jnp.log(jnp.expm1(1.0)))  # = log(e - 1) ≈ 0.5413


def terrain_value(p, h, c, sigma_t):
    """T(p) = Σ_k h_k · exp(−‖p − c_k‖² / (2 σ_k²)).

    p: [..., 2], c: [K, 2], h: [K], sigma_t: [K]. → [...]
    """
    diff = p[..., None, :] - c            # [..., K, 2]
    d2 = jnp.sum(diff * diff, axis=-1)     # [..., K]
    return jnp.sum(h * jnp.exp(-d2 / (2.0 * sigma_t**2)), axis=-1)


def terrain_grad(p, h, c, sigma_t, screw_b=None, screw_c=None, screw_eps=1e-3):
    """∇T(p) — RBF 부분(단가) + (선택) 나사 전위 기울기.

    ∇T_rbf = Σ_k h_k · (−(p − c_k)/σ_k²) · exp(...)
    ∇T_screw += Σ_m b_m · (−(y − cy_m), (x − cx_m)) / max(r_m², ε)   (∮∇T·dℓ = 2π b_m)

    p: [..., 2] → [..., 2].
    """
    diff = p[..., None, :] - c                          # [..., K, 2]
    d2 = jnp.sum(diff * diff, axis=-1)                   # [..., K]
    w = h * jnp.exp(-d2 / (2.0 * sigma_t**2))            # [..., K]
    g = jnp.sum(w[..., None] * (-diff / sigma_t[..., :, None] ** 2), axis=-2)  # [..., 2]

    if screw_b is not None and screw_c is not None:
        sdiff = p[..., None, :] - screw_c                # [..., M, 2]
        r2 = jnp.sum(sdiff * sdiff, axis=-1)             # [..., M]
        denom = jnp.maximum(r2, screw_eps)
        # (−(y − cy), (x − cx)) / r²
        rot = jnp.stack([-sdiff[..., 1], sdiff[..., 0]], axis=-1)  # [..., M, 2]
        g = g + jnp.sum(screw_b[..., :, None] * rot / denom[..., :, None], axis=-2)

    return g


def gain_value(p, a, d, sigma_g):
    """G(p) = softplus(GAIN_BIAS + Σ_k a_k · exp(−‖p − d_k‖²/(2 σ_g,k²)))  ≥ 0.

    a_k=0 초기화 → G≈1 (GAIN_BIAS 오프셋, §3.4). σ_g 는 이득장 RBF 폭(지형 폭과 구별).
    p: [..., 2] → [...].
    """
    diff = p[..., None, :] - d
    d2 = jnp.sum(diff * diff, axis=-1)
    raw = jnp.sum(a * jnp.exp(-d2 / (2.0 * sigma_g**2)), axis=-1)
    return jax.nn.softplus(GAIN_BIAS + raw)
