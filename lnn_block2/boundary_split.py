"""Phase 7 수선 2 — 경계 폭 분리 (병목 B: 좁은 u 압축 해결).

현 경계: 매질→u(저차원 압축)→φ→재방사. u 가 forward 유일 통로라 정보 병목(깊이↑면 압축 누적).
새 경계: **φ 를 전폭 readout r 에 적용**(폭 보존 재방사), **u = r 의 저차원 사영(옆가지)** — 해석·프로빙
자산은 유지하되 forward 폭은 안 깎음. 깊이가 안 통한 이유(Phase 5 압축 누적)를 제거.

정직한 대가: "출력 = u 경로 기여 합"이 더는 완전치 않음 → **u_outside_ratio**(능력 중 u 밖 비율)로
기둥②(폭) 교환비를 정량(Phase 6 path_decomp_residual 이 기둥①에 한 것의 짝).
"""

from __future__ import annotations

import jax.numpy as jnp


def full_width_regen(r):
    """φ 를 전폭 r 에 적용(부호 보존 tanh). 경계에서 폭 안 깎음."""
    return jnp.tanh(r)


def project_u(r, proj):
    """전폭 r:[..., R] → 저차원 u:[..., u_dim] 옆가지 사영. proj:[u_dim, R]."""
    return r @ proj.T


def u_outside_energy(r, proj):
    """기하적 u-밖 에너지 비율: ‖r − r의_u부분공간_사영‖² / ‖r‖² (proj 행이 정규직교 가정)."""
    u = project_u(r, proj)                      # [..., u_dim]
    rec = u @ proj                              # [..., R] (proj 정규직교면 P^T P 사영)
    num = jnp.sum((r - rec) ** 2, axis=-1)
    den = jnp.sum(r ** 2, axis=-1) + 1e-12
    return float(jnp.mean(num / den))
