"""Block II 학습 보조 — ε 스케줄(§4.1) + ρ<1 정규화(§5.3, 주기적 사영).

ρ 페널티를 매 스텝 미분해 넣으려면 [E·L]² 야코비안 고유값을 매번 계산해야 해 비현실적.
대신 사양이 허용한 **사영(projection)** 방식을 주기적으로 적용: 몇 epoch 마다 ρ 를 측정해
target 초과면 이득을 스케일다운(일양 ρ<1 강제). self-gating ε 은 0부터 점진 증가.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from .spectral import measure_rho


def epsilon_schedule(eps_max, warmup_frac=0.5):
    """ε 어닐링: 0 → eps_max (전체의 warmup_frac 까지 선형 증가 후 유지). §4.1."""
    def sched(ep, epochs):
        if epochs <= 1 or eps_max == 0.0:
            return eps_max
        frac = ep / max(1, int(epochs * warmup_frac))
        return float(min(1.0, frac) * eps_max)
    return sched


def project_area_gain(area, geo, rho_target=0.95):
    """ρ<1 사영: 측정 ρ>target 이면 gain_a 를 줄여(이득 압축) ρ 를 낮춘다. (area, rho_before, rho_after)."""
    rd = measure_rho(area, geo)
    if rd["rho"] <= rho_target:
        return area, rd["rho"], rd["rho"]
    # 이득 기여 contrib_G 가 주범이면 gain_a 를 비율만큼 축소(보수적 0.7x 반복은 호출부).
    shrink = max(0.3, rho_target / rd["rho"])
    area2 = eqx.tree_at(lambda a: a.gain_a, area, area.gain_a * shrink)
    rd2 = measure_rho(area2, geo)
    return area2, rd["rho"], rd2["rho"]
