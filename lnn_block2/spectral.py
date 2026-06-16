"""§5 ρ(스펙트럼 반경) 1급화 — 측정(2층) + 3기여 분해 + ρ<1 정규화.

Block I 은 루프 없는 DAG 라 ρ 가 비관심사였다. Block II 에서 다중 Area 되먹임·나사 전위
루프가 등장하는 순간 ρ 는 1급(루프가 ρ→1 이면 발산). 여기서 ρ 를 명시 측정·정규화한다.

기호 규약(§0.4): ``ρ`` 는 **스펙트럼 반경 전용**. ``clamp_τ``·``σ_g`` 와 글자 공유 금지.

2층 정의(§5.1):
- (a) 선형 영역(ε=0/IP): 한 스텝 펄스 전파 선형연산자 A 의 ``max|eig(A)|`` — 문자 그대로.
- (b) self-gating(ε>0): 단일 A 없음 → 동작점에서 **국소 야코비안** 선형화 후 ρ.
두 경우 모두 ``step`` 의 버퍼→버퍼 야코비안으로 통일 구현(선형이면 동작점 무관).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lnn import fields
from lnn.delay import clamp_tau, tobler_f
from lnn.dynamics import buffer_length, make_step_constants, step


def spectral_radius(A) -> float:
    """행렬 A 의 스펙트럼 반경 max|eigenvalue|. (검증: 대각·회전 행렬에서 정확.)"""
    eig = np.linalg.eigvals(np.asarray(A))
    return float(np.max(np.abs(eig)))


def _area_tau_gcell(area, geo, gain_override=None):
    s = jnp.sum(
        fields.terrain_grad(geo.edge_mid, area.terrain_h, area.terrain_c, area.terrain_sigma)
        * geo.edge_hat, axis=-1)
    tau = clamp_tau(tobler_f(s, area.tau_base, area.gamma, area.s_star), area.tau_min, area.tau_max)
    if gain_override is not None:
        g_cell = jnp.full((geo.N,), float(gain_override))
    else:
        g_cell = fields.gain_value(geo.pos, area.gain_a, area.gain_d, area.gain_sigma)
    return tau, g_cell


def step_jacobian(area, geo, buf_state=None, gain_override=None):
    """한 스텝 ``step`` 의 버퍼→버퍼 야코비안 A:[E·L, E·L] (동작점 buf_state).

    buf_state=None 이면 0(선형 영역 동작점). gain_override 로 이득=상수 강제(3기여 분해용).
    """
    tau, g_cell = _area_tau_gcell(area, geo, gain_override)
    L = buffer_length(area.tau_max)
    sc = make_step_constants(geo, tau, g_cell, L)
    if buf_state is None:
        buf_state = jnp.zeros((geo.E, L))
    inj0 = jnp.zeros((geo.N,))

    def f(buf_flat):
        buf = buf_flat.reshape(geo.E, L)
        buf2, _ = step(buf, inj0, sc)
        return buf2.reshape(-1)

    return jax.jacrev(f)(buf_state.reshape(-1))


def measure_rho(area, geo, buf_state=None, mode="linear"):
    """ρ 측정 + 3기여 분해(§5.2). 작은 격자(R≤4) 권장(야코비안 [E·L]² 고유값).

    - rho        : 전체 한-스텝 전파 연산자의 스펙트럼 반경.
    - contrib_T  : 이득=1 로 둔 순수 전파(transport) 연산자의 ρ.
    - contrib_G  : rho / contrib_T (이득의 곱셈 기여).
    - contrib_phase: 순환(루프) 기여 — 단일 Area(루프 없음)는 1.0; 루프 모델이 명시 설정.
    """
    A = step_jacobian(area, geo, buf_state)
    rho = spectral_radius(A)
    A_T = step_jacobian(area, geo, buf_state, gain_override=1.0)
    contrib_T = spectral_radius(A_T)
    contrib_G = rho / max(contrib_T, 1e-9)
    return dict(rho=rho, contrib_T=contrib_T, contrib_G=contrib_G,
                contrib_phase=1.0, method=mode)


def rho_regularization_loss(rho, rho_target=0.95):
    """ρ<1 정규화 항(§5.3): max(0, ρ − ρ_target)². 손실에 가산해 일양 ρ<1 강제."""
    return jnp.maximum(0.0, rho - rho_target) ** 2


def project_rho(A, rho_target=0.95):
    """사영 방식(§5.3): A 의 ρ 가 target 초과 시 전체를 스케일다운해 ρ=target 으로 클립."""
    r = spectral_radius(A)
    if r <= rho_target:
        return A
    return A * (rho_target / r)


def rho_report_block(rho_dict, regularized=True, rho_target=0.95):
    """§5.4 ρ 진단 산출물 블록."""
    return dict(value=rho_dict["rho"], contrib_T=rho_dict["contrib_T"],
                contrib_G=rho_dict["contrib_G"], contrib_phase=rho_dict["contrib_phase"],
                regularized=regularized, method=rho_dict.get("method", "linear"),
                rho_target=rho_target)
