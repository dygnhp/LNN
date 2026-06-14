"""§3.5 셀 갱신 — 한 스텝 동역학 + ``lax.scan`` 롤아웃.

버퍼는 **변(directed edge) 소속**이다: ``buf[e, k]`` = 변 e의 머리(head) 셀에 k 스텝
뒤 도착할 진폭. 한 스텝:

1. 수신: 셀 i는 입사 변 ``e_in[i,d]`` 의 슬롯 0 샘플을 모은다.
2. 중첩: 입사 샘플 합산 ``a``. 신호가 들어온 방향을 ``banned`` 기록(역류 금지).
3. 분배 준비: 허용 방향 수 ``n = 6 − |banned| − 경계``.
4. 방출: 허용 변 i→k 에 ``per = G(p_i)·a/n`` 을 그 변의 τ에 **분수 삽입**(선형 보간).
   - **n=0 보호(무조건 준수, §3.5/§12)**: ``per = where(n>0, G·a/n, 0)``. JAX에서
     0/0 → NaN 이 ``jax.grad`` 를 타고 전체 그래디언트를 조용히 오염시킨다.
5. 포화는 경계 재생에서만(§3.7) — Area 내부는 선형. (디버그용 internal_saturation 플래그.)

τ 테이블·이득장은 지형 고정이라 **scan 바깥에서 1회 계산**(미분 가능, detach 금지, §3.2 ①).
분수 지연은 인접 슬롯 선형 보간(초판). # TODO: Thiran/Lagrange 올패스.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .geometry import Geometry

_BAN_EPS = 1e-8  # banned 판정 임계(구조적 라우팅 결정 → stop_gradient)


class StepConstants(NamedTuple):
    e_in: jnp.ndarray          # [N, 6] 들어오는 변 인덱스
    nbr_exists: jnp.ndarray    # [N, 6] bool
    g_cell: jnp.ndarray        # [N] 셀별 이득 G(p_i)
    i0: jnp.ndarray            # [E] deposit 하한 슬롯 (정수, 미분 불가 — floor)
    i1: jnp.ndarray            # [E] = i0+1
    w0: jnp.ndarray            # [E] = 1−frac (미분 가능)
    w1: jnp.ndarray            # [E] = frac   (미분 가능)
    arangeE: jnp.ndarray       # [E]
    L: int                     # 버퍼 길이
    E: int


def buffer_length(tau_max: float) -> int:
    """변 버퍼 길이: 최대 지연을 담을 만큼 + 여유. (n_steps 와 무관 — 매 스텝 재삽입.)

    tau_max 는 정적 Python float — math.ceil 로 호스트 계산(jit 트레이싱 무관).
    """
    return int(math.ceil(float(tau_max))) + 2


def make_step_constants(geo: Geometry, tau, g_cell, L: int) -> StepConstants:
    """τ(변별 지연)·이득에서 분수-삽입 상수를 구성한다.

    deposit 인덱스는 **delay_idx = τ − 1** 기준: 스텝 t에 방출한 펄스는 t+τ 에 도착하고,
    shift-left 이후 다음 상태 인덱스로는 (τ−1) 슬롯에 해당하기 때문(아래 step 주석 참조).
    """
    delay_idx = tau - 1.0                       # [E]
    i0 = jnp.floor(delay_idx).astype(jnp.int32)  # 정수 슬롯(미분 불가 — frac 으로만 grad 흐름)
    frac = delay_idx - i0
    i0 = jnp.clip(i0, 0, L - 2)
    i1 = i0 + 1
    return StepConstants(
        e_in=geo.e_in,
        nbr_exists=geo.nbr_exists,
        g_cell=g_cell,
        i0=i0,
        i1=i1,
        w0=1.0 - frac,
        w1=frac,
        arangeE=jnp.arange(geo.E),
        L=L,
        E=geo.E,
    )


def step(buf, inject_t, sc: StepConstants, internal_saturation: bool = False):
    """한 스텝 갱신. buf:[E,L], inject_t:[N] → (buf', out_all:[N]).

    시간 모델: buf[e,k] = 스텝 t+k 에 변 e 머리에 도착할 양.
      - 슬롯 0 이 현재(t) 도착분.
      - shift-left 로 다음 상태(t+1) 기준으로 인덱스 1 감소.
      - 스텝 t 방출 펄스는 t+τ 도착 → 다음 상태 인덱스 (τ−1) = delay_idx 에 삽입.
    """
    arr_edge = buf[:, 0]                                 # [E] 각 변 머리 도착분
    arr_in = arr_edge[sc.e_in] * sc.nbr_exists           # [N,6] 방향별 입사(경계 마스크)
    a = arr_in.sum(axis=1) + inject_t                    # [N] 중첩 + 주입

    # banned: 신호가 들어온 방향(역류 금지). 구조적 라우팅 → stop_gradient.
    banned = jax.lax.stop_gradient(jnp.abs(arr_in) > _BAN_EPS) & sc.nbr_exists
    allowed = sc.nbr_exists & (~banned)                  # [N,6]
    n = allowed.sum(axis=1)                              # [N]

    per = jnp.where(n > 0, a / jnp.maximum(n, 1), 0.0)   # n=0 보호 (무조건 준수)
    if internal_saturation:                              # 디버그 전용(기본 비활성)
        per = jnp.tanh(per)
    emit_cell = sc.g_cell * per                          # [N] G(p_i)·a/n
    emit = jnp.where(allowed, emit_cell[:, None], 0.0).reshape(sc.E)  # [E]

    # 시간 전진(shift-left) + 새 슬롯 0
    buf = jnp.concatenate([buf[:, 1:], jnp.zeros((sc.E, 1), buf.dtype)], axis=1)
    # 분수 지연 deposit (i0,i1 고정 / w0,w1 미분가능 — fractional-delay 사슬)
    buf = buf.at[sc.arangeE, sc.i0].add(emit * sc.w0)
    buf = buf.at[sc.arangeE, sc.i1].add(emit * sc.w1)

    return buf, a


def rollout(sc: StepConstants, inject_TN, out_cells, n_cells: int,
            use_remat: bool = True, internal_saturation: bool = False):
    """``lax.scan`` 롤아웃. inject_TN:[T,N] → out_TS:[T, len(out_cells)].

    BPTT 메모리 절감을 위해 step 에 ``jax.checkpoint``(remat) 적용(§6).
    """
    buf0 = jnp.zeros((sc.E, sc.L), dtype=inject_TN.dtype)

    def _step(buf, inj_t):
        return step(buf, inj_t, sc, internal_saturation)

    if use_remat:
        _step = jax.checkpoint(_step)

    _, out_all = jax.lax.scan(_step, buf0, inject_TN)    # out_all: [T, N]
    return out_all[:, out_cells]                          # [T, n_out]
