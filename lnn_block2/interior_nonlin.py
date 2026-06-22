"""Phase 6 실험 II — 분산 비선형 노드 (★ frozen interface 진단 분기).

LNN 근본 베팅: Area 내부 선형(LTI), 비선형은 경계 φ 에만(기둥①=경로 분해). 본 모듈은 이 베팅을
*국소적으로* 깨서(매질 내부에 MLP식 비선형 노드 분산) "선형 내부가 천장의 binding constraint인가"를
직접 묻는다. **density="none" 이면 코어 dynamics.step 과 비트 동일**(회귀 보장) — 신규 파일·토글만,
ARIS 코어 파일 불변.

펄스 물리 주의(§1.2.3, 무조건 준수): 펄스는 평균 0 wavelet. 생짜 ReLU 는 반파 정류로 ① 극성 파괴
② DC 누적(발산) ③ 고조파. 따라서 **기본 = 부호 보존** signed_relu(x)=sign(x)·relu(|x|−θ)
(평균 0·극성 보존, 경계 φ 와 같은 부류). raw ReLU 는 명시적 ablation + DC 모니터.

단자(Generator·Output) 제외(interior_only) → encode/decode frozen 시그니처 보존.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from lnn.dynamics import _BAN_EPS

THETA_DEFAULT = 0.1


def nl_apply(x, kind, theta=THETA_DEFAULT):
    """분산 비선형. none=항등. signed_relu/soft_threshold=부호보존 연성임계. relu=생짜. tanh."""
    if kind == "none":
        return x
    if kind in ("signed_relu", "soft_threshold"):
        return jnp.sign(x) * jax.nn.relu(jnp.abs(x) - theta)
    if kind == "relu":
        return jax.nn.relu(x)
    if kind == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"unknown kind: {kind}")


def build_nl_mask(geo, exclude_cells, density, sparse_k=3):
    """비선형 적용 셀 마스크 [N] bool. 단자 제외. density: none/sparse/dense."""
    mask = np.zeros(geo.N, dtype=bool)
    if density == "none":
        return jnp.asarray(mask)
    excl = set(int(c) for c in exclude_cells)
    for i in range(geo.N):
        if i in excl:
            continue
        if density == "dense" or (density == "sparse" and i % sparse_k == 0):
            mask[i] = True
    return jnp.asarray(mask)


def interior_step(buf, inject_t, sc, kind, theta, placement, nl_mask):
    """dynamics.step 복제 + 분산 비선형(cell_sum/edge_emit). kind="none" → 코어와 동일.

    cell_sum: 합산 직후 `a` 에 NL(마스크 셀) → 분배·readout 에 반영(§3.5 단계 3↔4).
    edge_emit: 방출 진폭 `per` 에 NL(마스크 셀) → 버퍼 삽입 직전.
    """
    arr_edge = buf[:, 0]
    arr_in = arr_edge[sc.e_in] * sc.nbr_exists
    a = arr_in.sum(axis=1) + inject_t                          # [N]

    if placement == "cell_sum":
        a = jnp.where(nl_mask, nl_apply(a, kind, theta), a)    # 합산 직후 비선형

    banned = jax.lax.stop_gradient(jnp.abs(arr_in) > _BAN_EPS) & sc.nbr_exists
    allowed = sc.nbr_exists & (~banned)
    n = allowed.sum(axis=1)
    per = jnp.where(n > 0, a / jnp.maximum(n, 1), 0.0)         # [N]
    if placement == "edge_emit":
        per = jnp.where(nl_mask, nl_apply(per, kind, theta), per)
    emit_cell = sc.g_cell * per
    emit = jnp.where(allowed, emit_cell[:, None], 0.0).reshape(sc.E)

    buf = jnp.concatenate([buf[:, 1:], jnp.zeros((sc.E, 1), buf.dtype)], axis=1)
    buf = buf.at[sc.arangeE, sc.i0].add(emit * sc.w0)
    buf = buf.at[sc.arangeE, sc.i1].add(emit * sc.w1)
    return buf, a


def interior_rollout(sc, inject_TN, out_cells, n_cells, kind, theta, placement,
                     nl_mask, use_remat=True):
    """interior_step 롤아웃. kind="none" → dynamics.rollout 과 동일 경로."""
    buf0 = jnp.zeros((sc.E, sc.L), dtype=inject_TN.dtype)

    def _step(buf, inj_t):
        return interior_step(buf, inj_t, sc, kind, theta, placement, nl_mask)

    if use_remat:
        _step = jax.checkpoint(_step)
    _, out_all = jax.lax.scan(_step, buf0, inject_TN)
    return out_all[:, out_cells]


def interior_area_forward(area, geo, inject_BTN, window, kind, theta, placement, nl_mask):
    """단일 Area 를 분산 비선형 동역학으로 forward → u:[B, n_out] (부호 있는 정합 피크)."""
    from lnn import readout
    sc = area.step_constants(geo)
    out_arr = jnp.asarray(area.out_cells)
    w = readout.wavelet(area.P)

    def single(inj_TN):
        o = interior_rollout(sc, inj_TN, out_arr, geo.N, kind, theta, placement, nl_mask)
        c_TS = readout.matched_filter(o, w)
        return readout.signed_readout(c_TS, window)

    return jax.vmap(single)(inject_BTN)


def dc_drift(area, geo, inject_TN, kind, theta, placement, nl_mask):
    """매질 총 진폭(DC) 추이 — 생짜 ReLU 의 DC 누적/발산 감시(§3.2.2 가드)."""
    sc = area.step_constants(geo)
    buf = jnp.zeros((sc.E, sc.L))
    means = []
    for t in range(inject_TN.shape[0]):
        buf, a = interior_step(buf, inject_TN[t], sc, kind, theta, placement, nl_mask)
        means.append(float(jnp.mean(a)))
    return float(np.max(np.abs(means)))
