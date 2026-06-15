"""Experiment 2 계측 — 모델 구성(§6.2)·파라미터 수·시간·FLOPs.

- ``config_of(model, geo, extra)``: Area별 격자/단자/RBF 수, Cluster 위상, 인코더/디코더
  종류·D·m, 롤아웃, 그룹별 파라미터 수를 자동 수집(사람이 수기 입력하지 않음).
- ``time_call``: JIT 워밍업 1회 후 측정 → (compile_sec, run_sec) 분리(필수, §2.1).
- ``flops_xla`` / ``flops_analytic``: 두 방법(XLA cost_analysis + 해석적 카운트, §2.2).
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp

_GROUP = {
    "terrain_h": "terrain_h", "gain_a": "gain_a", "embedding": "embedding",
    "diag_gains": "diag_gain", "weight": "decoder_code", "bias": "decoder_code",
}
_FIXED = {"terrain_c", "terrain_sigma", "gain_d", "gain_sigma"}
_TRAINABLE = ("terrain_h", "gain_a", "embedding", "decoder_code", "diag_gain")


def count_params(model):
    """그룹별 파라미터 개수 + 학습 총합 + 고정(RBF 기하) 개수. (§6.2 파라미터 수)."""
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    counts = {g: 0 for g in _TRAINABLE}
    counts["fixed_rbf"] = 0

    def visit(path, leaf):
        last = next((k.name for k in reversed(path)
                     if isinstance(k, jax.tree_util.GetAttrKey)), "")
        if last in _FIXED:
            counts["fixed_rbf"] += int(leaf.size)
        else:
            g = _GROUP.get(last)
            if g:
                counts[g] += int(leaf.size)
        return leaf

    jax.tree_util.tree_map_with_path(visit, params)
    counts["total_trainable"] = sum(counts[g] for g in _TRAINABLE)
    return counts


def _areas_of(model):
    """model 튜플에서 cluster 와 (있다면) 그 areas 추출."""
    for m in model:
        if hasattr(m, "areas"):
            return m
    return None


def config_of(model, geo, encoder_type, decoder_type, codebook_shared,
              coherence_cycles=1, batch=None):
    """§6.2 모델 구성 블록(dict)을 자동 수집한다."""
    cluster = _areas_of(model)
    encoder = model[0]
    areas = []
    for a in cluster.areas:
        areas.append(dict(
            role=a.role, R=geo.R, N_cells=geo.N, N_edges=geo.E,
            n_generators=len(a.gen_cells), n_output_cells=len(a.out_cells),
            K_terrain=int(a.terrain_h.shape[0]), K_gain=int(a.gain_a.shape[0]),
        ))
    n_proc = max(0, len(cluster.areas) - 2)
    D = int(encoder.embedding.shape[1]) if hasattr(encoder, "embedding") else len(encoder.gen_cells)
    vocab = int(encoder.embedding.shape[0]) if hasattr(encoder, "embedding") else None
    # 디코더 출력 채널 m = 마지막 Area 출력 셀 수
    m = len(cluster.areas[-1].out_cells)
    params = count_params(model)
    return dict(
        areas=areas,
        cluster=dict(n_areas=len(cluster.areas), L_processors=n_proc, topology="serial"),
        encoder=dict(type=encoder_type, D=D, vocab=vocab,
                     injection="spatial" if encoder_type == "image" else "temporal",
                     coherence_cycles=coherence_cycles),
        decoder=dict(type=decoder_type, m_channels=m, codebook_shared=bool(codebook_shared)),
        rollout=dict(n_steps=int(cluster.n_steps), P=int(cluster.P), batch=batch),
        params=params,
    )


# ─────────────────────────────── 시간 ────────────────────────────────────────
def time_call(fn, *args, warmup=1, repeat=2):
    """JIT 워밍업 후 측정. (compile_sec, run_sec) — compile≈첫호출−평균실행 (§2.1)."""
    f = jax.jit(fn)
    t = time.perf_counter()
    r = f(*args)
    jax.block_until_ready(r)
    first = time.perf_counter() - t
    runs = []
    for _ in range(max(1, repeat)):
        t = time.perf_counter()
        r = f(*args)
        jax.block_until_ready(r)
        runs.append(time.perf_counter() - t)
    run_sec = sum(runs) / len(runs)
    return max(first - run_sec, 0.0), run_sec


# ─────────────────────────────── FLOPs ───────────────────────────────────────
def flops_xla(fn, *args):
    """방법 1 — XLA 컴파일러 추정. (flops|None, cost_analysis 원본 dict)."""
    try:
        compiled = jax.jit(fn).lower(*args).compile()
        ca = compiled.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0] if ca else {}
        ca = dict(ca) if ca else {}
        flops = ca.get("flops")
        return (float(flops) if flops is not None else None,
                {k: float(v) for k, v in ca.items() if isinstance(v, (int, float))})
    except Exception as e:  # pragma: no cover - 버전별 차이
        return None, {"error": f"{type(e).__name__}: {e}"}


def flops_analytic(geo, n_steps, batch, n_areas, n_out, P, k_eff=4):
    """방법 2 — 해석적 카운트(폴백·투명). 코어 비용 ∝ 변 수 × 스텝 수.

    edges(directed) = E = 6·N. per_step ≈ E×(분수보간4 + 이득곱1 + 합산1)=E×6.
    rollout = per_step×n_steps. forward = rollout×batch×n_areas.
    + readout: n_out×n_steps×P (정합필터)  + 지형평가: E×k_eff (국소지지).
    """
    E = geo.E
    per_step = E * 6
    rollout = per_step * n_steps
    core = rollout * batch * n_areas
    readout = n_out * n_steps * P * batch * n_areas
    terrain = E * k_eff * n_areas
    return int(core + readout + terrain)
