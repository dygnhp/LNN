"""§6 텔레스코핑 검증 (Exit gate ③) — 무차원 상대분산 정량 기준.

동일 끝점·상이 경로 N≥20 쌍에 대해 각 경로 총지연 Στ 측정 →
    V_rel = Var(Στ) / (Δτ_endpoint)²        (무차원 — τ_base·R 불변)

- 선형 f(s)=a+b·s : V_rel < ε_rel            (텔레스코핑 — 끝점에만 의존)
- Tobler   f       : V_rel > c·ε_rel, c ≥ 10 (경사 분산이 경로를 구분 — 복원)
- 두 조건 동시 충족 시 통과. ε_rel ~ 1e-6 (상대분산 → 스케일 불변).

주의: 판정은 "그럴듯한 그림"이 아니라 위 분산 비교 수치다.
clamp_τ 는 적용하지 않는다 — softplus 자체가 볼록이라 선형 대조군의 텔레스코핑을 가린다.
"""

import itertools

import jax.numpy as jnp
import numpy as np

from lnn import delay, fields
from lnn.geometry import DIRS, build_geometry, cell_index

EPS_REL = 1e-6


def _setup():
    geo = build_geometry(5)
    rng = np.random.default_rng(0)
    K = 8
    h = jnp.asarray(rng.uniform(-1, 1, K), jnp.float32)
    c = jnp.asarray(rng.uniform(-6, 6, (K, 2)), jnp.float32)
    sig = jnp.asarray(rng.uniform(2.5, 4.0, K), jnp.float32)
    s = (fields.terrain_grad(geo.edge_mid, h, c, sig) * geo.edge_hat).sum(-1)  # [E]
    return geo, s


def _paths():
    """(0,0) → (3,-3) 사이 길이 6 의 단조 최단경로 C(6,3)=20 개 (방향 d0,d2 의 순열)."""
    return list(set(itertools.permutations([0, 0, 0, 2, 2, 2])))


def _path_edges(geo, seq):
    q, r = 0, 0
    edges = []
    for d in seq:
        i = cell_index(geo, q, r)
        edges.append(i * 6 + d)
        dq, dr = DIRS[d]
        q, r = q + dq, r + dr
    return edges, cell_index(geo, q, r)


def _v_rel(geo, s, law, params):
    seqs = _paths()
    totals = []
    end = None
    for sq in seqs:
        edges, e = _path_edges(geo, sq)
        end = e if end is None else end
        assert e == end  # 모든 경로가 같은 끝점
        se = s[jnp.array(edges)]
        tau = delay.edge_delay_from_slope(se, params, law=law, clamp=False)
        totals.append(float(jnp.sum(tau)))
    totals = np.array(totals)
    dtau_endpoint = totals.min()  # 끝점 간 특성(최소경로) 지연 스케일
    return float(np.var(totals) / (dtau_endpoint**2)), len(seqs)


def test_linear_telescopes():
    geo, s = _setup()
    v_lin, n = _v_rel(geo, s, "linear", dict(a=1.0, b=1.0))
    assert n >= 20
    assert v_lin < EPS_REL, f"linear V_rel={v_lin:.3e} not < {EPS_REL:.0e}"


def test_tobler_restores():
    geo, s = _setup()
    v_tob, _ = _v_rel(geo, s, "tobler", dict(tau_base=1.0, gamma=1.5, s_star=-0.2))
    assert v_tob > 10 * EPS_REL, f"tobler V_rel={v_tob:.3e} not > {10 * EPS_REL:.0e}"


def test_telescoping_gap():
    """선형 붕괴 vs Tobler 복원: 비율이 10× 이상."""
    geo, s = _setup()
    v_lin, _ = _v_rel(geo, s, "linear", dict(a=1.0, b=1.0))
    v_tob, _ = _v_rel(geo, s, "tobler", dict(tau_base=1.0, gamma=1.5, s_star=-0.2))
    assert v_tob / max(v_lin, 1e-30) >= 10.0
