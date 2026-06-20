"""Phase 4 §3 — Cell(격자) 스케일링 + R↔K 동반 규칙.

네 천장 확증이 모두 R=5 위에서 돌았다. 다섯 번째 축 = 격자 부피(cell 수). R 을 키워 입력→출력
**계산 경로 다양성**을 늘리고(H-CELL), K 를 동반 확장(H6 계승 — cell 단독은 이득 아님)한다.

- ``CellScaler(R, K_per_cell)``: N=1+3R(R+1), K=round(K_per_cell·N), n_steps∝R(펄스 도달 시간).
- Phase 3(차원=시간 무비용)와 대비: **cell 확장은 시간 비용이 있다**(cell·롤아웃 동반↑) — 정직 측정.
- ``path_diversity``: 입력→출력 도착 파형의 다양성 proxy(유효 응답 빈 수). R↑로 증가하는지 검증(C1 전제).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from lnn import dynamics
from lnn.cluster import DEFAULT_HP, _make_area
from lnn.geometry import build_geometry
from lnn.readout import matched_filter, wavelet

K_PER_CELL = 0.09   # R5 N=91 → K≈8 (Block I 기본과 정합)


def n_steps_for(R: int) -> int:
    """R↑면 입력→출력 거리↑ → 롤아웃 비례 증가(펄스 도달 + 여유)."""
    return 30 + 8 * R          # R5→70, R8→94, R12→126


@dataclass
class CellScaler:
    R: int
    K_per_cell: float = K_PER_CELL

    @property
    def n_cells(self):
        return 1 + 3 * self.R * (self.R + 1)

    @property
    def K(self):
        return max(4, round(self.K_per_cell * self.n_cells))

    @property
    def n_steps(self):
        return n_steps_for(self.R)

    def hp(self, base=None):
        h = dict(base or DEFAULT_HP)
        h["n_steps"] = self.n_steps
        h["n_hills_terrain"] = self.K
        h["n_hills_gain"] = max(4, self.K // 2)
        return h


def _count_shortest_paths(geo, src, dst):
    """격자 그래프에서 src→dst 최단경로 수(정확, BFS DP). 구별 가능한 기하적 우회로 수."""
    from collections import deque
    nbr = np.asarray(geo.nbr)
    dist = {src: 0}
    order = []
    q = deque([src])
    while q:
        u = q.popleft()
        order.append(u)
        for d in range(6):
            v = int(nbr[u, d])
            if v >= 0 and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    cnt = {src: 1}
    for u in order:
        for d in range(6):
            v = int(nbr[u, d])
            if v >= 0 and dist.get(v, -1) == dist[u] + 1:
                cnt[v] = cnt.get(v, 0) + cnt.get(u, 0)
    return dist.get(dst, float("inf")), cnt.get(dst, 0)


def path_diversity(R, seed=0, k_per_cell=K_PER_CELL, n_gen=4, n_out=6):
    """입력→출력 **구별 가능한 우회로 수**(log10 평균). H-CELL 전제: 격자 부피↑ → 경로 수↑.

    gen×out 단자쌍의 최단경로 수(조합적으로 증가)를 세어 log10 평균. 격자가 크고 단자가
    멀수록 우회로가 조합적으로 늘어 단조 증가 — "정합필터가 도착 시각으로 구별할 특징 수"의
    기하 상한 proxy(진폭 무관·순수 기하).
    """
    from lnn.area import pick_cells
    geo = build_geometry(R)
    gen = pick_cells(geo, n_gen, phase=0.0)
    out = pick_cells(geo, n_out, phase=2.6)
    logs = []
    for s in gen:
        for t in out:
            if s == t:
                continue
            _d, n = _count_shortest_paths(geo, int(s), int(t))
            logs.append(np.log10(max(n, 1)))
    return float(np.mean(logs)) if logs else 0.0


def free_evolution_stability(R, seed=0, k_per_cell=K_PER_CELL, gain=1.0):
    """임펄스 후 자유전개 버퍼 노름 성장비(<1=유계/안정). 큰 격자 발산 점검(ρ proxy)."""
    from lnn.area import pick_cells
    geo = build_geometry(R)
    cs = CellScaler(R, k_per_cell)
    gen = list(pick_cells(geo, 1))
    area = _make_area(geo, jax.random.PRNGKey(seed), "processor", gen, pick_cells(geo, 3), cs.hp())
    tau = jax.lax.stop_gradient(area.edge_tau(geo))
    L = dynamics.buffer_length(area.tau_max)
    sc = dynamics.make_step_constants(geo, tau, jnp.full((geo.N,), gain), L)
    P = area.P
    w = wavelet(P)
    buf = jnp.zeros((geo.E, L))
    norms = []
    for t in range(cs.n_steps):
        inj = jnp.zeros((geo.N,))
        if t < P:
            for c in gen:
                inj = inj.at[c].add(float(w[t]))
        buf, _o = dynamics.step(buf, inj, sc)
        norms.append(float(jnp.linalg.norm(buf)))
    norms = np.asarray(norms)
    return float(norms[-1] / (norms[P] + 1e-12))
