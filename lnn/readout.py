"""§3.6 Readout — 정합필터 + 적분 윈도 + logsumexp 점수.

    c(t)   = Σ_{k=0}^{P-1} o(t+k)·wavelet(k)              # 정합필터(웨이블릿 에너지 정규화)
    window = Dijkstra t_min (변지연 그래프) ± 여유          # stop-gradient (※1)
    score  = (1/λ)·logsumexp_{t∈win}( λ·ĉ(t)² )           # λ=8, max-subtraction (※2)
    coherence_ratio = peak² / mean_background              # < 4 이면 무판정(abstain)

무조건 준수(§12):
- ※1 Dijkstra t_min 은 미분 불가 → ``stop_gradient`` + 윈도 폭을 넉넉히. (여기서는
  윈도를 t_min−2P 부터 끝까지로 잡아 학습 중 도착 시각 이동을 흡수.)
- ※2 logsumexp 는 ``jax.scipy.special.logsumexp`` 로 max-subtraction 안정화. λ·ĉ² 를
  지수에 넣으므로 ĉ 가 1을 조금만 넘어도 float 오버플로(inf→NaN) → 학습이 조용히 죽는다.
"""

from __future__ import annotations

import heapq

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp

DEFAULT_P = 8
DEFAULT_LAMBDA = 8.0


def wavelet(P: int = DEFAULT_P):
    """펄스 파형 wavelet(k) = sin(2π k / P), 1사이클, 평균 0 (§3.5)."""
    k = jnp.arange(P)
    return jnp.sin(2.0 * jnp.pi * k / P)


def matched_filter(o_TS, w):
    """o:[T, S] (출력 셀 진폭 시계열) → c:[valid, S] 정합필터 응답.

    c(t) = Σ_k o(t+k)·w(k) / Σ_k w(k)²  (웨이블릿 에너지 정규화 → 진폭 A 매칭시 c≈A).
    """
    T = o_TS.shape[0]
    P = w.shape[0]
    valid = T - P + 1
    idx = jnp.arange(valid)[:, None] + jnp.arange(P)[None, :]  # [valid, P]
    o_win = o_TS[idx]                                          # [valid, P, S]
    norm = jnp.sum(w * w) + 1e-12
    c = jnp.sum(o_win * w[None, :, None], axis=1) / norm       # [valid, S]
    return c


def window_mask(valid: int, t_min: float, P: int = DEFAULT_P, pre: int = 2):
    """적분 윈도 마스크 [valid] (bool). t_min−pre·P 부터 끝까지(넉넉히, ※1).

    t_min 은 Dijkstra 결과(미분 불가) — 호출부에서 stop_gradient 한 호스트 정수로 넘긴다.
    """
    start = max(0, int(np.floor(t_min)) - pre * P)
    m = np.zeros(valid, dtype=bool)
    m[start:] = True
    if not m.any():
        m[:] = True
    return jnp.asarray(m)


def score_from_c(c_TS, mask, lam: float = DEFAULT_LAMBDA):
    """채널별 점수 score:[S] = (1/λ)·logsumexp_{t∈win}(λ·ĉ²)  (max-subtraction 안정화)."""
    neg = jnp.where(mask[:, None], 0.0, -jnp.inf)  # 윈도 밖은 −inf 로 배제
    return logsumexp(lam * c_TS**2 + neg, axis=0) / lam  # [S]


def peak_and_sign(c_TS, mask):
    """채널별 (|c| 피크값, 피크에서의 부호). 이진·어휘 디코딩의 극성 복원용."""
    masked = jnp.where(mask[:, None], c_TS, 0.0)
    t_star = jnp.argmax(jnp.abs(masked), axis=0)         # [S]
    peak_vals = jnp.take_along_axis(masked, t_star[None, :], axis=0)[0]  # 부호 있는 값
    return jnp.abs(peak_vals), jnp.sign(peak_vals)


def coherence_ratio(c_TS, mask):
    """coherence_ratio = peak² / mean_background (§3.6). < 4 이면 무판정 권고."""
    masked_sq = jnp.where(mask[:, None], c_TS**2, jnp.nan)
    peak2 = jnp.nanmax(masked_sq, axis=0)
    mean_bg = jnp.nanmean(masked_sq, axis=0) + 1e-9
    return peak2 / mean_bg


def signed_readout(c_TS, mask):
    """경계 재생 입력용 u_k = sign(peak_k)·|c_k(t*_k)| (§3.7). → [S]."""
    peak, sgn = peak_and_sign(c_TS, mask)
    return sgn * peak


def dijkstra_tmin(geo, tau, sources, targets):
    """변지연 그래프에서 source 집합 → 각 target 의 최소 도착 지연 t_min (호스트 계산).

    이산 그래프 알고리즘이라 미분 불가 → 결과는 readout 윈도 선택에만 쓰고
    ``stop_gradient`` 로 미분 경로에서 차단(§3.6 ※1). tau:[E] (호스트로 변환됨).
    """
    tau_np = np.asarray(tau)
    nbr = np.asarray(geo.nbr)
    N = geo.N
    dist = np.full(N, np.inf)
    pq = []
    for s in sources:
        dist[s] = 0.0
        heapq.heappush(pq, (0.0, int(s)))
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for di in range(6):
            v = int(nbr[u, di])
            if v < 0:
                continue
            w = float(tau_np[u * 6 + di])
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return np.array([dist[t] for t in targets])
