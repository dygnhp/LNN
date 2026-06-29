"""Phase 7 수선 1 — 동적 Query Readout (병목 A: Q 부재 해결).

현 readout 은 출력 셀이 *언제나 같은* wavelet(k) 와 상관을 잡는다(고정 질문). 동적 Query 는
읽는 파형 q(k) 를 **내용 의존**으로 만든다: c(t)=Σ o(t+k)·q(k).

- **Cross-Query(권장·선형 보존)**: q 를 *다른* Area/토큰의 u(또는 임베딩)가 정한다. A 의 forward
  동안 q 가 상수 → 내부 선형성(기둥①) 안 깨짐. **O4 의 정확한 처방**(동사 문맥 → 명사 readout).
- **Self-Query(능력↑·해석↓)**: q = q0 + ε·δq(o), 매질 자신 상태 함수. ε-전개로 1차 분해 보존.

생성 코어 불변 — 본 모듈은 ``dynamics.rollout`` 의 출력 시계열 o 에만 작용한다(추출 층).
무조건 준수: Dijkstra t_min 창 안에서만 동적 q 적용(창은 stop-grad 유지).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from lnn.readout import wavelet


def query_matched_filter(o_TS, q):
    """o:[T, S], 질문 파형 q:[P] → c:[valid, S]. 고정 wavelet 대신 q 로 상관(에너지 정규화)."""
    T = o_TS.shape[0]
    P = q.shape[0]
    valid = T - P + 1
    idx = jnp.arange(valid)[:, None] + jnp.arange(P)[None, :]
    o_win = o_TS[idx]                                   # [valid, P, S]
    norm = jnp.sum(q * q) + 1e-12
    return jnp.sum(o_win * q[None, :, None], axis=1) / norm   # [valid, S]


def signed_query_readout(o_TS, q, mask):
    """동적 질문 q 로 읽은 부호 있는 정합 피크 u:[S]."""
    c = query_matched_filter(o_TS, q)
    masked = jnp.where(mask[:, None], c, 0.0)
    t_star = jnp.argmax(jnp.abs(masked), axis=0)
    peak = jnp.take_along_axis(masked, t_star[None, :], axis=0)[0]
    return peak


class CrossQuery(eqx.Module):
    """문맥 벡터 ctx(예: 동사 임베딩/다른 Area u) → 질문 파형 q. forward 동안 상수(기둥① 보존)."""

    W: jax.Array            # [P, ctx_dim]  ctx→q 선형 사상(학습)
    P: int = eqx.field(static=True)

    def query(self, ctx):
        """ctx:[..., ctx_dim] → q:[..., P]. (배치면 예시별 질문.)"""
        return ctx @ self.W.T            # [..., P]

    def base_query(self):
        return wavelet(self.P)


def make_cross_query(P, ctx_dim, key, scale=0.3):
    # wavelet 근처에서 출발하도록 작은 난수 + (열 0 에 wavelet 바이어스는 생략, 학습으로 형성)
    W = scale * jax.random.normal(key, (P, ctx_dim))
    return CrossQuery(W=W, P=P)


def self_query(o_TS, q0, epsilon):
    """Self-Query(ε-전개): q = q0 + ε·δq(o). δq = 출력 평균 파형 형태(내용 의존). ε=0 → static."""
    P = q0.shape[0]
    T = o_TS.shape[0]
    if epsilon == 0.0:
        return q0
    # 매질 출력의 평균 P-윈도 패턴을 δq 로(자기 상태 의존)
    seg = o_TS[:P].mean(axis=1) if T >= P else jnp.zeros(P)
    dq = seg / (jnp.linalg.norm(seg) + 1e-9)
    return q0 + epsilon * dq
