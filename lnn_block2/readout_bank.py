"""Phase 7 수선 3 — 정합필터 뱅크 일반화 (병목 C: 저랭크 readout 해결).

KEI freq_readout(D 주파수 필터)을 출력 셀당 **M개 *명명된* 템플릿**으로 일반화. 추출 랭크가
(출력 셀 수)→(셀 수 × M)로 오르되, 각 템플릿이 "도착 패턴 X 검출"로 읽혀 dense superposition
없이 랭크만↑(기둥③ 채널 분리 유지). KEI E1 에서 다중 템플릿이 top1 0→0.42 올린 것의 일반화.

생성 코어 불변 — o(출력 시계열)에만 작용.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from lnn_block2.dynamic_query import query_matched_filter


def named_templates(M, P):
    """명명된 기저 [M, P]: 주파수 하모닉 sin(2π m t/P) (m=1..M). freq 뱅크의 일반형."""
    t = jnp.arange(P)
    return jnp.stack([jnp.sin(2 * jnp.pi * (m + 1) * t / P) for m in range(M)])


class MatchedFilterBank(eqx.Module):
    """출력 셀당 M 템플릿. read → 전폭 r:[n_out, M] (셀×템플릿 부호 있는 정합 피크)."""

    templates: jax.Array          # [M, P] (학습 가능; 명명 초기화)
    P: int = eqx.field(static=True)
    M: int = eqx.field(static=True)

    def read(self, o_TS, mask):
        """o:[T, n_out] → r:[n_out, M]. 각 템플릿으로 정합 후 창 내 부호 있는 피크."""
        def per_template(q):
            c = query_matched_filter(o_TS, q)               # [valid, n_out]
            masked = jnp.where(mask[:, None], c, 0.0)
            t_star = jnp.argmax(jnp.abs(masked), axis=0)
            return jnp.take_along_axis(masked, t_star[None, :], axis=0)[0]   # [n_out]
        r = jax.vmap(per_template)(self.templates)          # [M, n_out]
        return r.T                                          # [n_out, M]


def make_bank(M, P, key, learnable_scale=0.0):
    """명명 템플릿 뱅크. learnable_scale>0 이면 명명 기저 + 작은 학습 섭동."""
    base = named_templates(M, P)
    if learnable_scale > 0:
        base = base + learnable_scale * jax.random.normal(key, base.shape)
    return MatchedFilterBank(templates=base, P=P, M=M)
