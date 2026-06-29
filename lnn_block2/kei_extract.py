"""Phase 7 — 추출 층 조립 (생성 코어 불변 + Dynamic Extraction Layer).

생성 코어(geometry~dynamics·fields·area 매질부·encode)는 import 만(불변). 추출 층(수선 1·2·3)을
config 플래그로 on/off → 사다리 스윕. **수선 모두 off = KEI Phase5/6 비트 동일**(회귀 보장):
static query(=wavelet) + bank M=1(=wavelet) + width_split off → 기존 signed_readout 와 동일.

생성 코어는 dynamics.rollout 으로 출력 시계열 o 를 얻는 데까지만 쓰고, o→결정의 *추출*만 재설계.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax.numpy as jnp

from lnn.readout import signed_readout, wavelet
from lnn_block2.dynamic_query import query_matched_filter, signed_query_readout
from lnn_block2.readout_bank import MatchedFilterBank


@dataclass
class ExtractConfig:
    query: str = "static"        # static | cross | self
    use_bank: bool = False
    M: int = 1
    width_split: bool = False
    epsilon: float = 0.0         # self-query 차수


def baseline_u(o_TS, mask, P):
    """수선 모두 off 기준선: 기존 KEI readout (정합필터 wavelet → 부호 피크)."""
    from lnn.readout import matched_filter
    c = matched_filter(o_TS, wavelet(P))
    return signed_readout(c, mask)


def static_query_u(o_TS, mask, P):
    """static 동적-Query(=wavelet) — baseline 과 비트 동일해야 함(회귀 검증)."""
    return signed_query_readout(o_TS, wavelet(P), mask)


def cross_query_u(o_TS, mask, q):
    """Cross-Query: 외부 문맥이 정한 질문 q 로 읽기 (q forward 상수 → 기둥① 보존)."""
    return signed_query_readout(o_TS, q, mask)


def bank_r(o_TS, mask, bank: MatchedFilterBank):
    """뱅크 전폭 readout r:[n_out, M] (수선 3)."""
    return bank.read(o_TS, mask)
