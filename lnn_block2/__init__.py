"""LNN Block II — KEI (경로 A 완성 + 다중 Area + 동적 라우팅 + ρ 1급화).

ARIS(Block I) 코어를 **재사용**(import from ``lnn.*``)하고 Block II 신규 요소만 추가한다.
frozen interface(Cell 갱신·변지연 τ·경계 재생 φ·통신 벡터 u·encode/decode)는 불변 —
어떤 Block II 코드도 ``lnn/`` 코어를 수정하지 않는다(§0.3).
"""

__all__ = [
    "freq_encoding", "freq_readout", "multi_area",
    "routing", "spectral", "growth2", "kei", "train2",
]
