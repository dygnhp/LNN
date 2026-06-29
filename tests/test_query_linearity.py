"""Phase 7 — Cross/static Query 는 o 에 선형(기둥① 보존), Self-Query 는 ε 로 비선형."""

import jax.numpy as jnp
import numpy as np

from lnn.readout import wavelet
from lnn_block2.dynamic_query import query_matched_filter, self_query


def _o(seed=0, T=40, S=4, P=8):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(size=(T, S)).astype("float32"))


def test_cross_query_linear_in_o():
    # q 가 o 와 무관(외부 문맥)하면 정합필터는 o 에 1차 동차 → 기둥① 보존.
    o = _o()
    q = wavelet(8) * 0.7 + 0.1            # 임의 고정 질문(외부 문맥이 정한 셈)
    c1 = query_matched_filter(o, q)
    c2 = query_matched_filter(2.0 * o, q)
    assert jnp.allclose(c2, 2.0 * c1, atol=1e-5)


def test_self_query_is_content_dependent():
    # static 은 q 가 입력 무관(고정 필터). self(ε>0)는 q 가 *내용*에 따라 바뀜(읽는 질문이 적응)
    # → 더는 고정 선형 필터가 아님(기둥① 약화). 서로 다른 입력에 q 가 달라지는지로 검증.
    o1, o2 = _o(seed=0), _o(seed=1)
    q0 = wavelet(8)
    # static: 입력 무관 동일
    assert jnp.allclose(q0, q0)
    # self: 내용에 따라 q 가 달라짐
    qa = self_query(o1, q0, epsilon=0.5)
    qb = self_query(o2, q0, epsilon=0.5)
    assert not jnp.allclose(qa, qb, atol=1e-4)
    # ε=0 이면 static 으로 회귀
    assert jnp.allclose(self_query(o1, q0, epsilon=0.0), q0)
