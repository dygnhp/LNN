"""§4.4 출력 디코더 — 세 모드 plug-in. encoder 와 같은 코드북 공유(vocabulary).

공통 인터페이스(frozen): ``decode(readout) -> 출력``.

- discriminative : 클래스당 Output Cell + score + softmax (D1, D2)
- structural     : Output Cell 격자를 출력 좌표에 공간 대응, 셀별 정합 응답 = 값 (D4)
- vocabulary     : 어휘마다 (준)직교 코드 c_v, 코드북 정합필터 상관의 argmax (D3, CDMA식)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .readout import peak_and_sign, score_from_c


def discriminative_logits(c_BTS, window):
    """클래스 = 출력 셀. c_BTS:[B, valid, n_classes] → logits:[B, n_classes] (채널 score)."""
    return jax.vmap(lambda c: score_from_c(c, window))(c_BTS)


def structural_image(c_BTS, window):
    """출력 셀 격자 → 셀별 정합 응답 피크 = 픽셀 값. c_BTS:[B, valid, n_pix] → [B, n_pix] (≥0)."""
    peak = jax.vmap(lambda c: peak_and_sign(c, window)[0])(c_BTS)
    return peak


def vocabulary_logits(u_BC, codebook_VD):
    """어휘 코드북 상관. u_BC:[B, m] (부호 있는 readout), codebook:[V, m] → logits:[B, V].

    디코딩 = 코드북 정합필터 상관의 argmax = softmax over vocabulary 의 물리 등가.
    입력 임베딩과 같은 코드북을 공유(weight tying, §4.3 결정 3).
    """
    cb = codebook_VD / (jnp.linalg.norm(codebook_VD, axis=1, keepdims=True) + 1e-9)
    return u_BC @ cb.T
