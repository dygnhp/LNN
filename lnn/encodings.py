"""§4.3 입력 인코딩 — plug-in. TimeEncoder(경로 B, 구현) / FreqEncoder(경로 A, 예약).

공통 인터페이스(frozen): ``encode(x) -> 주입 파형 inject_BTN``. 코어 동역학·readout·
학습 루프는 인코더 구현에 무관. 플래그 ``encoding="time"|"frequency"`` 로 교체.

핵심 사상: 토큰 임베딩 = **토큰의 파형**. D차원 벡터 = D개 입력 셀에 실리는 진폭 패턴
(작은 D, D ≤ 코히어런스 길이). 텍스트는 토큰을 일정 간격 **주입 시점**으로 순차 주입
(위치 = 주입 시점), 이미지는 픽셀을 공간 대응 셀에 **동시 방사**(초기 파면 = 이미지).
임베딩 학습 = 파형 학습, 입출력 코드북 공유(weight tying, §4.4와 닫힘).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .readout import wavelet


class TextTimeEncoder(eqx.Module):
    """경로 B 텍스트 인코더. 임베딩 [V, D] 가 곧 토큰별 파형 사전(학습 가능)."""

    embedding: jax.Array                       # [V, D]
    gen_cells: tuple = eqx.field(static=True)  # 길이 D 의 입력 셀(임베딩 캔버스)
    stride: int = eqx.field(static=True)       # 토큰 간 주입 간격(스텝)
    P: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)

    @property
    def D(self):
        return self.embedding.shape[1]

    def encode(self, tokens_BS):
        """tokens:[B,S] int → inject:[B, n_steps, N]. 토큰 p 를 시점 p·stride 에 방사."""
        emb = self.embedding[tokens_BS]                 # [B, S, D]
        B, S, D = emb.shape
        w = wavelet(self.P)                             # [P]
        inj = jnp.zeros((B, self.n_steps, self.n_cells), emb.dtype)
        for p in range(S):
            t0 = p * self.stride
            if t0 + self.P > self.n_steps:
                break
            for k, cell in enumerate(self.gen_cells):
                inj = inj.at[:, t0:t0 + self.P, cell].add(emb[:, p, k][:, None] * w[None, :])
        return inj


class ImageEncoder(eqx.Module):
    """이미지 인코더(학습 파라미터 없음). 픽셀 밝기 → 공간 대응 셀에 동시 방사(wavelet)."""

    gen_cells: tuple = eqx.field(static=True)   # 길이 = 픽셀 수
    P: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)

    def encode(self, images_BP):
        """images:[B, n_pixels] (0–1 정규화) → inject:[B, n_steps, N], 모두 t=0 방사."""
        w = wavelet(self.P)
        B = images_BP.shape[0]
        inj = jnp.zeros((B, self.n_steps, self.n_cells), images_BP.dtype)
        for k, cell in enumerate(self.gen_cells):
            inj = inj.at[:, 0:self.P, cell].add(images_BP[:, k][:, None] * w[None, :])
        return inj


class FreqEncoder(eqx.Module):
    """경로 A(OFDM식) — Block II 예약. Block I 에서는 인터페이스/직교성 테스트만.

    벡터 성분을 서로 다른 주파수의 진폭에 실어 합성 → 차원을 시간에서 분리(큰 D 확장의
    유일한 길). 단일 매질 직교 주파수 수 한계 때문에 다중 Area 차원 분할과 반드시 결합.
    """

    embedding: jax.Array
    gen_cells: tuple = eqx.field(static=True)
    P: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)

    def encode(self, tokens_BS):  # pragma: no cover - Block II
        raise NotImplementedError(
            "FreqEncoder 는 Block II 예약(경로 A). Block I 은 TimeEncoder 사용. # TODO"
        )

    def channel_basis(self):
        """주파수 채널 직교 기저(단위 테스트용). 주기 1..D 의 sin 파형 [D, P]."""
        D = self.embedding.shape[1]
        t = jnp.arange(self.P)
        return jnp.stack([jnp.sin(2 * jnp.pi * (m + 1) * t / self.P) for m in range(D)])


def channel_orthogonality(codes):
    """코드 행렬 [K, d] 의 **최대 비대각 절대 코사인 유사도**(§11★ D3 직교성 측정).

    값이 작을수록 채널이 직교. D3(D≈8)에서 임계 초과 시 선택성 붕괴 신호.
    """
    n = codes / (jnp.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
    g = jnp.abs(n @ n.T)
    K = g.shape[0]
    off = g - jnp.eye(K)
    return float(jnp.max(off))
