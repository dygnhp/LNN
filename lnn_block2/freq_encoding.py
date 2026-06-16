"""§2.1 경로 A 입구 — FreqEncoder (OFDM 직교 주파수 다중화).

토큰 임베딩 D개 성분을 **서로 다른 직교 주파수 대역**의 wavelet 합으로 매핑:
    s(t) = Σ_{d=1..D} a_d · sin(2π·d·t / W),  t ∈ [0, W)
주파수 d(= 윈도 W 안의 정수 사이클)는 DFT 기저라 W 구간에서 **엄밀히 직교**(주파수 간격 ×
윈도 길이 = 정수, §2.1). 출력은 Block I 과 같은 펄스 자료형([B, T, N] 주입 텐서).

frozen interface: ``encode(tokens) -> pulses`` 시그니처 불변(§0.3). 이 직교는 **선형 운용**
에서만 유지(포화 시 혼변조 — §5 ρ 세 기둥과 연결).

freq_readout 의 정합필터 뱅크가 같은 ``ofdm_basis`` 를 써야 채널이 분리 복원된다(§2.2).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


def ofdm_basis(D: int, window: int):
    """[D, window] — 주파수 d=1..D 의 sin 기저(윈도 W 에서 직교). 인코더·readout 공유.

    **직교 조건**: ``D < window/2``. 주파수 W/2(Nyquist)는 정수 t 에서 sin(π t)=0 이라
    빈(영) 채널이 되어 정규화 시 float 잡음으로 직교가 깨진다 → 호출부가 W ≥ 2(D+1) 보장.
    """
    t = jnp.arange(window)
    return jnp.stack([jnp.sin(2.0 * jnp.pi * (d + 1) * t / window) for d in range(D)])


class FreqEncoder(eqx.Module):
    """경로 A 인코더. 임베딩 [V, D] 가 토큰별 주파수-스펙트럼 사전(학습 가능)."""

    embedding: jax.Array                        # [V, D]
    gen_cells: tuple = eqx.field(static=True)   # 운반(carrier) 셀(들)
    P: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)
    window: int = eqx.field(static=True)        # OFDM 심볼 길이 W (≥ 2D 권장)
    stride: int = eqx.field(static=True)

    @property
    def D(self):
        return self.embedding.shape[1]

    def encode(self, tokens_BS):
        """tokens:[B,S] → pulses:[B, n_steps, N]. 토큰별 OFDM 심볼을 운반 셀에 주입."""
        emb = self.embedding[tokens_BS]                 # [B, S, D]
        B, S, D = emb.shape
        basis = ofdm_basis(D, self.window)              # [D, W]
        inj = jnp.zeros((B, self.n_steps, self.n_cells), emb.dtype)
        for p in range(S):
            t0 = p * self.stride
            if t0 + self.window > self.n_steps:
                break
            wave = jnp.einsum("bd,dw->bw", emb[:, p, :], basis)   # [B, W]
            for cell in self.gen_cells:
                inj = inj.at[:, t0:t0 + self.window, cell].add(wave)
        return inj
