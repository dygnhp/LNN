"""작업 1 — 임베딩 분산 (E1 의 학습 임베딩 colinear 붕괴 처방, §1).

단일 코드북 c_v ∈ R^D 를 M개 부분코드 c_v^(m) ∈ R^(D/M) 로 분할, 각 부분을 **다른 처리
Area** 가 담당. 각 Area 가 일부 차원만 책임 → 전체 코드가 한 직선으로 뭉치려면 *모든* Area 가
동시에 정렬돼야 함(자유도↓ → colinear 억제). 임베딩은 **하나의 [V,D] 배열**을 슬라이스로
공유(부분 복사 아님 — 전체 임베딩에 그래디언트가 흐름).

frozen interface: encode 시그니처 보존 — 반환을 Area별로 분배하는 래퍼(§1.1).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .freq_encoding import ofdm_basis


class DistributedEmbedding(eqx.Module):
    embedding: jax.Array                         # [V, D]  (공유 코드북)
    n_areas: int = eqx.field(static=True)        # M
    dim_per_area: int = eqx.field(static=True)   # D/M
    carriers: tuple = eqx.field(static=True)     # Area별 운반 셀 tuple of tuple
    window: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)

    @property
    def D(self):
        return self.embedding.shape[1]

    def _slice(self, m):
        dm = self.dim_per_area
        return self.embedding[:, m * dm:(m + 1) * dm]    # [V, dm]

    def encode(self, tokens_BS):
        """tokens:[B,S] → Area별 주입 list (각 [B, n_steps, N]). Area m 은 자기 부분코드만 방사."""
        injs = []
        dm = self.dim_per_area
        basis = ofdm_basis(dm, self.window)              # [dm, W] (Area 공통 기저)
        for m in range(self.n_areas):
            sub = self._slice(m)[tokens_BS]              # [B, S, dm]
            inj = jnp.zeros((tokens_BS.shape[0], self.n_steps, self.n_cells), self.embedding.dtype)
            for p in range(tokens_BS.shape[1]):
                t0 = p * self.stride
                if t0 + self.window > self.n_steps:
                    break
                wave = jnp.einsum("bd,dw->bw", sub[:, p, :], basis)   # [B, W]
                for cell in self.carriers[m]:
                    inj = inj.at[:, t0:t0 + self.window, cell].add(wave)
            injs.append(inj)
        return injs


def orthogonality_penalty(embedding, token_ids):
    """부분코드 직교 정규화 항(소프트, 선택 — §1.2). 약하게(λ 작게) 과제손실과 병행.

    토큰 코드 간 평균 제곱 코사인 유사도(비대각). Block I 원칙대로 '직교성은 부산물'이라
    약한 보조항으로만 둔다.
    """
    c = embedding[jnp.asarray(token_ids)]
    n = c / (jnp.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    g = n @ n.T
    K = g.shape[0]
    off = g - jnp.eye(K) * jnp.diag(g)
    return jnp.sum(off ** 2) / (K * (K - 1) + 1e-9)
