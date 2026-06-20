"""Phase 3 §2 — D 스케일링 + 다중 Area 차원 분할 (경로 A).

단일 매질(한 Area)이 직교로 담을 수 있는 주파수 채널 수는 코히어런스·격자로 제한된다.
D 를 키우면 단일 매질이 모든 채널을 직교 유지 못 함(혼변조) → **여러 Area 가 채널을 나눠
나르는** 차원 분할이 필수(SPEC §4.3). 각 Area 가 ``freq_per_area`` 개 채널만 담당.

DimSplitEncoder: D 주파수 채널을 M=ceil(D/freq_per_area) Area 에 분배(Phase 2
DistributedEmbedding 과 같은 래퍼 패턴, frozen encode 시그니처 보존). 읽기는 Area별 주파수
뱅크로 자기 채널 복원 → 합쳐 전체 D 차원 logits.

ρ 주의(§2.3): D·Area↑ 로 결합↑ → ρ↑. spectral.ρ<1 정규화 계승. 경로 A 는 선형 영역에서만
비간섭(포화 시 혼변조) → 큰 D 에서 선형 운용 유지.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .freq_encoding import ofdm_basis
from .freq_readout import signed_bank_readout


def n_areas_for(D: int, freq_per_area: int = 8) -> int:
    """D 를 직교 유지 가능한 Area 수로 분할: M = ceil(D / freq_per_area)."""
    return max(1, (D + freq_per_area - 1) // freq_per_area)


class DimSplitEncoder(eqx.Module):
    embedding: jax.Array                         # [V, D]
    n_areas: int = eqx.field(static=True)        # M
    freq_per_area: int = eqx.field(static=True)  # D/M (Area당 채널)
    carriers: tuple = eqx.field(static=True)     # Area별 운반 셀
    window: int = eqx.field(static=True)
    stride: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    n_cells: int = eqx.field(static=True)

    @property
    def D(self):
        return self.embedding.shape[1]

    def encode(self, tokens_BS):
        """tokens:[B,S] → Area별 주입 list (각 [B,n_steps,N]). Area m 은 자기 주파수 블록만."""
        fpa = self.freq_per_area
        basis = ofdm_basis(fpa, self.window)             # [fpa, W]
        injs = []
        for m in range(self.n_areas):
            sub = self.embedding[:, m * fpa:(m + 1) * fpa][tokens_BS]   # [B, S, fpa]
            inj = jnp.zeros((tokens_BS.shape[0], self.n_steps, self.n_cells), self.embedding.dtype)
            for p in range(tokens_BS.shape[1]):
                t0 = p * self.stride
                if t0 + self.window > self.n_steps:
                    break
                wave = jnp.einsum("bd,dw->bw", sub[:, p, :], basis)     # [B, W]
                for cell in self.carriers[m]:
                    inj = inj.at[:, t0:t0 + self.window, cell].add(wave)
            injs.append(inj)
        return injs


def dim_split_feats(areas, geo, injs, fpa, window, masks, out_arr):
    """Area별 주파수 뱅크 복원 → 전체 D 차원 feature [B, D] (Area 축으로 concat)."""
    from lnn import dynamics
    basis = ofdm_basis(fpa, window)
    parts = []
    for m, area in enumerate(areas):
        sc = area.step_constants(geo)

        def single(inj_TN):
            o = dynamics.rollout(sc, inj_TN, out_arr, geo.N)
            return jnp.mean(signed_bank_readout(o, basis, masks[m]), axis=0)   # [fpa]
        parts.append(jax.vmap(single)(injs[m]))                                # [B, fpa]
    return jnp.concatenate(parts, axis=-1)                                     # [B, D]
