"""§3 다중 Area 차원 분할 — IP(대조군) / RP 두 위상.

Block I 은 단일 처리 Area. 능력을 키우려면 처리 Area 를 복수로 두고 차원을 분할한다.
각 Area 는 Block I ``Area`` 자료형 그대로(서로 다른 지형 초기화로 비중복 특징 추출).

- **IP (Independent Parallel, 대조군)**: Area 들이 경계 재생 φ 로 분리, **u 로만 통신**.
  Area 간 직접 간섭 없음 — 깨끗·해석 가능.
- **RP (Related Parallel)**: Area 들이 상호작용(coupling). 능력↑ 가능, ρ 비용.

> 정직성: 본 초판 RP 는 **u-수준 coupling**(채널 평균 공유)으로 구현 — "펄스 수준 공통 시계
> 공유"의 근사다. 완전한 공통-시계 펄스 공유는 더 깊은 Block II 작업(# TODO). E2 의 판정은
> "IP 대비 측정 가능한 차이"(부호 무관·정량)이므로 이 근사로 충분히 측정된다.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class MultiAreaCluster(eqx.Module):
    areas: tuple                      # 병렬 처리 Area 들(서로 다른 지형)
    mode: str = eqx.field(static=True)        # "IP" | "RP"
    coupling: float = eqx.field(static=True)  # RP 공유 정도 ∈ [0,1]

    def run(self, geo, inj_BTN, windows):
        """inj:[B,T,N] → u_concat:[B, n_areas·C]. windows: Area별 마스크 list."""
        us = [a.forward(geo, inj_BTN, w)[1] for a, w in zip(self.areas, windows)]  # 각 [B,C]
        if self.mode == "RP" and len(us) > 1:
            mean_u = jnp.mean(jnp.stack(us, 0), axis=0)            # [B,C] 공유 성분
            us = [(1.0 - self.coupling) * u + self.coupling * mean_u for u in us]
        return jnp.concatenate(us, axis=-1)                        # [B, n_areas·C]

    def run_list(self, geo, inj_list, windows):
        """라우터가 Area별로 변조한 주입 inj_list(각 [B,T,N]) → u_concat:[B, n_areas·C]."""
        us = [a.forward(geo, inj, w)[1] for a, inj, w in zip(self.areas, inj_list, windows)]
        if self.mode == "RP" and len(us) > 1:
            mean_u = jnp.mean(jnp.stack(us, 0), axis=0)
            us = [(1.0 - self.coupling) * u + self.coupling * mean_u for u in us]
        return jnp.concatenate(us, axis=-1)

    def make_windows(self, geo):
        return [a.tmin_window(geo) for a in self.areas]


def build_multi_area(geo, gen_cells, out_cells, n_areas, mode, coupling, key, hp):
    """병렬 처리 Area n_areas 개(서로 다른 지형 초기화) 구성."""
    from lnn.cluster import _make_area
    keys = jax.random.split(key, n_areas)
    areas = tuple(_make_area(geo, keys[i], "processor", gen_cells, out_cells, hp)
                  for i in range(n_areas))
    return MultiAreaCluster(areas=areas, mode=mode, coupling=coupling)
