"""§6 자율 성장 다중 Area 확장 — Exp3 ``lnn.growth`` 계승.

Block I 은 단일 Area 지형만 키웠다. Block II 는 **어느 Area 가 용량 부족인가**를 Area별
PlateauDetector 로 감지해 그 Area 의 K 만 성장(라우팅이 특정 Area 에 과부하면 그 Area 우선).
성장 후 ARIS 패턴대로 상태 재구축(재컴파일·optax 재초기화)은 호출부(train2)가 담당.
"""

from __future__ import annotations

from lnn.growth import GrowthConfig, PlateauDetector, grow_area_gain, grow_area_terrain  # 계승


class PerAreaPlateau:
    """Area별 plateau 감지기 묶음 — Area별 손실 기여로 부족 Area 를 특정."""

    def __init__(self, n_areas, window=3, threshold=0.02):
        self.detectors = [PlateauDetector(window, threshold) for _ in range(n_areas)]

    def update(self, per_area_losses):
        for d, lo in zip(self.detectors, per_area_losses):
            d.update(float(lo))

    def plateaued_areas(self):
        """plateau 인 Area 인덱스 목록."""
        return [i for i, d in enumerate(self.detectors) if d.is_plateau()]

    def reset(self, i):
        self.detectors[i].reset()


def grow_one_area(multi_cluster, area_idx, n_grow_t, n_grow_g, geo, key):
    """MultiAreaCluster 의 특정 Area 만 성장시켜 새 cluster 반환."""
    import equinox as eqx
    import jax

    k1, k2 = jax.random.split(key)
    new_areas = list(multi_cluster.areas)
    a = new_areas[area_idx]
    a = grow_area_terrain(a, n_grow_t, geo, k1)
    a = grow_area_gain(a, n_grow_g, geo, k2)
    new_areas[area_idx] = a
    return eqx.tree_at(lambda c: c.areas, multi_cluster, tuple(new_areas))


__all__ = ["GrowthConfig", "PlateauDetector", "PerAreaPlateau", "grow_one_area",
           "grow_area_terrain", "grow_area_gain"]
