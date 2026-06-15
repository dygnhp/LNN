"""Experiment 3 성장 단위 테스트 (§3 절차 3).

- grow_terrain_K 후 지형 T(p) 가 연속·미분가능하고, 기존 언덕이 보존되는지.
- grow_grid 후 기존 언덕이 보존되는지(같은 좌표에서 T값 불변).
- PlateauDetector 동작.
"""

import jax
import jax.numpy as jnp
import numpy as np

from lnn import fields, growth
from lnn.cluster import build_serial_cluster
from lnn.geometry import build_geometry


def _area():
    geo = build_geometry(5)
    clus = build_serial_cluster(geo, (0, 1, 2, 3), (10, 11, 12, 13),
                                jax.random.PRNGKey(0), {"n_channels": 8, "n_proc": 1})
    return geo, clus.areas[0]


def test_plateau_detector():
    d = growth.PlateauDetector(window=3, threshold=0.02)
    for v in [0.800, 0.790, 0.785, 0.783, 0.782, 0.781]:  # 양 윈도 모두 정체 구간
        d.update(v)
    assert d.is_plateau()
    d2 = growth.PlateauDetector(window=3, threshold=0.02)
    for v in [1.0, 0.9, 0.8, 0.6, 0.4, 0.2]:       # 계속 개선
        d2.update(v)
    assert not d2.is_plateau()


def test_grow_terrain_K_appends_and_preserves():
    geo, area = _area()
    K0 = area.terrain_h.shape[0]
    grown = growth.grow_area_terrain(area, 4, geo, jax.random.PRNGKey(1))
    assert grown.terrain_h.shape[0] == K0 + 4
    assert grown.terrain_c.shape == (K0 + 4, 2)
    # 기존 언덕(앞 K0개) 보존
    assert jnp.allclose(grown.terrain_h[:K0], area.terrain_h)
    assert jnp.allclose(grown.terrain_c[:K0], area.terrain_c)


def test_grow_terrain_K_terrain_continuous_differentiable():
    geo, area = _area()
    grown = growth.grow_area_terrain(area, 4, geo, jax.random.PRNGKey(2))
    p = geo.edge_mid

    def Tsum(h):
        return jnp.sum(fields.terrain_value(p, h, grown.terrain_c, grown.terrain_sigma))

    val = Tsum(grown.terrain_h)
    g = jax.grad(Tsum)(grown.terrain_h)
    assert jnp.isfinite(val)
    assert bool(jnp.all(jnp.isfinite(g)))
    assert g.shape == grown.terrain_h.shape


def test_grow_grid_preserves_hills():
    geo, area = _area()
    # 격자 확대 전, 임의 좌표에서의 T값
    probe = jnp.asarray([[0.0, 0.0], [2.0, -1.0], [-3.0, 1.5]], jnp.float32)
    T_before = fields.terrain_value(probe, area.terrain_h, area.terrain_c, area.terrain_sigma)
    geo_new = growth.grow_grid(geo, R_new=8)
    assert geo_new.N > geo.N
    # 지형 파라미터는 연속 좌표 위에 정의 → 격자 확대 후 같은 좌표 T값 불변
    T_after = fields.terrain_value(probe, area.terrain_h, area.terrain_c, area.terrain_sigma)
    assert jnp.allclose(T_before, T_after)


def test_grow_cluster_and_model():
    geo, _ = _area()
    clus = build_serial_cluster(geo, (0, 1, 2, 3), (10, 11, 12, 13),
                                jax.random.PRNGKey(0), {"n_channels": 8, "n_proc": 1})
    kt0, kg0 = growth.current_K((None, clus))
    grown = growth.grow_cluster(clus, 4, 2, geo, jax.random.PRNGKey(3))
    for a in grown.areas:
        assert a.terrain_h.shape[0] == kt0 + 4
        assert a.gain_a.shape[0] == kg0 + 2
