"""자율 성장 (Experiment 3) — CHM ``FINAL/growth.py`` 구조를 LNN 지형에 이식.

가져온 것: PlateauDetector(이동 윈도 상대개선), grow_K(RBF 추가·교대부호·fill),
성장 후 옵티마이저 상태 재구축 패턴. **버린 것**: CHM 입자 동역학·접촉 해밀토니안,
그리고 grow_D 의 ``σ·√(D_new/D_old)`` 재스케일 — LNN 은 차원이 안 변하고 **격자만**
커지며 지형은 연속 좌표 위에 정의되므로 거리 분포가 불변, 재스케일 불요(코드 주석).

ARIS 코어 동역학은 불변 — 지형/이득 RBF 의 **개수**만 학습 중 변한다.

CHM 의 K_grow 는 끌개/장벽 부호(w<0/w>0)로 RBF 를 심지만, LNN 지형 T 는 지연을 만드는
언덕 높이 h_k(골/봉우리 = 빠른/느린 지연)다. 초판은 사양 허용대로 **교대 부호 + 작은
초기값**으로 fill 한다(guided 배치는 diagnostic 훅으로 열어둠).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .geometry import build_geometry


# ─────────────────────────── Plateau 감지 (CHM 그대로) ───────────────────────
class PlateauDetector:
    """이동 윈도 상대개선 plateau 감지. (LNN 은 epoch 단위 → 작은 window.)"""

    def __init__(self, window: int = 3, threshold: float = 0.02):
        self.window = window
        self.threshold = threshold
        self.losses: List[float] = []

    def update(self, loss: float) -> None:
        self.losses.append(float(loss))

    def is_plateau(self) -> bool:
        n = len(self.losses)
        if n < 2 * self.window:
            return False
        old = float(np.mean(self.losses[-2 * self.window:-self.window]))
        new = float(np.mean(self.losses[-self.window:]))
        if abs(old) < 1e-12:
            return False
        return (old - new) / abs(old) < self.threshold

    def reset(self) -> None:
        self.losses = []


@dataclass
class GrowthConfig:
    """성장 하이퍼파라미터 (CHM config.py 패턴, LNN epoch 스케일로 축소)."""
    plateau_window: int = 3
    plateau_threshold: float = 0.02
    min_epochs_before_grow: int = 4
    cooldown_after_grow: int = 3
    K_terrain_grow: int = 4
    K_gain_grow: int = 2
    K_terrain_max: int = 64
    K_gain_max: int = 32
    K_grows_before_grid: int = 3
    R_grow: int = 3
    R_max: int = 16
    grow_grid_enabled: bool = False  # Exp3 실험 B/C 는 격자 고정(변수를 K 성장으로 격리)


# ─────────────────────────── grow_terrain_K / grow_gain_K ────────────────────
def _bbox(geo):
    pos = np.asarray(geo.pos)
    return pos.min(0), pos.max(0)


def grow_area_terrain(area, n_grow, geo, key):
    """Area 지형에 n_grow 개 RBF 언덕 추가. 교대 부호 + 작은 초기값(§1.1)."""
    if n_grow <= 0:
        return area
    lo, hi = _bbox(geo)
    cc = np.asarray(jax.random.uniform(key, (n_grow, 2))) * (hi - lo) + lo
    signs = np.where(np.arange(n_grow) % 2 == 0, 1.0, -1.0).astype(np.float32)
    new_h = jnp.asarray(0.05 * signs)
    new_c = jnp.asarray(cc, jnp.float32)
    sig_mean = float(jnp.mean(area.terrain_sigma)) if area.terrain_sigma.size else 3.0
    new_sig = jnp.full((n_grow,), sig_mean, jnp.float32)
    return eqx.tree_at(
        lambda a: (a.terrain_h, a.terrain_c, a.terrain_sigma), area,
        (jnp.concatenate([area.terrain_h, new_h]),
         jnp.concatenate([area.terrain_c, new_c], axis=0),
         jnp.concatenate([area.terrain_sigma, new_sig])),
    )


def grow_area_gain(area, n_grow, geo, key):
    """Area 이득장에 n_grow 개 RBF 추가. a_k=0(초기 이득 중립 유지) → 학습으로 채움."""
    if n_grow <= 0:
        return area
    lo, hi = _bbox(geo)
    cc = np.asarray(jax.random.uniform(key, (n_grow, 2))) * (hi - lo) + lo
    new_a = jnp.zeros((n_grow,))
    new_d = jnp.asarray(cc, jnp.float32)
    sig_mean = float(jnp.mean(area.gain_sigma)) if area.gain_sigma.size else 3.0
    new_sig = jnp.full((n_grow,), sig_mean, jnp.float32)
    return eqx.tree_at(
        lambda a: (a.gain_a, a.gain_d, a.gain_sigma), area,
        (jnp.concatenate([area.gain_a, new_a]),
         jnp.concatenate([area.gain_d, new_d], axis=0),
         jnp.concatenate([area.gain_sigma, new_sig])),
    )


def grow_cluster(cluster, n_grow_t, n_grow_g, geo, key):
    """Cluster 의 모든 Area 지형(+이득)을 동시에 키운다. 새 Cluster 반환."""
    keys = jax.random.split(key, 2 * len(cluster.areas))
    new_areas = []
    for i, a in enumerate(cluster.areas):
        a = grow_area_terrain(a, n_grow_t, geo, keys[2 * i])
        a = grow_area_gain(a, n_grow_g, geo, keys[2 * i + 1])
        new_areas.append(a)
    return eqx.tree_at(lambda c: c.areas, cluster, tuple(new_areas))


def _cluster_index(model):
    for i, m in enumerate(model):
        if hasattr(m, "areas"):
            return i
    raise ValueError("no cluster in model")


def grow_model(model, n_grow_t, n_grow_g, geo, key):
    """model 튜플 안의 Cluster 를 찾아 키운 뒤 새 model 튜플 반환."""
    i = _cluster_index(model)
    grown = grow_cluster(model[i], n_grow_t, n_grow_g, geo, key)
    return tuple(grown if j == i else m for j, m in enumerate(model))


def current_K(model):
    """(K_terrain, K_gain) — encoder Area 기준(모든 Area 동일 보폭으로 성장)."""
    i = _cluster_index(model)
    a0 = model[i].areas[0]
    return int(a0.terrain_h.shape[0]), int(a0.gain_a.shape[0])


# ─────────────────────────── grow_grid (격자 확대) ───────────────────────────
def grow_grid(geo_old, R_new):
    """격자 반지름 R→R_new 확대. 새 geometry 반환.

    지형/이득 RBF 는 **연속 좌표**(c_k) 위에 정의되므로 격자가 커져도 그대로 보존하면
    같은 연속 함수 T(p)·G(p) 가 새 격자에서 재평가된다(CHM zero-padding 보다 자연스러움).
    **CHM 의 σ·√(D_new/D_old) 재스케일은 불요** — 차원이 아니라 격자만 커지므로 거리 분포 불변.
    단자(Generator/Output) 셀 인덱스는 격자 의존이라 호출부가 배치 규칙으로 재계산해야 한다.
    """
    return build_geometry(R_new)
