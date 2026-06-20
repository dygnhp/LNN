"""Phase 5 §3.5 — RAL 단위 검증: φ 가 깊이를 만든다 + ρ 정규화 없으면 발산.

① φ 존재 검증: phi=identity(선형 RAL)는 입력에 1차 동차(homogeneous, x→2x ⇒ 출력 2배) —
   깊이 0(단일 선형변환). phi=tanh 는 동차성 깨짐(포화) = 비선형 깊이 발생.
② ρ 발산 재현: 선형 루프 + gain>1 은 루프마다 증폭→발산, tanh 는 유계(E4 메커니즘).
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import _common as C  # noqa: E402

from lnn.area import pick_cells  # noqa: E402
from lnn_block2.ral import build_ral_classifier, free_loop_growth  # noqa: E402


def _ral(n_loops=4, seed=0):
    geo = C.make_geometry(5)
    hp = {**C.classify_hp(), "n_steps": 70}
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, 8, phase=2.6)
    ral = build_ral_classifier(geo, pix, feat, 10, n_loops, jax.random.PRNGKey(seed), hp)
    win = ral.make_window(geo)
    return geo, ral, win


def test_phi_creates_depth_linear_is_homogeneous():
    geo, ral, win = _ral(n_loops=4)
    X = jnp.asarray(np.random.default_rng(0).uniform(0, 1, (3, 64)).astype("float32"))
    # 선형 RAL(phi=identity): 1차 동차 — 입력 2배 → 출력 ~2배(깊이 0, 단일 선형변환).
    u1 = ral.run_u(geo, X, win, phi=lambda z: z)
    u2 = ral.run_u(geo, 2.0 * X, win, phi=lambda z: z)
    lin_ratio = float(jnp.linalg.norm(u2) / (jnp.linalg.norm(u1) + 1e-9))
    assert abs(lin_ratio - 2.0) < 0.1, f"linear RAL not homogeneous: ratio={lin_ratio:.3f}"
    # tanh RAL: 동차성 깨짐(포화) = 비선형 깊이 발생 → 비율이 2 에서 유의 이탈.
    t1 = ral.run_u(geo, X, win, phi=jnp.tanh)
    t2 = ral.run_u(geo, 2.0 * X, win, phi=jnp.tanh)
    tanh_ratio = float(jnp.linalg.norm(t2) / (jnp.linalg.norm(t1) + 1e-9))
    assert abs(tanh_ratio - 2.0) > 0.05, f"tanh RAL stayed linear: ratio={tanh_ratio:.3f}"


def test_rho_divergence_without_regularization():
    geo, ral, win = _ral(n_loops=8)
    X = jnp.asarray(np.random.default_rng(1).uniform(0, 1, (2, 64)).astype("float32"))
    # 선형 루프 + gain>1 → 발산(성장비≫1). tanh → 유계.
    g_lin, _ = free_loop_growth(ral, geo, X, win, phi=lambda z: z, gain=1.6)
    g_tanh, _ = free_loop_growth(ral, geo, X, win, phi=jnp.tanh, gain=1.6)
    assert g_lin > g_tanh, f"linear loop not diverging vs tanh: lin={g_lin:.2e} tanh={g_tanh:.2e}"
