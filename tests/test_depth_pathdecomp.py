"""Phase 5 §3.2 — 깊이↑에서 경로 분해(세 기둥①)가 끊기는 양 측정.

L=1(얕음)은 잔차 작음(거의 선형·분해 가능), L↑로 잔차 증가 = 깊이가 해석 가능성을 깎는 양.
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
from lnn_block2.depth_scaling import build_depth_classifier, path_decomp_residual  # noqa: E402


def _resid_at_L(L, seed=0):
    geo = C.make_geometry(5)
    hp = {**C.classify_hp(), "n_steps": 70}
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, 8, phase=2.6)
    enc, cluster, _head = build_depth_classifier(geo, pix, feat, 10, L, jax.random.PRNGKey(seed), hp)
    X = jnp.asarray(np.random.default_rng(0).uniform(0, 1, (4, 64)).astype("float32"))
    windows = cluster.make_windows(geo)
    inj = enc.encode(X)
    return path_decomp_residual(cluster, geo, inj, windows)


def test_residual_nonnegative_and_grows_with_depth():
    r1 = _resid_at_L(1)
    r4 = _resid_at_L(4)
    assert r1 >= 0.0 and r4 >= 0.0
    # 깊이↑ → 경계 φ 통과 횟수↑ → 선형 경로-합에서 더 벗어남(해석 가능성 손실↑).
    assert r4 > r1, f"path-decomp residual not growing with depth: L1={r1:.4f}, L4={r4:.4f}"
