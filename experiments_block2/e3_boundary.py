"""E3 — Block I 천장(~0.56) 돌파 시도 (§8, 핵심).

경로 A 정신(다중 채널) + 다중 Area + 동적 라우팅(ε>0) + 이득장 적극화로 MNIST 8x8 에서
Block I 천장(Exp3 ~0.56)을 넘는지. 넘으면 "물리 확장이 경계를 민다", 못 넘어도 정량 측정이 목표.
세 기둥 ① 감쇠(ε별 경로분해 손실)도 함께 기록(§4.2).
"""

from __future__ import annotations

import os
import sys

import jax

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import _common as C  # noqa: E402

from lnn.area import pick_cells  # noqa: E402
from lnn.data.mnist_data import load_mnist_split  # noqa: E402
from lnn_block2.kei import build_kei_image  # noqa: E402
from lnn_block2.routing import DynamicRouter, pathway_decomposition_decay  # noqa: E402
from _kei_common import train_kei_image  # noqa: E402

N_CLASSES = 10
BLOCK1_CEILING = 0.56


def run(seed=0, R=5, per_class=100, n_areas=3, epsilon=0.3, e_res=5, e_open=16, batch=32):
    print(f"[E3] Block I 천장(~{BLOCK1_CEILING}) 돌파 시도 (다중Area+라우팅ε={epsilon}+이득)")
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 70, "n_proc": 1}
    Xtr, ytr, Xte, yte, src = load_mnist_split(n_per_class=per_class, size=8,
                                               test_per_class=30, seed=seed)
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, 10, phase=2.6)
    model = build_kei_image(geo, pix, feat, n_areas, "RP", coupling=0.4, epsilon=epsilon,
                            n_classes=N_CLASSES, key=jax.random.PRNGKey(seed), hp=hp)
    _, acc_res, acc_open, _ = train_kei_image(model, geo, Xtr, ytr, Xte, yte, e_res=e_res,
                                              e_open=e_open, batch=batch, open_gain=True,
                                              seed=seed, prefix="[E3] ")
    broke = acc_open > BLOCK1_CEILING
    print(f"    acc {acc_res:.3f} -> {acc_open:.3f}  vs Block I {BLOCK1_CEILING} "
          f"({'돌파' if broke else '미돌파'})")

    # 세 기둥 ① 감쇠: ε별 경로분해 손실
    import jax.numpy as jnp
    inj = model.encoder.encode(jnp.asarray(Xtr[:8]))
    decay = {eps: round(pathway_decomposition_decay(DynamicRouter(epsilon=eps), inj, n_areas), 4)
             for eps in (0.0, 0.3, 0.6)}
    print(f"    세 기둥① 경로분해 감쇠 (ε별): {decay}")
    return dict(exp="E3", acc_res=acc_res, acc_open=acc_open, block1_ceiling=BLOCK1_CEILING,
                broke_ceiling=bool(broke), pathway_decay=decay, n_areas=n_areas,
                epsilon=epsilon, passed=True)


if __name__ == "__main__":
    run()
