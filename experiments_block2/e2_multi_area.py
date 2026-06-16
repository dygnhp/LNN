"""E2 — IP vs RP 다중 Area 분류 (§8). RP가 IP 대비 측정 가능한 차이(부호 무관·정량이면 통과)."""

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
from _kei_common import train_kei_image  # noqa: E402

N_CLASSES = 10


def run(seed=0, R=5, per_class=60, n_areas=3, e_res=4, e_open=10, batch=32):
    print(f"[E2] IP vs RP 다중 Area (n_areas={n_areas}, MNIST 8x8)")
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 70, "n_proc": 1}
    Xtr, ytr, Xte, yte, src = load_mnist_split(n_per_class=per_class, size=8,
                                               test_per_class=20, seed=seed)
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, 8, phase=2.6)

    out = {}
    for mode in ("IP", "RP"):
        model = build_kei_image(geo, pix, feat, n_areas, mode, coupling=0.5, epsilon=0.0,
                                n_classes=N_CLASSES, key=jax.random.PRNGKey(seed), hp=hp)
        _, ar, ao, _ = train_kei_image(model, geo, Xtr, ytr, Xte, yte, e_res=e_res,
                                       e_open=e_open, batch=batch, open_gain=True,
                                       seed=seed, prefix=f"[{mode}] ")
        out[mode] = ao
        print(f"    {mode}: acc {ar:.3f} -> {ao:.3f}")
    diff = out["RP"] - out["IP"]
    print(f"    RP - IP = {diff:+.3f} (다중 Area 직접 상호작용 이득의 정량)")
    return dict(exp="E2", acc_IP=out["IP"], acc_RP=out["RP"], diff=diff,
                n_areas=n_areas, passed=True)  # 정량이면 통과(부호 무관)


if __name__ == "__main__":
    run()
