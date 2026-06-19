"""P2 — RP 펄스 수준 공유 (§4). IP vs RP(펄스) acc 차가 Phase1 +0.005 대비 확대되는가."""

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
from lnn_block2.kei import build_rp_image  # noqa: E402
from _kei_common import train_kei_image  # noqa: E402

N_CLASSES = 10
PHASE1_DIFF = 0.005


def run(seed=0, R=5, per_class=60, n_areas=3, coupling=0.4, e_res=4, e_open=10, batch=32):
    print(f"[P2] IP vs RP 펄스 공유 (n_areas={n_areas}, coupling={coupling}, MNIST 8x8)")
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 70}
    Xtr, ytr, Xte, yte, src = load_mnist_split(n_per_class=per_class, size=8,
                                               test_per_class=20, seed=seed)
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, 8, phase=2.6)
    out = {}
    for name, c in (("IP", 0.0), ("RP", coupling)):
        model = build_rp_image(geo, pix, feat, n_areas, c, N_CLASSES, jax.random.PRNGKey(seed), hp)
        _, ar, ao, _ = train_kei_image(model, geo, Xtr, ytr, Xte, yte, e_res=e_res,
                                       e_open=e_open, batch=batch, open_gain=True,
                                       seed=seed, prefix=f"[{name}] ")
        out[name] = ao
        print(f"    {name}: acc {ar:.3f} -> {ao:.3f}")
    diff = out["RP"] - out["IP"]
    expanded = abs(diff) > abs(PHASE1_DIFF)
    print(f"    RP - IP = {diff:+.3f} (Phase1 u-수준 {PHASE1_DIFF:+.3f} 대비 "
          f"{'확대' if expanded else '유사/축소'})")
    return dict(exp="P2", acc_IP=out["IP"], acc_RP=out["RP"], diff=diff,
                phase1_diff=PHASE1_DIFF, expanded=bool(expanded), coupling=coupling,
                n_areas=n_areas, passed=True)


if __name__ == "__main__":
    run()
