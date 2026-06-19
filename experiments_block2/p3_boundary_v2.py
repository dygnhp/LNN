"""P3 — 통합 후 천장 재시험 (§3, 가장 중요). 0.56 돌파 or 미돌파 정량 확정(둘 다 통과).

작업 2(RP 펄스 수준 공유)를 켠 다중 Area KEI 로 MNIST 8×8·14×14 재학습, Block I 천장 0.56·
Phase 1 KEI 0.557 과 대조. (작업 1 임베딩 분산은 토큰 어휘 전용 → P1 에서 측정. 이미지 천장
시험의 통합 대상은 이미지에 적용되는 펄스 수준 공유다.)

판정: >0.56 돌파면 '완전통합이 경계를 민다'; ~0.56 정체면 '지연-전용+위상-제약 물리 상한'의
세 번째 독립 확증(용량 Exp3 / 구조 Phase1 / 완전통합 Phase2).
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
from lnn_block2.kei import build_rp_image  # noqa: E402
from _kei_common import train_kei_image  # noqa: E402

N_CLASSES = 10
CEILING = 0.56


def _one(geo, size, per_class, n_areas, coupling, e_res, e_open, batch, seed):
    hp = {**C.classify_hp(), "n_steps": 70}
    Xtr, ytr, Xte, yte, src = load_mnist_split(n_per_class=per_class, size=size,
                                               test_per_class=30, seed=seed)
    pix = C.map_image_cells(geo, size, size)
    feat = pick_cells(geo, 10, phase=2.6)
    model = build_rp_image(geo, pix, feat, n_areas, coupling, N_CLASSES,
                           jax.random.PRNGKey(seed), hp)
    _, ar, ao, _ = train_kei_image(model, geo, Xtr, ytr, Xte, yte, e_res=e_res,
                                   e_open=e_open, batch=batch, open_gain=True,
                                   seed=seed, prefix=f"[{size}x{size}] ")
    return ar, ao


def run(seed=0, per_class=100, n_areas=3, coupling=0.4, e_res=5, e_open=16, batch=32):
    print(f"[P3] 통합 천장 재시험 (RP 펄스 공유 c={coupling}, n_areas={n_areas})")
    res = {}
    # 8x8 (R=5) · 14x14 (R=8) — 규모 의존 단서
    ar8, ao8 = _one(C.make_geometry(5), 8, per_class, n_areas, coupling, e_res, e_open, batch, seed)
    print(f"    8x8 : {ar8:.3f} -> {ao8:.3f}")
    ar14, ao14 = _one(C.make_geometry(8), 14, per_class, n_areas, coupling, e_res, e_open, batch, seed)
    print(f"    14x14: {ar14:.3f} -> {ao14:.3f}")
    best = max(ao8, ao14)
    broke = best > CEILING
    print(f"    best {best:.3f} vs Block I {CEILING} ({'돌파' if broke else '미돌파'}); "
          f"Phase1 KEI=0.557")
    return dict(exp="P3", acc_8x8=ao8, acc_14x14=ao14, best=best, ceiling=CEILING,
                phase1_kei=0.557, broke_ceiling=bool(broke), passed=True)


if __name__ == "__main__":
    run()
