"""실험 공통 — 기하, 픽셀↔셀 매핑, 모델 빌더, 평가/저장 유틸."""

from __future__ import annotations

import json
import os
import sys

try:  # Windows 콘솔(cp949)에서 한글·기호 출력 깨짐/크래시 방지
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lnn.area import pick_cells  # noqa: E402
from lnn.cluster import DEFAULT_HP, build_serial_cluster  # noqa: E402
from lnn.geometry import build_geometry  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def classify_hp():
    return {**DEFAULT_HP, "n_steps": 70, "n_channels": 8, "n_proc": 1}


def gen_hp():
    return {**DEFAULT_HP, "n_steps": 60, "n_channels": 8, "n_proc": 1}


def map_image_cells(geo, H=8, W=8):
    """HxW 픽셀을 격자 셀에 공간 대응(고유, 결정론적). 초기 파면 = 이미지 구조."""
    pos = np.asarray(geo.pos)
    mn = pos.min(0)
    ext = pos.max(0) - mn
    ext[ext == 0] = 1.0
    gp = (pos - mn) / ext
    cells, used = [], set()
    for r in range(H):
        for c in range(W):
            ty, tx = r / (H - 1), c / (W - 1)
            d = (gp[:, 0] - tx) ** 2 + (gp[:, 1] - ty) ** 2
            for idx in np.argsort(d):
                if int(idx) not in used:
                    used.add(int(idx))
                    cells.append(int(idx))
                    break
    return tuple(cells)


def build_classifier(geo, encoder, n_feat, n_classes, key, hp):
    """(encoder, cluster, head) — discriminative. head: Linear(n_feat→n_classes)."""
    k1, k2 = jax.random.split(key)
    dec_out = pick_cells(geo, n_feat, phase=2.6)
    cluster = build_serial_cluster(geo, encoder.gen_cells, dec_out, k1, hp)
    head = eqx.nn.Linear(n_feat, n_classes, key=k2)
    return (encoder, cluster, head)


def save_metrics(name, metrics):
    path = os.path.join(OUT_DIR, f"{name}_metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return path


def make_geometry(R):
    return build_geometry(R)
