"""E-INTERIOR — 실험 II: 분산 비선형성 (Phase 6 §3.3). 천장이 '선형 내부' 제약인가 지연 기질인가.

매질 내부에 MLP식 비선형 노드 분산(density: none/sparse/dense). density=none=Phase 5 회귀.
II2(핵심): acc 가 0.83 접근=선형내부 제약(H-INTERIOR), 0.56 정체=지연 기질(7번째 확증).
II3: 중첩 위반 잔차=기둥① 잠식(교환비). II4: signed vs raw(부호 보존 효과·DC).
단자 제외(interior_only) → encode/decode frozen 보존. 단일 변수=density(2차 kind).
"""

from __future__ import annotations

import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import _common as C  # noqa: E402

from lnn import train  # noqa: E402
from lnn.area import pick_cells  # noqa: E402
from lnn.cluster import _make_area  # noqa: E402
from lnn.data.mnist_data import load_mnist_split  # noqa: E402
from lnn.encodings import ImageEncoder  # noqa: E402
from lnn_block2 import interior_nonlin as IN  # noqa: E402

N_CLASSES = 10
N_FEAT = 12
CEILING = 0.56
MLP_REF = 0.83
THETA = 0.1
PLACEMENT = "cell_sum"


def _interior_residual(u_nl, u_lin):
    """기둥① 잠식량: 내부 비선형 출력 vs 같은 모델의 내부-선형화(kind=none) 출력의 상대차.

    density=none → 두 출력 동일 → 0. density↑ → 증가(내부 비선형이 경로-합에서 벗어난 양).
    Phase 5 방식(NL vs 선형화) 계승 — readout 의 argmax/sign 비선형을 상쇄해 내부 효과만 격리.
    """
    num = jnp.linalg.norm(u_nl - u_lin, axis=-1)
    den = jnp.linalg.norm(u_nl, axis=-1) + 1e-9
    return float(jnp.mean(num / den))


def _train(density, kind, seed, per_class, e_res, e_open, R=5):
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 70}
    Xtr, ytr, Xte, yte, _s = load_mnist_split(n_per_class=per_class, size=8,
                                              test_per_class=30, seed=seed)
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, N_FEAT, phase=2.6)
    ke, kh = jax.random.split(jax.random.PRNGKey(seed))
    area = _make_area(geo, ke, "processor", pix, feat, hp)
    enc = ImageEncoder(gen_cells=pix, P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    head = eqx.nn.Linear(N_FEAT, N_CLASSES, key=kh)
    nl_mask = IN.build_nl_mask(geo, tuple(pix) + tuple(feat), density)
    model = (enc, area, head)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def u_of(params, Xb, win):
        e, ar, _h = eqx.combine(params, static)
        return IN.interior_area_forward(ar, geo, e.encode(Xb), win, kind, THETA, PLACEMENT, nl_mask)

    def loss_fn(params, Xb, Yb, windows, lam):
        e, ar, h = eqx.combine(params, static)
        u = IN.interior_area_forward(ar, geo, e.encode(Xb), windows, kind, THETA, PLACEMENT, nl_mask)
        logits = jax.vmap(h)(u)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    rw = lambda m: m[1].tmin_window(geo)

    def predict(m, win):
        e, ar, h = m
        u = IN.interior_area_forward(ar, geo, e.encode(jnp.asarray(Xte)), win, kind, THETA, PLACEMENT, nl_mask)
        return jax.vmap(h)(u)

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res, batch_size=32,
                                    open_terrain=False, open_gain=False, lrs=None,
                                    lam_schedule=train.anneal_lambda(2, 8), recompute_windows=rw,
                                    seed=seed, log_prefix=f"[{density}/{kind} res] ")
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=32,
                                    open_terrain=True, open_gain=True, lrs=None,
                                    lam_schedule=train.const_lambda(8.0), recompute_windows=rw,
                                    seed=seed + 1, log_prefix=f"[{density}/{kind} open] ")
    acc = train.accuracy(predict(model, win), yte)
    params = eqx.partition(model, eqx.is_inexact_array)[0]
    # 기둥① 잠식: 내부 비선형 출력 vs 내부-선형화(kind=none) 출력
    Xr = jnp.asarray(Xte[:16])
    e0, ar0, _h0 = eqx.combine(params, static)
    inj0 = e0.encode(Xr)
    u_nl = IN.interior_area_forward(ar0, geo, inj0, win, kind, THETA, PLACEMENT, nl_mask)
    u_lin = IN.interior_area_forward(ar0, geo, inj0, win, "none", THETA, PLACEMENT, nl_mask)
    resid = _interior_residual(u_nl, u_lin)
    n_nl = int(np.asarray(nl_mask).sum())
    inj1 = model[0].encode(jnp.asarray(Xte[:1]))[0]   # [T, N] 단일 예시 주입
    dc = IN.dc_drift(model[1], geo, inj1, kind, THETA, PLACEMENT, nl_mask) if kind == "relu" else 0.0
    return dict(density=density, kind=kind, acc=acc, residual=round(resid, 4),
                n_nonlin_nodes=n_nl, dc_drift=round(float(dc), 4))


def run(seed=0, per_class=40, e_res=3, e_open=12):
    print("[E-INTERIOR] 분산 비선형 — density {none,sparse,dense} (signed_relu) + kind {signed,raw}@dense")
    rows = []
    for density in ("none", "sparse", "dense"):
        r = _train(density, "signed_relu", seed, per_class, e_res, e_open)
        rows.append(r)
        print(f"    density={density:6s} signed_relu: acc={r['acc']:.3f} resid={r['residual']} "
              f"nodes={r['n_nonlin_nodes']}")
    # 2차: raw ReLU @ dense (부호 보존 효과)
    r_raw = _train("dense", "relu", seed, per_class, e_res, e_open)
    rows.append(r_raw)
    print(f"    density=dense  relu(raw)  : acc={r_raw['acc']:.3f} resid={r_raw['residual']} "
          f"dc_drift={r_raw['dc_drift']}")
    return dict(exp="E-INTERIOR", rows=rows)


if __name__ == "__main__":
    run()
