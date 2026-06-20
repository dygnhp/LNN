"""E-DEPTH — 비선형 깊이 스윕 (Phase 5 §4). depth∈{1,2,4,6}, 방식 A·C 둘 다.

A(경계 깊이 L 확장, 독립 지형) 주 + C(RAL 시간 순환, 파라미터 공유) 핵심 대조.
G1 천장(acc)·G2 해석 대가(경로분해 잔차)·G4 의미(within/across)·G5 A vs C(깊이의 출처).
acc 와 경로분해 잔차를 함께 측정 — 능력-해석 trade-off.
"""

from __future__ import annotations

import os
import sys
import time

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
from lnn.data.mnist_data import load_mnist_split  # noqa: E402
from lnn.instrument import count_params  # noqa: E402
from lnn_block2.depth_scaling import build_depth_classifier, path_decomp_residual  # noqa: E402
from lnn_block2.ral import build_ral_classifier, free_loop_growth  # noqa: E402

N_CLASSES = 10
N_FEAT = 8
CEILING = 0.56
MLP_REF = 0.83


def _data(per_class, seed):
    Xtr, ytr, Xte, yte, _s = load_mnist_split(n_per_class=per_class, size=8,
                                              test_per_class=20, seed=seed)
    return Xtr, ytr, Xte, yte


def _train_A(L, geo, hp, data, e_res, e_open, seed):
    Xtr, ytr, Xte, yte = data
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, N_FEAT, phase=2.6)
    model = build_depth_classifier(geo, pix, feat, N_CLASSES, L, jax.random.PRNGKey(seed), hp)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params, Xb, Yb, windows, lam):
        enc, clus, head = eqx.combine(params, static)
        _c, u = clus.forward(geo, enc.encode(Xb), windows)
        logits = jax.vmap(head)(u)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    rw = lambda m: m[1].make_windows(geo)

    def predict(m, win):
        enc, clus, head = m
        _c, u = clus.forward(geo, enc.encode(jnp.asarray(Xte)), win)
        return jax.vmap(head)(u)

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res, batch_size=32,
                                    open_terrain=False, open_gain=False, lrs=None,
                                    lam_schedule=train.anneal_lambda(2, 8), recompute_windows=rw,
                                    seed=seed, log_prefix=f"[A L{L} res] ")
    t = time.perf_counter()
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=32,
                                    open_terrain=True, open_gain=True, lrs=None,
                                    lam_schedule=train.const_lambda(8.0), recompute_windows=rw,
                                    seed=seed + 1, log_prefix=f"[A L{L} open] ")
    train_sec = time.perf_counter() - t
    acc = train.accuracy(predict(model, win), yte)
    # 경로분해 잔차(G2)
    enc, clus, _h = model
    resid = path_decomp_residual(clus, geo, enc.encode(jnp.asarray(Xte[:16])), win)
    return dict(method="A", depth=L, acc=acc, residual=round(resid, 4),
                params=count_params(model)["total_trainable"], train_sec=round(train_sec, 1))


def _train_C(n, geo, hp, data, e_res, e_open, seed):
    Xtr, ytr, Xte, yte = data
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, N_FEAT, phase=2.6)
    model = build_ral_classifier(geo, pix, feat, N_CLASSES, n, jax.random.PRNGKey(seed), hp)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params, Xb, Yb, windows, lam):
        m = eqx.combine(params, static)
        logits = m.forward(geo, Xb, windows[0])
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    rw = lambda m: [m.make_window(geo)]

    def predict(m, win):
        return m.forward(geo, jnp.asarray(Xte), win[0])

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res, batch_size=32,
                                    open_terrain=False, open_gain=False, lrs=None,
                                    lam_schedule=train.anneal_lambda(2, 8), recompute_windows=rw,
                                    seed=seed, log_prefix=f"[C n{n} res] ")
    t = time.perf_counter()
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=32,
                                    open_terrain=True, open_gain=True, lrs=None,
                                    lam_schedule=train.const_lambda(8.0), recompute_windows=rw,
                                    seed=seed + 1, log_prefix=f"[C n{n} open] ")
    train_sec = time.perf_counter() - t
    acc = train.accuracy(predict(model, win), yte)
    growth, _ = free_loop_growth(model, geo, jnp.asarray(Xte[:8]), win[0], jnp.tanh, 1.0)
    return dict(method="C", depth=n, acc=acc, params=count_params(model)["total_trainable"],
                loop_growth=round(float(growth), 3), train_sec=round(train_sec, 1))


def run(seed=0, R=5, per_class=20, e_res=2, e_open=8, depths=(1, 2, 4, 6)):
    print(f"[E-DEPTH] 비선형 깊이 스윕 depth={depths} (방식 A·C)")
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 70}
    data = _data(per_class, seed)
    A, Cc = [], []
    for d in depths:
        print(f"\n===== depth={d} =====")
        a = _train_A(d, geo, hp, data, e_res, e_open, seed)
        c = _train_C(d, geo, hp, data, e_res, e_open, seed)
        A.append(a)
        Cc.append(c)
        print(f"    A(L={d}): acc={a['acc']:.3f} resid={a['residual']} params={a['params']} | "
              f"C(n={d}): acc={c['acc']:.3f} params={c['params']} growth={c['loop_growth']}")
    return dict(exp="E-DEPTH", depths=list(depths), A=A, C=Cc)


if __name__ == "__main__":
    run()
