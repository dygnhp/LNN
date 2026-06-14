"""D2 — 이미지 분류 (sklearn digits 8×8 → class). 산출물: 정확도·혼동행렬·지형 히트맵.

Exit gate ①: reservoir 기준선 대비 지형 개방 후 정확도 상승을 측정한다.
"""

from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import _common as C

from lnn import train
from lnn.data.image_data import load_digits_split
from lnn.decoders import discriminative_logits
from lnn.encodings import ImageEncoder
from lnn.readout import score_from_c

N_CLASSES = 10
N_FEAT = 12


def _predict(model, static, geo, X, windows):
    enc, clus, head = model
    inj = enc.encode(jnp.asarray(X))
    _c, u = clus.forward(geo, inj, windows)   # u = 출력 셀별 부호 있는 정합 피크(§3.7)
    logits = jax.vmap(head)(u)
    return logits


def run(seed=0, R=5, n_per_class=40, epochs_res=6, epochs_open=12, batch=32):
    print("[D2] 이미지 분류 (digits 8×8)")
    geo = C.make_geometry(R)
    hp = C.classify_hp()
    Xtr, ytr, Xte, yte = load_digits_split(n_per_class=n_per_class, seed=seed)
    print(f"    train={len(Xtr)} test={len(Xte)} cells N={geo.N}")

    pix_cells = C.map_image_cells(geo, 8, 8)
    encoder = ImageEncoder(gen_cells=pix_cells, P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    key = jax.random.PRNGKey(seed)
    model = C.build_classifier(geo, encoder, N_FEAT, N_CLASSES, key, hp)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params, Xb, Yb, windows, lam):
        enc, clus, head = eqx.combine(params, static)
        inj = enc.encode(Xb)
        _c, u = clus.forward(geo, inj, windows)   # 부호 있는 정합 피크 features
        logits = jax.vmap(head)(u)
        ll = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(ll[jnp.arange(Yb.shape[0]), Yb]), logits

    def recompute_windows(m):
        return m[1].make_windows(geo)

    # ── 1) Reservoir 기준선 ──────────────────────────────────────────────
    model, _, win = train.run_phase(
        model, static, loss_fn, Xtr, ytr,
        epochs=epochs_res, batch_size=batch, open_terrain=False, open_gain=False,
        lrs=None, lam_schedule=train.anneal_lambda(2.0, 8.0),
        recompute_windows=recompute_windows, seed=seed, log_prefix="[res] ")
    acc_res = train.accuracy(_predict(model, static, geo, Xte, win), yte)
    print(f"    reservoir test acc = {acc_res:.3f}")

    # ── 2) 지형 개방 (+ gain) ────────────────────────────────────────────
    model, _, win = train.run_phase(
        model, static, loss_fn, Xtr, ytr,
        epochs=epochs_open, batch_size=batch, open_terrain=True, open_gain=True,
        lrs=None, lam_schedule=train.const_lambda(8.0),
        recompute_windows=recompute_windows, seed=seed + 1, log_prefix="[open] ")
    logits_te = _predict(model, static, geo, Xte, win)
    acc_open = train.accuracy(logits_te, yte)
    print(f"    opened   test acc = {acc_open:.3f}  (Δ={acc_open - acc_res:+.3f})")

    _save_outputs(model, geo, logits_te, yte, acc_res, acc_open)
    metrics = dict(demo="D2", acc_reservoir=acc_res, acc_opened=acc_open,
                   improved=bool(acc_open > acc_res), n_classes=N_CLASSES)
    C.save_metrics("D2_image_classify", metrics)
    return metrics


def _save_outputs(model, geo, logits_te, yte, acc_res, acc_open):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 혼동행렬
    pred = np.asarray(jnp.argmax(logits_te, axis=-1))
    cm = np.zeros((N_CLASSES, N_CLASSES), int)
    for t, p in zip(np.asarray(yte), pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"D2 confusion (res={acc_res:.2f} → open={acc_open:.2f})")
    ax.set_xlabel("pred")
    ax.set_ylabel("true")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "D2_confusion.png"), dpi=110)
    plt.close(fig)

    # 지형 히트맵 (processor area)
    from lnn import fields
    clus = model[1]
    area = clus.areas[len(clus.areas) // 2]
    pos = np.asarray(geo.pos)
    gx = np.linspace(pos[:, 0].min(), pos[:, 0].max(), 60)
    gy = np.linspace(pos[:, 1].min(), pos[:, 1].max(), 60)
    GX, GY = np.meshgrid(gx, gy)
    P = jnp.asarray(np.stack([GX.ravel(), GY.ravel()], -1), jnp.float32)
    T = np.asarray(fields.terrain_value(P, area.terrain_h, area.terrain_c, area.terrain_sigma))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(T.reshape(60, 60), origin="lower", cmap="terrain",
                   extent=[gx.min(), gx.max(), gy.min(), gy.max()])
    ax.scatter(pos[:, 0], pos[:, 1], s=4, c="k", alpha=0.3)
    ax.set_title("D2 learned terrain T (processor)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "D2_terrain.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    run()
