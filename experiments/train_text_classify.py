"""D1 — 텍스트 분류 (합성 한국어 → 의미 범주 3종). 산출물: 정확도, 혼동행렬 PNG.

Exit gate ①: reservoir 기준선 대비 지형 개방 후 정확도 상승.
"""

from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import _common as C

from lnn import train
from lnn.area import pick_cells
from lnn.data.text_corpus import (N_CLASSES, SEQ_LEN, VOCAB_SIZE,
                                  make_classification_dataset)
from lnn.encodings import TextTimeEncoder

D = 4         # 경로 B 임베딩 차원(작은 D)
N_FEAT = 8
STRIDE = 12


def _predict(model, geo, X, windows):
    enc, clus, head = model
    _c, u = clus.forward(geo, enc.encode(jnp.asarray(X)), windows)
    return jax.vmap(head)(u)


def run(seed=0, R=5, n_total=300, epochs_res=6, epochs_open=16, batch=32):
    print("[D1] 텍스트 분류 (합성 한국어, 3범주)")
    geo = C.make_geometry(R)
    hp = C.classify_hp()
    X, y = make_classification_dataset(n_total=n_total, seed=seed)
    ntr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    print(f"    train={len(Xtr)} test={len(Xte)} vocab={VOCAB_SIZE} N={geo.N}")

    key = jax.random.PRNGKey(seed)
    ke, kb = jax.random.split(key)
    emb = 0.5 * jax.random.normal(ke, (VOCAB_SIZE, D))
    gen_cells = pick_cells(geo, D)
    encoder = TextTimeEncoder(embedding=emb, gen_cells=gen_cells, stride=STRIDE,
                              P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    model = C.build_classifier(geo, encoder, N_FEAT, N_CLASSES, kb, hp)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params, Xb, Yb, windows, lam):
        enc, clus, head = eqx.combine(params, static)
        _c, u = clus.forward(geo, enc.encode(Xb), windows)
        logits = jax.vmap(head)(u)
        ll = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(ll[jnp.arange(Yb.shape[0]), Yb]), logits

    def recompute_windows(m):
        return m[1].make_windows(geo)

    model, _, win = train.run_phase(
        model, static, loss_fn, Xtr, ytr, epochs=epochs_res, batch_size=batch,
        open_terrain=False, open_gain=False, lrs=None,
        lam_schedule=train.anneal_lambda(2.0, 8.0),
        recompute_windows=recompute_windows, seed=seed, log_prefix="[res] ")
    acc_res = train.accuracy(_predict(model, geo, Xte, win), yte)
    print(f"    reservoir test acc = {acc_res:.3f}")

    # §4.3 결정 2: 지형 개방과 함께 임베딩 학습. 임베딩 lr 을 높이고 지형 lr 은 완만히
    # (joint 최적화에서 지형 섭동이 임베딩 학습을 흔드는 것 방지). gain 은 이 단계 미개방.
    model, _, win = train.run_phase(
        model, static, loss_fn, Xtr, ytr, epochs=epochs_open, batch_size=batch,
        open_terrain=True, open_gain=False, lrs=dict(embedding=6e-2, terrain_h=2e-2),
        lam_schedule=train.const_lambda(8.0),
        recompute_windows=recompute_windows, seed=seed + 1, log_prefix="[open] ")
    logits_te = _predict(model, geo, Xte, win)
    acc_open = train.accuracy(logits_te, yte)
    print(f"    opened   test acc = {acc_open:.3f}  (Δ={acc_open - acc_res:+.3f})")

    _save_confusion(logits_te, yte, acc_res, acc_open)
    metrics = dict(demo="D1", acc_reservoir=acc_res, acc_opened=acc_open,
                   improved=bool(acc_open > acc_res), n_classes=N_CLASSES)
    C.save_metrics("D1_text_classify", metrics)
    return metrics


def _save_confusion(logits_te, yte, acc_res, acc_open):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pred = np.asarray(jnp.argmax(logits_te, axis=-1))
    cm = np.zeros((N_CLASSES, N_CLASSES), int)
    for t, p in zip(np.asarray(yte), pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_title(f"D1 confusion (res={acc_res:.2f}→open={acc_open:.2f})")
    ax.set_xlabel("pred"); ax.set_ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "D1_confusion.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    run()
