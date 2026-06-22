"""E-BUDGET — 실험 I: 최적화 충분성 (Phase 6 §3.1). ~0.56 이 최적화 인공산물인가.

epoch 예산만 ×{1,3,5} 확장(모델·시드·격자·D 고정), PlateauDetector 로 *수렴 도달*을 측정.
I2(핵심): acc 가 0.56 을 넘으면 H-OPT 참(천장 서사 붕괴). 미돌파면 천장 표현적 → 실험 II.
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
from lnn.data.mnist_data import load_mnist_split  # noqa: E402
from lnn.growth import PlateauDetector  # noqa: E402

N_CLASSES = 10
N_FEAT = 12
CEILING = 0.56


def _train(epoch_mult, seed, per_class, base_open, R=5):
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 70, "n_proc": 1}
    Xtr, ytr, Xte, yte, _s = load_mnist_split(n_per_class=per_class, size=8,
                                              test_per_class=30, seed=seed)
    pix = C.map_image_cells(geo, 8, 8)
    from lnn.encodings import ImageEncoder
    enc = ImageEncoder(gen_cells=pix, P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    model = C.build_classifier(geo, enc, N_FEAT, N_CLASSES, jax.random.PRNGKey(seed), hp)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params, Xb, Yb, windows, lam):
        e, cl, h = eqx.combine(params, static)
        _c, u = cl.forward(geo, e.encode(Xb), windows)
        logits = jax.vmap(h)(u)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    rw = lambda m: m[1].make_windows(geo)

    def predict(m, win):
        e, cl, h = m
        _c, u = cl.forward(geo, e.encode(jnp.asarray(Xte)), win)
        return jax.vmap(h)(u)

    model, _hr, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=4, batch_size=32,
                                      open_terrain=False, open_gain=False, lrs=None,
                                      lam_schedule=train.anneal_lambda(2, 8), recompute_windows=rw,
                                      seed=seed, log_prefix=f"[m{epoch_mult} res] ")
    e_open = base_open * epoch_mult
    model, hist, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=32,
                                       open_terrain=True, open_gain=True, lrs=None,
                                       lam_schedule=train.const_lambda(8.0), recompute_windows=rw,
                                       seed=seed + 1, log_prefix=f"[m{epoch_mult} open] ")
    acc = train.accuracy(predict(model, win), yte)
    # 수렴 판정: 마지막 윈도에서 plateau 인가 + 정체 도달 epoch
    det = PlateauDetector(window=max(2, e_open // 6), threshold=0.01)
    conv_epoch = None
    for i, lv in enumerate(hist):
        det.update(lv)
        if conv_epoch is None and det.is_plateau():
            conv_epoch = i
    return dict(epoch_mult=epoch_mult, e_open=e_open, acc=acc,
                converged=bool(conv_epoch is not None), conv_epoch=conv_epoch,
                final_loss=round(hist[-1], 4))


def run(seed=0, per_class=50, base_open=8, mults=(1, 3, 5), do_basin=True):
    print(f"[E-BUDGET] 최적화 충분성 — epoch_mult={mults} (per_class={per_class})")
    rows = [_train(m, seed, per_class, base_open) for m in mults]
    for r in rows:
        print(f"    mult={r['epoch_mult']} (e_open={r['e_open']}): acc={r['acc']:.3f} "
              f"converged={r['converged']}@{r['conv_epoch']} loss={r['final_loss']}")
    best = max(r["acc"] for r in rows)
    broke = best > CEILING
    basin = []
    if (not broke) and do_basin:
        print("    [I3] basin 강건성 (seed×3, 최대 예산)")
        for s in (seed + 10, seed + 20, seed + 30):
            r = _train(mults[-1], s, per_class, base_open)
            basin.append(r["acc"])
            print(f"      seed={s}: acc={r['acc']:.3f}")
    return dict(exp="E-BUDGET", rows=rows, best=best, broke=bool(broke), basin=basin)


if __name__ == "__main__":
    run()
