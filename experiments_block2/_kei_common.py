"""E2/E3 공통 — KEIImage 학습 헬퍼 (reservoir → 지형개방, ARIS run_phase 재사용)."""

from __future__ import annotations

import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

from lnn import train  # noqa: E402


def train_kei_image(model, geo, Xtr, ytr, Xte, yte, *, e_res, e_open, batch,
                    open_gain, seed=0, prefix=""):
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params, Xb, Yb, windows, lam):
        m = eqx.combine(params, static)
        logits = m.forward(geo, Xb, windows)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    rw = lambda m: m.make_windows(geo)

    def predict(m, win):
        return m.forward(geo, jnp.asarray(Xte), win)

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res,
                                    batch_size=batch, open_terrain=False, open_gain=False,
                                    lrs=None, lam_schedule=train.anneal_lambda(2, 8),
                                    recompute_windows=rw, seed=seed, log_prefix=prefix + "[res] ")
    acc_res = train.accuracy(predict(model, win), yte)
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open,
                                    batch_size=batch, open_terrain=True, open_gain=open_gain,
                                    lrs=None, lam_schedule=train.const_lambda(8.0),
                                    recompute_windows=rw, seed=seed + 1, log_prefix=prefix + "[open] ")
    acc_open = train.accuracy(predict(model, win), yte)
    return model, acc_res, acc_open, win
