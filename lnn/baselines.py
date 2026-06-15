"""Experiment 2-B 비교 기준선 — 작은 MLP (표준 신경망 대조, §2.3).

같은 과제·같은 데이터에 작은 MLP 를 학습해 wall-clock·FLOPs·정확도·파라미터 수를 측정.
산출: "LNN 은 같은 정확도에 MLP 대비 몇 배의 FLOPs·시간을 쓰는가"의 분모(Block III 출발 수치).
"""

from __future__ import annotations

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .instrument import flops_xla


class MLP(eqx.Module):
    layers: list

    def __init__(self, sizes, key):
        keys = jax.random.split(key, len(sizes) - 1)
        self.layers = [eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
                       for i in range(len(sizes) - 1)]

    def __call__(self, x):
        for lyr in self.layers[:-1]:
            x = jax.nn.relu(lyr(x))
        return self.layers[-1](x)


def n_params(model):
    return int(sum(p.size for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))))


def train_mlp(Xtr, ytr, Xte, yte, hidden=(32,), epochs=30, batch=32, lr=1e-2, seed=0):
    """작은 MLP 학습. 반환: dict(acc, params, train_sec, compile_sec, infer_run_sec, flops_fwd)."""
    in_dim = Xtr.shape[1]
    n_cls = int(max(ytr.max(), yte.max())) + 1
    model = MLP((in_dim, *hidden, n_cls), jax.random.PRNGKey(seed))
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    Xtr_j, ytr_j = jnp.asarray(Xtr), jnp.asarray(ytr)

    def loss_fn(m, xb, yb):
        logits = jax.vmap(m)(xb)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(yb.shape[0]), yb])

    @eqx.filter_jit
    def step(m, st, xb, yb):
        loss, g = eqx.filter_value_and_grad(loss_fn)(m, xb, yb)
        up, st = opt.update(g, st, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, up), st, loss

    rng = np.random.default_rng(seed)
    # 워밍업(컴파일) 1회
    t = time.perf_counter()
    model, opt_state, _ = step(model, opt_state, Xtr_j[:batch], ytr_j[:batch])
    jax.block_until_ready(_)
    compile_sec = time.perf_counter() - t

    t = time.perf_counter()
    for _ in range(epochs):
        idx = np.arange(len(Xtr)); rng.shuffle(idx)
        for s in range(0, len(Xtr), batch):
            bi = idx[s:s + batch]
            model, opt_state, _l = step(model, opt_state, Xtr_j[bi], ytr_j[bi])
    jax.block_until_ready(_l)
    train_sec = time.perf_counter() - t

    # 추론 정확도 + 시간
    predict = eqx.filter_jit(lambda m, x: jax.vmap(m)(x))
    Xte_j = jnp.asarray(Xte)
    r = predict(model, Xte_j); jax.block_until_ready(r)
    t = time.perf_counter()
    logits = predict(model, Xte_j); jax.block_until_ready(logits)
    infer_sec = time.perf_counter() - t
    acc = float(np.mean(np.asarray(jnp.argmax(logits, -1)) == np.asarray(yte)))

    # forward FLOPs (단일 샘플)
    flops, _ca = flops_xla(lambda x: jax.vmap(model)(x), Xte_j[:1])

    return dict(acc=acc, params=n_params(model), train_sec=round(train_sec, 3),
                compile_sec=round(compile_sec, 3), infer_run_sec=round(infer_sec, 4),
                flops_fwd=flops, hidden=list(hidden), epochs=epochs)
