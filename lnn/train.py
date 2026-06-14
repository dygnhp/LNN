"""§6 학습 — optax multi_transform 파라미터 그룹 + 커리큘럼 + 손실.

파라미터 그룹(§6): ``{terrain_h, gain_a, embedding, decoder_head, diag_gain}`` 분리.
지형 RBF 중심/폭·이득장 중심/폭은 동결(label "frozen" → ``set_to_zero``).

커리큘럼(§6):
1. **Reservoir 기준선**: 지형/이득 고정, readout/디코더(embedding·head·diag_gain)만 학습.
2. **코히어런스 어닐링**: logsumexp λ 를 작게(매끄러운 손실)→8 로 키운다(피크 선명화).
3. **지형 개방**: terrain_h → (선택) gain_a 동시 학습.

손실 함수는 실험에서 self-contained 클로저로 만들어 ``run_phase`` 에 직접 넘긴다.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

# ─────────────────────────────── 파라미터 그룹 ────────────────────────────────

_GROUP_OF_FIELD = {
    "terrain_h": "terrain_h",
    "gain_a": "gain_a",
    "embedding": "embedding",
    "diag_gains": "diag_gain",
    "weight": "decoder_head",
    "bias": "decoder_head",
}

DEFAULT_LRS = dict(terrain_h=5e-2, gain_a=2e-2, embedding=2e-2,
                   decoder_head=1e-2, diag_gain=1e-2)


def build_labels(params, open_terrain: bool, open_gain: bool):
    """params 와 같은 구조의 그룹 라벨 트리. 동결 단계는 'frozen'."""

    def label(path, _leaf):
        names = [k.name for k in path if isinstance(k, jax.tree_util.GetAttrKey)]
        last = names[-1] if names else ""
        grp = _GROUP_OF_FIELD.get(last, "frozen")
        if grp == "terrain_h" and not open_terrain:
            return "frozen"
        if grp == "gain_a" and not open_gain:
            return "frozen"
        # §4.3 결정 2: reservoir 단계에선 임베딩 고정, 지형 개방과 함께 학습
        # (파형과 그 라우팅이 정합해야 하므로). decoder_head·diag_gain 은 readout 으로 항상 학습.
        if grp == "embedding" and not open_terrain:
            return "frozen"
        return grp

    return jax.tree_util.tree_map_with_path(label, params)


def make_optimizer(params, lrs, open_terrain, open_gain):
    """그룹별 adam + 동결 set_to_zero 의 multi_transform."""
    lrs = {**DEFAULT_LRS, **(lrs or {})}
    labels = build_labels(params, open_terrain, open_gain)
    transforms = {
        "terrain_h": optax.adam(lrs["terrain_h"]),
        "gain_a": optax.adam(lrs["gain_a"]),
        "embedding": optax.adam(lrs["embedding"]),
        "decoder_head": optax.adam(lrs["decoder_head"]),
        "diag_gain": optax.adam(lrs["diag_gain"]),
        "frozen": optax.set_to_zero(),
    }
    opt = optax.multi_transform(transforms, labels)
    return opt, opt.init(params)


# ─────────────────────────────── λ 스케줄 ────────────────────────────────────


def anneal_lambda(lo=1.0, hi=8.0):
    def sched(ep, epochs):
        return hi if epochs <= 1 else lo + (hi - lo) * ep / (epochs - 1)
    return sched


def const_lambda(v=8.0):
    return lambda ep, epochs: v


# ─────────────────────────────────── 루프 ────────────────────────────────────


def _batches(n, batch_size, rng):
    idx = np.arange(n)
    rng.shuffle(idx)
    for s in range(0, n, batch_size):
        yield idx[s:s + batch_size]


def run_phase(model, static, loss_fn, X, Y, *, epochs, batch_size,
              open_terrain, open_gain, lrs, lam_schedule, recompute_windows,
              seed=0, log_prefix=""):
    """한 커리큘럼 단계 학습.

    loss_fn(params, Xb, Yb, windows, lam) -> (scalar_loss, aux). model 갱신본·손실이력 반환.
    """
    rng = np.random.default_rng(seed)
    params, _ = eqx.partition(model, eqx.is_inexact_array)
    opt, opt_state = make_optimizer(params, lrs, open_terrain, open_gain)
    grad_fn = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    Xj, Yj = jnp.asarray(X), jnp.asarray(Y)
    history = []
    windows = recompute_windows(model)
    for ep in range(epochs):
        lam = float(lam_schedule(ep, epochs))
        if ep > 0 and ep % 5 == 0:
            windows = recompute_windows(model)  # 지형 변화에 윈도 갱신(generous, stop_gradient)
        ep_loss, nb = 0.0, 0
        for bi in _batches(len(X), batch_size, rng):
            params, _ = eqx.partition(model, eqx.is_inexact_array)
            (lval, _aux), grads = grad_fn(params, Xj[bi], Yj[bi], windows, lam)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            model = eqx.combine(params, static)
            ep_loss += float(lval)
            nb += 1
        history.append(ep_loss / max(nb, 1))
        if ep == 0 or ep == epochs - 1:
            print(f"    {log_prefix}ep{ep:02d} loss={history[-1]:.4f} lam={lam:.2f}")
    return model, history, windows


def accuracy(logits, y):
    return float(np.mean(np.asarray(jnp.argmax(logits, axis=-1)) == np.asarray(y)))
