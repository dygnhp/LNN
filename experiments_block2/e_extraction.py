"""E-EXTRACT — 추출 층 재설계 사다리 (Phase 7). 생성 코어 불변, readout 만 재설계.

F1(핵심): Cross-Query 가 O4(동사→명사) top1 을 static 대비 올리는가(진단 직접 검증).
F3: 정합필터 뱅크 랭크 M↑ 가 MNIST acc 를 올리는가. F6(FINAL): 결합 best acc vs 0.56.
F4: 능력 중 u-밖 비율(기둥② 교환비).
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

from lnn import dynamics, train  # noqa: E402
from lnn.area import pick_cells  # noqa: E402
from lnn.cluster import _make_area  # noqa: E402
from lnn.data.mnist_data import load_mnist_split  # noqa: E402
from lnn.data.text_corpus_variant import (VOCAB_SIZE, make_mask_dataset_variant,  # noqa: E402
                                          noun_token_ids_variant)
from lnn.decoders import vocabulary_logits  # noqa: E402
from lnn.encodings import TextTimeEncoder, ImageEncoder  # noqa: E402
from lnn.readout import matched_filter, signed_readout, wavelet  # noqa: E402
from lnn_block2.boundary_split import u_outside_energy  # noqa: E402
from lnn_block2.dynamic_query import make_cross_query, signed_query_readout  # noqa: E402
from lnn_block2.readout_bank import make_bank  # noqa: E402

D = 8
P = 8
CEILING = 0.56


# ───────────────────────── F1 — Cross-Query on O4 ──────────────────────────
def run_F1(seed=0, n_total=300, e_res=4, e_open=16):
    geo = C.make_geometry(5)
    hp = {**C.classify_hp(), "n_channels": D, "n_steps": 90}
    X, y = make_mask_dataset_variant(n_total=n_total, seed=seed, variant_level="L2")
    ntr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    ke, ka, kq = jax.random.split(jax.random.PRNGKey(seed), 3)
    emb = 0.5 * jax.random.normal(ke, (VOCAB_SIZE, D))
    gen = pick_cells(geo, D, phase=0.0)
    out = pick_cells(geo, D, phase=2.6)
    encoder = TextTimeEncoder(embedding=emb, gen_cells=gen, stride=12, P=P,
                              n_steps=hp["n_steps"], n_cells=geo.N)
    area = _make_area(geo, ka, "processor", gen, out, hp)
    out_arr = jnp.asarray(out)

    def _train(mode):  # "static" | "cross"
        cq = make_cross_query(P, D, kq)
        model = (encoder, area, cq) if mode == "cross" else (encoder, area)
        _, static = eqx.partition(model, eqx.is_inexact_array)

        def logits_of(params, Xb, win):
            parts = eqx.combine(params, static)
            e, ar = parts[0], parts[1]
            sc = ar.step_constants(geo)
            inj = e.encode(Xb)
            if mode == "cross":
                q = parts[2].query(e.embedding[Xb[:, 3]])         # 동사(pos3) 문맥 → 질문 [B,P]

                def single(inj_TN, q_vec):
                    o = dynamics.rollout(sc, inj_TN, out_arr, geo.N)
                    return signed_query_readout(o, q_vec, win)
                u = jax.vmap(single)(inj, q)                      # [B, D]
            else:
                w = wavelet(P)

                def single(inj_TN):
                    o = dynamics.rollout(sc, inj_TN, out_arr, geo.N)
                    return signed_readout(matched_filter(o, w), win)
                u = jax.vmap(single)(inj)
            return vocabulary_logits(u, e.embedding)

        def loss_fn(params, Xb, Yb, windows, lam):
            lg = logits_of(params, Xb, windows)
            return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(Yb.shape[0]), Yb]), lg

        rw = lambda m: (m[1] if isinstance(m, tuple) else m).tmin_window(geo)
        model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res, batch_size=32,
                                        open_terrain=False, open_gain=False, lrs=None,
                                        lam_schedule=train.anneal_lambda(2, 8), recompute_windows=rw,
                                        seed=seed, log_prefix=f"[F1 {mode} res] ")
        model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=32,
                                        open_terrain=True, open_gain=False,
                                        lrs=dict(embedding=4e-2, terrain_h=2e-2, decoder_code=2e-2),
                                        lam_schedule=train.const_lambda(8.0), recompute_windows=rw,
                                        seed=seed + 1, log_prefix=f"[F1 {mode} open] ")
        params = eqx.partition(model, eqx.is_inexact_array)[0]
        return train.accuracy(logits_of(params, jnp.asarray(Xte), win), yte)

    static_top1 = _train("static")
    cross_top1 = _train("cross")
    print(f"    F1 O4 top1: static={static_top1:.3f}  cross-query={cross_top1:.3f}  "
          f"(Δ{cross_top1 - static_top1:+.3f})")
    return dict(static_top1=static_top1, cross_top1=cross_top1,
                improved=bool(cross_top1 > static_top1 + 0.02))


# ───────────────────────── F3/F6/F4 — bank on MNIST ────────────────────────
def run_bank(M, seed=0, per_class=30, e_res=3, e_open=12):
    geo = C.make_geometry(5)
    hp = {**C.classify_hp(), "n_steps": 70}
    Xtr, ytr, Xte, yte, _s = load_mnist_split(n_per_class=per_class, size=8, test_per_class=30, seed=seed)
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, 8, phase=2.6)
    ke, ka, kh = jax.random.split(jax.random.PRNGKey(seed), 3)
    area = _make_area(geo, ka, "processor", pix, feat, hp)
    enc = ImageEncoder(gen_cells=pix, P=P, n_steps=hp["n_steps"], n_cells=geo.N)
    bank = make_bank(M, P, ke, learnable_scale=0.1)
    head = eqx.nn.Linear(8 * M, 10, key=kh)
    out_arr = jnp.asarray(feat)
    model = (enc, area, bank, head)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def feats_of(params, Xb, win):
        e, ar, bk, _h = eqx.combine(params, static)
        sc = ar.step_constants(geo)
        inj = e.encode(Xb)

        def single(inj_TN):
            o = dynamics.rollout(sc, inj_TN, out_arr, geo.N)
            return bk.read(o, win).reshape(-1)       # [8*M] 전폭 r
        return jax.vmap(single)(inj)

    def loss_fn(params, Xb, Yb, windows, lam):
        e, ar, bk, h = eqx.combine(params, static)
        r = feats_of(params, Xb, windows)
        lg = jax.vmap(h)(r)
        return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(Yb.shape[0]), Yb]), lg

    rw = lambda m: m[1].tmin_window(geo)

    def predict(params, win):
        e, ar, bk, h = eqx.combine(params, static)
        r = feats_of(params, jnp.asarray(Xte), win)
        return jax.vmap(h)(r)

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res, batch_size=32,
                                    open_terrain=False, open_gain=False, lrs=None,
                                    lam_schedule=train.anneal_lambda(2, 8), recompute_windows=rw,
                                    seed=seed, log_prefix=f"[bank M{M} res] ")
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=32,
                                    open_terrain=True, open_gain=True,
                                    lrs=dict(decoder_code=1e-2), lam_schedule=train.const_lambda(8.0),
                                    recompute_windows=rw, seed=seed + 1, log_prefix=f"[bank M{M} open] ")
    params = eqx.partition(model, eqx.is_inexact_array)[0]
    acc = train.accuracy(predict(params, win), yte)
    # F4: u-밖 에너지(전폭 r 중 u_dim=8 사영 밖 비율)
    r = feats_of(params, jnp.asarray(Xte[:32]), win)            # [B, 8M]
    Rdim = r.shape[1]
    udim = min(8, Rdim)
    proj = jnp.asarray(np.linalg.qr(np.random.default_rng(0).normal(size=(Rdim, udim)))[0].T)
    u_out = u_outside_energy(r, proj)
    return dict(M=M, acc=acc, u_outside=round(u_out, 3))


if __name__ == "__main__":
    run_F1(n_total=120, e_res=2, e_open=6)
    print(run_bank(4, per_class=12, e_res=1, e_open=3))
