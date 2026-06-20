"""E-CELL — 격자 부피 스윕 (Phase 4 §4). R∈{5,8,12}, K 동반(K_per_cell·N), n_steps∝R.

다섯 번째 축(cell 수)으로 ~0.56 천장이 물리 절대 한계인지 공간 규모 한계인지 판별.
C1 path_diversity(전제)·C2 acc_mnist(핵심)·C3 stability·C4 cost(정직 측정).
단일 변수=R(K·n_steps 는 R 종속). MLP 0.83 기준선과 대조.
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import _common as C  # noqa: E402

from lnn import train  # noqa: E402
from lnn.area import pick_cells  # noqa: E402
from lnn.cluster import build_serial_cluster  # noqa: E402
from lnn.data.text_corpus import (CATEGORIES, VOCAB_SIZE, make_mask_dataset,  # noqa: E402
                                  noun_token_ids, TOKEN2ID)
from lnn.decoders import vocabulary_logits  # noqa: E402
from lnn.encodings import TextTimeEncoder, channel_orthogonality  # noqa: E402
from lnn_block2.cell_scaling import CellScaler, free_evolution_stability, path_diversity  # noqa: E402
from run_experiment2 import measure_cost, train_image_stage  # noqa: E402

CEILING = 0.56
MLP_REF = 0.83


def _within_across(embedding):
    cat_of = {TOKEN2ID[n]: c for c, d in CATEGORIES.items() for n in d["nouns"]}
    ids = noun_token_ids()
    codes = np.asarray(embedding[jnp.asarray(ids)])
    n = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
    g = np.abs(n @ n.T)
    win, acr = [], []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            (win if cat_of[ids[a]] == cat_of[ids[b]] else acr).append(g[a, b])
    return float(np.mean(win)), float(np.mean(acr))


def _vocab_at_R(R, K, n_steps, seed, n_total, e_res, e_open):
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_channels": 8, "n_steps": n_steps,
          "n_hills_terrain": K, "n_hills_gain": max(4, K // 2)}
    X, y = make_mask_dataset(n_total=n_total, seed=seed)
    ntr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    import equinox as eqx
    ke, kc = jax.random.split(jax.random.PRNGKey(seed))
    emb = 0.5 * jax.random.normal(ke, (VOCAB_SIZE, 8))
    gen = pick_cells(geo, 8, phase=0.0)
    dec = pick_cells(geo, 8, phase=2.6)
    enc = TextTimeEncoder(embedding=emb, gen_cells=gen, stride=12, P=hp["P"],
                          n_steps=n_steps, n_cells=geo.N)
    clus = build_serial_cluster(geo, gen, dec, kc, hp)
    model = (enc, clus)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    nouns = jnp.asarray(noun_token_ids())

    def logits_of(params, Xb, win):
        e, cl = eqx.combine(params, static)
        _c, u = cl.forward(geo, e.encode(Xb), win)
        return vocabulary_logits(u, e.embedding)

    def loss_fn(params, Xb, Yb, win, lam):
        lg = logits_of(params, Xb, win)
        return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(Yb.shape[0]), Yb]), lg

    rw = lambda m: m[1].make_windows(geo)
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res, batch_size=32,
                                    open_terrain=False, open_gain=False, lrs=None,
                                    lam_schedule=train.anneal_lambda(2, 8), recompute_windows=rw,
                                    seed=seed, log_prefix=f"[R{R} voc-res] ")
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=32,
                                    open_terrain=True, open_gain=False,
                                    lrs=dict(embedding=4e-2, terrain_h=2e-2),
                                    lam_schedule=train.const_lambda(8.0), recompute_windows=rw,
                                    seed=seed + 1, log_prefix=f"[R{R} voc-open] ")
    params = eqx.partition(model, eqx.is_inexact_array)[0]
    top1 = train.accuracy(logits_of(params, jnp.asarray(Xte), win), yte)
    within, across = _within_across(model[0].embedding)
    return top1, within, across


def run(seed=0, mnist_per_class=30, m_res=3, m_open=10,
        voc_total=200, v_res=2, v_open=8, Rs=(5, 8, 12)):
    print(f"[E-CELL] 격자 부피 스윕 R={Rs} (K 동반, n_steps∝R, 단일 변수=R)")
    rows = []
    for R in Rs:
        cs = CellScaler(R)
        print(f"\n===== R={R} (N={cs.n_cells} K={cs.K} n_steps={cs.n_steps}) =====")
        stage = dict(name=f"R{R}", per_class=mnist_per_class, R=R, size=8, n_steps=cs.n_steps,
                     batch=32, e_res=m_res, e_open=m_open, K_terrain=cs.K, K_gain=max(4, cs.K // 2))
        res = train_image_stage(stage)
        cost = measure_cost(res, stage)
        pdiv = path_diversity(R, seed=seed)
        stab = free_evolution_stability(R, seed=seed)
        top1, within, across = _vocab_at_R(R, cs.K, cs.n_steps, seed, voc_total, v_res, v_open)
        row = dict(R=R, n_cells=cs.n_cells, K=cs.K, n_steps=cs.n_steps,
                   acc_mnist=res["acc_open"], acc_res=res["acc_res"],
                   path_diversity=round(pdiv, 3), stability=round(stab, 3),
                   top1=top1, within=round(within, 3), across=round(across, 3),
                   fwd_sec=cost["time"]["fwd_run_sec"],
                   flops_xla=cost["flops"]["method1_xla"],
                   flops_analytic=cost["flops"]["method2_analytic"],
                   params=cost["model_config"]["params"]["total_trainable"])
        rows.append(row)
        print(f"    R={R}: acc_mnist={res['acc_open']:.3f} pdiv={pdiv:.2f} top1={top1:.3f} "
              f"within={within:.3f} stab={stab:.2e} fwd={cost['time']['fwd_run_sec']}s "
              f"params={row['params']}")
    return dict(exp="E-CELL", rows=rows, Rs=list(Rs))


if __name__ == "__main__":
    run()
