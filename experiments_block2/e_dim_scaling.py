"""E-DIM — 경로 A 차원 스윕 (Phase 3 §3). D∈{8,16,32}, 단일 변수=D.

각 D 에서 세 매듭을 측정: Q1 기하바닥(orth_max), Q2 라우팅(top1), Q3 의미구조(within/across),
Q4 천장(acc_mnist). D 외 모든 것(시드·격자·스케줄) 고정. M=ceil(D/8) Area 로 차원 분할.
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
from lnn.cluster import _make_area  # noqa: E402
from lnn.data.text_corpus import (CATEGORIES, VOCAB_SIZE, make_mask_dataset,  # noqa: E402
                                  noun_token_ids, TOKEN2ID)
from lnn.data.mnist_data import load_mnist_split  # noqa: E402
from lnn.decoders import vocabulary_logits  # noqa: E402
from lnn.encodings import channel_orthogonality  # noqa: E402
from lnn.readout import dijkstra_tmin  # noqa: E402
from lnn_block2.dim_scaling import DimSplitEncoder, dim_split_feats, n_areas_for  # noqa: E402
from lnn_block2.freq_readout import make_bank_mask  # noqa: E402
from lnn_block2.kei import build_rp_image  # noqa: E402
from lnn_block2.rp_pulse_coupling import measure_rho_coupled  # noqa: E402

WINDOW = 20
STRIDE = 20
FREQ_PER_AREA = 8
N_CLASSES = 10


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


def _vocab_sweep(D, seed, R, n_total, e_res, e_open):
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 90}
    M = n_areas_for(D, FREQ_PER_AREA)
    fpa = D // M
    X, y = make_mask_dataset(n_total=n_total, seed=seed)
    ntr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    keys = jax.random.split(jax.random.PRNGKey(seed), M + 1)
    emb = 0.5 * jax.random.normal(keys[0], (VOCAB_SIZE, D))
    carriers = tuple((int(pick_cells(geo, 1, phase=0.8 * m)[0]),) for m in range(M))
    out_cells = pick_cells(geo, 3, phase=2.6)
    enc = DimSplitEncoder(embedding=emb, n_areas=M, freq_per_area=fpa, carriers=carriers,
                          window=WINDOW, stride=STRIDE, P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    areas = tuple(_make_area(geo, keys[1 + m], "processor", carriers[m], out_cells, hp)
                  for m in range(M))
    model = (enc, areas)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    out_arr = jnp.asarray(out_cells)
    nouns = jnp.asarray(noun_token_ids())

    def logits_of(params, Xb, masks):
        e, ars = eqx.combine(params, static)
        feat = dim_split_feats(ars, geo, e.encode(Xb), fpa, WINDOW, masks, out_arr)
        return vocabulary_logits(feat, e.embedding)

    def recompute_windows(mdl):
        e, ars = mdl
        ms = []
        for m, area in enumerate(ars):
            tau = jax.lax.stop_gradient(area.edge_tau(geo))
            tmin = dijkstra_tmin(geo, tau, list(carriers[m]), list(out_cells)).min()
            ms.append(make_bank_mask(hp["n_steps"], WINDOW, float(tmin), hp["P"]))
        return ms

    def loss_fn(params, Xb, Yb, windows, lam):
        logits = logits_of(params, Xb, windows)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res, batch_size=32,
                                    open_terrain=False, open_gain=False, lrs=None,
                                    lam_schedule=train.anneal_lambda(2, 8),
                                    recompute_windows=recompute_windows, seed=seed, log_prefix=f"[D{D} res] ")
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=32,
                                    open_terrain=True, open_gain=False,
                                    lrs=dict(embedding=4e-2, terrain_h=2e-2),
                                    lam_schedule=train.const_lambda(8.0),
                                    recompute_windows=recompute_windows, seed=seed + 1, log_prefix=f"[D{D} open] ")
    params = eqx.partition(model, eqx.is_inexact_array)[0]
    # cost: forward 시간(워밍업 후)
    pf = jax.jit(lambda p, Xb: logits_of(p, Xb, win))
    r = pf(params, jnp.asarray(Xte)); jax.block_until_ready(r)
    t = time.perf_counter(); r = pf(params, jnp.asarray(Xte)); jax.block_until_ready(r)
    fwd = time.perf_counter() - t
    top1 = train.accuracy(np.asarray(r), yte)
    embf = model[0].embedding
    orth_max = float(channel_orthogonality(embf[nouns]))
    within, across = _within_across(embf)
    return dict(M=M, fpa=fpa, orth_max=orth_max, within=within, across=across,
                top1=top1, fwd_sec=round(fwd, 4))


def _mnist_ceiling(D, seed, R, per_class, e_res, e_open):
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 70}
    M = n_areas_for(D, FREQ_PER_AREA)
    Xtr, ytr, Xte, yte, _s = load_mnist_split(n_per_class=per_class, size=8, test_per_class=20, seed=seed)
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, 8, phase=2.6)
    from _kei_common import train_kei_image
    model = build_rp_image(geo, pix, feat, M, 0.4, N_CLASSES, jax.random.PRNGKey(seed), hp)
    _, _ar, ao, _ = train_kei_image(model, geo, Xtr, ytr, Xte, yte, e_res=e_res, e_open=e_open,
                                    batch=32, open_gain=True, seed=seed, prefix=f"[D{D} mnist] ")
    return ao


def _rho_proxy(D, seed, coupling=0.4):
    """R3 M-Area 결합계 ρ(G=1) — D(=M Area) 가 ρ 를 미는지(작은 격자 proxy)."""
    geo = C.make_geometry(3)
    M = n_areas_for(D, FREQ_PER_AREA)
    from lnn.cluster import DEFAULT_HP
    keys = jax.random.split(jax.random.PRNGKey(seed), M)
    areas = tuple(_make_area(geo, keys[m], "processor", pick_cells(geo, 1), pick_cells(geo, 3),
                             dict(DEFAULT_HP)) for m in range(M))
    return float(measure_rho_coupled(areas, geo, coupling, gain_override=1.0))


def run(seed=0, R=5, n_total=240, e_res=3, e_open=12, mnist_per_class=40,
        m_res=3, m_open=10, dims=(8, 16, 32)):
    print(f"[E-DIM] 경로 A 차원 스윕 D={dims} (단일 변수=D, M=ceil(D/{FREQ_PER_AREA}))")
    rows = []
    for D in dims:
        print(f"\n===== D={D} (M={n_areas_for(D, FREQ_PER_AREA)} Areas) =====")
        v = _vocab_sweep(D, seed, R, n_total, e_res, e_open)
        acc = _mnist_ceiling(D, seed, R, mnist_per_class, m_res, m_open)
        rho = _rho_proxy(D, seed)
        row = dict(D=D, **v, acc_mnist=acc, rho=rho)
        rows.append(row)
        print(f"    D={D}: orth_max={v['orth_max']:.3f} top1={v['top1']:.3f} "
              f"within={v['within']:.3f} across={v['across']:.3f} acc_mnist={acc:.3f} "
              f"rho={rho:.3f} fwd={v['fwd_sec']}s")
    return dict(exp="E-DIM", rows=rows, dims=list(dims))


if __name__ == "__main__":
    run()
