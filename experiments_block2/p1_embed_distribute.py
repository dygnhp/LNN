"""P1 — 임베딩 분산 (§4). orth_after 0.996 → <0.5 목표(colinear 붕괴 해소).

D3 빈칸 채우기를 M개 Area로 분산: 각 Area 가 임베딩의 D/M 차원만 담당(경로 A 주파수 뱅크).
임베딩만 분산하고 나머지는 Phase 1 그대로 → colinear 해소를 격리 측정.
"""

from __future__ import annotations

import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import _common as C  # noqa: E402

from lnn import dynamics, train  # noqa: E402
from lnn.area import pick_cells  # noqa: E402
from lnn.cluster import _make_area  # noqa: E402
from lnn.data.text_corpus import (VOCAB_SIZE, make_mask_dataset, noun_token_ids)  # noqa: E402
from lnn.decoders import vocabulary_logits  # noqa: E402
from lnn.encodings import channel_orthogonality  # noqa: E402
from lnn.readout import dijkstra_tmin  # noqa: E402
from lnn_block2.distributed_embedding import (DistributedEmbedding,  # noqa: E402
                                              orthogonality_penalty)
from lnn_block2.freq_encoding import ofdm_basis  # noqa: E402
from lnn_block2.freq_readout import make_bank_mask, signed_bank_readout  # noqa: E402

D = 8
M = 4              # Area 수 (각 dm=2 차원 담당)
WINDOW = 20
STRIDE = 20


def run(seed=0, R=5, n_total=240, e_res=4, e_open=16):
    print(f"[P1] 임베딩 분산 (D={D}, M={M} Areas, dm={D // M}) — D3 빈칸")
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 90}
    dm = D // M
    X, y = make_mask_dataset(n_total=n_total, seed=seed)
    ntr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    keys = jax.random.split(jax.random.PRNGKey(seed), M + 1)
    emb = 0.5 * jax.random.normal(keys[0], (VOCAB_SIZE, D))
    carriers = tuple((int(pick_cells(geo, 1, phase=0.9 * m)[0]),) for m in range(M))
    out_cells = pick_cells(geo, 3, phase=2.6)
    de = DistributedEmbedding(embedding=emb, n_areas=M, dim_per_area=dm, carriers=carriers,
                              window=WINDOW, stride=STRIDE, P=hp["P"], n_steps=hp["n_steps"],
                              n_cells=geo.N)
    areas = tuple(_make_area(geo, keys[1 + m], "processor", carriers[m], out_cells, hp)
                  for m in range(M))
    model = (de, areas)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    basis = ofdm_basis(dm, WINDOW)
    out_arr = jnp.asarray(out_cells)
    nouns = jnp.asarray(noun_token_ids())

    def logits_of(params, Xb, masks):
        d, ars = eqx.combine(params, static)
        injs = d.encode(Xb)
        feats = []
        for m, area in enumerate(ars):
            sc = area.step_constants(geo)

            def single(inj_TN):
                o = dynamics.rollout(sc, inj_TN, out_arr, geo.N)
                return jnp.mean(signed_bank_readout(o, basis, masks[m]), axis=0)  # [dm]
            feats.append(jax.vmap(single)(injs[m]))                                # [B, dm]
        feat = jnp.concatenate(feats, axis=-1)                                     # [B, D]
        return vocabulary_logits(feat, d.embedding)

    def recompute_windows(mdl):
        d, ars = mdl
        masks = []
        for m, area in enumerate(ars):
            tau = jax.lax.stop_gradient(area.edge_tau(geo))
            tmin = dijkstra_tmin(geo, tau, list(carriers[m]), list(out_cells)).min()
            masks.append(make_bank_mask(hp["n_steps"], WINDOW, float(tmin), hp["P"]))
        return masks

    def loss_fn(params, Xb, Yb, windows, lam):
        d, _ars = eqx.combine(params, static)
        logits = logits_of(params, Xb, windows)
        ce = -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb])
        # 약한 직교 정규화(§1.2 선택 안전장치, λ 작게 — 직교성은 부산물 원칙 유지)
        orth_pen = orthogonality_penalty(d.embedding, noun_token_ids())
        return ce + 0.1 * orth_pen, logits

    orth_before = float(channel_orthogonality(emb[nouns]))
    print(f"    orth (before) = {orth_before:.3f}")

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res,
                                    batch_size=32, open_terrain=False, open_gain=False, lrs=None,
                                    lam_schedule=train.anneal_lambda(2, 8),
                                    recompute_windows=recompute_windows, seed=seed, log_prefix="[res] ")
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open,
                                    batch_size=32, open_terrain=True, open_gain=False,
                                    lrs=dict(embedding=4e-2, terrain_h=2e-2),
                                    lam_schedule=train.const_lambda(8.0),
                                    recompute_windows=recompute_windows, seed=seed + 1, log_prefix="[open] ")
    params = eqx.partition(model, eqx.is_inexact_array)[0]
    top1 = train.accuracy(logits_of(params, jnp.asarray(Xte), win), yte)
    orth_after = float(channel_orthogonality(model[0].embedding[nouns]))
    passed = bool(orth_after < 0.5)
    print(f"    top1={top1:.3f}  orth {orth_before:.3f}->{orth_after:.3f}  "
          f"(E1 단일=0.996) {'PASS' if passed else 'CHECK'}")
    return dict(exp="P1", top1=top1, orth_before=orth_before, orth_after=orth_after,
                M=M, D=D, e1_orth=0.996, passed=passed)


if __name__ == "__main__":
    run()
