"""E1 — 경로 A 완성으로 D3 빈칸 채우기 재시험 (§8).

경로 A = FreqEncoder(입구) + FreqBankReadout(출구) 둘 다. Exp2 H4 가 인코더만 바꿔선
부족함을 보였으므로, 출력단 주파수 정합필터 뱅크로 채널을 분리 복원한다.
통과: orth_after < 0.5 (직교 유지) **또는** top-1 > 0.22 (Block I 경로 B 한계 돌파).
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
from lnn.data.text_corpus import (VOCAB_SIZE, decode_ids, make_mask_dataset,  # noqa: E402
                                  noun_token_ids)
from lnn.decoders import vocabulary_logits  # noqa: E402
from lnn.encodings import channel_orthogonality  # noqa: E402
from lnn_block2.freq_encoding import FreqEncoder, ofdm_basis  # noqa: E402
from lnn_block2.freq_readout import make_bank_mask, signed_bank_readout  # noqa: E402

D = 8
WINDOW = 20    # D < W/2 (직교)
STRIDE = 20


def run(seed=0, R=5, n_total=240, e_res=4, e_open=16):
    print("[E1] 경로 A (FreqEncoder + 주파수 뱅크 readout) — D3 빈칸 채우기")
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": 90}   # 4토큰×stride20 = 80 < 90
    X, y = make_mask_dataset(n_total=n_total, seed=seed)
    ntr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    ke, ka = jax.random.split(jax.random.PRNGKey(seed))
    emb = 0.5 * jax.random.normal(ke, (VOCAB_SIZE, D))
    carrier = pick_cells(geo, 1, phase=0.0)
    out_cells = pick_cells(geo, 3, phase=2.6)
    encoder = FreqEncoder(embedding=emb, gen_cells=carrier, P=hp["P"], n_steps=hp["n_steps"],
                          n_cells=geo.N, window=WINDOW, stride=STRIDE)
    area = _make_area(geo, ka, "processor", carrier, out_cells, hp)
    model = (encoder, area)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    basis = ofdm_basis(D, WINDOW)
    out_arr = jnp.asarray(out_cells)
    nouns = jnp.asarray(noun_token_ids())

    def logits_of(params, Xb, mask):
        enc, ar = eqx.combine(params, static)
        inj = enc.encode(Xb)                                   # [B,T,N]
        sc = ar.step_constants(geo)

        def single(inj_TN):
            o = dynamics.rollout(sc, inj_TN, out_arr, geo.N)   # [T, n_out]
            u = signed_bank_readout(o, basis, mask)            # [n_out, D]
            return jnp.mean(u, axis=0)                          # [D]
        feats = jax.vmap(single)(inj)                          # [B, D]
        return vocabulary_logits(feats, enc.embedding)         # [B, V]

    def recompute_windows(m):
        enc, ar = m
        tau = jax.lax.stop_gradient(ar.edge_tau(geo))
        from lnn.readout import dijkstra_tmin
        tmin = dijkstra_tmin(geo, tau, list(carrier), list(out_cells)).min()
        return [make_bank_mask(hp["n_steps"], WINDOW, float(tmin), hp["P"])]

    def loss_fn(params, Xb, Yb, windows, lam):
        logits = logits_of(params, Xb, windows[0])
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    orth_before = channel_orthogonality(emb[nouns])
    print(f"    [§11★] noun-code orth (before) = {orth_before:.3f}  (D={D}, 경로 A)")

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res,
                                    batch_size=32, open_terrain=False, open_gain=False,
                                    lrs=None, lam_schedule=train.anneal_lambda(2, 8),
                                    recompute_windows=recompute_windows, seed=seed, log_prefix="[res] ")
    acc_res = train.accuracy(logits_of(eqx.partition(model, eqx.is_inexact_array)[0],
                                       jnp.asarray(Xte), win[0]), yte)
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open,
                                    batch_size=32, open_terrain=True, open_gain=False,
                                    lrs=dict(embedding=4e-2, terrain_h=2e-2),
                                    lam_schedule=train.const_lambda(8.0),
                                    recompute_windows=recompute_windows, seed=seed + 1, log_prefix="[open] ")
    params = eqx.partition(model, eqx.is_inexact_array)[0]
    acc_open = train.accuracy(logits_of(params, jnp.asarray(Xte), win[0]), yte)
    orth_after = channel_orthogonality(model[0].embedding[nouns])
    passed = bool(orth_after < 0.5 or acc_open > 0.22)
    print(f"    top1 {acc_res:.3f}->{acc_open:.3f}  orth {orth_before:.3f}->{orth_after:.3f}  "
          f"{'PASS' if passed else 'CHECK'}")
    return dict(exp="E1", acc_res=acc_res, acc_open=acc_open,
                orth_before=float(orth_before), orth_after=float(orth_after),
                passed=passed, D=D, path="A(freq encoder+bank readout)")


if __name__ == "__main__":
    run()
