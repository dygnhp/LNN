"""E-ORTH — 변별 문맥 통제 대조 (SPEC-EXT §2). H-ORTH 검증.

D3 빈칸 채우기를 변별 강도 3레벨(L0/L1/L2)로 동일 모델·동일 D(=8)·동일 시드로 학습.
단일 변수(데이터 변별 강도)만 바꿈. orth_after 단조성(O2)·범주 내/간 분리(O3)로 colinear 가
데이터 보상 구조의 반영인지(H-ORTH 참) 임베딩 기하 한계인지(거짓) 확정.

M=1(임베딩 분산 없음 — 분산이 아니라 데이터가 변수임을 격리). 경로 B(TimeEncoder) = Block I D3 과 동일.
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
from lnn.area import pick_cells  # noqa: E402
from lnn.cluster import build_serial_cluster  # noqa: E402
from lnn.data import text_corpus_variant as TV  # noqa: E402
from lnn.decoders import vocabulary_logits  # noqa: E402
from lnn.encodings import TextTimeEncoder, channel_orthogonality  # noqa: E402

D = 8
STRIDE = 12


def _within_across(embedding):
    """범주 내/간 명사쌍 평균 절대 교차상관 (O3)."""
    by_cat = TV.noun_ids_by_category()
    cat_of = {i: c for c, ids in by_cat.items() for i in ids}
    ids = TV.noun_token_ids_variant()
    codes = np.asarray(embedding[jnp.asarray(ids)])
    n = codes / (np.linalg.norm(codes, axis=1, keepdims=True) + 1e-9)
    g = np.abs(n @ n.T)
    win, acr = [], []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            (win if cat_of[ids[a]] == cat_of[ids[b]] else acr).append(g[a, b])
    return float(np.mean(win)), float(np.mean(acr))


def run_level(level, seed=0, R=5, n_total=300, e_res=4, e_open=16):
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_channels": D}
    X, y = TV.make_mask_dataset_variant(n_total=n_total, seed=seed, variant_level=level)
    ntr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    ke, kc = jax.random.split(jax.random.PRNGKey(seed))
    emb = 0.5 * jax.random.normal(ke, (TV.VOCAB_SIZE, D))
    gen = pick_cells(geo, D, phase=0.0)
    dec = pick_cells(geo, D, phase=2.6)
    encoder = TextTimeEncoder(embedding=emb, gen_cells=gen, stride=STRIDE,
                              P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    cluster = build_serial_cluster(geo, gen, dec, kc, hp)
    model = (encoder, cluster)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    nouns = jnp.asarray(TV.noun_token_ids_variant())

    def logits_of(params, Xb, windows):
        e, cl = eqx.combine(params, static)
        _c, u = cl.forward(geo, e.encode(Xb), windows)
        return vocabulary_logits(u, e.embedding)

    def loss_fn(params, Xb, Yb, windows, lam):
        logits = logits_of(params, Xb, windows)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    rw = lambda m: m[1].make_windows(geo)
    orth_before = float(channel_orthogonality(emb[nouns]))

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res,
                                    batch_size=32, open_terrain=False, open_gain=False, lrs=None,
                                    lam_schedule=train.anneal_lambda(2, 8),
                                    recompute_windows=rw, seed=seed, log_prefix=f"[{level} res] ")
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open,
                                    batch_size=32, open_terrain=True, open_gain=False,
                                    lrs=dict(embedding=4e-2, terrain_h=2e-2),
                                    lam_schedule=train.const_lambda(8.0),
                                    recompute_windows=rw, seed=seed + 1, log_prefix=f"[{level} open] ")
    params = eqx.partition(model, eqx.is_inexact_array)[0]
    top1 = train.accuracy(logits_of(params, jnp.asarray(Xte), win), yte)
    emb_final = model[0].embedding
    orth_after = float(channel_orthogonality(emb_final[nouns]))
    within, across = _within_across(emb_final)
    print(f"    [{level}] orth {orth_before:.3f}->{orth_after:.3f}  top1={top1:.3f}  "
          f"within={within:.3f} across={across:.3f}")
    return dict(level=level, orth_before=orth_before, orth_after=orth_after, top1=top1,
                orth_within=within, orth_across=across)


def run(seed=0, R=5, n_total=300, e_res=4, e_open=16):
    print("[E-ORTH] 변별 문맥 통제 대조 (L0/L1/L2, D=8 고정, M=1)")
    res = [run_level(lvl, seed, R, n_total, e_res, e_open) for lvl in ("L0", "L1", "L2")]
    oa = {r["level"]: r["orth_after"] for r in res}
    monotone = oa["L0"] >= oa["L1"] >= oa["L2"]
    recovered = oa["L2"] < oa["L0"] - 0.05
    L0 = next(r for r in res if r["level"] == "L0")
    within_gg_across = L0["orth_within"] > L0["orth_across"] + 0.05
    print(f"    O2 단조성 L0≥L1≥L2: {monotone} (L0={oa['L0']:.3f} L1={oa['L1']:.3f} L2={oa['L2']:.3f})")
    print(f"    O3 L0 within({L0['orth_within']:.3f}) ≫ across({L0['orth_across']:.3f}): {within_gg_across}")
    verdict = "H-ORTH 참(데이터 원인)" if recovered else "H-ORTH 거짓(기하 원인 — 경로 A 필요)"
    print(f"    판정: {verdict}")
    return dict(exp="E-ORTH", levels=res, monotone=bool(monotone),
                recovered=bool(recovered), within_gg_across=bool(within_gg_across),
                verdict=verdict, D=D)


if __name__ == "__main__":
    run()
