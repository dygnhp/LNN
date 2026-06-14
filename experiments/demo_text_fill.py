"""D3 — 텍스트→텍스트 생성: "나는 ___를 쓴다" 빈칸 채우기.

vocabulary 디코더(CDMA식, 입출력 코드북 공유 = weight tying). 마스크드 토큰 CE 학습.
§11★ 우선 측정: D=8 인코더의 채널 직교성(코드 간 최대 교차상관)을 학습 전후 로깅 —
경로 B 의 D ≤ 코히어런스 길이 제약과 충돌하는 첫 신호인지 확인.
"""

from __future__ import annotations

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import _common as C

from lnn import train
from lnn.area import pick_cells
from lnn.cluster import build_serial_cluster
from lnn.data.text_corpus import (MASK_ID, SEQ_LEN, TOKEN2ID, VOCAB_SIZE,
                                  decode_ids, make_mask_dataset, noun_token_ids)
from lnn.decoders import vocabulary_logits
from lnn.encodings import channel_orthogonality
from lnn.encodings import TextTimeEncoder

D = 8          # §10: D3 는 D≈log2|V|+마진 ≈ 8 불가피 (§11★ 직교성 충돌 주시)
STRIDE = 12


def _logits(model, geo, X, windows):
    enc, clus = model
    _c, u = clus.forward(geo, enc.encode(jnp.asarray(X)), windows)
    return vocabulary_logits(u, enc.embedding)


def run(seed=0, R=5, n_total=300, epochs_res=5, epochs_open=18, batch=32):
    print("[D3] 빈칸 채우기 (나는 ___를 쓴다)")
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_channels": D}
    X, y = make_mask_dataset(n_total=n_total, seed=seed)
    ntr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]

    key = jax.random.PRNGKey(seed)
    ke, kc = jax.random.split(key)
    emb = 0.5 * jax.random.normal(ke, (VOCAB_SIZE, D))
    gen_cells = pick_cells(geo, D, phase=0.0)
    dec_out = pick_cells(geo, D, phase=2.6)
    encoder = TextTimeEncoder(embedding=emb, gen_cells=gen_cells, stride=STRIDE,
                              P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    cluster = build_serial_cluster(geo, gen_cells, dec_out, kc, hp)
    model = (encoder, cluster)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    # §11★ 직교성: 명사(예측 대상) 코드 행들의 최대 교차상관
    nouns = noun_token_ids()
    orth_before = channel_orthogonality(emb[jnp.asarray(nouns)])
    print(f"    [§11★] noun-code max cross-corr (before) = {orth_before:.3f}  (D={D})")

    def loss_fn(params, Xb, Yb, windows, lam):
        enc, clus = eqx.combine(params, static)
        _c, u = clus.forward(geo, enc.encode(Xb), windows)
        logits = vocabulary_logits(u, enc.embedding)
        ll = jax.nn.log_softmax(logits, axis=-1)
        return -jnp.mean(ll[jnp.arange(Yb.shape[0]), Yb]), logits

    def recompute_windows(m):
        return m[1].make_windows(geo)

    model, _, win = train.run_phase(
        model, static, loss_fn, Xtr, ytr, epochs=epochs_res, batch_size=batch,
        open_terrain=False, open_gain=False, lrs=None,
        lam_schedule=train.anneal_lambda(2.0, 8.0),
        recompute_windows=recompute_windows, seed=seed, log_prefix="[res] ")
    acc_res = train.accuracy(_logits(model, geo, Xte, win), yte)
    print(f"    reservoir test top-1 = {acc_res:.3f}")

    model, _, win = train.run_phase(
        model, static, loss_fn, Xtr, ytr, epochs=epochs_open, batch_size=batch,
        open_terrain=True, open_gain=True, lrs=None,
        lam_schedule=train.const_lambda(8.0),
        recompute_windows=recompute_windows, seed=seed + 1, log_prefix="[open] ")
    acc_open = train.accuracy(_logits(model, geo, Xte, win), yte)
    print(f"    opened   test top-1 = {acc_open:.3f}  (Δ={acc_open - acc_res:+.3f})")

    orth_after = channel_orthogonality(model[0].embedding[jnp.asarray(nouns)])
    print(f"    [§11★] noun-code max cross-corr (after)  = {orth_after:.3f}")

    # 빈칸 top-k 예측: "나는 <MASK>를 쓴다"
    sent = [TOKEN2ID["나는"], MASK_ID, TOKEN2ID["를"], TOKEN2ID["쓴다"]]
    sent = sent[:SEQ_LEN] + [0] * (SEQ_LEN - len(sent))
    logits = _logits(model, geo, np.asarray([sent], np.int32), win)[0]
    topk = np.argsort(-np.asarray(logits))[:5]
    preds = decode_ids(topk)
    print(f"    '나는 ___를 쓴다' top-5 = {preds}")

    metrics = dict(demo="D3", acc_reservoir=acc_res, acc_opened=acc_open,
                   improved=bool(acc_open >= acc_res), top5=preds,
                   orth_before=orth_before, orth_after=orth_after, D=D)
    with open(os.path.join(C.OUT_DIR, "D3_text_fill.txt"), "w", encoding="utf-8") as f:
        f.write(f"D3 fill-in-the-blank\nreservoir top1={acc_res:.3f} opened top1={acc_open:.3f}\n")
        f.write(f"[§11*] noun-code max cross-corr before={orth_before:.3f} after={orth_after:.3f}\n")
        f.write(f"'나는 ___를 쓴다' top-5 = {preds}\n")
    C.save_metrics("D3_text_fill", metrics)
    return metrics


if __name__ == "__main__":
    run()
