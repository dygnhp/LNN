"""D4 — 텍스트→이미지 생성: 글자 토큰 "A" → 8×8 비트맵.

structural 디코더(출력 셀 격자 → 8×8 공간 대응). 지도학습(글자→목표 비트맵, MSE).
지속 구동형 생성(Block I 은 nilpotent → 자유전개 어트랙터 무의미, §7): 조건 글자를
매 스텝 구동하고 정상 응답을 readout. # TODO: 시간역전 생성·정상파 어트랙터.
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
from lnn.data.image_data import make_char_dataset
from lnn.decoders import structural_image
from lnn.encodings import TextTimeEncoder

D = 4


def _images(model, geo, X, windows):
    enc, clus = model
    c, _u = clus.forward(geo, enc.encode(jnp.asarray(X)), windows)
    return structural_image(c, windows[-1])


def run(seed=0, R=5, epochs_res=10, epochs_open=40):  # 8x8 출력 셀 64개 필요 → N>=64 (R>=5)
    print("[D4] 글자->이미지 (A-Z 일부 -> 8x8)")
    geo = C.make_geometry(R)
    hp = {**C.gen_hp(), "n_channels": 8}
    chars, targets = make_char_dataset()
    K = len(chars)
    X = np.arange(K, dtype=np.int32)[:, None]   # 글자 토큰(길이 1 시퀀스)
    Y = targets                                  # [K, 64]
    print(f"    chars={chars} N={geo.N}")

    key = jax.random.PRNGKey(seed)
    ke, kc = jax.random.split(key)
    emb = 0.6 * jax.random.normal(ke, (K, D))
    gen_cells = pick_cells(geo, D, phase=0.0)
    dec_out = C.map_image_cells(geo, 8, 8)
    encoder = TextTimeEncoder(embedding=emb, gen_cells=gen_cells, stride=10,
                              P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    cluster = build_serial_cluster(geo, gen_cells, dec_out, kc, hp)
    model = (encoder, cluster)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params, Xb, Yb, windows, lam):
        enc, clus = eqx.combine(params, static)
        c, _u = clus.forward(geo, enc.encode(Xb), windows)
        img = structural_image(c, windows[-1])
        return jnp.mean((img - Yb) ** 2), img

    def recompute_windows(m):
        return m[1].make_windows(geo)

    model, _, win = train.run_phase(
        model, static, loss_fn, X, Y, epochs=epochs_res, batch_size=K,
        open_terrain=False, open_gain=False, lrs=dict(embedding=5e-2),
        lam_schedule=train.const_lambda(8.0),
        recompute_windows=recompute_windows, seed=seed, log_prefix="[res] ")
    mse_res = float(jnp.mean((_images(model, geo, X, win) - jnp.asarray(Y)) ** 2))

    model, _, win = train.run_phase(
        model, static, loss_fn, X, Y, epochs=epochs_open, batch_size=K,
        open_terrain=True, open_gain=True, lrs=dict(terrain_h=5e-2, gain_a=3e-2, embedding=3e-2),
        lam_schedule=train.const_lambda(8.0),
        recompute_windows=recompute_windows, seed=seed + 1, log_prefix="[open] ")
    imgs = np.asarray(_images(model, geo, X, win))
    mse_open = float(np.mean((imgs - Y) ** 2))
    print(f"    MSE reservoir={mse_res:.4f} → opened={mse_open:.4f}")

    _save_gallery(chars, imgs, Y, mse_res, mse_open)
    metrics = dict(demo="D4", mse_reservoir=mse_res, mse_opened=mse_open,
                   improved=bool(mse_open < mse_res), chars=chars)
    C.save_metrics("D4_text_to_image", metrics)
    return metrics


def _save_gallery(chars, imgs, Y, mse_res, mse_open):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    K = len(chars)
    fig, axes = plt.subplots(2, K, figsize=(1.6 * K, 3.4))
    for k in range(K):
        axes[0, k].imshow(Y[k].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_title(f"target {chars[k]}")
        axes[1, k].imshow(imgs[k].reshape(8, 8), cmap="gray")
        axes[1, k].set_title("LNN out")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"D4 char→8x8  MSE {mse_res:.3f}→{mse_open:.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(C.OUT_DIR, "D4_chars.png"), dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    run()
