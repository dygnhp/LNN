"""Experiment 2-A 데이터 — MNIST (대규모) + downscale/native 모드 + fallback.

- 로드: ``sklearn.datasets.fetch_openml('mnist_784')`` (캐시됨). 실패 시 digits 업샘플 fallback(§1.5).
- 정규화 0–255 → 0–1.
- ``img_mode``: downscale(28→8/14, PIL resize) | native(28×28 그대로).
  28×28 은 R=5 격자에 직접 안 맞으므로 격자 크기에 맞춰 해상도를 고른다(§1.1).
"""

from __future__ import annotations

import numpy as np

_CACHE = {}


def _load_mnist_raw():
    """(X:[70000,784] 0-1 float32, y:[70000] int). 실패 시 (None, None, reason)."""
    if "mnist" in _CACHE:
        return _CACHE["mnist"]
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                            as_frame=False, parser="auto")
        X = (X.astype(np.float32) / 255.0)
        y = y.astype(np.int64)
        _CACHE["mnist"] = (X, y, "mnist")
    except Exception as e:  # pragma: no cover
        _CACHE["mnist"] = (None, None, f"fetch_openml failed: {type(e).__name__}: {e}")
    return _CACHE["mnist"]


def _resize(img_28, size):
    """28×28 → size×size. PIL BILINEAR (8·14 모두 정수배 아님 → 보간)."""
    if size == 28:
        return img_28
    from PIL import Image
    im = Image.fromarray((img_28 * 255).astype(np.uint8))
    im = im.resize((size, size), Image.BILINEAR)
    return np.asarray(im, np.float32) / 255.0


def _digits_fallback(n_per_class, size, seed):
    """MNIST 로드 실패 시: digits(8×8)를 size로 업샘플 + 가우시안 잡음(유사 대규모, §1.5)."""
    from .image_data import load_digits_split
    Xtr, ytr, Xte, yte = load_digits_split(n_per_class=n_per_class, seed=seed)
    rng = np.random.default_rng(seed)

    def up(X):
        out = []
        for row in X:
            img = _resize(row.reshape(8, 8), size)
            img = np.clip(img + rng.normal(0, 0.05, img.shape), 0, 1)
            out.append(img.reshape(-1))
        return np.asarray(out, np.float32)

    return up(Xtr), ytr, up(Xte), yte, "digits_fallback(upsampled+noise)"


def load_mnist_split(n_per_class=100, size=8, test_per_class=20, seed=0):
    """MNIST 부분집합을 size×size 로. 반환: (Xtr, ytr, Xte, yte, source).

    size: 8|14|28. source: "mnist" 또는 fallback 사유 문자열.
    """
    X, y, src = _load_mnist_raw()
    if X is None:
        return _digits_fallback(n_per_class, size, seed)

    rng = np.random.default_rng(seed)
    tr_idx, te_idx = [], []
    for c in range(10):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        tr_idx.extend(idx[:n_per_class].tolist())
        te_idx.extend(idx[n_per_class:n_per_class + test_per_class].tolist())
    tr_idx, te_idx = np.array(tr_idx), np.array(te_idx)
    rng.shuffle(tr_idx)

    def to_res(rows):
        return np.asarray([_resize(r.reshape(28, 28), size).reshape(-1) for r in rows],
                          np.float32)

    return (to_res(X[tr_idx]), y[tr_idx].astype(np.int32),
            to_res(X[te_idx]), y[te_idx].astype(np.int32), src)
