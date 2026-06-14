"""§5.2 이미지 데이터(D2 sklearn digits) · §5.3 글자 비트맵(D4 PIL 렌더).

외부 다운로드 금지: digits 는 sklearn 내장, 글자는 PIL 렌더(폰트 미가용 시 하드코딩 fallback).
"""

from __future__ import annotations

import numpy as np


def load_digits_split(n_per_class=40, test_frac=0.25, seed=0):
    """sklearn digits (8×8, 10클래스). 0–16 명암 → 0–1 정규화, 소규모 분할.

    반환: (Xtr:[Ntr,64], ytr, Xte:[Nte,64], yte) — float32, 0–1.
    """
    from sklearn.datasets import load_digits

    d = load_digits()
    X = (d.images.reshape(len(d.images), 64) / 16.0).astype(np.float32)
    y = d.target.astype(np.int32)
    rng = np.random.default_rng(seed)
    idx = []
    for cls in range(10):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        idx.extend(cls_idx[:n_per_class].tolist())
    idx = np.asarray(idx)
    rng.shuffle(idx)
    n_test = int(len(idx) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


# A,B,C,O,X 하드코딩 8×8 비트맵 fallback (폰트 미가용 시).
_HARDCODED = {
    "A": [
        "00111100",
        "01100110",
        "01100110",
        "01111110",
        "01100110",
        "01100110",
        "01100110",
        "00000000",
    ],
    "B": [
        "01111100",
        "01100110",
        "01100110",
        "01111100",
        "01100110",
        "01100110",
        "01111100",
        "00000000",
    ],
    "C": [
        "00111110",
        "01100000",
        "01100000",
        "01100000",
        "01100000",
        "01100000",
        "00111110",
        "00000000",
    ],
    "O": [
        "00111100",
        "01100110",
        "01100110",
        "01100110",
        "01100110",
        "01100110",
        "00111100",
        "00000000",
    ],
    "X": [
        "01100110",
        "01100110",
        "00111100",
        "00011000",
        "00111100",
        "01100110",
        "01100110",
        "00000000",
    ],
}


def _hardcoded_bitmap(ch):
    rows = _HARDCODED[ch]
    return np.array([[float(c) for c in row] for row in rows], np.float32)


def render_char_8x8(ch):
    """글자 ch 를 8×8 그레이스케일(0–1)로 렌더. PIL 폰트 우선, 실패 시 하드코딩 fallback."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.new("L", (8, 8), 0)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 8)
        except Exception:
            font = ImageFont.load_default()
        draw.text((1, 0), ch, fill=255, font=font)
        arr = np.asarray(img, np.float32) / 255.0
        if arr.sum() < 1e-3 and ch in _HARDCODED:  # 폰트가 빈 글리프를 그린 경우
            return _hardcoded_bitmap(ch)
        return arr
    except Exception:
        if ch in _HARDCODED:
            return _hardcoded_bitmap(ch)
        raise


def make_char_dataset(chars=("A", "B", "C", "O", "X")):
    """글자 토큰(공간 코드) → 8×8 목표 비트맵. 반환: (chars, targets:[K,64] float32)."""
    targets = np.stack([render_char_8x8(c).reshape(64) for c in chars])
    return list(chars), targets.astype(np.float32)
