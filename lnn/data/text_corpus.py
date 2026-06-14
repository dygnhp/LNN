"""§5.1 텍스트 코퍼스 (D1, D3) — 합성 한국어 소코퍼스(결정론적, 외부 다운로드 금지).

문형: ``{주어} {목적어}를 {동사}``. 어절 단위 고정 vocab(~25 토큰, 특수토큰 포함).
라벨(D1): 의미 범주 3종 — 문자(글류·쓰기/읽기) / 음료(마시기) / 예술(그리기·부르기).
목적어·동사가 범주로 정렬되어 문장에서 범주가 학습 가능.
D3: ``나는 <MASK>를 쓴다`` → <MASK> 위치 목적어 예측(마스크드 토큰 교차엔트로피).
"""

from __future__ import annotations

import numpy as np

SUBJECTS = ["나는", "그는", "그녀는", "우리는"]

# 범주별 (목적어, 동사) — 문장에서 범주가 식별 가능하도록 정렬.
CATEGORIES = {
    0: {  # 문자
        "nouns": ["시", "글", "편지", "일기", "소설"],
        "verbs": ["쓴다", "읽는다"],
    },
    1: {  # 음료
        "nouns": ["물", "커피", "차"],
        "verbs": ["마신다"],
    },
    2: {  # 예술
        "nouns": ["노래", "그림"],
        "verbs": ["부른다", "그린다"],
    },
}
N_CLASSES = len(CATEGORIES)

SPECIALS = ["<PAD>", "<BOS>", "<MASK>"]
PARTICLE = "를"

SEQ_LEN = 4  # [주어, 목적어, 를, 동사]


def _build_vocab():
    toks = list(SPECIALS) + list(SUBJECTS) + [PARTICLE]
    for cat in CATEGORIES.values():
        toks += cat["nouns"] + cat["verbs"]
    # 중복 제거(순서 보존)
    seen, vocab = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            vocab.append(t)
    return vocab


VOCAB = _build_vocab()
TOKEN2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = TOKEN2ID["<PAD>"]
MASK_ID = TOKEN2ID["<MASK>"]


def _encode(tokens):
    ids = [TOKEN2ID[t] for t in tokens]
    ids = ids[:SEQ_LEN] + [PAD_ID] * (SEQ_LEN - len(ids))
    return ids


def _sample_sentence(rng):
    cat = int(rng.integers(N_CLASSES))
    noun = CATEGORIES[cat]["nouns"][int(rng.integers(len(CATEGORIES[cat]["nouns"])))]
    verb = CATEGORIES[cat]["verbs"][int(rng.integers(len(CATEGORIES[cat]["verbs"])))]
    subj = SUBJECTS[int(rng.integers(len(SUBJECTS)))]
    return [subj, noun, PARTICLE, verb], cat, noun


def make_classification_dataset(n_total=300, seed=0):
    """(token_ids:[N, SEQ_LEN] int, labels:[N] int). D1."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_total):
        tokens, cat, _ = _sample_sentence(rng)
        X.append(_encode(tokens))
        y.append(cat)
    return np.asarray(X, np.int32), np.asarray(y, np.int32)


def make_mask_dataset(n_total=300, seed=1):
    """빈칸 채우기 (token_ids:[N,SEQ_LEN] with <MASK>, target:[N] 목적어 id). D3."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_total):
        tokens, _, noun = _sample_sentence(rng)
        masked = list(tokens)
        masked[1] = "<MASK>"           # 목적어 자리를 가림
        X.append(_encode(masked))
        y.append(TOKEN2ID[noun])
    return np.asarray(X, np.int32), np.asarray(y, np.int32)


def noun_token_ids():
    """목적어(예측 대상) 토큰 id 집합 — D3 top-k 를 명사로 제한할 때 사용."""
    ids = []
    for cat in CATEGORIES.values():
        ids += [TOKEN2ID[n] for n in cat["nouns"]]
    return sorted(ids)


def decode_ids(ids):
    return [ID2TOKEN[int(i)] for i in ids]
