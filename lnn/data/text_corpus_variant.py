"""SPEC-EXT §1 — 변별 문맥 코퍼스 (직교성 붕괴 원인 규명).

기존 ``text_corpus.py`` 는 **건드리지 않는다**(대조군 보존). 본 모듈은 명사마다 고유 동사를
부여해 "명사 구별을 보상하는" 데이터를 만든다. 변별 강도를 3레벨로 연속 조절:

- **L0 (shared)**  : 범주 공유 동사만 — 기존 D3 재현(colinear 최대). distinct_ratio=0.0
- **L1 (partial)** : 고유 동사 일부 혼합(변별 중간).                  distinct_ratio=0.3
- **L2 (distinct)**: 고유 동사 위주(변별 강함).                       distinct_ratio=0.8

가설 H-ORTH 참이면 orth_after 가 L0 ≥ L1 ≥ L2 로 단조 감소(변별↑ → 코드 직교 회복).
인터페이스·반환형은 ``make_mask_dataset`` 과 동일(드롭인). 결정론적 시드.
"""

from __future__ import annotations

import numpy as np

SUBJECTS = ["나는", "그는", "그녀는", "우리는"]
PARTICLE = "를"
SEQ_LEN = 4
SPECIALS = ["<PAD>", "<BOS>", "<MASK>"]

# 범주 → 명사 (기존 D3 구조 유지)
CATEGORIES = {
    0: ["시", "글", "편지", "일기", "소설"],   # 문자
    1: ["물", "커피", "차"],                    # 음료
    2: ["노래", "그림"],                        # 예술
}
# 범주 공유 동사 (L0/L1 — 범주 신호 보존)
CATEGORY_VERBS = {0: ["쓴다", "읽는다"], 1: ["마신다"], 2: ["부른다", "그린다"]}
# 명사별 고유 동사 (L2 변별 핵심 — 동사가 명사를 거의 유일 지정)
NOUN_VERBS = {
    "시": "짓는다", "글": "적는다", "편지": "부친다", "일기": "남긴다", "소설": "펴낸다",
    "물": "붓는다", "커피": "내린다", "차": "우린다", "노래": "흥얼댄다", "그림": "칠한다",
}

N_CLASSES = len(CATEGORIES)
_LEVEL_RATIO = {"L0": 0.0, "L1": 0.3, "L2": 0.8}


def _noun_to_cat():
    return {n: c for c, ns in CATEGORIES.items() for n in ns}


NOUN_CAT = _noun_to_cat()
ALL_NOUNS = [n for ns in CATEGORIES.values() for n in ns]


def build_variant_vocab():
    toks = list(SPECIALS) + list(SUBJECTS) + [PARTICLE] + ALL_NOUNS
    for vs in CATEGORY_VERBS.values():
        toks += vs
    toks += list(NOUN_VERBS.values())
    seen, vocab = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            vocab.append(t)
    return vocab


VOCAB = build_variant_vocab()
TOKEN2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
VOCAB_SIZE = len(VOCAB)
PAD_ID = TOKEN2ID["<PAD>"]
MASK_ID = TOKEN2ID["<MASK>"]


def _encode(tokens):
    ids = [TOKEN2ID[t] for t in tokens]
    return ids[:SEQ_LEN] + [PAD_ID] * (SEQ_LEN - len(ids))


def make_mask_dataset_variant(n_total=300, seed=1, variant_level="L0", distinct_ratio=None):
    """변별 문맥 빈칸 데이터. (X[N,SEQ_LEN] with <MASK>, y[N] 목적어 id). 드롭인 호환."""
    ratio = _LEVEL_RATIO[variant_level] if distinct_ratio is None else float(distinct_ratio)
    rng = np.random.default_rng(seed)
    X, y = [], []
    for _ in range(n_total):
        noun = ALL_NOUNS[int(rng.integers(len(ALL_NOUNS)))]
        cat = NOUN_CAT[noun]
        if rng.random() < ratio:
            verb = NOUN_VERBS[noun]                                  # 고유(변별) 동사
        else:
            cv = CATEGORY_VERBS[cat]
            verb = cv[int(rng.integers(len(cv)))]                    # 범주 공유 동사
        subj = SUBJECTS[int(rng.integers(len(SUBJECTS)))]
        tokens = [subj, "<MASK>", PARTICLE, verb]                    # 목적어 자리 가림
        X.append(_encode(tokens))
        y.append(TOKEN2ID[noun])
    return np.asarray(X, np.int32), np.asarray(y, np.int32)


def noun_token_ids_variant():
    return [TOKEN2ID[n] for n in ALL_NOUNS]


def noun_ids_by_category():
    """범주별 명사 토큰 id (orth_within/across 측정용)."""
    return {c: [TOKEN2ID[n] for n in ns] for c, ns in CATEGORIES.items()}


def decode_ids(ids):
    return [ID2TOKEN[int(i)] for i in ids]
