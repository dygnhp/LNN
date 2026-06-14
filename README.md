# LNN — ARIS (Area-Routed Interference System)

격자 위 펄스의 **전파·지연·간섭**으로 계산하는 신경망의 **존재 증명(proof of concept)**.
파라미터는 행렬이 아니라 **매질의 지형(scalar field)**이며, 출력은 입력 펄스가 거쳐 온 **경로 기여의 합**으로 분해된다.

> 본 저장소의 1차 빌드 대상은 **Block I (모델명 ARIS)** 한정이다. 단일 명령 `/goal`로 전체 코드베이스를 생성·검증·시연한다.
> 유일한 사양은 [`LNN_SPEC.md`](LNN_SPEC.md) (v0.7)이며, 모든 설계 결정은 확정되어 있다.

---

## 현재 상태

📋 **사양 확정 · 빌드 대기 (scaffold only)**

이 저장소에는 현재 사양(`LNN_SPEC.md`)과 빌드 커맨드(`.claude/commands/goal.md`)만 들어 있다.
코어 엔진·실험·테스트는 아직 생성되지 않았다 — `/goal` 실행이 그 작업을 수행한다.

## 빌드 방법

Claude Code에서:

```
/goal
```

`/goal`은 사용자에게 되묻지 않고 `LNN_SPEC.md`만으로 §8 구조 전체를 생성하고,
코어 엔진 → 테스트 → 인코더/디코더/데이터/학습 → 네 시연(run_all) → README 결과 기록 순으로 완주한다.

## Definition of Done — 네 시연 (Block I)

| # | 시연 | 양식 | 데이터 | 산출물 |
|---|------|------|--------|--------|
| D1 | 텍스트 분류 | text → class | 합성 한국어 소코퍼스 | 정확도, 혼동행렬 PNG |
| D2 | 이미지 분류 | image(8×8) → class | sklearn digits | 정확도, 지형 히트맵 PNG |
| D3 | 텍스트→텍스트 생성 | "나는 ___를 쓴다" 빈칸 채우기 | D1 코퍼스 | 빈칸 top-k 예측 |
| D4 | 텍스트→이미지 생성 | 글자 토큰 "A" → 8×8 비트맵 | PIL 렌더 A–Z | 글자별 8×8 출력 PNG |

세 계층 **입력 Area → 처리 Cluster → 출력 Area**가 네 시연 공통 엔진으로 동작하면 성공.

## Exit Gate (Block II 진입 조건)

1. D1·D2가 reservoir 기준선을 넘어 **지형 학습으로 정확도 상승**
2. `tests/test_gradient_fd.py` 통과 (유한차분 상대오차 < 1e-3)
3. `tests/test_telescoping.py` 통과 (무차원 상대분산 `V_rel`: 선형 < ε_rel, Tobler > 10·ε_rel)

## 무조건 준수 (fallback 대상 아님 — 가장 비싼 실수들)

- ① 셀 분배 `n=0` 보호: `per = where(n>0, G·a/n, 0)` (§3.5) — NaN이 grad 전체 오염
- ② readout: Dijkstra `t_min`에 `stop_gradient`, logsumexp는 max-subtraction 안정화 (§3.6)
- ③ 기호 분리: `ρ`는 스펙트럼 반경 전용 / 변지연 하한 `clamp_τ` / 이득장 폭 `σ_g` (§3)

## 알려진 한계

- 코히어런스 길이가 직접 간섭 가능한 토큰 거리의 상한 → **장문맥 불가** (단문·소규모만).
- 선형 보간 분수 지연의 진폭-지연 허위 결합 잔존 (Thiran 미적용).
- 진폭 라우팅은 Gain Field 의존, 완전 동적 어텐션 미구현.
- 전 과정은 **존재 증명**이며 현대 LLM/비전 모델과 성능 비교 대상이 아님.
- ★ D3는 `D≈8`이 불가피해 인코더 채널 직교성이 무너질 수 있음 — 빌드 시 D3 직교성을 우선 측정.

## 저장소 구조 (`/goal`이 생성)

```
LNN_SPEC.md                  # 유일한 사양 (v0.7)
README.md
pyproject.toml               # jax, optax, equinox, numpy, scikit-learn, pillow, matplotlib
lnn/
  geometry.py fields.py delay.py dynamics.py readout.py
  area.py cluster.py encodings.py decoders.py train.py
  data/ text_corpus.py image_data.py
experiments/
  train_text_classify.py    # D1
  train_image_classify.py   # D2
  demo_text_fill.py         # D3
  demo_text_to_image.py     # D4
scripts/run_all.py          # 네 시연 일괄 실행 (= /goal 최종 타깃)
outputs/                    # 학습 곡선·히트맵·생성 PNG·메트릭 JSON
tests/ test_geometry.py test_gradient_fd.py test_telescoping.py test_readout.py
.claude/commands/goal.md    # /goal 슬래시 커맨드
```
