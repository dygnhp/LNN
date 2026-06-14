# LNN — ARIS (Area-Routed Interference System)

격자 위 펄스의 **전파·지연·간섭**으로 계산하는 신경망의 **존재 증명(proof of concept)**.
파라미터는 행렬이 아니라 **매질의 지형(scalar field)**이며, 출력은 입력 펄스가 거쳐 온
**경로 기여의 합**으로 분해된다. 비선형성은 **Area 경계 재생(`φ=tanh`)에만** 둔다 —
그래서 Area 내부는 정확한 선형계이고 경로 분해·그래디언트가 깨끗하게 유지된다.

> 1차 빌드 대상은 **Block I (모델명 ARIS)** 한정. 유일한 사양은 [`LNN_SPEC.md`](LNN_SPEC.md) (v0.7).
> 단일 명령 `/goal`(.claude/commands/goal.md)로 전체를 생성·검증·시연하도록 설계되었고, 본 저장소가 그 산출물이다.

---

## 빠른 시작

```bash
pip install -e .          # jax(CPU)·optax·equinox·scikit-learn·pillow·matplotlib
pytest tests -q           # 코어 게이트(15개): gradient_fd·telescoping 포함
python scripts/run_all.py # D1~D4 학습/시연 + outputs/ 산출물 + 합격 표  (CPU, ~90s)
python scripts/run_all.py --quick   # 더 짧은 설정
```

개별 시연: `python experiments/train_text_classify.py` (D1) ·
`train_image_classify.py` (D2) · `demo_text_fill.py` (D3) · `demo_text_to_image.py` (D4).

## 결과 (full `run_all`, CPU, 결정론적 시드)

| # | 시연 | 기준 | reservoir → opened | 합격 |
|---|------|------|--------------------|------|
| D1 | 텍스트 분류(3범주) | 지형 개방이 reservoir 기준선 위로 정확도 상승 (Exit gate ①) | acc **0.41 → 1.00** | ✅ |
| D2 | 이미지 분류(digits 10클래스) | 동상 (gate ①) | acc **0.12 → 0.49** (chance 0.10) | ✅ |
| D3 | 빈칸 채우기 | top-k 가시적 출력(생성) | top1 0.00 → **0.22**, 아래 §11★ | ✅ |
| D4 | 글자→8×8 이미지 | 8×8 가시적 출력(생성) | MSE **0.124 → 0.084** | ✅ |

- **코어 게이트**: `pytest` 15개 통과 — `test_gradient_fd`(유한차분 상대오차 `<1e-3`, float64),
  `test_telescoping`(무차원 상대분산: **선형 `V_rel≈1.5e-7 < 1e-6`**, **Tobler `V_rel≈1.2e-3 > 10×`**,
  비율 ~7900×), geometry, readout.
- **D3 `나는 ___를 쓴다` top-5** = `['노래', '그림', '시', '편지', '차']` (쓰기류 명사 시·편지 포함).
- 산출물: `outputs/`(혼동행렬·지형 히트맵·글자 갤러리 PNG, 메트릭 JSON, `summary.json`).

### ★ §11 — "가장 먼저 깨질 가정"의 경험적 확인 (D3)

D3 의 vocabulary 디코더는 `D ≈ log₂|V|+마진 ≈ 8` 을 요구하는데, 경로 B 는 `D ≤ 코히어런스 길이`를
강제한다. 빌드 측정 결과 명사 코드 간 **최대 교차상관이 학습 중 0.78 → 0.99 로 상승**(채널 직교성
붕괴) — 사양 §11★ 이 예고한 충돌이 정확도 저하 이전에 인코더 단위에서 먼저 드러났다. 이는 경로 A
(FreqEncoder)를 Block I 후반으로 앞당기라는 첫 경험적 신호로 README 에 기록한다(§11★·§9 5단계).

## 무조건 준수한 수치 안전망 (§12 — fallback 대상 아님)

- **`n=0` 분배 보호**: `per = where(n>0, G·a/n, 0)` ([lnn/dynamics.py](lnn/dynamics.py)). 0/0 NaN 의 grad 오염 차단.
- **readout**: Dijkstra `t_min` 에 `stop_gradient` + 넉넉한 윈도, logsumexp 는 max-subtraction 안정화
  ([lnn/readout.py](lnn/readout.py)).
- **기호 분리**: `ρ`(스펙트럼 반경)는 코드 어디에도 없음 / 변지연 하한 `clamp_τ`([lnn/delay.py](lnn/delay.py)) /
  이득장 폭 `σ_g`([lnn/fields.py](lnn/fields.py)).

## 아키텍처

```
입력 Area(encoder) → 처리 Cluster(processor ×L) → 출력 Area(decoder)
```

- **Cell/Area/Cluster** — 육각 격자, 변(face) 소속 분수지연 버퍼, `lax.scan` 동역학(+remat).
  지연은 Tobler형 볼록 법칙 `τ=clamp_τ(τ_base·exp(γ(|s−s*|−|s*|)))`, `s=∇T(m)·ê`
  (선형 법칙은 텔레스코핑으로 붕괴 → 금지, 이유는 코드 주석).
- **경계 재생**(유일한 비선형) — `u_k=sign·|정합피크|`, `다음 진폭 = g_k·tanh(u/u0)`.
- **인코딩/디코딩 plug-in** — `TimeEncoder`(경로 B, 구현) / `FreqEncoder`(경로 A, Block II 예약·인터페이스만).
  토큰 임베딩 = 토큰의 파형(작은 D), 입출력 코드북 공유(weight tying). 디코더: discriminative/structural/vocabulary.
- **학습** — `optax.multi_transform` 파라미터 그룹 `{terrain_h, gain_a, embedding, decoder_head, diag_gain}`,
  커리큘럼 = reservoir(지형·이득·임베딩 고정, readout 학습) → 지형+임베딩 개방, λ 어닐링(코히어런스 평활 등가).

저장소 구조는 [`LNN_SPEC.md` §8](LNN_SPEC.md) 참조.

## 알려진 한계 (§11)

- **코히어런스 길이**가 직접 간섭 가능한 토큰 거리의 상한 → 장문맥 불가(단문·소규모만).
- **선형 보간 분수 지연의 진폭-지연 허위 결합** 잔존(Thiran 미적용 → `# TODO`). 추론 사전합성 `h(t)`도 근사.
- **진폭 라우팅은 Gain Field 의존**, 완전 동적 어텐션 미구현(약화된 라우팅).
- **경로 B 의 차원-시간 결합**: D 를 키우면 롤아웃·격자·코히어런스 길이가 동시에 커져 비용이 초선형.
  큰 D 는 경로 A(차원↔시간 분리)+다중 Area 차원 분할로만 가능 → Block II.
- **D3 채널 직교성 붕괴**(위 §11★) — 경로 B 의 D=8 한계.
- **BPTT 비볼록성·기울기 사막** 가능 → reservoir 기준선이 안전망. 본 결과는 작은 격자(R=5)·짧은 학습의
  존재 증명이며 **현대 LLM/비전 모델과 성능 비교 대상이 아님**.
- 합격선은 SOTA 가 아니라 "**구조가 끝까지 흐른다**" — 세 계층이 네 시연 공통 엔진으로 동작함을 보인다.

## 미구현(`# TODO`, Block II·III) — 인터페이스만 예약

잔차 스트림 도파로 · 시간역전 생성/정상파 어트랙터 · Thiran 분수지연 · 그래프 일반 Cluster ·
FreqEncoder(경로 A) · 나사 전위 루프 + `ρ(A)<1` 정규화 · 추론 임펄스응답 사전합성. 상세는 `LNN_SPEC.md` §0.5.
