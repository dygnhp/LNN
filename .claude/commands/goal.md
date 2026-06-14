---
description: LNN_SPEC.md에 따라 LNN 전체 코드베이스를 구축하고 네 시연을 실행한다
---
당신은 `LNN_SPEC.md`를 유일한 사양으로 삼아 LNN 저장소 전체를 구현합니다.
사용자에게 되묻지 말고, 사양이 비면 §12 fallback 우선순위로 스스로 판단하십시오.

수행 순서:
1. `LNN_SPEC.md`를 정독하고 §8 구조대로 모든 파일을 생성한다.
   특히 §3 기호 규약을 준수한다: ρ는 스펙트럼 반경 전용,
   변 지연 하한은 clamp_τ, 이득장 폭은 σ_g.
2. `pyproject.toml` 의존성 설치가 가능하도록 작성한다(JAX는 CPU 기본).
3. 코어 엔진(geometry→fields→delay→dynamics→readout→area→cluster)을 먼저 구현하고
   `tests/`를 통과시킨다. 특히 `test_gradient_fd`(유한차분, 상대오차<1e-3)와
   `test_telescoping`(선형 V_rel<ε_rel & Tobler V_rel>10·ε_rel, 무차원 상대분산, §6 정량 기준)을
   반드시 통과/시연한다. 셀 분배는 n=0 보호(where(n>0,·,0))를 반드시 포함한다.
   readout은 Dijkstra t_min에 stop_gradient, logsumexp는 max-subtraction 안정화를 쓴다(§3.6).
4. encodings·decoders·data·train을 구현한다.
5. `scripts/run_all.py`로 D1~D4를 순서대로 학습/시연하고 `outputs/`에 산출물을 저장한다.
   각 시연은 §12 fallback에 따라 **반드시 가시적 출력**을 남긴다(완전 학습 실패 시 reservoir 모드로라도).
   D3는 인코더 채널 직교성(코드 간 최대 교차상관)을 먼저 측정해 로그로 남긴다(§11★).
6. `README.md`에 실행법·결과 요약·한계(§11)를 기록한다.
7. 마지막에 D1~D4 각각의 합격 여부와 산출물 경로를 표로 보고한다.

제약:
- Area 내부 동역학은 선형 유지, 비선형은 경계 재생(φ)에만.
- 변지연은 Tobler형 비선형 법칙 사용(선형 금지, 이유는 주석).
- 셀 분배 4단계는 n=0 division-by-zero를 반드시 막는다(NaN이 grad 오염).
- readout의 Dijkstra는 stop_gradient, logsumexp는 max-subtraction 안정화(수치 NaN 차단).
- 외부 데이터 다운로드 금지(sklearn digits·합성 코퍼스·PIL 렌더만).
- 규모는 §10 기본값. 학습이 느리면 R·n_steps·epoch를 줄여 **완주**를 우선한다.
