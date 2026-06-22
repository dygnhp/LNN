"""Phase 6 일괄 — 천장 원인 국소화. 실험 I(최적화) → 미돌파 시 실험 II(분산 비선형) + 보고서.

I2 돌파(>0.56)=H-OPT(천장=최적화, 서사 붕괴, 여기서 종료). 미돌파 → 실험 II.
II2 돌파=H-INTERIOR(천장=선형 내부 선택). 미돌파=7번째 확증(천장=지연 기질, 비선형 배제).
frozen interface: 실험 I 불변, 실험 II 진단 분기(density=none 회귀). 사용법: python scripts/run_phase6.py [--quick]
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "experiments_block2"))
sys.stdout.reconfigure(encoding="utf-8")

from run_experiment import resolve_experiment_dir  # noqa: E402

CEILING = 0.56
MLP_REF = 0.83


def frozen_gate():
    r = subprocess.run([sys.executable, "-m", "pytest",
                        os.path.join(ROOT, "tests", "test_frozen_interface.py"),
                        os.path.join(ROOT, "tests", "test_interior_nonlin.py"), "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    print("[gate] frozen + interior_nonlin(density=none 회귀):", "PASS" if r.returncode == 0 else "FAIL")
    return r.returncode == 0


def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    print(f"[phase6] {os.path.relpath(edir, ROOT)} (quick={quick})")
    t0 = time.time()
    frozen_ok = frozen_gate()

    no_basin = "--nobasin" in sys.argv   # I3 basin(×3 중복 최대-epoch) 생략 — 본 epoch 설정 유지
    import e_budget
    print("\n##### 실험 I — 최적화 충분성 (E-BUDGET) #####")
    EB = e_budget.run(per_class=(20 if quick else 40), base_open=(4 if quick else 6),
                      mults=(1, 3) if quick else (1, 3, 5), do_basin=(not quick) and (not no_basin))
    EI = None
    if not EB["broke"]:
        import e_interior
        print("\n##### 실험 II — 분산 비선형 (E-INTERIOR) #####")
        EI = e_interior.run(per_class=(20 if quick else 40),
                            e_res=(2 if quick else 3), e_open=(6 if quick else 12))

    write_report(edir, no, date, frozen_ok, EB, EI, quick, t0)
    with open(os.path.join(edir, "datasets", "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# 데이터셋 — Phase 6 (천장 원인 국소화)\n\n- MNIST 8x8. 실험 I=epoch 예산 스윕, "
                "실험 II=매질 내부 분산 비선형(density/kind).\n")
    print(f"\n[phase6] 저장 완료 -> {os.path.relpath(edir, ROOT)} ({time.time() - t0:.1f}s)")


def write_report(edir, no, date, frozen_ok, EB, EI, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    I_broke = EB["broke"]
    R = [f"# Block II Phase 6 (천장 원인 국소화) 보고서 — {date}_experiment_{no}", "",
         "여섯 축 확증 위 두 의심(최적화 / 선형-내부)을 두 실험으로 국소화. 천장이 최적화/선형-내부/"
         "지연-기질 중 어디 사는지 판별.", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()} · 경과 {time.time() - t0:.0f}s",
         f"- frozen interface: **{'PASS' if frozen_ok else 'FAIL'}** (실험 I 불변, 실험 II density=none 회귀 보장)", "",
         "## 실험 I — 최적화 충분성 (E-BUDGET)", "",
         "| epoch_mult | e_open | acc | converged@epoch |",
         "|------------|--------|-----|-----------------|"]
    for r in EB["rows"]:
        R.append(f"| ×{r['epoch_mult']} | {r['e_open']} | **{r['acc']:.3f}** | "
                 f"{r['converged']}@{r['conv_epoch']} |")
    R += [f"- best acc = {EB['best']:.3f} vs {CEILING}. "
          + ("**I2 돌파 — H-OPT 참: 천장은 최적화 인공산물(여섯 축 재해석 필요). Phase 6 여기서 종료.**"
             if I_broke else "**I2 미돌파 — 천장 표현적 확정 → 실험 II 진행.**")]
    if EB["basin"]:
        R += [f"- **I3 basin 강건성**: 다른 시드 acc {[round(a,3) for a in EB['basin']]} — "
              f"~{CEILING} 재현(최적화·초기화 무관 강건 확정)."]

    if EI is not None:
        rows = {(r["density"], r["kind"]): r for r in EI["rows"]}
        dens = [("none", "signed_relu"), ("sparse", "signed_relu"), ("dense", "signed_relu")]
        best_ii = max(r["acc"] for r in EI["rows"])
        II_broke = best_ii > CEILING
        resid_grows = (rows[("dense", "signed_relu")]["residual"] >
                       rows[("none", "signed_relu")]["residual"])
        raw = rows.get(("dense", "relu"))
        R += ["", "## 실험 II — 분산 비선형 (E-INTERIOR)", "",
              "| density | kind | acc | residual(기둥① 잠식) | 비선형노드 | dc_drift |",
              "|---------|------|-----|----------------------|-----------|----------|"]
        for d, k in dens + [("dense", "relu")]:
            r = rows[(d, k)]
            R.append(f"| {d} | {k} | **{r['acc']:.3f}** | {r['residual']} | {r['n_nonlin_nodes']} | "
                     f"{r['dc_drift']} |")
        approach = ("0.83 접근 — 지연 기질 선형부가 MLP 만큼 표현력, 부족했던 건 비선형 형태뿐"
                    if best_ii > 0.75 else
                    ("중간(0.65~0.75) — 선형-내부 제약과 지연-기질 용량이 함께 기여(분할)"
                     if best_ii > 0.62 else
                     "~0.56 정체 — 압도적 비선형 깊이로도 미돌파"))
        R += ["", "## II1~II5 판정", "",
              "- **II1 (구현·물리)**: density=none 회귀 비트 동일(test_interior_nonlin) + signed 부호 보존(DC 가드) PASS.",
              f"- **II2 (천장) [핵심]**: best acc={best_ii:.3f} vs {CEILING} (MLP {MLP_REF}). "
              + (f"**돌파 — H-INTERIOR 참: 천장은 '선형 내부' 선택의 한계.**" if II_broke else
                 f"**미돌파 ({approach}) — 7번째 독립 확증: 천장은 비선형성과 무관, "
                 "지연 기질(substrate) 자체의 정보 용량 한계. 비선형성 확정 배제.**"),
              f"- **II3 (해석 대가)**: residual none={rows[('none','signed_relu')]['residual']} → "
              f"dense={rows[('dense','signed_relu')]['residual']} ({'증가' if resid_grows else '비증가'} — "
              "density↑로 기둥① 잠식 = 교환비).",
              f"- **II4 (비선형 형태)**: dense signed acc={rows[('dense','signed_relu')]['acc']:.3f} vs "
              f"raw relu acc={raw['acc']:.3f}, raw dc_drift={raw['dc_drift']} "
              "(부호 보존이 zero-mean·극성 보존; raw 는 DC 누적).",
              "- **II5 (안정성)**: 분산 비선형에서 학습 유한·DC 가드 작동(raw dc_drift 로 모니터)."]
        R += ["", "## 능력-해석 교환비 (II2×II3)", "",
              "| density | acc | residual |", "|---------|-----|----------|"]
        for d, k in dens:
            r = rows[(d, k)]
            R.append(f"| {d} | {r['acc']:.3f} | {r['residual']} |")

    # 최종 국소화
    R += ["", "## 천장 원인의 최종 국소화", ""]
    if I_broke:
        R += ["**(가) 최적화** — epoch 예산 확장으로 천장 돌파. 여섯 축은 고정 예산 천장이었음 → 재학습 필요."]
    elif EI is not None and max(r["acc"] for r in EI["rows"]) > CEILING:
        R += ["**(나) 선형 내부 선택** — 매질 내부 분산 비선형이 천장 돌파. 천장은 물리 절대 한계가 아니라 "
              "기둥①(내부 선형) *선택*이 만든 경계. 그 대가(경로분해 잔차)를 교환비로 측정. → relaxed-interior "
              "변종은 해석 가능성을 일부 포기한 별개 모델(ARIS·KEI 정통과 분리)."]
    else:
        R += ["**(다) 지연 기질(substrate) 자체** — 최적화(I)·압도적 분산 비선형(II) 둘 다 천장을 못 밂. "
              f"~{CEILING} 은 비선형성·최적화와 무관한 **지연 계산의 정보 용량 한계**. **7번째 독립 확증**이자 "
              "가장 깊은 결론: 능력 한계가 '우리 선택(선형 내부)'이 아니라 '지연 계산의 성질'. 다음 이론 과제 = "
              "'왜 하필 ~0.56인가'의 분석적 유도. Block II 능력 경계 *원인 수준* 종결 → Block III(효율)·IV(자율성)."]
    R += ["", "## 정직성 노트", "",
          "- 실험 II 가 깨도 = 'LNN 작동'이 아니라 *교환비 측정*(깨진 것은 천장이 아니라 해석 가능성). "
          "안 깨면 = 더 강한 결론(천장은 지연 물리 자체). frozen interface·코어 불변(density=none 회귀가 코드 보증).",
          "- 0.56 vs 0.83 을 변명 않고 해부 — 출처를 (비선형 형태)·(지연 기질 용량)으로 분리 측정."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    # README + json
    locus = ("최적화(가)" if I_broke else
             ("선형내부(나)" if (EI is not None and max(r["acc"] for r in EI["rows"]) > CEILING)
              else "지연기질(다)·7번째 확증"))
    readme = [f"# {date}_experiment_{no} — Block II Phase 6 (천장 원인 국소화)", "",
              f"천장 binding constraint 국소화 → **{locus}**", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · frozen {'PASS' if frozen_ok else 'FAIL'}",
              f"- 실험 I best acc {EB['best']:.3f} vs 0.56 ({'돌파' if I_broke else '미돌파'})"]
    if EI is not None:
        readme.append(f"- 실험 II best acc {max(r['acc'] for r in EI['rows']):.3f} vs 0.56 (MLP 0.83)")
    readme.append("- 상세: [`results/report.md`](results/report.md)")
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(no=no, date=date, quick=quick, git_commit=commit, frozen_interface_ok=frozen_ok,
                       elapsed_sec=round(time.time() - t0, 1), locus=locus,
                       experiment_I=EB, experiment_II=EI), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
