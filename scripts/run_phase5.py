"""Phase 5 일괄 실행 — 비선형 깊이 (E-DEPTH) + G1~G6 판정 + 보고서. Block II 마지막 실험.

frozen gate → depth∈{1,2,4,6} 스윕(방식 A·C) → G1(천장)·G2(해석 대가)·G3(안정)·G5(A vs C) 판정.
G1 돌파=얕은 비선형 한계 / 미돌파=6번째 확증(Block II 완전 종결). exlog/ 저장.
사용법: python scripts/run_phase5.py [--quick]
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
                        os.path.join(ROOT, "tests", "test_depth_pathdecomp.py"),
                        os.path.join(ROOT, "tests", "test_ral_depth.py"), "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    print("[gate] frozen + depth_pathdecomp + ral_depth:", "PASS" if r.returncode == 0 else "FAIL")
    return r.returncode == 0


def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    print(f"[phase5] {os.path.relpath(edir, ROOT)} (quick={quick})")
    t0 = time.time()
    frozen_ok = frozen_gate()

    import e_depth_scaling
    depths = (1, 2, 4) if quick else (1, 2, 4, 6)
    E = e_depth_scaling.run(per_class=(12 if quick else 20),
                            e_res=(1 if quick else 2), e_open=(4 if quick else 8), depths=depths)
    write_report(edir, no, date, frozen_ok, E, quick, t0)
    with open(os.path.join(edir, "datasets", "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# 데이터셋 — Phase 5 (비선형 깊이)\n\n- MNIST 8x8(천장). depth∈"
                f"{list(depths)}, 방식 A(L 확장)·C(RAL 순환).\n")
    print(f"\n[phase5] 저장 완료 -> {os.path.relpath(edir, ROOT)} ({time.time() - t0:.1f}s)")


def write_report(edir, no, date, frozen_ok, E, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    A = {a["depth"]: a for a in E["A"]}
    Cc = {c["depth"]: c for c in E["C"]}
    depths = E["depths"]
    lo, hi = depths[0], depths[-1]
    best_acc = max(max(a["acc"] for a in E["A"]), max(c["acc"] for c in E["C"]))
    G1 = best_acc > CEILING
    G2 = A[hi]["residual"] > A[lo]["residual"]            # 깊이↑ → 경로분해 잔차↑(대가)
    G3 = all(c["loop_growth"] < 2.0 for c in E["C"])      # RAL 루프 유계(tanh 안정)
    # G5: 같은 최대 깊이에서 A vs C
    a_hi, c_hi = A[hi]["acc"], Cc[hi]["acc"]
    g5 = ("C≈A (파라미터 공유로 같은 깊이 효과 — RAL 효율적)" if abs(a_hi - c_hi) < 0.05
          else ("A>C (독립 파라미터 다양성 필요 — RAL 표현력 한계)" if a_hi > c_hi
                else "C>A (공유 순환이 더 나음)"))

    R = [f"# Block II Phase 5 (비선형 깊이) 보고서 — {date}_experiment_{no} · **Block II 마지막 실험**", "",
         "다섯 축(용량·구조·통합·차원·공간)이 모두 선형 매질 다이얼이었다 → 마지막 축 비선형 깊이"
         "(경계 φ 통과 횟수)로 ~0.56 이 얕은 비선형의 한계인지 물리 절대 한계인지 판별. 코어 불변.", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()} · 경과 {time.time() - t0:.0f}s",
         f"- frozen interface: **{'PASS' if frozen_ok else 'FAIL'}** · 단일 변수=깊이 · ρ<1: RAL 은 tanh 가 진폭 유계화", "",
         "## 깊이 스윕 (방식 A: L 확장 / 방식 C: RAL 순환)", "",
         "| depth | A.acc | A.resid(해석대가) | A.params | C.acc | C.params | C.loop_growth |",
         "|-------|-------|-------------------|----------|-------|----------|---------------|"]
    for d in depths:
        R.append(f"| {d} | **{A[d]['acc']:.3f}** | {A[d]['residual']} | {A[d]['params']} | "
                 f"**{Cc[d]['acc']:.3f}** | {Cc[d]['params']} | {Cc[d]['loop_growth']} |")
    R += ["", "## G1~G6 판정", "",
          f"- **G1 (천장) [핵심]**: best acc={best_acc:.3f} vs {CEILING} (MLP {MLP_REF}). "
          + ("**돌파 — 천장은 얕은 비선형의 한계. LNN 은 깊이로 능력↑(단 G2 해석 대가).**" if G1
             else f"**미돌파 — ~{CEILING} 은 비선형 깊이와도 무관(6번째 독립 확증: "
                  "용량·구조·통합·차원·공간·**깊이**). 지연-전용 물리의 진정 절대 상한.**"),
          f"- **G2 (해석 대가) [핵심]**: 경로분해 잔차 A L{lo}={A[lo]['residual']} → L{hi}={A[hi]['residual']} "
          + (f"(**증가** — 깊이↑로 세 기둥① 손실↑, 능력-해석 trade-off 정량)." if G2
             else "(유의 증가 미관측)."),
          f"- **G3 (안정성)**: RAL 루프 성장비 모두 유계 {G3} (tanh 가 진폭 유계화 — nilpotent 깬 루프에서 "
          "ρ<1 자연 충족; 선형 루프 발산은 test_ral_depth 에서 확인).",
          "- **G4 (의미 구조)**: 어휘 within/across 깊이 스윕은 런타임상 이번 회차 보류(MNIST 천장 G1 우선).",
          f"- **G5 (A vs C: 깊이의 출처) [핵심]**: 최대 깊이 {hi} 에서 A.acc={a_hi:.3f}(params {A[hi]['params']}) "
          f"vs C.acc={c_hi:.3f}(params {Cc[hi]['params']}) → {g5}.",
          "- **G6 (방식 B, ε)**: A·C 가 천장을 못 밀 때만 — " + ("G1 돌파로 불요." if G1 else "차순위(이번 회차 미실행)."),
          "",
          "## 능력-해석 trade-off (G1×G2)", "",
          "| depth | acc(A) | 경로분해 잔차(A) |", "|-------|--------|------------------|"]
    for d in depths:
        R.append(f"| {d} | {A[d]['acc']:.3f} | {A[d]['residual']} |")
    R += ["", "(깊이↑로 acc 변화 vs 해석 가능성 손실 — LNN 근본 긴장의 정면 측정.)", "",
          "## 종합 — Block II 종결", "",
          ("천장이 깊이로 **이동** → ~0.56 은 얕은 비선형의 한계였음. Block II 가 '능력-해석 trade-off 곡선'"
           "으로 재정의(원하는 해석 수준을 고르면 능력이 정해지는 다이얼). 이후 Block III(비용)·IV(자율성)."
           if G1 else
           f"비선형 깊이(L·n {lo}→{hi})에도 best acc {best_acc:.3f} < {CEILING} → **~0.56 은 여섯 축 "
           "(용량·구조·통합·차원·공간·깊이) 모두에서 견고**. 지연-전용+위상-제약 물리의 **진정 절대 표현력 "
           "상한**으로 확정. **Block II 능력 경계 측정 완전 종결.** Phase 4 가 보인 '재료는 늘었으나 활용 "
           "불가'(라우팅 병목)가 깊이로도 안 풀림 — 천장은 재료가 아니라 지연-전용 물리의 함수류 자체."),
          "",
          "## 정직성 노트 — LNN 가치 명제 최종 확정", "",
          ("천장 이동 → trade-off 곡선이 LNN 의 정체성." if G1 else
           "0.56 vs MLP 0.83 격차는 **여섯 축에서 확정된 표현력 대가**다. LNN 의 가치 명제는 '정확도 경쟁자'가 "
           "아니라 **'해석 가능하고 물리 실현 가능한 계산의 정량적 경계 증명'** — 6중 독립 확증 + 그 대가로 얻은 "
           "해석 가능성·파라미터 효율(MLP 대비 ~13×↓)·물리 사상 가능성. Block III(효율)·IV(자율성)로 축 전환."),
          "",
          "## 한계", "",
          "- 존재 증명·축소 학습(깊이 6 은 epoch·데이터 축소). G4 어휘 깊이 스윕·G6 ε 스윕은 차순위 보류.",
          "- 방식 C(RAL) 는 tanh 자연 유계 — 명시적 ρ 사영은 선형 루프 발산 테스트로 대체 검증."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    readme = [f"# {date}_experiment_{no} — Block II Phase 5 (비선형 깊이, 마지막 실험)", "",
              "깊이(φ 통과 횟수)로 ~0.56 판별. 방식 A(L 확장)·C(RAL). 코어 불변.", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · frozen {'PASS' if frozen_ok else 'FAIL'}", "",
              "| depth | A.acc | A.params | C.acc | C.params |", "|---|---|---|---|---|"]
    for d in depths:
        readme.append(f"| {d} | {A[d]['acc']:.3f} | {A[d]['params']} | {Cc[d]['acc']:.3f} | {Cc[d]['params']} |")
    readme += ["",
               f"- G1 천장 {'돌파' if G1 else '미돌파(6번째 확증 → Block II 완전 종결)'} · best acc {best_acc:.3f} vs 0.56 (MLP 0.83)",
               f"- G5 깊이의 출처: {g5}"]
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(no=no, date=date, quick=quick, git_commit=commit, frozen_interface_ok=frozen_ok,
                       elapsed_sec=round(time.time() - t0, 1),
                       verdicts=dict(G1=bool(G1), G2=bool(G2), G3=bool(G3), best_acc=best_acc, g5=g5),
                       e_depth=E), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
