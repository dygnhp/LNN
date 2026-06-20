"""Phase 3 일괄 실행 — 경로 A 차원 스케일링 (E-DIM) + Q1~Q4 판정 + 보고서.

frozen gate → D∈{8,16,32} 스윕 → Q1(기하바닥)·Q2(라우팅)·Q3(의미구조)·Q4(천장) 판정 → exlog/.
ARIS·KEI 코어 불변, dim_scaling 만 증축.  사용법: python scripts/run_phase3.py [--quick]
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


def frozen_gate():
    r = subprocess.run([sys.executable, "-m", "pytest",
                        os.path.join(ROOT, "tests", "test_frozen_interface.py"),
                        os.path.join(ROOT, "tests", "test_freq_orthogonality.py"),
                        os.path.join(ROOT, "tests", "test_dim_split.py"), "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    print("[gate] frozen + freq_orth + dim_split:", "PASS" if r.returncode == 0 else "FAIL")
    return r.returncode == 0


def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    print(f"[phase3] {os.path.relpath(edir, ROOT)} (quick={quick})")
    t0 = time.time()
    frozen_ok = frozen_gate()

    import e_dim_scaling
    dims = (8, 16) if quick else (8, 16, 32)
    E = e_dim_scaling.run(n_total=(120 if quick else 240),
                          e_res=(2 if quick else 3), e_open=(5 if quick else 12),
                          mnist_per_class=(15 if quick else 40),
                          m_res=(2 if quick else 3), m_open=(5 if quick else 10), dims=dims)
    write_report(edir, no, date, frozen_ok, E, quick, t0)
    with open(os.path.join(edir, "datasets", "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# 데이터셋 — Phase 3 (경로 A D 스윕)\n\n- 어휘(D3): text_corpus 빈칸(orth/top1).\n"
                "- 이미지: MNIST 8x8(천장).\n- D∈{8,16,32}, M=ceil(D/8) Area 차원 분할.\n")
    print(f"\n[phase3] 저장 완료 -> {os.path.relpath(edir, ROOT)} ({time.time() - t0:.1f}s)")


def write_report(edir, no, date, frozen_ok, E, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    rows = {r["D"]: r for r in E["rows"]}
    Ds = E["dims"]
    lo, hi = Ds[0], Ds[-1]
    # 판정
    Q1 = rows[lo]["orth_max"] > 0.9 and rows[hi]["orth_max"] < rows[lo]["orth_max"] - 0.2
    Q2 = rows[hi]["top1"] > rows[lo]["top1"] + 0.05
    Q3 = all(rows[d]["within"] >= rows[d]["across"] for d in Ds)
    best_acc = max(rows[d]["acc_mnist"] for d in Ds)
    Q4 = best_acc > CEILING

    R = [f"# Block II Phase 3 (경로 A 차원 스케일링) 보고서 — {date}_experiment_{no}", "",
         "경로 A 로 D 를 8→16→32 확장해 작은 D 의 세 매듭(기하바닥·라우팅·천장)을 시험. KEI 코어 불변.", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()} · 경과 {time.time() - t0:.0f}s",
         f"- frozen interface: **{'PASS' if frozen_ok else 'FAIL'}** · 단일 변수=D · M=ceil(D/8) Area 차원분할", "",
         "## D 스윕 (단일 변수 = D)", "",
         "| D | M(Area) | orth_max | within | across | top1 | acc_mnist | rho | fwd(s) |",
         "|---|---------|----------|--------|--------|------|-----------|-----|--------|"]
    for d in Ds:
        r = rows[d]
        R.append(f"| {d} | {r['M']} | **{r['orth_max']:.3f}** | {r['within']:.3f} | {r['across']:.3f} | "
                 f"{r['top1']:.3f} | **{r['acc_mnist']:.3f}** | {r['rho']:.3f} | {r['fwd_sec']} |")
    R += ["", "## Q1~Q4 판정", "",
          f"- **Q1 (기하 바닥)**: orth_max D{lo}={rows[lo]['orth_max']:.3f} → D{hi}={rows[hi]['orth_max']:.3f}. "
          + ("**풀림** — D≥명사수에서 비둘기집 해소(차원이 기하 바닥의 원인)." if Q1
             else "유의 하강 미관측 — 기하 바닥이 D 만으론 안 풀림(정량 확정)."),
          f"- **Q2 (라우팅 병목 O4)**: top1 D{lo}={rows[lo]['top1']:.3f} → D{hi}={rows[hi]['top1']:.3f}. "
          + ("**상승** — 코드 분리가 top1 로 활용됨." if Q2
             else "**미상승** — 코드 분리(Q1)≠활용(O4 재확인). 병목이 readout 너머(동적 라우팅)."),
          f"- **Q3 (의미 구조)**: within≥across 모든 D 유지 {Q3} "
          f"(D{hi}: within={rows[hi]['within']:.3f} ≥ across={rows[hi]['across']:.3f}).",
          f"- **Q4 (천장 0.56) [핵심]**: best acc_mnist={best_acc:.3f} vs {CEILING}. "
          + ("**돌파 — 차원이 천장의 원인.**" if Q4
             else f"**미돌파 — ~{CEILING} 은 차원과도 무관한 더 깊은 물리 한계(4번째 독립 확증: "
                  "용량 Exp3 / 구조 Phase1 / 완전통합 Phase2 / 차원 Phase3).**"),
          "",
          "## 종합 — Phase 3 가 답한 질문", "",
          ("차원이 천장의 원인 → 경로 A 다중 Area 차원 분할을 Block II 능력 경계의 본궤도로(다음: D 비용=Block III)."
           if Q4 else
           f"~{CEILING} 천장은 **차원 축에서도 견고**(4번째 확증). 지연-전용+위상-제약 물리의 근본 상한 확정 → "
           "Block II '능력 경계 측정' 종결. 다음은 능력이 아니라 **효율(Block III)·자율성(Block IV)** 으로 축 전환."),
          (f"- 단, Q1(기하 바닥)은 {'풀림' if Q1 else '미해소'}·Q2(라우팅)는 {'풀림' if Q2 else '미해소'} — "
           + ("기하는 차원이 풀지만 라우팅은 readout 너머(동적 라우팅) 문제로 분리됨(O4 정면 공략 대상)."
              if (Q1 and not Q2) else "두 매듭의 D 의존성을 위 표로 정량 기록.")),
          f"- **비용**: forward 시간 D{lo}={rows[lo]['fwd_sec']}s → D{hi}={rows[hi]['fwd_sec']}s "
          "(경로 A 가 D 를 키워도 롤아웃 길이 불변 — 시간 비용은 Area 수(M)에 선형, 경로 B 의 초선형과 대비).",
          "",
          "## 한계", "",
          "- 존재 증명·축소 학습. 직교성은 within/across 중심 해석(MAX 는 참고, Exp2 교정).",
          "- 경로 A 는 선형 영역 비간섭 — 큰 D 에서 선형 운용 유지(ε 작게). ρ 는 R3 결합계 proxy."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    readme = [f"# {date}_experiment_{no} — Block II Phase 3 (경로 A 차원 스케일링)", "",
              f"D 8→{hi} 스윕으로 기하바닥·라우팅·천장 시험. KEI 코어 불변.", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · frozen {'PASS' if frozen_ok else 'FAIL'}", "",
              "| D | M | orth_max | top1 | acc_mnist |", "|---|---|----------|------|-----------|"]
    for d in Ds:
        r = rows[d]
        readme.append(f"| {d} | {r['M']} | {r['orth_max']:.3f} | {r['top1']:.3f} | {r['acc_mnist']:.3f} |")
    readme += ["",
               f"- Q1 기하바닥 {'풀림' if Q1 else '미해소'} · Q2 라우팅 {'풀림' if Q2 else '미해소'} · "
               f"Q4 천장 {'돌파' if Q4 else '미돌파(4번째 확증)'}"]
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(no=no, date=date, quick=quick, git_commit=commit, frozen_interface_ok=frozen_ok,
                       elapsed_sec=round(time.time() - t0, 1),
                       verdicts=dict(Q1=bool(Q1), Q2=bool(Q2), Q3=bool(Q3), Q4=bool(Q4),
                                     best_acc_mnist=best_acc), e_dim=E),
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
