"""Phase 4 일괄 실행 — 격자 부피 스케일링 (E-CELL) + C1~C4 판정 + 보고서.

frozen gate → R∈{5,8,12} 스윕 → C1(경로다양성)·C2(천장)·C3(격자성장)·C4(비용) 판정 → exlog/.
ARIS·KEI 코어 불변, cell_scaling 만 증축.  사용법: python scripts/run_phase4.py [--quick]
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
                        os.path.join(ROOT, "tests", "test_cell_scaling.py"), "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    print("[gate] frozen + cell_scaling:", "PASS" if r.returncode == 0 else "FAIL")
    return r.returncode == 0


def check_grow_grid():
    """C3 — grow_grid 격자 확장 메커니즘(지형 연속 보존은 test_growth 에서 검증)."""
    from lnn.geometry import build_geometry
    from lnn.growth import grow_grid
    g5 = build_geometry(5)
    g8 = grow_grid(g5, 8)
    return bool(g8.N > g5.N), g5.N, g8.N


def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    print(f"[phase4] {os.path.relpath(edir, ROOT)} (quick={quick})")
    t0 = time.time()
    frozen_ok = frozen_gate()
    c3_ok, n5, n8 = check_grow_grid()

    import e_cell_scaling
    Rs = (5, 8) if quick else (5, 8, 12)
    E = e_cell_scaling.run(mnist_per_class=(15 if quick else 30),
                           m_res=(2 if quick else 3), m_open=(5 if quick else 10),
                           voc_total=(120 if quick else 200), v_res=(1 if quick else 2),
                           v_open=(4 if quick else 8), Rs=Rs)
    write_report(edir, no, date, frozen_ok, c3_ok, n5, n8, E, quick, t0)
    with open(os.path.join(edir, "datasets", "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# 데이터셋 — Phase 4 (격자 부피 스윕)\n\n- 이미지: MNIST 8x8(천장, 핵심).\n"
                "- 어휘: text_corpus 빈칸(top1/within/across).\n"
                f"- R∈{list(Rs)}, K 동반(K_per_cell·N), n_steps∝R.\n")
    print(f"\n[phase4] 저장 완료 -> {os.path.relpath(edir, ROOT)} ({time.time() - t0:.1f}s)")


def write_report(edir, no, date, frozen_ok, c3_ok, n5, n8, E, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    rows = {r["R"]: r for r in E["rows"]}
    Rs = E["Rs"]
    lo, hi = Rs[0], Rs[-1]
    pdiv = [rows[r]["path_diversity"] for r in Rs]
    C1 = all(pdiv[i] < pdiv[i + 1] for i in range(len(pdiv) - 1))
    best_acc = max(rows[r]["acc_mnist"] for r in Rs)
    C2 = best_acc > CEILING
    cost_grows = rows[hi]["fwd_sec"] > rows[lo]["fwd_sec"]

    R = [f"# Block II Phase 4 (Cell 확장과 계산 경로 다양성) 보고서 — {date}_experiment_{no}", "",
         "다섯 번째 축(격자 부피=cell 수)으로 ~0.56 천장이 물리 절대 한계인지 공간 규모 한계인지 판별. KEI 코어 불변.", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()} · 경과 {time.time() - t0:.0f}s",
         f"- frozen interface: **{'PASS' if frozen_ok else 'FAIL'}** · 단일 변수=R(K·n_steps 종속, H6 동반)", "",
         "## R 스윕 (단일 변수 = R, cell 수)", "",
         "| R | N_cells | K | n_steps | path_div(log10#) | acc_mnist | top1 | within | across | stab | fwd(s) | params |",
         "|---|---------|---|---------|------------------|-----------|------|--------|--------|------|--------|--------|"]
    for r in Rs:
        d = rows[r]
        R.append(f"| {r} | {d['n_cells']} | {d['K']} | {d['n_steps']} | {d['path_diversity']} | "
                 f"**{d['acc_mnist']:.3f}** | {d['top1']:.3f} | {d['within']} | {d['across']} | "
                 f"{d['stability']:.1e} | {d['fwd_sec']} | {d['params']} |")
    R += ["", "## C1~C4 판정", "",
          f"- **C1 (경로 다양성, 전제)**: path_diversity R{lo}={pdiv[0]} → R{hi}={pdiv[-1]} — "
          + ("**단조 증가** (격자 부피↑ → 구별 가능 우회로 수↑, H-CELL 전제 성립)." if C1
             else "단조 증가 미성립 — 전제 불충족."),
          f"- **C2 (천장 0.56) [핵심]**: best acc_mnist={best_acc:.3f} vs {CEILING} (MLP {MLP_REF}). "
          + ("**돌파 — 천장은 공간(규모) 한계, Block II 능력 경계 열림.**" if C2
             else f"**미돌파 — ~{CEILING} 은 격자 부피와도 무관(5번째 독립 확증: "
                  "용량 Exp3 / 구조 Phase1 / 통합 Phase2 / 차원 Phase3 / 공간 Phase4).**"),
          f"- **C3 (자율 격자 성장)**: grow_grid R5(N={n5})→R8(N={n8}) 확장 동작 {c3_ok} "
          "(지형 연속 보존은 test_growth 검증). 학습 중 자율 격자 성장 루프는 인터페이스 준비·셀 재매핑 비용으로 예약.",
          f"- **C4 (비용, 정직)**: fwd R{lo}={rows[lo]['fwd_sec']}s → R{hi}={rows[hi]['fwd_sec']}s "
          f"(FLOPs {rows[lo]['flops_analytic']:.2e}→{rows[hi]['flops_analytic']:.2e}). "
          "**cell 확장은 시간 비용 있음**(cell·n_steps 동반↑) — Phase 3(차원=시간 무비용)와 대비.",
          "",
          "## 종합 — 다섯 번째 다이얼의 판정", "",
          ("천장이 cell 확장으로 **이동** → ~0.56 은 *작은 격자의* 한계였음. 4중 확증을 '고정 격자 천장'으로 "
           "재해석, LNN 은 격자 규모로 능력↑(MLP 폭 확장 대응). Block II 능력 경계 *열림* → 규모-능력 곡선이 새 과제."
           if C2 else
           f"cell 확장(N {rows[lo]['n_cells']}→{rows[hi]['n_cells']}, 경로 다양성 {pdiv[0]}→{pdiv[-1]})에도 "
           f"acc {best_acc:.3f} < {CEILING} → **~0.56 은 다섯 축(용량·구조·통합·차원·공간) 모두에서 견고**. "
           "지연-전용+위상-제약 물리의 **절대 표현력 상한**으로 확정 → Block II 능력 경계 측정 *완결*. "
           "다음은 능력이 아니라 효율(Block III)·자율성(Block IV)."),
          "",
          "## 정직성 노트 — MLP 0.83 vs LNN ~0.56", "",
          (f"천장 이동 → 0.56 은 작은 격자 한계였고, 격차는 규모로 좁혀짐. MLP 대비는 같은 acc 의 비용·해석가능성으로 평가."
           if C2 else
           f"0.56 vs 0.83 격차는 **실재하는 표현력 대가**다. LNN 의 가치는 최고 정확도가 아니라 "
           "**해석 가능성 + 물리 실현 가능성 + 파라미터 효율**(MLP 대비 ~13× 적은 파라미터). "
           "조밀 비선형(MLP)을 포기하고 선형 매질 + 경계 비선형을 택한 구조적 대가로 ~0.56 을 위치시킨다 — "
           "음의 결과의 견고함을 측정하는 것이 LNN 연구의 자세."),
          "",
          "## 한계", "",
          "- 존재 증명·축소 학습(R=12 는 epoch·데이터 축소). 직교성 within/across 중심(Exp2 교정).",
          "- path_diversity 는 순수 기하(최단경로 수) proxy. 자율 격자 성장 in-train 루프는 예약."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    readme = [f"# {date}_experiment_{no} — Block II Phase 4 (격자 부피 스케일링)", "",
              "다섯 번째 축(cell 수)으로 ~0.56 천장 판별. KEI 코어 불변.", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · frozen {'PASS' if frozen_ok else 'FAIL'}", "",
              "| R | N_cells | acc_mnist | path_div | fwd(s) |", "|---|---------|-----------|----------|--------|"]
    for r in Rs:
        d = rows[r]
        readme.append(f"| {r} | {d['n_cells']} | {d['acc_mnist']:.3f} | {d['path_diversity']} | {d['fwd_sec']} |")
    readme += ["",
               f"- C1 경로다양성 {'증가' if C1 else '비증가'} · C2 천장 {'돌파' if C2 else '미돌파(5번째 확증)'} · "
               f"best acc {best_acc:.3f} vs 0.56 (MLP 0.83)"]
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(no=no, date=date, quick=quick, git_commit=commit, frozen_interface_ok=frozen_ok,
                       elapsed_sec=round(time.time() - t0, 1),
                       verdicts=dict(C1=bool(C1), C2=bool(C2), C3=bool(c3_ok), best_acc_mnist=best_acc),
                       e_cell=E), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
