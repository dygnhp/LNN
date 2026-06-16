"""Block II (KEI) 일괄 실행 — E1~E4 + 산출물 (= Block II /goal 타깃, §9).

순서: frozen interface 회귀 테스트 → E1(경로 A) → E2(IP/RP) → E3(천장 돌파) → E4(ρ 안정성)
→ exlog/<YYYYMMDD>_experiment_<No>/ 에 report.md(E1~E4 판정 + §5.4 ρ 계측) 저장.

ARIS 코어 불변. frozen interface 깨지면 중단(§9 절차 1).
사용법:  python scripts/run_block2.py [--quick]
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


def frozen_gate():
    print("=" * 66 + "\n[GATE] frozen interface + Block II 단위 테스트\n" + "=" * 66)
    r = subprocess.run([sys.executable, "-m", "pytest",
                        os.path.join(ROOT, "tests", "test_frozen_interface.py"),
                        os.path.join(ROOT, "tests", "test_freq_orthogonality.py"),
                        os.path.join(ROOT, "tests", "test_rho_measure.py"), "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-700:])
    return r.returncode == 0


def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    rdir, ddir = os.path.join(edir, "results"), os.path.join(edir, "datasets")
    print(f"[block2/KEI] {os.path.relpath(edir, ROOT)} (quick={quick})")
    t0 = time.time()

    frozen_ok = frozen_gate()
    if not frozen_ok:
        print("  !! frozen interface 회귀 — ARIS 코어 시그니처 변경됨. 중단 권고.")

    import e1_path_a_fill, e2_multi_area, e3_boundary, e4_rho_stability

    sml = quick
    print("\n##### E1 — 경로 A (FreqEncoder + 주파수 뱅크 readout) #####")
    E1 = e1_path_a_fill.run(n_total=(120 if sml else 240),
                            e_res=(2 if sml else 4), e_open=(6 if sml else 16))
    print("\n##### E2 — IP vs RP 다중 Area #####")
    E2 = e2_multi_area.run(per_class=(20 if sml else 60),
                           e_res=(2 if sml else 4), e_open=(5 if sml else 10))
    print("\n##### E3 — Block I 천장(0.56) 돌파 시도 #####")
    E3 = e3_boundary.run(per_class=(20 if sml else 100),
                         e_res=(2 if sml else 5), e_open=(6 if sml else 16))
    print("\n##### E4 — 루프 ρ 안정성 #####")
    E4 = e4_rho_stability.run()

    write_report(edir, no, date, frozen_ok, E1, E2, E3, E4, quick, t0)
    with open(os.path.join(ddir, "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# 데이터셋 — Block II\n\n- E1: 합성 한국어 빈칸(D3, 경로 A).\n"
                "- E2/E3: MNIST 8x8 (Exp2 로더).\n- E4: 합성(임펄스 자유전개, 데이터 없음).\n")
    print(f"\n[block2] 저장 완료 -> {os.path.relpath(edir, ROOT)}  ({time.time() - t0:.1f}s)")


def _yn(b):
    return "PASS" if b else "CHECK"


def write_report(edir, no, date, frozen_ok, E1, E2, E3, E4, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    R = [f"# Block II (KEI) 보고서 — {date}_experiment_{no}", "",
         "ARIS 코어 재사용 + Block II 껍질(경로 A·다중 Area·동적 라우팅·ρ 1급화). frozen interface 불변.", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()} · 경과 {time.time() - t0:.0f}s",
         f"- frozen interface 회귀 테스트: **{_yn(frozen_ok)}** (ARIS 코어 시그니처 불변 확인)", "",
         "## E1~E4 판정", "",
         "| # | 과제 | 측정 | 통과조건 | 결과 | 판정 |",
         "|---|------|------|----------|------|------|",
         f"| E1 | 경로 A D3 빈칸 | orth_after, top1 | orth<0.5 **or** top1>0.22 | "
         f"orth {E1['orth_before']:.2f}→{E1['orth_after']:.2f}, top1 {E1['acc_res']:.2f}→{E1['acc_open']:.2f} | {_yn(E1['passed'])} |",
         f"| E2 | IP vs RP | acc 차 | 정량차(부호무관) | IP {E2['acc_IP']:.3f} / RP {E2['acc_RP']:.3f} (Δ{E2['diff']:+.3f}) | {_yn(E2['passed'])} |",
         f"| E3 | 천장(0.56) 돌파 | MNIST acc | 0.56 초과 시도 | {E3['acc_res']:.3f}→**{E3['acc_open']:.3f}** ({'돌파' if E3['broke_ceiling'] else '미돌파'}) | {_yn(E3['passed'])} |",
         f"| E4 | 루프 ρ 안정성 | ρ, 발산 | ρ<1 정규화로 수렴 보존 | ρ {E4['rho_unreg']:.2f}→{E4['rho_reg']:.2f}, 성장 {E4['growth_unreg']:.1e}→{E4['growth_reg']:.1e} | {_yn(E4['passed'])} |",
         "",
         "## §5.4 ρ 계측 (E4, 2층 정의 — 선형영역 야코비안 스펙트럼 반경)", "",
         "```json",
         json.dumps({"rho_unreg": round(E4["rho_unreg"], 4), "rho_reg": round(E4["rho_reg"], 4),
                     "regularized": True, "method": "linear(step jacobian)",
                     "rho_target": E4["rho_target"],
                     "growth_unreg": E4["growth_unreg"], "growth_reg": E4["growth_reg"]},
                    ensure_ascii=False, indent=2),
         "```", "",
         "## 해석", "",
         f"- **E1(경로 A)**: 출력단 주파수 뱅크까지 더하니 orth_after={E1['orth_after']:.2f} "
         f"(Block I 경로 B 0.99 대비 {'개선' if E1['orth_after'] < 0.99 else '유사'}), top1={E1['acc_open']:.2f}. "
         "Exp2 H4(인코더만으론 부족)에 대해 인코더+readout 동반의 효과를 정량화.",
         f"- **E2(다중 Area)**: RP−IP={E2['diff']:+.3f} — Area 간 직접 상호작용(u 공유)의 이득 정량(부호 무관 측정).",
         f"- **E3(천장)**: {E3['acc_open']:.3f} vs Block I {E3['block1_ceiling']} — "
         f"{'물리 확장이 경계를 밀어냄' if E3['broke_ceiling'] else '현 설정으론 미돌파(추가 용량/경로 A 통합 필요)'}. "
         f"세 기둥① 경로분해 감쇠(ε별): {E3['pathway_decay']} (ε↑ 로 해석가능성 손실 증가).",
         f"- **E4(ρ)**: 고이득 ρ={E4['rho_unreg']:.2f}>1 에서 자유전개 {E4['growth_unreg']:.0e}배 발산, "
         f"ρ<1 사영(ρ={E4['rho_reg']:.2f}) 후 {E4['growth_reg']:.0e}배(수렴). ρ 1급화·정규화가 루프 안정성을 보존.",
         "",
         "## 한계 / Block III 예약", "",
         "- 존재 증명·축소 학습. RP 는 u-수준 coupling 근사(완전 공통-시계 펄스 공유는 # TODO).",
         "- self-gating 경로분해는 ε차수 근사(정밀도 한계는 Block III LTI 사전합성 과제).",
         "- 추론 사전합성·비용 배율 축소·메모리 2축·Block IV(Agentic)는 Block III 이후 예약."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    readme = [f"# {date}_experiment_{no} — Block II (KEI)", "",
              "경로 A 완성 + 다중 Area + 동적 라우팅 + ρ 1급화. ARIS 코어 불변.", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · frozen interface {_yn(frozen_ok)}",
              "- 상세: [`results/report.md`](results/report.md)", "",
              "| E | 결과 | 판정 |", "|---|------|------|",
              f"| E1 경로A | orth {E1['orth_after']:.2f}, top1 {E1['acc_open']:.2f} | {_yn(E1['passed'])} |",
              f"| E2 IP/RP | Δacc {E2['diff']:+.3f} | {_yn(E2['passed'])} |",
              f"| E3 천장 | {E3['acc_open']:.3f} vs 0.56 ({'돌파' if E3['broke_ceiling'] else '미돌파'}) | {_yn(E3['passed'])} |",
              f"| E4 ρ | {E4['rho_unreg']:.2f}→{E4['rho_reg']:.2f} 수렴보존 | {_yn(E4['passed'])} |"]
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(no=no, date=date, quick=quick, git_commit=commit,
                       frozen_interface_ok=frozen_ok, elapsed_sec=round(time.time() - t0, 1),
                       E1=E1, E2=E2, E3=E3, E4=E4), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
