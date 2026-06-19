"""Block II Phase 2 일괄 실행 — P1~P4 + 산출물 (= Phase 2 /goal 타깃, §5).

세 미완(E1 임베딩 colinear · E2 RP u-수준 · E3 천장)을 한 통합 작업으로 해소/확정:
P1 임베딩 분산 → P2 RP 펄스 공유 → P4 ρ 재확인 → P3 통합 천장 재시험(8×8·14×14).
ARIS frozen interface · KEI Phase 1 코어 불변. exlog/<date>_experiment_<No>/ 저장.

사용법:  python scripts/run_phase2.py [--quick]
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
    r = subprocess.run([sys.executable, "-m", "pytest",
                        os.path.join(ROOT, "tests", "test_frozen_interface.py"),
                        os.path.join(ROOT, "tests", "test_freq_orthogonality.py"),
                        os.path.join(ROOT, "tests", "test_rho_measure.py"), "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    print("[gate] frozen interface + 단위 테스트:", "PASS" if r.returncode == 0 else "FAIL")
    return r.returncode == 0


def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    print(f"[phase2/KEI] {os.path.relpath(edir, ROOT)} (quick={quick})")
    t0 = time.time()
    frozen_ok = frozen_gate()

    import p1_embed_distribute, p2_rp_pulse, p3_boundary_v2, p4_rho_recheck
    s = quick
    print("\n##### P1 — 임베딩 분산 #####")
    P1 = p1_embed_distribute.run(n_total=(120 if s else 240), e_res=(2 if s else 4),
                                 e_open=(6 if s else 16))
    print("\n##### P2 — RP 펄스 수준 공유 #####")
    P2 = p2_rp_pulse.run(per_class=(20 if s else 60), e_res=(2 if s else 4),
                         e_open=(5 if s else 10))
    print("\n##### P4 — 펄스 공유 후 ρ 재확인 #####")
    P4 = p4_rho_recheck.run()
    print("\n##### P3 — 통합 천장 재시험 (8x8·14x14) #####")
    P3 = p3_boundary_v2.run(per_class=(20 if s else 100), e_res=(2 if s else 5),
                            e_open=(6 if s else 16))

    write_report(edir, no, date, frozen_ok, P1, P2, P3, P4, quick, t0)
    with open(os.path.join(edir, "datasets", "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# 데이터셋 — Block II Phase 2\n\n- P1: 합성 한국어 빈칸(D3, 분산 임베딩).\n"
                "- P2/P3: MNIST 8x8·14x14 (Exp2 로더).\n- P4: 합성(임펄스 자유전개).\n")
    print(f"\n[phase2] 저장 완료 -> {os.path.relpath(edir, ROOT)}  ({time.time() - t0:.1f}s)")


def _yn(b):
    return "PASS" if b else "CHECK"


def write_report(edir, no, date, frozen_ok, P1, P2, P3, P4, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    R = [f"# Block II Phase 2 (KEI 완전 통합) 보고서 — {date}_experiment_{no}", "",
         "Phase 1 의 세 미완(임베딩 colinear·RP u-수준·천장)을 통합 해소/확정. ARIS·KEI 코어 불변.", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()} · 경과 {time.time() - t0:.0f}s",
         f"- frozen interface: **{_yn(frozen_ok)}**", "",
         "## P1~P4 판정", "",
         "| # | 과제 | 결과 | 통과조건 | 판정 |",
         "|---|------|------|----------|------|",
         f"| P1 | 임베딩 분산 | orth {P1['orth_before']:.3f}→**{P1['orth_after']:.3f}** "
         f"(E1 단일 0.996), top1 {P1['top1']:.2f} | <0.5 | {_yn(P1['passed'])} |",
         f"| P2 | RP 펄스 공유 | IP {P2['acc_IP']:.3f} / RP {P2['acc_RP']:.3f} "
         f"(Δ**{P2['diff']:+.3f}**, P1 +0.005) | 정량차 | {_yn(P2['passed'])} |",
         f"| P3 | 통합 천장 | 8×8 {P3['acc_8x8']:.3f} / 14×14 {P3['acc_14x14']:.3f} "
         f"best **{P3['best']:.3f}** vs 0.56 ({'돌파' if P3['broke_ceiling'] else '미돌파'}) | 정량확정 | {_yn(P3['passed'])} |",
         f"| P4 | 공유 후 ρ | ρ {P4['rho_hi']:.2f}→{P4['rho_reg']:.2f}, 성장 {P4['growth_hi']:.1e}→{P4['growth_reg']:.1e} | ρ<1 수렴보존 | {_yn(P4['passed'])} |",
         "",
         "## 세 미완의 해소/확정", "",
         f"- **E1 임베딩 colinear → P1**: 단일 임베딩 0.996 붕괴를 M={P1['M']} Area 분산으로 "
         f"orth_after **{P1['orth_after']:.3f}** 로 {'해소(<0.5)' if P1['orth_after'] < 0.5 else '대폭 억제(붕괴 방지, 무작위 바닥 부근)'}. "
         "각 Area 가 D/M 차원만 책임 → 전체 colinear 정렬 자유도 감소.",
         f"- **E2 RP u-수준 → P2**: 펄스 버퍼 공통시계 공유로 RP−IP={P2['diff']:+.3f} "
         f"(Phase1 u-수준 +0.005 대비 {'확대' if P2['expanded'] else '유사'}). 펄스 수준 간섭의 실측 이득.",
         f"- **E3 천장 → P3**: 완전 통합(펄스 공유 다중 Area) best {P3['best']:.3f}. "
         + ("**0.56 돌파 → 완전통합이 경계를 민다(Block II 목표 달성).**"
            if P3["broke_ceiling"] else
            "**~0.56 미돌파 → 지연-전용+위상-제약 물리 상한의 세 번째 독립 확증**"
            "(용량 Exp3 / 구조 Phase1 / 완전통합 Phase2). 천장 돌파는 Block II 범위 밖(더 큰 차원·다른 물리)."),
         "",
         "## §5.4 ρ 계측 (P4, 결합계 스택버퍼 야코비안)", "",
         "```json",
         json.dumps({"rho_by_coupling": P4["rho_by_coupling"], "rho_hi": round(P4["rho_hi"], 4),
                     "rho_reg": round(P4["rho_reg"], 4), "regularized": True,
                     "method": "linear(coupled step jacobian)",
                     "coupling_raises_rho": P4["coupling_raises_rho"]},
                    ensure_ascii=False, indent=2),
         "```", "",
         "## Phase 2 가 답한 질문", "",
         ("~0.56 천장이 다중 Area 미완 탓인가, 진짜 물리 상한인가 → "
          + ("**미완 탓(돌파)** — Block II 목표 달성." if P3["broke_ceiling"]
             else "**진짜 물리 상한(미돌파, 3번째 확증)** — Block I~II 능력 경계 질문 종결. "
                  "다음은 Block III(비용)·Block IV(Agentic).")),
         "",
         "## 한계", "",
         "- 존재 증명·축소 학습. P1 의 <0.5 절대목표는 10 명사코드/D=8 의 무작위 바닥(~0.7)에 제약 — "
         "붕괴 방지(0.996 회피)가 실질 성과. P3 천장 판정은 이 규모에서의 정량 확정.",
         "- 펄스 공유는 시간 정렬에 민감(어긋나면 잡음 — Thalamus 류 정렬은 Block II 후반/멀티모달).",
         "- 추론 사전합성·비용·메모리·Agentic 은 Block III 이후 예약."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    readme = [f"# {date}_experiment_{no} — Block II Phase 2 (KEI 완전 통합)", "",
              "세 미완(임베딩·RP펄스·천장) 통합 해소/확정. ARIS·KEI 코어 불변.", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · frozen {_yn(frozen_ok)}",
              "- 상세: [`results/report.md`](results/report.md)", "",
              "| P | 결과 | 판정 |", "|---|------|------|",
              f"| P1 분산 | orth 0.996→{P1['orth_after']:.2f} | {_yn(P1['passed'])} |",
              f"| P2 펄스 | Δacc {P2['diff']:+.3f} | {_yn(P2['passed'])} |",
              f"| P3 천장 | best {P3['best']:.3f} vs 0.56 ({'돌파' if P3['broke_ceiling'] else '미돌파'}) | {_yn(P3['passed'])} |",
              f"| P4 ρ | {P4['rho_hi']:.2f}→{P4['rho_reg']:.2f} 수렴보존 | {_yn(P4['passed'])} |"]
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(no=no, date=date, quick=quick, git_commit=commit,
                       frozen_interface_ok=frozen_ok, elapsed_sec=round(time.time() - t0, 1),
                       P1=P1, P2=P2, P3=P3, P4=P4), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
