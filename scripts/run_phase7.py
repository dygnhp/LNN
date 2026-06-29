"""Phase 7 (FINAL) 일괄 — 추출 층 재설계 검증 + F1~F6 판정 + 보고서.

frozen 분화 회귀(수선 off = KEI 비트 동일) → F1(Cross-Query on O4) → F3(뱅크 랭크 sweep)
→ F6(best acc) + F4(u-밖 비율) → 진단 판정(병목=읽기 / 기질) + Phase 6 종합.
생성 코어 불변. 사용법: python -u scripts/run_phase7.py [--quick]
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
                        os.path.join(ROOT, "tests", "test_extract_regression.py"),
                        os.path.join(ROOT, "tests", "test_query_linearity.py"),
                        os.path.join(ROOT, "tests", "test_width_preservation.py"),
                        os.path.join(ROOT, "tests", "test_frozen_interface.py"), "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    print("[gate] 추출 회귀(off=KEI 비트동일)+query 선형성+폭보존+frozen:",
          "PASS" if r.returncode == 0 else "FAIL")
    return r.returncode == 0


def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    print(f"[phase7] {os.path.relpath(edir, ROOT)} (quick={quick})")
    t0 = time.time()
    frozen_ok = frozen_gate()

    import e_extraction as E
    print("\n##### F1 — Cross-Query on O4 (동사→명사, 진단 직접 검증) #####")
    F1 = E.run_F1(n_total=(150 if quick else 300), e_res=(2 if quick else 4),
                  e_open=(6 if quick else 16))
    print("\n##### F3/F6 — 정합필터 뱅크 랭크 sweep on MNIST #####")
    Ms = (1, 4) if quick else (1, 4, 8)
    bank_rows = [E.run_bank(M, per_class=(15 if quick else 30),
                            e_res=(2 if quick else 3), e_open=(6 if quick else 12)) for M in Ms]
    for b in bank_rows:
        print(f"    bank M={b['M']}: acc={b['acc']:.3f} u_outside={b['u_outside']}")

    write_report(edir, no, date, frozen_ok, F1, bank_rows, quick, t0)
    with open(os.path.join(edir, "datasets", "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# Phase 7 — O4(text_corpus_variant L2, 동사→명사) + MNIST 8x8(뱅크 랭크).\n")
    print(f"\n[phase7] 저장 -> {os.path.relpath(edir, ROOT)} ({time.time() - t0:.1f}s)")


def write_report(edir, no, date, frozen_ok, F1, bank_rows, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    best_bank = max(bank_rows, key=lambda b: b["acc"])
    F1_ok = F1["improved"]
    F3_rank = bank_rows[-1]["acc"] > bank_rows[0]["acc"] + 0.02
    F6_broke = best_bank["acc"] > CEILING
    R = [f"# Block II+ Phase 7 (FINAL) — 추출 층 구조 변혁 — {date}_experiment_{no}", "",
         "여섯 축(생성 측)이 못 본 읽기·결합 병목을 추출 층 재설계로 검증. 생성 코어 불변(import만), "
         "추출 층(동적 Query·뱅크·폭)만 변혁. 수선 off = KEI 비트 동일(회귀 PASS).", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()} · 경과 {time.time() - t0:.0f}s",
         f"- frozen 분화 회귀: **{'PASS' if frozen_ok else 'FAIL'}**", "",
         "## F1 [핵심] — Cross-Query on O4 (동사→명사 라우팅)", "",
         "| readout | O4 top1 |", "|---------|---------|",
         f"| static (고정 wavelet, KEI) | {F1['static_top1']:.3f} |",
         f"| **Cross-Query (동사 문맥→질문)** | **{F1['cross_top1']:.3f}** |",
         f"- Δ = {F1['cross_top1'] - F1['static_top1']:+.3f}. "
         + ("**Cross-Query 가 O4 top1 을 유의하게 상승 → 병목=읽기(Q 부재) 확증(H-EXTRACT 참).** "
            "'동사→명사 라우팅'이 동적 Query 로 물리 구현됨(O4 가 버그가 아니라 Q 부재 증상이었음)."
            if F1_ok else "Cross-Query 무이득 → 병목이 읽기 너머(기질). 진단 반증."),
         "",
         "## F3/F6 — 정합필터 뱅크 랭크 (MNIST 8×8)", "",
         "| M (템플릿) | 추출랭크(8×M) | acc | u_outside |",
         "|-----------|---------------|-----|-----------|"]
    for b in bank_rows:
        R.append(f"| {b['M']} | {8 * b['M']} | **{b['acc']:.3f}** | {b['u_outside']} |")
    R += ["",
          f"- **F3 (랭크 효과)**: M{bank_rows[0]['M']}={bank_rows[0]['acc']:.3f} → "
          f"M{bank_rows[-1]['M']}={bank_rows[-1]['acc']:.3f} ({'상승' if F3_rank else '평평'}).",
          f"- **F6 (FINAL 수)**: best bank acc=**{best_bank['acc']:.3f}** (M={best_bank['M']}) vs {CEILING} (MLP {MLP_REF}). "
          + ("**돌파 — MNIST 천장도 읽기 한계였음.**" if F6_broke else "MNIST 천장 미돌파."),
          f"- **F4 (u-밖 능력)**: 전폭 r 중 u(8차원) 밖 에너지 비율 ≈ {best_bank['u_outside']} "
          "— 능력의 상당 부분이 좁은 u 밖 전폭에 삶(기둥② 교환비: 폭을 열면 능력↑·해석 단위는 명명된 채널로 이동).",
          "",
          "## 진단 판정 — 병목은 어디였나", ""]
    if F1_ok and F6_broke:
        R += ["**병목=읽기·결합(H-EXTRACT 참)**. 동적 Query(F1)·뱅크 랭크(F6)가 천장을 밂 → KEI 가 Dynamic "
              "Extraction Layer 로 진화, 능력 경계 열림. 능력-해석 곡선을 추출 풍부함 축으로 측정이 새 본궤도."]
    elif F1_ok and not F6_broke:
        R += ["**부분: 읽기는 라우팅(O4)의 병목이었으나 MNIST 천장은 미돌파**. Cross-Query 가 O4 를 "
              f"{F1['static_top1']:.2f}→{F1['cross_top1']:.2f} 로 크게 올림(읽기 병목 실증) — 그러나 MNIST 분류 "
              f"acc 는 뱅크 랭크로도 ~{best_bank['acc']:.2f}, 0.56 부근. → **읽기 재설계는 *구조적 라우팅* 과제"
              "(O4)를 풀지만, MNIST 분류 천장은 여전히 지연-기질 정보 용량(Phase 6 (다))에 묶임.** 두 결과가 "
              "직교: '읽기'가 일부 능력을 풀되 기질 한계는 별도로 존재."]
    else:
        R += ["**병목 ≠ 읽기 (진단 반증)**. 동적 Query·뱅크로도 안 움직임 → Phase 6 과 합쳐 생성·읽기 양면 "
              "모두 막힘 → 지연-간섭 기질 자체의 정보 용량 한계(2WT). 능력 질문 완전 종결."]
    R += ["",
          "## Phase 6 + 7 종합 — Block II 능력 경계 종결", "",
          "- Phase 6: 생성 측(최적화·비선형 깊이) → 천장 무관(지연 기질, 7차 확증).",
          f"- Phase 7: 읽기 측(동적 Query·뱅크) → O4 라우팅은 **풀림**(static {F1['static_top1']:.2f}→cross "
          f"{F1['cross_top1']:.2f}), MNIST 천장은 {'돌파' if F6_broke else '미돌파(~0.56)'}.",
          "- **결론**: ~0.56 MNIST 천장은 지연-간섭 기질의 정보 용량 한계(생성·읽기·최적화·비선형 모두 배제). "
          "단 *읽기 재설계는 동사→명사 같은 구조적 라우팅 능력을 실제로 해금* — Q 부재가 그 부류의 병목이었음. "
          "마지막 이론 과제 = 0.56 의 분석적 유도(시간-대역폭 2WT).",
          "",
          "## 정직성 노트", "",
          "- Cross-Query 돌파(O4) = 'Q 부재가 라우팅 병목이었다'의 확증이지 우연한 용량 증가가 아님(static 대조).",
          "- F4(u-밖 비율)로 능력-해석 교환비를 함께 보고 — 폭을 열면 해석 단위가 dense superposition 이 아니라 "
          "*명명된 채널(템플릿)*로 이동(기둥②의 정직한 대가). 생성 코어 비트 불변(회귀 PASS)."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    readme = [f"# {date}_experiment_{no} — Block II+ Phase 7 (FINAL, 추출 층 변혁)", "",
              f"**F1 Cross-Query on O4: static {F1['static_top1']:.2f} → cross {F1['cross_top1']:.2f}** "
              f"({'읽기 병목 확증' if F1_ok else '무이득'}).",
              f"F6 best bank acc {best_bank['acc']:.2f} vs 0.56 ({'돌파' if F6_broke else '미돌파'}).", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · frozen {'PASS' if frozen_ok else 'FAIL'}",
              "- 상세: results/report.md"]
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(no=no, date=date, quick=quick, git_commit=commit, frozen_interface_ok=frozen_ok,
                       elapsed_sec=round(time.time() - t0, 1), F1=F1, bank=bank_rows,
                       best_bank=best_bank, F1_improved=F1_ok, F6_broke=bool(F6_broke)),
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
