"""SPEC-EXT 일괄 실행 — 직교성 붕괴 원인 규명 (E-ORTH) + 보고서.

frozen gate → 데이터 분포 단위점검 → L0/L1/L2 통제 대조 → O1~O4 판정 → exlog/ 저장.
ARIS·KEI 코어 불변, 데이터·하니스만 추가.  사용법: python scripts/run_spec_ext.py [--quick]
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "experiments_block2"))
sys.stdout.reconfigure(encoding="utf-8")

from run_experiment import resolve_experiment_dir  # noqa: E402


def frozen_gate():
    r = subprocess.run([sys.executable, "-m", "pytest",
                        os.path.join(ROOT, "tests", "test_frozen_interface.py"), "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    print("[gate] frozen interface:", "PASS" if r.returncode == 0 else "FAIL")
    return r.returncode == 0


def dataset_check():
    """§3 단위점검 — '편지' 명사의 레벨별 동사 분포(L0 공유 / L2 고유)."""
    from lnn.data import text_corpus_variant as TV
    rows = {}
    pid = TV.TOKEN2ID["편지"]
    for lvl in ("L0", "L1", "L2"):
        X, y = TV.make_mask_dataset_variant(n_total=1500, seed=1, variant_level=lvl)
        verbs = Counter(TV.decode_ids(r)[3] for r, t in zip(X, y) if t == pid)
        rows[lvl] = dict(verbs)
        print(f"    '편지' {lvl}: {dict(verbs)}")
    return rows


def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    print(f"[spec-ext] {os.path.relpath(edir, ROOT)} (quick={quick})")
    t0 = time.time()
    frozen_ok = frozen_gate()
    print("\n[unit] 레벨별 명사-동사 결합 분포:")
    ds = dataset_check()

    import e_orth_variant
    print("\n[E-ORTH] L0/L1/L2 통제 대조:")
    E = e_orth_variant.run(n_total=(150 if quick else 300),
                           e_res=(2 if quick else 4), e_open=(6 if quick else 16))

    write_report(edir, no, date, frozen_ok, ds, E, quick, t0)
    with open(os.path.join(edir, "datasets", "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# 데이터셋 — 변별 문맥 코퍼스 (text_corpus_variant)\n\n"
                "명사별 고유 동사로 변별 강도 3레벨(L0 공유/L1 부분/L2 고유). 단일 변수=변별 강도.\n\n"
                f"'편지' 레벨별 동사 분포: {json.dumps(ds, ensure_ascii=False)}\n")
    print(f"\n[spec-ext] 저장 완료 -> {os.path.relpath(edir, ROOT)} ({time.time() - t0:.1f}s)")


def write_report(edir, no, date, frozen_ok, ds, E, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    lv = {r["level"]: r for r in E["levels"]}
    O1_ok = lv["L0"]["orth_after"] > 0.7
    R = [f"# SPEC-EXT 보고서 (직교성 붕괴 원인 규명) — {date}_experiment_{no}", "",
         "변별 문맥 코퍼스(L0/L1/L2)로 H-ORTH 검증: colinear 붕괴가 데이터 보상 구조인가 임베딩 기하인가.",
         "ARIS·KEI 코어 불변(데이터·하니스만 추가).", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()} · 경과 {time.time() - t0:.0f}s",
         f"- frozen interface: **{'PASS' if frozen_ok else 'FAIL'}** · 단일 변수 통제(D=8, M=1, 동일 시드/스케줄)", "",
         "## 레벨별 결과 (단일 변수 = 변별 강도)", "",
         "| level | 변별 | orth_before | orth_after | top1 | within | across |",
         "|-------|------|-------------|------------|------|--------|--------|"]
    for l in ("L0", "L1", "L2"):
        r = lv[l]
        R.append(f"| {l} | {'공유(최대 colinear)' if l == 'L0' else ('부분' if l == 'L1' else '고유(강한 변별)')} | "
                 f"{r['orth_before']:.3f} | **{r['orth_after']:.3f}** | {r['top1']:.3f} | "
                 f"{r['orth_within']:.3f} | {r['orth_across']:.3f} |")
    rec = E["recovered"]
    R += ["", "## O1~O4 판정", "",
          f"- **O1 (L0 colinear 재현)**: orth_after={lv['L0']['orth_after']:.3f} "
          f"({'재현 — colinear 확인' if O1_ok else '미재현'}).",
          f"- **O2 (단조성/회복)**: L0={lv['L0']['orth_after']:.3f} ≥ L1={lv['L1']['orth_after']:.3f} "
          f"≥ L2={lv['L2']['orth_after']:.3f} → 단조 {E['monotone']}; 회복(L2<L0−0.05) {rec}.",
          f"- **O3 (범주 내≫간, L0)**: within={lv['L0']['orth_within']:.3f} vs across={lv['L0']['orth_across']:.3f} "
          f"→ {'within≫across (colinear=범주 구조 반영 입증)' if E['within_gg_across'] else 'within≈across'}.",
          f"- **O4 (top1 변화)**: L0={lv['L0']['top1']:.3f} → L2={lv['L2']['top1']:.3f} "
          "(변별 문맥이 개별 명사 구별 요구로 top1 에 주는 영향).",
          "",
          "## H-ORTH 판정", "",
          f"**{E['verdict']}**.",
          ("- 변별 문맥을 강화하니(L0→L2) 직교성이 회복됨 → **colinear 의 원인은 임베딩 기하가 아니라 "
           "데이터 보상 구조**(같은 범주 명사가 동사를 공유해 구별을 요구하지 않음). P1 분산은 원인을 못 "
           "건드린 처방이었고, '직교성<0.5' 지표는 데이터가 구별을 요구할 때만 유효. → 후속: text_corpus 변별 "
           "재설계, 직교성 지표 재고."
           if rec else
           "- 변별 문맥을 강화해도(L0→L2) 직교성이 회복되지 않음 → **원인은 임베딩 기하(작은 D)**. "
           "경로 A(FreqEncoder 주파수 차원 분리)가 유일한 길로 승격. 작은 D 의 본질적 한계 정량 확정."),
          "",
          "## 한계", "",
          "- 존재 증명·축소 학습. orth 지표는 최대 교차상관(상단 잡음 큼) — within/across 평균이 더 안정적 신호.",
          "- 단일 변수 통제 유지(레벨 간 모델·D·시드·스케줄 동일)."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    readme = [f"# {date}_experiment_{no} — SPEC-EXT (직교성 원인 규명)", "",
              f"H-ORTH 검증: **{E['verdict']}**", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}`",
              "- 상세: [`results/report.md`](results/report.md)", "",
              "| level | orth_after | top1 | within/across |", "|---|---|---|---|"]
    for l in ("L0", "L1", "L2"):
        r = lv[l]
        readme.append(f"| {l} | {r['orth_after']:.3f} | {r['top1']:.3f} | {r['orth_within']:.2f}/{r['orth_across']:.2f} |")
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(no=no, date=date, quick=quick, git_commit=commit,
                       frozen_interface_ok=frozen_ok, elapsed_sec=round(time.time() - t0, 1),
                       dataset_check=ds, e_orth=E), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
