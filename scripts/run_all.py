"""네 시연 일괄 실행 + 산출물 저장 (= /goal 최종 타깃, §8/§9).

순서: (1) Exit-gate 테스트(test_gradient_fd·test_telescoping 포함) →
(2) D1~D4 학습/시연, outputs/ 저장 → (3) 합격 여부·산출물 경로 표 보고.

각 시연은 §12 fallback 에 따라 반드시 가시적 출력을 남긴다(완전 학습 실패 시 reservoir 라도).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

OUT = os.path.join(ROOT, "outputs")


def run_gates():
    print("=" * 70)
    print("[GATES] core tests (gradient_fd, telescoping, geometry, readout)")
    print("=" * 70)
    r = subprocess.run([sys.executable, "-m", "pytest", os.path.join(ROOT, "tests"), "-q"],
                       cwd=ROOT, capture_output=True, text=True)
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-1500:])
    return r.returncode == 0


def main(quick=False):
    t0 = time.time()
    gates_ok = run_gates()

    import demo_text_fill
    import demo_text_to_image
    import train_image_classify
    import train_text_classify

    if quick:
        cfg = dict(d1=dict(n_total=160, epochs_res=3, epochs_open=8),
                   d2=dict(n_per_class=15, epochs_res=3, epochs_open=8),
                   d3=dict(n_total=160, epochs_res=3, epochs_open=8),
                   d4=dict(epochs_res=5, epochs_open=20))
    else:
        cfg = dict(d1=dict(n_total=320, epochs_res=6, epochs_open=18),
                   d2=dict(n_per_class=30, epochs_res=6, epochs_open=18),
                   d3=dict(n_total=320, epochs_res=5, epochs_open=18),
                   d4=dict(epochs_res=10, epochs_open=40))

    results = []
    print("\n" + "=" * 70 + "\n[DEMOS]\n" + "=" * 70)
    results.append(train_text_classify.run(**cfg["d1"]))
    results.append(train_image_classify.run(**cfg["d2"]))
    results.append(demo_text_fill.run(**cfg["d3"]))
    results.append(demo_text_to_image.run(**cfg["d4"]))

    # ── 합격 판정 ────────────────────────────────────────────────────────
    # D1·D2: Exit gate ① — 지형 개방이 reservoir 기준선 위로 정확도 상승.
    # D3·D4: 생성 — 가시적 출력 산출(완주). 정량치는 참고로 함께 보고.
    print("\n" + "=" * 70 + "\n[SUMMARY] Block I (ARIS)\n" + "=" * 70)
    print(f"{'demo':<5}{'criterion':<34}{'result':<22}{'pass'}")
    print("-" * 70)

    def line(demo, crit, res, ok):
        print(f"{demo:<5}{crit:<34}{res:<22}{'PASS' if ok else 'CHECK'}")

    d1, d2, d3, d4 = results
    line("D1", "acc up vs reservoir (gate1)",
         f"{d1['acc_reservoir']:.2f}->{d1['acc_opened']:.2f}", d1["improved"])
    line("D2", "acc up vs reservoir (gate1)",
         f"{d2['acc_reservoir']:.2f}->{d2['acc_opened']:.2f}", d2["improved"])
    line("D3", "produces top-k fill (visible)",
         f"top1={d3['acc_opened']:.2f}", True)
    line("D4", "produces 8x8 images (visible)",
         f"MSE {d4['mse_reservoir']:.3f}->{d4['mse_opened']:.3f}", True)
    print("-" * 70)
    print(f"gates (gradient_fd, telescoping, ...) : {'PASS' if gates_ok else 'FAIL'}")
    print(f"\n[§11★] D3 noun-code max cross-corr: "
          f"{d3['orth_before']:.2f} -> {d3['orth_after']:.2f} (D={d3['D']})")
    print("\noutputs/:")
    for fn in sorted(os.listdir(OUT)):
        if fn != ".gitkeep":
            print("  ", os.path.join("outputs", fn))
    print(f"\nelapsed {time.time() - t0:.1f}s")

    import json
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(dict(gates_ok=gates_ok, results=results), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
