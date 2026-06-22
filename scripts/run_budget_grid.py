"""실험 I 확장 — 2^3 그리드: cell 확장{off:R5, on:R8} × RAL{off,on} × epoch{×5,×10}.

Experiment I(E-BUDGET)이 cell·RAL 미적용(R5·표준 분류기)이었으므로 8개 조합을 모두 실행.
cell-on = R8 + K·n_steps 동반 확장(Phase 4 CellScaler), RAL-on = 단일 Area n_loops회 순환(Phase 5).
MNIST 8x8 acc 를 측정해 현재 baseline(R5·RALoff·×5 = 0.493)·천장 0.56·MLP 0.83 과 대조.

진행 가시화를 위해 `python -u` 권장. exlog/<date>_experiment_<No>/ 저장.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
sys.path.insert(0, os.path.join(ROOT, "experiments_block2"))
sys.stdout.reconfigure(encoding="utf-8")

import _common as C  # noqa: E402
from lnn import train  # noqa: E402
from lnn.area import pick_cells  # noqa: E402
from lnn.data.mnist_data import load_mnist_split  # noqa: E402
from lnn.encodings import ImageEncoder  # noqa: E402
from lnn_block2.cell_scaling import CellScaler  # noqa: E402
from lnn_block2.ral import build_ral_classifier  # noqa: E402
from run_experiment import resolve_experiment_dir  # noqa: E402

N_FEAT = 8
N_CLASSES = 10
CEILING = 0.56
MLP_REF = 0.83
CURRENT_BASELINE = 0.493   # Phase 6 실험 I, R5·RALoff·×5


def _train_one(ral_on, R, mult, base_open, n_loops, per_class, seed):
    cs = CellScaler(R)
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": cs.n_steps, "n_channels": 8, "n_proc": 1,
          "n_hills_terrain": cs.K, "n_hills_gain": max(4, cs.K // 2)}
    Xtr, ytr, Xte, yte, _s = load_mnist_split(n_per_class=per_class, size=8,
                                              test_per_class=20, seed=seed)
    pix = C.map_image_cells(geo, 8, 8)
    feat = pick_cells(geo, N_FEAT, phase=2.6)
    e_open = base_open * mult

    if ral_on:
        model = build_ral_classifier(geo, pix, feat, N_CLASSES, n_loops, jax.random.PRNGKey(seed), hp)
        _, static = eqx.partition(model, eqx.is_inexact_array)

        def loss_fn(params, Xb, Yb, windows, lam):
            m = eqx.combine(params, static)
            lg = m.forward(geo, Xb, windows[0])
            return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(Yb.shape[0]), Yb]), lg
        rw = lambda m: [m.make_window(geo)]
        predict = lambda m, win: m.forward(geo, jnp.asarray(Xte), win[0])
    else:
        enc = ImageEncoder(gen_cells=pix, P=hp["P"], n_steps=cs.n_steps, n_cells=geo.N)
        model = C.build_classifier(geo, enc, N_FEAT, N_CLASSES, jax.random.PRNGKey(seed), hp)
        _, static = eqx.partition(model, eqx.is_inexact_array)

        def loss_fn(params, Xb, Yb, windows, lam):
            e, cl, h = eqx.combine(params, static)
            _c, u = cl.forward(geo, e.encode(Xb), windows)
            lg = jax.vmap(h)(u)
            return -jnp.mean(jax.nn.log_softmax(lg)[jnp.arange(Yb.shape[0]), Yb]), lg
        rw = lambda m: m[1].make_windows(geo)

        def predict(m, win):
            e, cl, h = m
            _c, u = cl.forward(geo, e.encode(jnp.asarray(Xte)), win)
            return jax.vmap(h)(u)

    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=4, batch_size=32,
                                    open_terrain=False, open_gain=False, lrs=None,
                                    lam_schedule=train.anneal_lambda(2, 8), recompute_windows=rw,
                                    seed=seed, log_prefix="")
    t = time.perf_counter()
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=32,
                                    open_terrain=True, open_gain=True, lrs=None,
                                    lam_schedule=train.const_lambda(8.0), recompute_windows=rw,
                                    seed=seed + 1, log_prefix="")
    sec = time.perf_counter() - t
    acc = train.accuracy(predict(model, win), yte)
    return dict(cell=("on(R%d)" % R if R != 5 else "off(R5)"), R=R, ral=bool(ral_on),
                mult=mult, e_open=e_open, acc=round(acc, 3), n_cells=cs.n_cells, K=cs.K,
                n_steps=cs.n_steps, train_sec=round(sec, 1))


def run(per_class=12, base_open=6, n_loops=3, seed=0):
    print(f"[BUDGET-GRID] 2^3 = cell{{R5,R8}} x RAL{{off,on}} x epoch{{x5,x10}} "
          f"(per_class={per_class}, base_open={base_open}, RAL n_loops={n_loops})")
    rows = []
    for R in (5, 8):
        for ral_on in (False, True):
            for mult in (5, 10):
                r = _train_one(ral_on, R, mult, base_open, n_loops, per_class, seed)
                rows.append(r)
                print(f"    cell={r['cell']:7s} RAL={'on ' if ral_on else 'off'} x{mult:2d} "
                      f"(e_open={r['e_open']}): acc={r['acc']:.3f}  [N={r['n_cells']} K={r['K']} "
                      f"{r['train_sec']}s]")
    best = max(rows, key=lambda r: r["acc"])
    print(f"\n    BEST: {best['cell']} RAL={best['ral']} x{best['mult']} acc={best['acc']:.3f} "
          f"vs baseline {CURRENT_BASELINE} / ceiling {CEILING} / MLP {MLP_REF}")
    return dict(rows=rows, best=best)


def main():
    edir, no, date = resolve_experiment_dir()
    print(f"[budget-grid] {os.path.relpath(edir, ROOT)}")
    t0 = time.time()
    res = run()
    write_report(edir, no, date, res, t0)
    print(f"\n[budget-grid] 저장 -> {os.path.relpath(edir, ROOT)} ({time.time() - t0:.1f}s)")


def write_report(edir, no, date, res, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    rows, best = res["rows"], res["best"]
    broke = best["acc"] > CEILING
    improved = best["acc"] > CURRENT_BASELINE
    R = [f"# 실험 I 확장 그리드 (cell x RAL x epoch) — {date}_experiment_{no}", "",
         "Experiment I(R5·RALoff·표준)이 cell·RAL 미적용이라 2^3=8 조합 실행. "
         f"현재 baseline(R5·RALoff·x5)={CURRENT_BASELINE}, 천장={CEILING}, MLP={MLP_REF}.", "",
         f"- git `{commit}` · 경과 {time.time() - t0:.0f}s · per_class=12, base_open=6, RAL n_loops=3", "",
         "## 8개 조합 결과", "",
         "| cell | RAL | epoch | e_open | N_cells | K | acc |",
         "|------|-----|-------|--------|---------|---|-----|"]
    for r in sorted(rows, key=lambda x: (-x["acc"])):
        R.append(f"| {r['cell']} | {'on' if r['ral'] else 'off'} | x{r['mult']} | {r['e_open']} | "
                 f"{r['n_cells']} | {r['K']} | **{r['acc']:.3f}** |")
    R += ["", "## 판정", "",
          f"- **best**: cell={best['cell']}, RAL={best['ral']}, x{best['mult']} → acc **{best['acc']:.3f}**.",
          f"- vs 현재 baseline({CURRENT_BASELINE}): {'개선' if improved else '비개선'} "
          f"(Δ{best['acc'] - CURRENT_BASELINE:+.3f}).",
          f"- vs 천장 {CEILING}: {'**돌파**' if broke else '미돌파'}.",
          "- 해석: cell 확장·RAL·epoch 증배가 ~0.56 천장에 대해 추가 능력을 주는지의 직접 측정. "
          + ("천장 돌파 — 재해석 필요." if broke else
             "미돌파면 (cell·RAL·epoch 어느 조합으로도) ~0.56 견고 재확인 — Phase 6 (다) 지연-기질 결론 보강."),
          "",
          "## 한계", "",
          "- 축소 학습(per_class=12). 일부 조합 미수렴 가능 — acc 는 동일 예산 비교용."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {date}_experiment_{no} — 실험 I 확장 그리드 (cell x RAL x epoch)\n\n"
                f"best acc {best['acc']:.3f} (cell={best['cell']}, RAL={best['ral']}, x{best['mult']}) "
                f"vs baseline {CURRENT_BASELINE} / 천장 {CEILING}.\n\n상세: results/report.md\n")
    with open(os.path.join(edir, "datasets", "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# MNIST 8x8. cell{R5,R8}xRAL{off,on}xepoch{x5,x10} 그리드.\n")
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(dict(no=no, date=date, git_commit=commit, rows=rows, best=best,
                       baseline=CURRENT_BASELINE, ceiling=CEILING), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
