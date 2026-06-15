"""Experiment 3 — 자율 성장 + 용량-격자 동반 스케일링 (H2 반증 직접 재시험).

세 갈래:
  A) 수동 동반 스케일링 S0'~S3' (K 를 격자와 함께 키워 Exp2 의 격자↑→정확도↓ 를 뒤집는가).
  B) 자율 성장 (K_init=4, plateau 시 grow_terrain_K) — 모델이 스스로 용량을 키우는가.
  C) 용량 천장 (K_max 까지 성장) — Exp2 의 0.49 가 표현력 상한인가 K 부족인가.

CHM(`research_main/FINAL`) 성장 메커니즘을 LNN 지형에 이식. ARIS 코어 불변 — 지형/이득
RBF 개수만 학습 중 가변. exlog/<YYYYMMDD>_experiment_<No>/ 에 정리 저장.

사용법:  python scripts/run_experiment3.py [--quick]
"""

from __future__ import annotations

import json
import os
import platform
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
sys.stdout.reconfigure(encoding="utf-8")

import _common as C  # noqa: E402
from lnn import instrument, train  # noqa: E402
from lnn.data.mnist_data import load_mnist_split  # noqa: E402
from lnn.encodings import ImageEncoder  # noqa: E402
from lnn.growth import GrowthConfig  # noqa: E402
from run_experiment import resolve_experiment_dir  # noqa: E402
from run_experiment2 import (measure_cost, save_confusion, save_curve,  # noqa: E402
                             save_terrain, train_image_stage)

N_CLASSES = 10
N_FEAT = 12

# 실험 A — 수동 동반 스케일링 (K 를 격자와 함께)
STAGES_A = [
    dict(name="S0p", per_class=30, R=5, size=8, n_steps=70, batch=32, e_res=4, e_open=10,
         K_terrain=8, K_gain=8),
    dict(name="S1p", per_class=100, R=5, size=8, n_steps=70, batch=32, e_res=4, e_open=10,
         K_terrain=16, K_gain=8),
    dict(name="S2p", per_class=100, R=8, size=14, n_steps=70, batch=32, e_res=4, e_open=10,
         K_terrain=32, K_gain=16),
    dict(name="S3p", per_class=20, R=16, size=28, n_steps=60, batch=16, e_res=2, e_open=3,
         K_terrain=64, K_gain=24),
]
# Exp2 대조값(같은 stage 의 K-고정 결과)
EXP2_REF = {"S0p": 0.35, "S1p": 0.49, "S2p": 0.38, "S3p": 0.17}


def _quick_A(stages):
    for s in stages:
        s["per_class"] = min(s["per_class"], 12)
        s["e_res"], s["e_open"] = 2, 3
    return stages


# ─────────────────── 자율 성장 이미지 분류 (실험 B·C) ────────────────────────
def train_image_growth(per_class, R, size, K_init, gconf, e_res, e_open,
                       n_steps=70, batch=32, seed=0):
    geo = C.make_geometry(R)
    hp = {**C.classify_hp(), "n_steps": n_steps, "n_proc": 1,
          "n_hills_terrain": K_init, "n_hills_gain": K_init}
    Xtr, ytr, Xte, yte, src = load_mnist_split(n_per_class=per_class, size=size,
                                               test_per_class=max(10, per_class // 3), seed=seed)
    pix = C.map_image_cells(geo, size, size)
    enc = ImageEncoder(gen_cells=pix, P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    model = C.build_classifier(geo, enc, N_FEAT, N_CLASSES, jax.random.PRNGKey(seed), hp)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    def loss_fn(params, Xb, Yb, windows, lam):
        e, cl, h = eqx.combine(params, static)
        _c, u = cl.forward(geo, e.encode(Xb), windows)
        logits = jax.vmap(h)(u)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    rw = lambda m: m[1].make_windows(geo)

    def predict(m, win):
        e, cl, h = m
        _c, u = cl.forward(geo, e.encode(jnp.asarray(Xte)), win)
        return jax.vmap(h)(u)

    # 1) reservoir (지형 고정, K 고정)
    model, hist_r, win = train.run_phase(
        model, static, loss_fn, Xtr, ytr, epochs=e_res, batch_size=batch,
        open_terrain=False, open_gain=False, lrs=None,
        lam_schedule=train.anneal_lambda(2.0, 8.0), recompute_windows=rw,
        seed=seed, log_prefix="[res] ")
    acc_res = train.accuracy(predict(model, win), yte)
    _, static = eqx.partition(model, eqx.is_inexact_array)

    # 2) 지형 개방 + 자율 성장
    model, hist_o, glog = train.run_growth_phase(
        model, static, loss_fn, Xtr, ytr, epochs=e_open, batch_size=batch,
        open_terrain=True, open_gain=True, lrs=None,
        lam_schedule=train.const_lambda(8.0), recompute_windows=rw,
        geo=geo, gconf=gconf, seed=seed + 1, log_prefix="[grow] ")
    win = rw(model)
    acc_open = train.accuracy(predict(model, win), yte)
    from lnn.growth import current_K
    kt, kg = current_K(model)
    return dict(geo=geo, model=model, acc_res=acc_res, acc_open=acc_open,
                hist=hist_r + hist_o, growth_log=glog, K_terrain=kt, K_gain=kg,
                e_res=e_res, src=src)


def save_growth_curve(glog, e_res, total_epochs, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    eps = [e_res + g["epoch"] for g in glog]
    kts = [g["K_terrain"] for g in glog]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    if eps:
        ax.step([0] + eps + [total_epochs], [4] + kts + [kts[-1]], where="post", marker="o")
    ax.set_xlabel("epoch"); ax.set_ylabel("K_terrain"); ax.set_title(title)
    ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    rdir, ddir = os.path.join(edir, "results"), os.path.join(edir, "datasets")
    print(f"[exp3] {os.path.relpath(edir, ROOT)} (quick={quick})")
    t0 = time.time()

    # ── 실험 A — 수동 동반 스케일링 ───────────────────────────────────────────
    stages = _quick_A([dict(s) for s in STAGES_A]) if quick else [dict(s) for s in STAGES_A]
    A = []
    print("\n##### 실험 A — 수동 동반 스케일링 (K↑ with grid) #####")
    for st in stages:
        print(f"\n===== {st['name']}: R={st['R']} size={st['size']} "
              f"K_t={st['K_terrain']} K_g={st['K_gain']} per_class={st['per_class']} =====")
        res = train_image_stage(st)
        cost = measure_cost(res, st)
        cfg = cost["model_config"]
        print(f"    acc {res['acc_res']:.3f} -> {res['acc_open']:.3f} | "
              f"N={res['geo'].N} K_t={cfg['areas'][0]['K_terrain']} "
              f"params={cfg['params']['total_trainable']} | Exp2 ref={EXP2_REF.get(st['name'])}")
        save_confusion(res["logits_te"], res["yte"], os.path.join(rdir, f"{st['name']}_confusion.png"),
                       f"{st['name']} {res['acc_open']:.2f}")
        save_terrain(res["model"], res["geo"], os.path.join(rdir, f"{st['name']}_terrain.png"),
                     f"{st['name']} terrain K={cfg['areas'][0]['K_terrain']}")
        A.append(dict(stage=st["name"], R=st["R"], size=st["size"],
                      K_terrain=st["K_terrain"], K_gain=st["K_gain"],
                      acc_res=res["acc_res"], acc_open=res["acc_open"],
                      N_cells=res["geo"].N, params=cfg["params"]["total_trainable"],
                      flops_xla=cost["flops"]["method1_xla"],
                      fwd_sec=cost["time"]["fwd_run_sec"], exp2_ref=EXP2_REF.get(st["name"])))

    # ── 실험 B — 자율 성장 (K_init=4, S1 설정 고정) ──────────────────────────
    print("\n##### 실험 B — 자율 성장 (K_init=4, MNIST 8x8 고정) #####")
    gconf_B = GrowthConfig(plateau_window=2, plateau_threshold=0.02,
                           min_epochs_before_grow=3, cooldown_after_grow=2,
                           K_terrain_grow=4, K_gain_grow=2, K_terrain_max=40)
    eB = (3, 10) if quick else (3, 26)
    B = train_image_growth(per_class=(20 if quick else 100), R=5, size=8, K_init=4,
                           gconf=gconf_B, e_res=eB[0], e_open=eB[1])
    print(f"    B: acc {B['acc_res']:.3f} -> {B['acc_open']:.3f}, "
          f"K_terrain 4->{B['K_terrain']}, {len(B['growth_log'])} grow events")
    save_growth_curve(B["growth_log"], B["e_res"], B["e_res"] + eB[1],
                      os.path.join(rdir, "B_growth_curve.png"), "Exp3-B autonomous K growth")
    save_curve(B["hist"], os.path.join(rdir, "B_loss.png"), "Exp3-B loss (grow markers in log)")

    # ── 실험 C — 용량 천장 (K_max 까지) ──────────────────────────────────────
    print("\n##### 실험 C — 용량 천장 탐색 (K_max=64) #####")
    gconf_C = GrowthConfig(plateau_window=2, plateau_threshold=0.02,
                           min_epochs_before_grow=3, cooldown_after_grow=2,
                           K_terrain_grow=6, K_gain_grow=3, K_terrain_max=64)
    eC = (3, 14) if quick else (3, 44)
    Cc = train_image_growth(per_class=(20 if quick else 100), R=5, size=8, K_init=4,
                            gconf=gconf_C, e_res=eC[0], e_open=eC[1])
    print(f"    C: acc {Cc['acc_res']:.3f} -> {Cc['acc_open']:.3f}, "
          f"K_terrain 4->{Cc['K_terrain']} (vs Exp2 H1=0.49 @ K=8)")
    save_growth_curve(Cc["growth_log"], Cc["e_res"], Cc["e_res"] + eC[1],
                      os.path.join(rdir, "C_growth_curve.png"), "Exp3-C K growth to ceiling")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    with open(os.path.join(ddir, "dataset.md"), "w", encoding="utf-8") as f:
        f.write("# 데이터셋 — MNIST (Exp2 와 동일 로더, 동반 스케일링)\n\n"
                "- 실험 A: per_class·해상도·격자를 단계별로(S0'~S3'), K_terrain/K_gain 동반 증가.\n"
                "- 실험 B/C: MNIST 8x8 per_class 고정, K 만 자율 성장.\n"
                f"- 소스: {B['src']}\n")
    write_report(edir, no, date, A, B, Cc, quick, t0)
    print(f"\n[exp3] 저장 완료 -> {os.path.relpath(edir, ROOT)}  ({time.time() - t0:.1f}s)")


def write_report(edir, no, date, A, B, Cc, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    s2p = next((a for a in A if a["stage"] == "S2p"), None)
    R = [f"# Experiment 3 보고서 — {date}_experiment_{no}", "",
         "자율 성장 + 용량-격자 동반 스케일링 (H2 반증 직접 재시험). CHM 성장 메커니즘 이식. ARIS 코어 불변.", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()}, jax {jax.__version__} · 경과 {time.time() - t0:.0f}s", "",
         "## 실험 A — 수동 동반 스케일링 (K↑ with grid)", "",
         "| stage | R | size | K_t | K_g | N_cells | params | acc_open | Exp2(K고정) | Δ vs Exp2 |",
         "|-------|---|------|-----|-----|---------|--------|----------|-------------|-----------|"]
    for a in A:
        ref = a["exp2_ref"]
        dv = (f"{a['acc_open'] - ref:+.3f}" if ref is not None else "-")
        R.append(f"| {a['stage']} | {a['R']} | {a['size']} | {a['K_terrain']} | {a['K_gain']} | "
                 f"{a['N_cells']} | {a['params']} | {a['acc_open']:.3f} | {ref} | {dv} |")
    R += ["", "## 실험 B — 자율 성장 (K_init=4, MNIST 8x8)", "",
          f"- 최종: acc {B['acc_res']:.3f} → {B['acc_open']:.3f}, K_terrain **4 → {B['K_terrain']}**, "
          f"K_gain → {B['K_gain']}, 성장 이벤트 {len(B['growth_log'])}회.",
          f"- 성장 로그: {[{'ep': g['epoch'], 'K_t': g['K_terrain']} for g in B['growth_log']]}",
          "- 성장 곡선: ![B](./B_growth_curve.png) · 손실: ![Bloss](./B_loss.png)",
          f"- 수동 대조(S2'): acc_open={s2p['acc_open'] if s2p else '-'} "
          f"(자율이 수동에 준하는가 = H8).", "",
          "## 실험 C — 용량 천장 (K_max=64)", "",
          f"- 최종: acc {Cc['acc_res']:.3f} → **{Cc['acc_open']:.3f}**, K_terrain 4 → {Cc['K_terrain']}.",
          f"- Exp2 H1 기준: K=8 에서 0.49. 천장 후 {Cc['acc_open']:.3f} "
          f"({'0.49 크게 초과 → 0.49 는 K 부족이었음(H7)' if Cc['acc_open'] > 0.55 else '0.49 근처 정체 → 지연-전용 물리의 상한(H7)'}).",
          "- 성장 곡선: ![C](./C_growth_curve.png)", "",
          "## 가설 판정 (§4)", "",
          _verdict_H6(A), _verdict_H5(B), _verdict_H7(Cc), _verdict_H8(B, s2p), "",
          "## 한계", "",
          "- 존재 증명. S3'(R16·K64)는 비용/완주 목적의 축소 학습. 성장은 지형 RBF 개수만 변경(코어 불변).",
          "- 게이트(gradient_fd·telescoping)·무조건 준수(n=0·stop-grad·logsumexp) 유지."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    # README + experiment.json
    readme = [f"# {date}_experiment_{no} — Experiment 3 (자율 성장)", "",
              "CHM 성장 이식 + 용량-격자 동반 스케일링. ARIS 코어 불변.", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}`",
              "- 상세: [`results/report.md`](results/report.md)", "",
              "| 실험 | 핵심 결과 |", "|------|-----------|",
              f"| A 동반스케일링 | " + " / ".join(f"{a['stage']}:{a['acc_open']:.2f}(K{a['K_terrain']})" for a in A) + " |",
              f"| B 자율성장 | acc→{B['acc_open']:.2f}, K 4→{B['K_terrain']} ({len(B['growth_log'])}회) |",
              f"| C 천장 | acc→{Cc['acc_open']:.2f}, K 4→{Cc['K_terrain']} (Exp2 H1=0.49) |"]
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))
    meta = dict(no=no, date=date, quick=quick, git_commit=commit,
                elapsed_sec=round(time.time() - t0, 1), experiment_A=A,
                experiment_B=dict(acc_res=B["acc_res"], acc_open=B["acc_open"],
                                  K_terrain=B["K_terrain"], K_gain=B["K_gain"],
                                  growth_log=B["growth_log"]),
                experiment_C=dict(acc_res=Cc["acc_res"], acc_open=Cc["acc_open"],
                                  K_terrain=Cc["K_terrain"], growth_log=Cc["growth_log"]))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def _verdict_H6(A):
    s1 = next((a for a in A if a["stage"] == "S1p"), None)
    s2 = next((a for a in A if a["stage"] == "S2p"), None)
    up = (s2 and s1 and s2["acc_open"] >= s1["acc_open"])
    return ("- **H6(H2 수정) — " + ("지지" if up else "미지지") + "**: K 동반 시 격자 확장이 "
            f"정확도를 {'올림' if up else '못올림'}. S1'={s1['acc_open']:.2f}→S2'={s2['acc_open']:.2f} "
            f"(Exp2 는 S1 0.49→S2 0.38 하락). " +
            ("격자↑→정확도↑ 로 뒤집힘 → H2 가 'K 동반 시 격자 유효'로 수정." if up
             else "동반에도 미상승 — report 본문/추가 epoch 필요."))


def _verdict_H5(B):
    n = len(B["growth_log"])
    return (f"- **H5(자율 성장 작동) — {'지지' if n > 0 else '미지지'}**: plateau→grow_terrain_K→재수렴 "
            f"사이클이 {n}회 작동(K 4→{B['K_terrain']}). CHM 의 성장 패턴이 LNN 지형에서도 동작" +
            ("함." if n > 0 else " 안 함(plateau 미감지 — epoch/threshold 조정 필요)."))


def _verdict_H7(Cc):
    over = Cc["acc_open"] > 0.55
    return (f"- **H7(H1 재조명) — {'K 부족이었음' if over else '상한 확인'}**: K 천장({Cc['K_terrain']})에서 "
            f"acc {Cc['acc_open']:.3f}. " +
            ("Exp2 0.49 를 크게 초과 → 0.49 는 표현력 상한이 아니라 K 부족이었다." if over
             else "0.49 근처 정체 → 그것이 지연-전용 물리의 진짜 상한(K 더 키워도 한계)."))


def _verdict_H8(B, s2p):
    if not s2p:
        return "- **H8(자율 vs 수동)**: 수동 S2' 부재로 판정 보류."
    close = B["acc_open"] >= s2p["acc_open"] - 0.05
    return (f"- **H8(자율 vs 수동) — {'지지' if close else '미지지'}**: 자율 B(acc {B['acc_open']:.2f}) vs "
            f"수동 S2'(acc {s2p['acc_open']:.2f}). 자율이 수동에 " +
            ("준함(더 적은 사람 개입)." if close else "못 미침."))


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
