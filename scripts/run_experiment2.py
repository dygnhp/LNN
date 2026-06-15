"""Experiment 2 — 대규모(MNIST 스케일링) + 비용 측정(시간·FLOPs·MLP 대비) + D3 경로 A 시범.

산출은 exlog/<YYYYMMDD>_experiment_<No>/ 에:
  results/gates.json     게이트 ②③ 수치(선결)
  results/cost.json      시연·규모별 시간·FLOPs·model_config
  results/*.png          혼동행렬·지형 히트맵·학습곡선·비용-정확도 산점도
  results/report.md      정확도 표 + 비용 표 + 산점도 + MLP 배율 + 모델 구성 표
  datasets/dataset.md    스케일링 단계별 데이터 스냅샷
  README.md / experiment.json

ARIS 코어 불변 — 데이터 규모와 계측만 확장(§Experiment2). CPU 완주, 결정론.
사용법:  python scripts/run_experiment2.py [--quick]
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

import _common as C  # noqa: E402  (stdout utf-8, matplotlib Agg, OUT_DIR)
from lnn import instrument, train  # noqa: E402
from lnn.baselines import train_mlp  # noqa: E402
from lnn.cluster import build_serial_cluster  # noqa: E402
from lnn.area import pick_cells  # noqa: E402
from lnn.data.mnist_data import load_mnist_split  # noqa: E402
from lnn.data.text_corpus import (VOCAB_SIZE, make_mask_dataset,  # noqa: E402
                                  noun_token_ids)
from lnn.decoders import vocabulary_logits  # noqa: E402
from lnn.encodings import (FreqEncoder, ImageEncoder, TextTimeEncoder,  # noqa: E402
                           channel_orthogonality)
from run_experiment import resolve_experiment_dir  # noqa: E402

N_CLASSES = 10
N_FEAT = 12

STAGES = [
    dict(name="S0", per_class=30, R=5, size=8, n_steps=70, batch=32, e_res=4, e_open=10),
    dict(name="S1", per_class=100, R=5, size=8, n_steps=70, batch=32, e_res=4, e_open=10),
    dict(name="S2", per_class=100, R=8, size=14, n_steps=70, batch=32, e_res=4, e_open=10),
    dict(name="S3", per_class=20, R=16, size=28, n_steps=60, batch=16, e_res=2, e_open=3),
]


def _quick(stages):
    for s in stages:
        s["per_class"] = min(s["per_class"], 12)
        s["e_res"], s["e_open"] = 2, 3
    return stages


# ─────────────────────────── 게이트 ②③ 선결 ─────────────────────────────────
def compute_gates(rdir):
    from tests.test_telescoping import _setup, _v_rel, EPS_REL  # x64 미설정(안전)
    geo, s = _setup()
    v_lin, _ = _v_rel(geo, s, "linear", dict(a=1.0, b=1.0))
    v_tob, _ = _v_rel(geo, s, "tobler", dict(tau_base=1.0, gamma=1.5, s_star=-0.2))

    pin = subprocess.run([sys.executable, "-m", "pytest",
                          os.path.join(ROOT, "tests", "test_gradient_fd.py"),
                          os.path.join(ROOT, "tests", "test_telescoping.py"), "-q"],
                         cwd=ROOT, capture_output=True, text=True)
    # FD 상대오차(대표값) — x64 격리 위해 서브프로세스에서 계산
    fd = subprocess.run([sys.executable, "-c", _FD_SNIPPET], cwd=ROOT,
                        capture_output=True, text=True)
    try:
        fd_rel = float(fd.stdout.strip().splitlines()[-1])
    except Exception:
        fd_rel = None

    gates = dict(
        telescoping=dict(V_rel_linear=v_lin, V_rel_tobler=v_tob,
                         eps_rel=EPS_REL, ratio=v_tob / max(v_lin, 1e-30),
                         pass_linear=bool(v_lin < EPS_REL),
                         pass_tobler=bool(v_tob > 10 * EPS_REL)),
        gradient_fd=dict(rel_error=fd_rel, threshold=1e-3,
                         pass_=(fd_rel is not None and fd_rel < 1e-3)),
        pytest_returncode=pin.returncode,
        pytest_pass=bool(pin.returncode == 0),
    )
    with open(os.path.join(rdir, "gates.json"), "w", encoding="utf-8") as f:
        json.dump(gates, f, ensure_ascii=False, indent=2)
    print(f"[gates] telescoping V_rel lin={v_lin:.2e} tob={v_tob:.2e} (ratio {gates['telescoping']['ratio']:.0f}x); "
          f"FD rel_err={fd_rel}; pytest {'PASS' if gates['pytest_pass'] else 'FAIL'}")
    return gates


_FD_SNIPPET = (
    "import jax; jax.config.update('jax_enable_x64', True)\n"
    "import jax.numpy as jnp, numpy as np\n"
    "import sys; sys.path.insert(0, r'%s')\n" % ROOT +
    "from lnn import delay\n"
    "rng=np.random.default_rng(1); E=12; L=7\n"
    "s=jnp.asarray(rng.uniform(-0.5,0.5,E)); tgt=jnp.asarray(rng.uniform(-1,1,(E,L)))\n"
    "dp=dict(tau_base=2.4,gamma=0.8,s_star=-0.2,tau_min=1.0,tau_max=5.0)\n"
    "def loss(th):\n"
    "    tau=delay.edge_delay_from_slope(s*th,dp,law='tobler',clamp=True)\n"
    "    di=tau-1.0; i0=jnp.floor(di).astype(jnp.int32); i0=jnp.clip(i0,0,L-2); fr=di-i0\n"
    "    buf=jnp.zeros((E,L)); ar=jnp.arange(E)\n"
    "    buf=buf.at[ar,i0].add(1.0-fr); buf=buf.at[ar,i0+1].add(fr)\n"
    "    return jnp.sum(tgt*buf)\n"
    "g=float(jax.grad(loss)(1.0)); eps=1e-4\n"
    "fd=(float(loss(1.0+eps))-float(loss(1.0-eps)))/(2*eps)\n"
    "print(abs(g-fd)/(abs(fd)+1e-12))\n"
)


# ─────────────────────────── 이미지 분류 단계 ───────────────────────────────
def train_image_stage(stage, seed=0):
    geo = C.make_geometry(stage["R"])
    hp = {**C.classify_hp(), "n_steps": stage["n_steps"], "n_proc": 1}
    Xtr, ytr, Xte, yte, src = load_mnist_split(
        n_per_class=stage["per_class"], size=stage["size"],
        test_per_class=max(10, stage["per_class"] // 3), seed=seed)
    pix = C.map_image_cells(geo, stage["size"], stage["size"])
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

    t = time.perf_counter()
    model, hist_r, win = train.run_phase(
        model, static, loss_fn, Xtr, ytr, epochs=stage["e_res"], batch_size=stage["batch"],
        open_terrain=False, open_gain=False, lrs=None,
        lam_schedule=train.anneal_lambda(2.0, 8.0), recompute_windows=rw,
        seed=seed, log_prefix=f"[{stage['name']} res] ")
    reservoir_sec = time.perf_counter() - t
    acc_res = train.accuracy(predict(model, win), yte)

    t = time.perf_counter()
    model, hist_o, win = train.run_phase(
        model, static, loss_fn, Xtr, ytr, epochs=stage["e_open"], batch_size=stage["batch"],
        open_terrain=True, open_gain=True, lrs=None,
        lam_schedule=train.const_lambda(8.0), recompute_windows=rw,
        seed=seed + 1, log_prefix=f"[{stage['name']} open] ")
    opened_sec = time.perf_counter() - t
    logits_te = predict(model, win)
    acc_open = train.accuracy(logits_te, yte)

    # 추론 시간(워밍업 후)
    pj = eqx.filter_jit(lambda m, x: (lambda r: r)(m[1].forward(geo, m[0].encode(x), win)[1]))
    Xte_j = jnp.asarray(Xte)
    r = pj(model, Xte_j); jax.block_until_ready(r)
    t = time.perf_counter(); r = pj(model, Xte_j); jax.block_until_ready(r)
    infer_sec = time.perf_counter() - t

    return dict(geo=geo, model=model, hp=hp, src=src, pix=pix,
                Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                acc_res=acc_res, acc_open=acc_open, logits_te=logits_te,
                hist=hist_r + hist_o, reservoir_sec=reservoir_sec,
                opened_sec=opened_sec, infer_sec=infer_sec, win=win)


def measure_cost(res, stage):
    geo, model, win = res["geo"], res["model"], res["win"]
    enc, clus, _h = model
    inj = enc.encode(jnp.asarray(res["Xtr"][:stage["batch"]]))
    fwd = lambda x: clus.forward(geo, x, win)
    compile_sec, run_sec = instrument.time_call(fwd, inj)
    flops_xla, ca = instrument.flops_xla(fwd, inj)
    flops_an = instrument.flops_analytic(geo, stage["n_steps"], stage["batch"],
                                         n_areas=len(clus.areas), n_out=N_FEAT, P=clus.P)
    cfg = instrument.config_of(model, geo, "image", "discriminative", False,
                               batch=stage["batch"])
    return dict(
        time=dict(compile_sec=round(compile_sec, 3), reservoir_sec=round(res["reservoir_sec"], 2),
                  opened_sec=round(res["opened_sec"], 2), infer_sec=round(res["infer_sec"], 4),
                  fwd_run_sec=round(run_sec, 4),
                  sec_per_step=round(run_sec / max(stage["n_steps"], 1), 6)),
        flops=dict(method1_xla=flops_xla, method2_analytic=flops_an),
        model_config=cfg,
        config=dict(R=stage["R"], n_steps=stage["n_steps"], N_cells=geo.N,
                    edges=geo.E, batch=stage["batch"], img_size=stage["size"]),
        cost_analysis_raw=ca,
    )


# ─────────────────────────── D3 경로 A 시범 ─────────────────────────────────
def run_d3(encoding, seed=0, n_total=240, e_res=4, e_open=16):
    geo = C.make_geometry(5)
    D = 8
    hp = {**C.classify_hp(), "n_channels": D}
    X, y = make_mask_dataset(n_total=n_total, seed=seed)
    ntr = int(len(X) * 0.8)
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    ke, kc = jax.random.split(jax.random.PRNGKey(seed))
    emb = 0.5 * jax.random.normal(ke, (VOCAB_SIZE, D))

    if encoding == "time":
        gen = pick_cells(geo, D, phase=0.0)
        encoder = TextTimeEncoder(embedding=emb, gen_cells=gen, stride=12,
                                  P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    else:  # frequency (경로 A)
        gen = pick_cells(geo, 1, phase=0.0)            # 단일 운반 셀
        encoder = FreqEncoder(embedding=emb, gen_cells=gen, P=hp["P"],
                              n_steps=hp["n_steps"], n_cells=geo.N, window=16, stride=16)
    dec_out = pick_cells(geo, D, phase=2.6)
    cluster = build_serial_cluster(geo, gen, dec_out, kc, hp)
    model = (encoder, cluster)
    _, static = eqx.partition(model, eqx.is_inexact_array)
    nouns = jnp.asarray(noun_token_ids())
    orth_before = channel_orthogonality(emb[nouns])

    def loss_fn(params, Xb, Yb, windows, lam):
        e, cl = eqx.combine(params, static)
        _c, u = cl.forward(geo, e.encode(Xb), windows)
        logits = vocabulary_logits(u, e.embedding)
        return -jnp.mean(jax.nn.log_softmax(logits)[jnp.arange(Yb.shape[0]), Yb]), logits

    rw = lambda m: m[1].make_windows(geo)

    def predict(m, win):
        e, cl = m
        _c, u = cl.forward(geo, e.encode(jnp.asarray(Xte)), win)
        return vocabulary_logits(u, e.embedding)

    t = time.perf_counter()
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_res,
                                    batch_size=32, open_terrain=False, open_gain=False,
                                    lrs=None, lam_schedule=train.anneal_lambda(2, 8),
                                    recompute_windows=rw, seed=seed, log_prefix=f"[D3-{encoding} res] ")
    model, _, win = train.run_phase(model, static, loss_fn, Xtr, ytr, epochs=e_open,
                                    batch_size=32, open_terrain=True, open_gain=False,
                                    lrs=dict(embedding=4e-2, terrain_h=2e-2),
                                    lam_schedule=train.const_lambda(8.0),
                                    recompute_windows=rw, seed=seed + 1, log_prefix=f"[D3-{encoding} open] ")
    elapsed = time.perf_counter() - t
    top1 = train.accuracy(predict(model, win), yte)
    orth_after = channel_orthogonality(model[0].embedding[nouns])
    cfg = instrument.config_of(model, geo, encoding, "vocabulary", True, batch=32)
    return dict(encoding=encoding, top1=top1, orth_before=float(orth_before),
                orth_after=float(orth_after), elapsed_sec=round(elapsed, 1), model_config=cfg)


# ─────────────────────────── 산출물 ─────────────────────────────────────────
def save_terrain(model, geo, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lnn import fields
    area = model[1].areas[len(model[1].areas) // 2]
    pos = np.asarray(geo.pos)
    gx = np.linspace(pos[:, 0].min(), pos[:, 0].max(), 60)
    gy = np.linspace(pos[:, 1].min(), pos[:, 1].max(), 60)
    GX, GY = np.meshgrid(gx, gy)
    P = jnp.asarray(np.stack([GX.ravel(), GY.ravel()], -1), jnp.float32)
    T = np.asarray(fields.terrain_value(P, area.terrain_h, area.terrain_c, area.terrain_sigma))
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(T.reshape(60, 60), origin="lower", cmap="terrain",
                   extent=[gx.min(), gx.max(), gy.min(), gy.max()])
    ax.set_title(title); fig.colorbar(im); fig.tight_layout()
    fig.savefig(path, dpi=100); plt.close(fig)


def save_curve(hist, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(hist, "-o", ms=3); ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def save_scatter(stage_costs, mlp, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for sc in stage_costs:
        f = sc["flops"]["method2_analytic"]
        a = sc["acc_open"]
        ax.scatter(f, a, s=60)
        cfg = sc["cost"]["model_config"]
        ax.annotate(f"{sc['stage']}\nN={cfg['areas'][0]['N_cells']} "
                    f"p={cfg['params']['total_trainable']}",
                    (f, a), fontsize=7, xytext=(5, 5), textcoords="offset points")
    if mlp and mlp.get("flops_fwd"):
        ax.scatter(mlp["flops_fwd"], mlp["acc"], marker="*", s=160, c="red")
        ax.annotate(f"MLP\np={mlp['params']}", (mlp["flops_fwd"], mlp["acc"]),
                    fontsize=7, c="red", xytext=(5, -10), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("FLOPs (analytic, forward/batch)")
    ax.set_ylabel("accuracy (opened)"); ax.set_title("Exp2: cost vs accuracy (MNIST scaling)")
    ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_confusion(logits, yte, path, title):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pred = np.asarray(jnp.argmax(logits, -1))
    cm = np.zeros((N_CLASSES, N_CLASSES), int)
    for t, p in zip(np.asarray(yte), pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(4, 3.6))
    im = ax.imshow(cm, cmap="Blues"); ax.set_title(title)
    ax.set_xlabel("pred"); ax.set_ylabel("true"); fig.colorbar(im)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


# ─────────────────────────── 메인 ───────────────────────────────────────────
def main(quick=False):
    edir, no, date = resolve_experiment_dir()
    rdir = os.path.join(edir, "results")
    ddir = os.path.join(edir, "datasets")
    print(f"[exp2] {os.path.relpath(edir, ROOT)}  (quick={quick})")
    t0 = time.time()

    gates = compute_gates(rdir)

    stages = _quick([dict(s) for s in STAGES]) if quick else [dict(s) for s in STAGES]
    stage_results, ds_md = [], ["# 데이터셋 스냅샷 — MNIST 스케일링", ""]
    for st in stages:
        print(f"\n===== {st['name']}: per_class={st['per_class']} R={st['R']} "
              f"size={st['size']}x{st['size']} =====")
        res = train_image_stage(st)
        cost = measure_cost(res, st)
        cfg = cost["model_config"]
        print(f"    acc {res['acc_res']:.3f} -> {res['acc_open']:.3f} | "
              f"N={res['geo'].N} params={cfg['params']['total_trainable']} | "
              f"fwd {cost['time']['fwd_run_sec']}s flops_xla={cost['flops']['method1_xla']} "
              f"flops_an={cost['flops']['method2_analytic']}")
        tag = st["name"]
        save_confusion(res["logits_te"], res["yte"], os.path.join(rdir, f"{tag}_confusion.png"),
                       f"{tag} conf {res['acc_res']:.2f}->{res['acc_open']:.2f}")
        save_terrain(res["model"], res["geo"], os.path.join(rdir, f"{tag}_terrain.png"),
                     f"{tag} terrain (N={res['geo'].N})")
        save_curve(res["hist"], os.path.join(rdir, f"{tag}_loss.png"), f"{tag} loss")
        stage_results.append(dict(stage=tag, acc_res=res["acc_res"], acc_open=res["acc_open"],
                                  src=res["src"], cost=cost,
                                  flops=cost["flops"], config=st))
        ds_md += [f"## {tag}", f"- 출처: {res['src']}, per_class={st['per_class']}, "
                  f"해상도 {st['size']}x{st['size']}, 격자 R={st['R']} (N={res['geo'].N})",
                  f"- train {len(res['Xtr'])} / test {len(res['Xte'])}", ""]

    # MLP 기준선 (S0 데이터 = digits/MNIST 8x8)
    print("\n===== MLP baseline (8x8) =====")
    s0 = stages[0]
    Xtr, ytr, Xte, yte, _src = load_mnist_split(n_per_class=s0["per_class"], size=8,
                                                test_per_class=max(10, s0["per_class"] // 3))
    mlp = train_mlp(Xtr, ytr, Xte, yte, hidden=(32,), epochs=30, batch=32)
    print(f"    MLP acc={mlp['acc']:.3f} params={mlp['params']} "
          f"train={mlp['train_sec']}s flops_fwd={mlp['flops_fwd']}")

    # D3 경로 A 시범
    print("\n===== D3 path B (time) vs path A (frequency) =====")
    d3_time = run_d3("time", e_res=2 if quick else 4, e_open=6 if quick else 16)
    d3_freq = run_d3("frequency", e_res=2 if quick else 4, e_open=6 if quick else 16)
    print(f"    time : top1={d3_time['top1']:.3f} orth {d3_time['orth_before']:.2f}->{d3_time['orth_after']:.2f}")
    print(f"    freq : top1={d3_freq['top1']:.3f} orth {d3_freq['orth_before']:.2f}->{d3_freq['orth_after']:.2f}")

    save_scatter(stage_results, mlp, os.path.join(rdir, "cost_vs_accuracy.png"))

    # cost.json
    cost_json = dict(stages=[dict(stage=s["stage"], acc_open=s["acc_open"], **s["cost"])
                            for s in stage_results],
                     mlp_baseline=mlp,
                     d3_path_trial=dict(time=d3_time, frequency=d3_freq))
    with open(os.path.join(rdir, "cost.json"), "w", encoding="utf-8") as f:
        json.dump(cost_json, f, ensure_ascii=False, indent=2)
    with open(os.path.join(ddir, "dataset.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(ds_md))

    write_report(edir, no, date, gates, stage_results, mlp, d3_time, d3_freq, quick, t0)
    print(f"\n[exp2] 저장 완료 -> {os.path.relpath(edir, ROOT)}  ({time.time() - t0:.1f}s)")


def write_report(edir, no, date, gates, stages, mlp, d3t, d3f, quick, t0):
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    R = [f"# Experiment 2 보고서 — {date}_experiment_{no}", "",
         "대규모(MNIST 스케일링) + 비용 측정(시간·FLOPs·MLP 대비) + D3 경로 A 시범. ARIS 코어 불변.", "",
         f"- 모드 {'quick' if quick else 'full'} · git `{commit}` · Python {platform.python_version()}, jax {jax.__version__}",
         f"- 경과 {time.time() - t0:.0f}s", "",
         "## 게이트 ②③ (선결)", "",
         f"- telescoping: V_rel 선형={gates['telescoping']['V_rel_linear']:.2e} "
         f"(<{gates['telescoping']['eps_rel']:.0e} {'OK' if gates['telescoping']['pass_linear'] else 'FAIL'}), "
         f"Tobler={gates['telescoping']['V_rel_tobler']:.2e} "
         f"(비율 {gates['telescoping']['ratio']:.0f}x {'OK' if gates['telescoping']['pass_tobler'] else 'FAIL'})",
         f"- gradient_fd: 상대오차={gates['gradient_fd']['rel_error']} (<1e-3 "
         f"{'OK' if gates['gradient_fd']['pass_'] else 'FAIL'}) · pytest "
         f"{'PASS' if gates['pytest_pass'] else 'FAIL'}", "",
         "## 실험 A — MNIST 스케일링 (정확도)", "",
         "| stage | per_class | R | size | N_cells | params | acc_res | acc_open | Δ |",
         "|-------|-----------|---|------|---------|--------|---------|----------|---|"]
    for s in stages:
        cfg = s["cost"]["model_config"]
        cf = s["config"]
        R.append(f"| {s['stage']} | {cf['per_class']} | {cf['R']} | {cf['size']} | "
                 f"{cfg['areas'][0]['N_cells']} | {cfg['params']['total_trainable']} | "
                 f"{s['acc_res']:.3f} | {s['acc_open']:.3f} | {s['acc_open'] - s['acc_res']:+.3f} |")
    R += ["", "## 실험 B — 비용 (시간·FLOPs·MLP 대비)", "",
          "| stage | fwd_run_s | sec/step | FLOPs(xla) | FLOPs(analytic) | xla/MLP |",
          "|-------|-----------|----------|------------|-----------------|---------|"]
    mlp_f = mlp.get("flops_fwd")
    for s in stages:
        c = s["cost"]
        xla = c["flops"]["method1_xla"]
        ratio = (f"{xla / mlp_f:.0f}x" if (xla and mlp_f) else "-")
        R.append(f"| {s['stage']} | {c['time']['fwd_run_sec']} | {c['time']['sec_per_step']} | "
                 f"{xla} | {c['flops']['method2_analytic']} | {ratio} |")
    R += ["", f"- **MLP 기준선(8x8)**: acc {mlp['acc']:.3f}, params {mlp['params']}, "
          f"train {mlp['train_sec']}s, FLOPs_fwd {mlp_f}.",
          "- 비용-정확도 산점도: ![scatter](./cost_vs_accuracy.png)", "",
          "## D3 경로 A 시범 (time vs frequency, §1.3 / H4)", "",
          "| encoding | top-1 | orth_before | orth_after | 경과s |",
          "|----------|-------|-------------|------------|-------|",
          f"| time (경로 B) | {d3t['top1']:.3f} | {d3t['orth_before']:.2f} | {d3t['orth_after']:.2f} | {d3t['elapsed_sec']} |",
          f"| frequency (경로 A) | {d3f['top1']:.3f} | {d3f['orth_before']:.2f} | {d3f['orth_after']:.2f} | {d3f['elapsed_sec']} |",
          "",
          f"- H4 판정: 경로 A 가 orth_after 를 {d3t['orth_after']:.2f}→{d3f['orth_after']:.2f}, "
          f"top-1 을 {d3t['top1']:.2f}→{d3f['top1']:.2f} 로 변화. "
          f"{'직교성 유지+top1 상승 → H4 지지' if (d3f['orth_after'] < 0.5 and d3f['top1'] > d3t['top1']) else '판정은 report 본문 참조(Block I 단일-wavelet readout 한계 동반)'}.",
          "",
          "## 가설 메모 (§4)", "",
          "- H1(용량): MNIST 에서 LNN < MLP 예상 — 위 표의 acc 격차로 측정.",
          "- H2(스케일링): 데이터(S0→S1)보다 격자(S1→S2→S3)에 더 민감한지 — acc 표 비교.",
          "- H3(비용): LNN FLOPs 가 MLP 대비 1~3자릿수 — 위 xla/MLP 배율.",
          "- H4(경로 A): D3 경로 A 직교성/ top-1 — 위 D3 표.", "",
          "## 한계", "",
          "- 존재 증명. native(S3)는 비용 측정 목적의 축소 학습. 경로 A 는 Block I 단일-wavelet "
          "readout 하의 시범(완전한 주파수 뱅크 분리는 Block II)."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    readme = [f"# {date}_experiment_{no} — Experiment 2", "",
              "MNIST 스케일링 + 비용 측정 + D3 경로 A 시범. ARIS 코어 불변.", "",
              f"- 모드 {'quick' if quick else 'full'} · git `{commit}`",
              "- 상세: [`results/report.md`](results/report.md) · 비용 [`results/cost.json`](results/cost.json) · "
              "게이트 [`results/gates.json`](results/gates.json)", "",
              "| stage | N_cells | params | acc_res→open | FLOPs(xla) |",
              "|-------|---------|--------|--------------|------------|"]
    for s in stages:
        cfg = s["cost"]["model_config"]
        readme.append(f"| {s['stage']} | {cfg['areas'][0]['N_cells']} | "
                      f"{cfg['params']['total_trainable']} | {s['acc_res']:.2f}→{s['acc_open']:.2f} | "
                      f"{s['cost']['flops']['method1_xla']} |")
    readme += ["", f"- MLP 기준선: acc {mlp['acc']:.2f}, params {mlp['params']}",
               f"- D3 경로 A: orth time {d3t['orth_after']:.2f} vs freq {d3f['orth_after']:.2f}, "
               f"top1 {d3t['top1']:.2f} vs {d3f['top1']:.2f}"]
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(readme))

    meta = dict(no=no, date=date, quick=quick, git_commit=commit,
                elapsed_sec=round(time.time() - t0, 1), gates=gates,
                stages=[dict(stage=s["stage"], acc_res=s["acc_res"], acc_open=s["acc_open"],
                             cost=s["cost"]) for s in stages],
                mlp=mlp, d3=dict(time=d3t, frequency=d3f))
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
