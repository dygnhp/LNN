"""Ready-to-experiment 러너 — 실험 실행 + 데이터셋 정리 + 결과 MD 를 exlog/ 에 저장.

매 실험마다 ``exlog/<YYYYMMDD>_experiment_<No>/`` 디렉토리를 만들고(또는 비어 있는
최신 디렉토리를 재사용) 다음을 채운다:

    <YYYYMMDD>_experiment_<No>/
      datasets/     # 학습에 쓰인 데이터셋 스냅샷 + dataset.md (출처·규모·분할·라벨·표본)
      results/      # 결과 PNG/JSON, 학습 로그, report.md (개요·과정·결과·한계)
      experiment.json   # 설정·환경·git·메트릭 (기계 판독용)
      README.md         # 이 실험 한 장 요약

사용법:
    python scripts/run_experiment.py                  # D1~D4 전체, full 설정
    python scripts/run_experiment.py --demo D2        # 특정 시연만
    python scripts/run_experiment.py --demo D1 D3 --quick --note "코히어런스 어닐링 비교"
    python scripts/run_experiment.py --no 5           # 번호 강제 지정

데이터셋·실험·결과는 시연이 실제로 쓰는 데이터 함수·시드로 재구성해 스냅샷하므로,
저장된 datasets/ 는 그 실험이 학습한 데이터와 동일하다(결정론적 시드, §12).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXLOG = os.path.join(ROOT, "exlog")
OUTPUTS = os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))

# ── 시연 레지스트리 ──────────────────────────────────────────────────────────
# kind: 데이터셋 스냅샷 종류. prefix: outputs/ 산출물 파일 접두사.
DEMOS = {
    "D1": dict(module="train_text_classify", title="텍스트 분류 (합성 한국어, 3범주)",
               kind="text_classify", prefix="D1",
               full=dict(seed=0, R=5, n_total=320, epochs_res=6, epochs_open=24, batch=32),
               quick=dict(seed=0, R=5, n_total=160, epochs_res=3, epochs_open=8, batch=32)),
    "D2": dict(module="train_image_classify", title="이미지 분류 (sklearn digits 8x8)",
               kind="image_classify", prefix="D2",
               full=dict(seed=0, R=5, n_per_class=30, epochs_res=6, epochs_open=18, batch=32),
               quick=dict(seed=0, R=5, n_per_class=15, epochs_res=3, epochs_open=8, batch=32)),
    "D3": dict(module="demo_text_fill", title="빈칸 채우기 (나는 ___를 쓴다)",
               kind="text_mask", prefix="D3",
               full=dict(seed=0, R=5, n_total=320, epochs_res=5, epochs_open=20, batch=32),
               quick=dict(seed=0, R=5, n_total=160, epochs_res=3, epochs_open=8, batch=32)),
    "D4": dict(module="demo_text_to_image", title="글자->이미지 (A-Z 일부 -> 8x8)",
               kind="char_image", prefix="D4",
               full=dict(seed=0, R=5, epochs_res=10, epochs_open=40),
               quick=dict(seed=0, R=5, epochs_res=5, epochs_open=20)),
}

LABEL_NAMES = {0: "문자(글류·쓰기/읽기)", 1: "음료(마시기)", 2: "예술(그리기·부르기)"}


# ── 디렉토리 선택 ────────────────────────────────────────────────────────────
def _dir_is_empty(path):
    for _root, _dirs, files in os.walk(path):
        if files:
            return False
    return True


def resolve_experiment_dir(force_no=None):
    """``exlog/<date>_experiment_<No>``. 비어 있는 최신 디렉토리는 재사용, 아니면 다음 번호."""
    os.makedirs(EXLOG, exist_ok=True)
    date = _dt.date.today().strftime("%Y%m%d")
    prefix = f"{date}_experiment_"
    existing = {}
    for name in os.listdir(EXLOG):
        if name.startswith(prefix) and name[len(prefix):].isdigit():
            existing[int(name[len(prefix):])] = os.path.join(EXLOG, name)

    if force_no is not None:
        no = force_no
    elif not existing:
        no = 1
    else:
        hi = max(existing)
        no = hi if _dir_is_empty(existing[hi]) else hi + 1  # 빈 최신 디렉토리 재사용

    d = os.path.join(EXLOG, f"{prefix}{no}")
    os.makedirs(os.path.join(d, "datasets"), exist_ok=True)
    os.makedirs(os.path.join(d, "results"), exist_ok=True)
    return d, no, date


# ── 데이터셋 스냅샷 ──────────────────────────────────────────────────────────
def _save_montage(images_HW, titles, path, ncol=10, cmap="gray"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(images_HW)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(1.1 * ncol, 1.2 * nrow))
    axes = axes.ravel() if n > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < n:
            ax.imshow(images_HW[i], cmap=cmap, vmin=0, vmax=1)
            ax.set_title(str(titles[i]), fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def snapshot_dataset(kind, cfg, ddir):
    """시연이 실제로 쓰는 데이터를 같은 시드로 재구성해 스냅샷. dataset.md 본문(str) 반환."""
    import numpy as np

    if kind == "text_classify":
        from lnn.data.text_corpus import (VOCAB, VOCAB_SIZE, decode_ids,
                                          make_classification_dataset)
        X, y = make_classification_dataset(n_total=cfg["n_total"], seed=cfg["seed"])
        rows = ["idx,sentence,label,label_name"]
        for i in range(len(X)):
            sent = " ".join(t for t in decode_ids(X[i]) if t != "<PAD>")
            rows.append(f'{i},"{sent}",{int(y[i])},"{LABEL_NAMES[int(y[i])]}"')
        with open(os.path.join(ddir, "d1_corpus.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows))
        counts = {LABEL_NAMES[c]: int((y == c).sum()) for c in sorted(set(y.tolist()))}
        md = [f"- 출처: 합성 한국어 소코퍼스(`lnn/data/text_corpus.py`, seed={cfg['seed']}, 외부 다운로드 없음)",
              f"- 규모: {len(X)} 문장, vocab {VOCAB_SIZE} 토큰, 문장 길이 {X.shape[1]}",
              f"- 라벨(3범주) 분포: {counts}",
              f"- vocab: {VOCAB}",
              "- 파일: `d1_corpus.csv` (idx, 문장, 라벨, 라벨명)",
              "- 표본: " + " / ".join(
                  '"' + " ".join(t for t in decode_ids(X[i]) if t != "<PAD>") + f'" -> {LABEL_NAMES[int(y[i])]}'
                  for i in range(3))]
        return "\n".join(md)

    if kind == "image_classify":
        from lnn.data.image_data import load_digits_split
        Xtr, ytr, Xte, yte = load_digits_split(n_per_class=cfg["n_per_class"], seed=cfg["seed"])
        np.savez_compressed(os.path.join(ddir, "d2_digits.npz"),
                            Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        # 클래스별 첫 표본 몽타주
        firsts = []
        for c in range(10):
            idx = np.where(ytr == c)[0]
            firsts.append(Xtr[idx[0]].reshape(8, 8) if len(idx) else np.zeros((8, 8)))
        _save_montage(firsts, list(range(10)), os.path.join(ddir, "d2_samples.png"), ncol=10)
        md = [f"- 출처: `sklearn.datasets.load_digits()` (8x8, 10클래스), 0-16 -> 0-1 정규화",
              f"- 규모: train {len(Xtr)} / test {len(Xte)} (클래스당 {cfg['n_per_class']}, seed={cfg['seed']})",
              "- 파일: `d2_digits.npz` (Xtr/ytr/Xte/yte), `d2_samples.png` (클래스별 표본)"]
        return "\n".join(md)

    if kind == "text_mask":
        from lnn.data.text_corpus import (decode_ids, make_mask_dataset,
                                          noun_token_ids)
        X, y = make_mask_dataset(n_total=cfg["n_total"], seed=cfg["seed"])
        rows = ["idx,masked_sentence,target_noun"]
        for i in range(len(X)):
            sent = " ".join(t for t in decode_ids(X[i]) if t != "<PAD>")
            rows.append(f'{i},"{sent}","{decode_ids([y[i]])[0]}"')
        with open(os.path.join(ddir, "d3_mask.csv"), "w", encoding="utf-8") as f:
            f.write("\n".join(rows))
        md = [f"- 출처: 합성 한국어 코퍼스의 빈칸 버전(`make_mask_dataset`, seed={cfg['seed']})",
              f"- 규모: {len(X)} 문장 (목적어 자리를 <MASK>로 가림)",
              f"- 예측 대상 명사 id: {noun_token_ids()}",
              "- 파일: `d3_mask.csv` (idx, 마스크 문장, 정답 명사)",
              "- 표본: " + " / ".join(
                  '"' + " ".join(t for t in decode_ids(X[i]) if t != "<PAD>") + f'" -> {decode_ids([y[i]])[0]}'
                  for i in range(3))]
        return "\n".join(md)

    if kind == "char_image":
        from lnn.data.image_data import make_char_dataset
        chars, targets = make_char_dataset()
        _save_montage([targets[k].reshape(8, 8) for k in range(len(chars))],
                      chars, os.path.join(ddir, "d4_targets.png"), ncol=len(chars))
        np.savez_compressed(os.path.join(ddir, "d4_targets.npz"),
                            chars=np.array(chars), targets=targets)
        md = [f"- 출처: PIL 렌더 글자 비트맵(폰트 미가용 시 하드코딩 fallback), 글자 {chars}",
              "- 규모: 글자별 8x8 그레이스케일 목표 비트맵",
              "- 파일: `d4_targets.png` (목표 비트맵), `d4_targets.npz`"]
        return "\n".join(md)

    return "- (스냅샷 미정의 데이터셋)"


# ── stdout tee ───────────────────────────────────────────────────────────────
class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def run_demo(demo, cfg):
    """시연 모듈 run(**cfg) 실행. (metrics, 학습로그 str) 반환."""
    mod = __import__(DEMOS[demo]["module"])
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = _Tee(real, buf)
    try:
        metrics = mod.run(**cfg)
    finally:
        sys.stdout = real
    return metrics, buf.getvalue()


def copy_outputs(prefix, rdir):
    """outputs/ 에서 해당 시연 산출물(PNG/JSON/TXT)을 results/ 로 복사. 복사 목록 반환."""
    copied = []
    if not os.path.isdir(OUTPUTS):
        return copied
    for fn in sorted(os.listdir(OUTPUTS)):
        if fn.startswith(prefix):
            shutil.copy2(os.path.join(OUTPUTS, fn), os.path.join(rdir, fn))
            copied.append(fn)
    return copied


# ── 보고서 ───────────────────────────────────────────────────────────────────
def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def _metric_line(m):
    if m["demo"] in ("D1", "D2"):
        return (f"| {m['demo']} | acc(reservoir→opened) | "
                f"{m['acc_reservoir']:.3f} → {m['acc_opened']:.3f} | "
                f"{'상승(gate①)' if m['improved'] else '비상승'} |")
    if m["demo"] == "D3":
        return (f"| D3 | top-1 / 명사코드 직교성(전→후) | "
                f"{m['acc_reservoir']:.3f} → {m['acc_opened']:.3f} / "
                f"{m['orth_before']:.2f} → {m['orth_after']:.2f} | "
                f"top-5={m.get('top5')} |")
    if m["demo"] == "D4":
        return (f"| D4 | MSE(reservoir→opened) | "
                f"{m['mse_reservoir']:.4f} → {m['mse_opened']:.4f} | "
                f"{'개선' if m['improved'] else '비개선'} |")
    return f"| {m['demo']} | - | - | - |"


def write_reports(edir, no, date, ran, note, quick, started):
    """results/report.md, README.md, experiment.json 작성."""
    commit = _git_commit()
    env = dict(python=platform.python_version(), platform=platform.platform())
    try:
        import jax
        env["jax"] = jax.__version__
    except Exception:
        pass

    # experiment.json
    meta = dict(no=no, date=date, note=note, quick=quick, git_commit=commit,
                env=env, elapsed_sec=round(time.time() - started, 1),
                demos=[{**{"demo": d, "config": ran[d]["config"]}, **ran[d]["metrics"]} for d in ran])
    with open(os.path.join(edir, "experiment.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # results/report.md
    R = ["# 실험 보고서 — {0}_experiment_{1}".format(date, no), ""]
    if note:
        R += [f"> **메모**: {note}", ""]
    R += ["## 개요", "",
          f"- 일시: {date}  · 경과 {meta['elapsed_sec']}s  · 모드: {'quick' if quick else 'full'}",
          f"- 코드: LNN/ARIS (Block I), git `{commit}`",
          f"- 환경: Python {env['python']}, jax {env.get('jax','?')}, {env['platform']}",
          f"- 실행 시연: {', '.join(ran)}", "",
          "## 결과 요약", "",
          "| # | 지표 | reservoir → opened | 비고 |",
          "|---|------|--------------------|------|"]
    R += [_metric_line(ran[d]["metrics"]) for d in ran]
    R += ["", "## 시연별 상세", ""]
    for d in ran:
        info = DEMOS[d]
        R += [f"### {d} — {info['title']}", "",
              f"- 설정: `{ran[d]['config']}`",
              f"- 데이터셋: [`../datasets/`](../datasets/) (dataset.md 의 {d} 항목)",
              "- 산출물:"]
        for fn in ran[d]["outputs"]:
            if fn.lower().endswith(".png"):
                R += [f"  - ![{fn}](./{fn})"]
            else:
                R += [f"  - [`{fn}`](./{fn})"]
        R += ["", "<details><summary>학습 로그</summary>", "", "```",
              ran[d]["log"].strip(), "```", "</details>", ""]
    R += ["## 한계 (LNN_SPEC §11)", "",
          "- 존재 증명(proof of concept) — 현대 LLM/비전 모델과 성능 비교 대상 아님.",
          "- 코히어런스 길이 = 토큰 거리 상한(장문맥 불가), 선형보간 분수지연의 진폭-지연 허위결합 잔존.",
          "- 작은 격자(R=5)·짧은 학습. D3 는 경로 B D=8 채널 직교성 붕괴(§11★)를 동반."]
    with open(os.path.join(edir, "results", "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R))

    # README.md (한 장 요약)
    head = ["# {0}_experiment_{1}".format(date, no), "",
            (f"{note}" if note else "_(메모 없음)_"), "",
            f"- 시연: {', '.join(ran)} · 모드 {'quick' if quick else 'full'} · git `{commit}`",
            "- 상세 결과: [`results/report.md`](results/report.md)",
            "- 데이터셋: [`datasets/dataset.md`](datasets/dataset.md)", "",
            "| # | reservoir → opened |",
            "|---|--------------------|"]
    for d in ran:
        m = ran[d]["metrics"]
        if d in ("D1", "D2"):
            head.append(f"| {d} | acc {m['acc_reservoir']:.2f} → {m['acc_opened']:.2f} |")
        elif d == "D3":
            head.append(f"| D3 | top1 {m['acc_reservoir']:.2f} → {m['acc_opened']:.2f} (orth {m['orth_before']:.2f}→{m['orth_after']:.2f}) |")
        elif d == "D4":
            head.append(f"| D4 | MSE {m['mse_reservoir']:.3f} → {m['mse_opened']:.3f} |")
    with open(os.path.join(edir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(head))


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description="LNN/ARIS 실험 러너 — exlog 에 정리 저장")
    ap.add_argument("--demo", nargs="+", choices=list(DEMOS) + ["all"], default=["all"],
                    help="실행할 시연 (기본: all)")
    ap.add_argument("--quick", action="store_true", help="짧은 설정")
    ap.add_argument("--note", default="", help="실험 메모(보고서에 기록)")
    ap.add_argument("--no", type=int, default=None, help="실험 번호 강제 지정")
    args = ap.parse_args(argv)

    demos = list(DEMOS) if "all" in args.demo else args.demo
    edir, no, date = resolve_experiment_dir(args.no)
    print(f"[exlog] {os.path.relpath(edir, ROOT)}  (시연: {', '.join(demos)}, "
          f"{'quick' if args.quick else 'full'})")
    started = time.time()

    ddir = os.path.join(edir, "datasets")
    rdir = os.path.join(edir, "results")
    dataset_md = ["# 데이터셋 스냅샷", "",
                  "각 시연이 학습에 실제로 쓴 데이터를 같은 시드로 재구성해 저장(결정론적).", ""]
    ran = {}
    for d in demos:
        info = DEMOS[d]
        cfg = dict(info["full"])
        if args.quick:
            cfg.update(info["quick"])
        print(f"\n===== {d} — {info['title']} =====")

        dataset_md += [f"## {d} — {info['title']}", "",
                       snapshot_dataset(info["kind"], cfg, ddir), ""]
        metrics, log = run_demo(d, cfg)
        outs = copy_outputs(info["prefix"], rdir)
        ran[d] = dict(config=cfg, metrics=metrics, log=log, outputs=outs)

    with open(os.path.join(ddir, "dataset.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(dataset_md))

    write_reports(edir, no, date, ran, args.note, args.quick, started)
    print(f"\n[exlog] 저장 완료 -> {os.path.relpath(edir, ROOT)}")
    print(f"        README.md · results/report.md · datasets/dataset.md")


if __name__ == "__main__":
    main()
