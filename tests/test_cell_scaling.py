"""Phase 4 §3.2 — R↑로 경로 다양성이 실제 증가하는지(C1 전제) + 동반 규칙."""

from lnn_block2.cell_scaling import CellScaler, n_steps_for, path_diversity


def test_co_scaling_rule():
    # R↑로 cell·K·n_steps 동반 증가.
    s5, s8, s12 = CellScaler(5), CellScaler(8), CellScaler(12)
    assert s5.n_cells == 91 and s8.n_cells == 217 and s12.n_cells == 469
    assert s5.K < s8.K < s12.K                 # K 동반 확장
    assert s5.n_steps < s8.n_steps < s12.n_steps
    assert n_steps_for(5) == 70


def test_path_diversity_increases_with_R():
    # H-CELL 전제: 격자 부피↑ → 입력→출력 도착 파형 다양성↑.
    d5 = path_diversity(5, seed=0)
    d8 = path_diversity(8, seed=0)
    d12 = path_diversity(12, seed=0)
    assert d5 < d8 < d12, f"path diversity not monotone: {d5}, {d8}, {d12}"
