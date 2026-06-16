"""§0.3 frozen interface 회귀 방지 — Block II 가 ARIS 코어 시그니처를 바꾸지 않았는지.

Cell 갱신·변지연 τ·경계 재생 φ·통신 벡터 u·encode/decode 의 시그니처가 Block I 과
동일해야 한다. 깨지면 Block II 빌드를 중단해야 한다(§9 절차 1).
"""

import inspect

from lnn import area, cluster, delay, dynamics, encodings, readout


def _params(fn):
    return list(inspect.signature(fn).parameters)


def test_delay_clamp_tau_signature():
    assert _params(delay.clamp_tau)[:3] == ["u", "tau_min", "tau_max"]
    assert hasattr(delay, "tobler_f") and hasattr(delay, "edge_delay_from_slope")


def test_dynamics_step_signature():
    # Cell 갱신 1스텝 + 분수지연 상수 구성 규약 불변.
    assert _params(dynamics.make_step_constants)[:4] == ["geo", "tau", "g_cell", "L"]
    assert _params(dynamics.step)[:2] == ["buf", "inject_t"]
    assert _params(dynamics.rollout)[:4] == ["sc", "inject_TN", "out_cells", "n_cells"]


def test_readout_signatures():
    assert _params(readout.matched_filter)[:2] == ["o_TS", "w"]
    assert _params(readout.signed_readout)[:2] == ["c_TS", "mask"]
    assert _params(readout.dijkstra_tmin)[:4] == ["geo", "tau", "sources", "targets"]


def test_area_forward_signature():
    assert _params(area.Area.forward) == ["self", "geo", "inject_BTN", "window"]


def test_cluster_forward_and_regen():
    # u 통신 + 경계 재생(φ=tanh) — forward 시그니처와 tanh 사용(소스) 확인.
    assert _params(cluster.Cluster.forward) == ["self", "geo", "enc_inject_BTN", "windows"]
    src = inspect.getsource(cluster.Cluster.forward)
    assert "tanh" in src  # φ = 부호 보존 tanh (유일 비선형)


def test_encode_decode_signatures():
    assert _params(encodings.TextTimeEncoder.encode) == ["self", "tokens_BS"]
    assert _params(encodings.ImageEncoder.encode) == ["self", "images_BP"]
