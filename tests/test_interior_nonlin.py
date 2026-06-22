"""Phase 6 실험 II 단위 검증 — density=none 회귀·grad 흐름·zero-mean/DC 가드."""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import _common as C  # noqa: E402

from lnn import dynamics  # noqa: E402
from lnn.area import pick_cells  # noqa: E402
from lnn.cluster import DEFAULT_HP, _make_area  # noqa: E402
from lnn_block2 import interior_nonlin as IN  # noqa: E402


def _area_sc(R=4, seed=0):
    geo = C.make_geometry(R)
    gen = pick_cells(geo, 4)
    out = pick_cells(geo, 3, phase=2.6)
    area = _make_area(geo, jax.random.PRNGKey(seed), "processor", gen, out, dict(DEFAULT_HP))
    return geo, area, gen, out


def _inject(geo, gen, n_steps=50, P=8):
    from lnn.readout import wavelet
    w = wavelet(P)
    inj = np.zeros((n_steps, geo.N), np.float32)
    for c in gen:
        for k in range(P):
            inj[k, c] += float(w[k])
    return jnp.asarray(inj)


def test_density_none_regression():
    """density=none(마스크 전부 False) → 코어 dynamics.rollout 과 비트 동일."""
    geo, area, gen, out = _area_sc()
    sc = area.step_constants(geo)
    inj = _inject(geo, gen, area.n_steps, area.P)
    out_arr = jnp.asarray(out)
    mask = IN.build_nl_mask(geo, gen + out, "none")
    core = dynamics.rollout(sc, inj, out_arr, geo.N)
    interior = IN.interior_rollout(sc, inj, out_arr, geo.N, "signed_relu", 0.1, "cell_sum", mask)
    assert jnp.allclose(core, interior, atol=1e-6), "density=none must regress to core"


def test_grad_flows_through_interior_nonlin():
    geo, area, gen, out = _area_sc()
    inj = _inject(geo, gen, area.n_steps, area.P)[None]   # [1,T,N]
    win = area.tmin_window(geo)
    mask = IN.build_nl_mask(geo, gen + out, "dense")

    def loss(h):
        a2 = type(area)(terrain_h=h, gain_a=area.gain_a, terrain_c=area.terrain_c,
                        terrain_sigma=area.terrain_sigma, gain_d=area.gain_d,
                        gain_sigma=area.gain_sigma, role=area.role, gen_cells=area.gen_cells,
                        out_cells=area.out_cells, n_steps=area.n_steps, P=area.P,
                        tau_base=area.tau_base, gamma=area.gamma, s_star=area.s_star,
                        tau_min=area.tau_min, tau_max=area.tau_max)
        u = IN.interior_area_forward(a2, geo, inj, win, "signed_relu", 0.1, "cell_sum", mask)
        return jnp.sum(u ** 2)

    g = jax.grad(loss)(area.terrain_h)
    assert bool(jnp.all(jnp.isfinite(g)))


def test_signed_preserves_zero_mean_vs_raw_dc():
    """부호 보존(signed_relu)은 DC 누적이 작고, 생짜 relu 는 크다(반파 정류 DC)."""
    geo, area, gen, out = _area_sc()
    inj = _inject(geo, gen, area.n_steps, area.P)
    mask = IN.build_nl_mask(geo, gen + out, "dense")
    dc_signed = IN.dc_drift(area, geo, inj, "signed_relu", 0.1, "cell_sum", mask)
    dc_raw = IN.dc_drift(area, geo, inj, "relu", 0.1, "cell_sum", mask)
    assert dc_raw > dc_signed, f"raw ReLU should accrue more DC: raw={dc_raw:.3e} signed={dc_signed:.3e}"
