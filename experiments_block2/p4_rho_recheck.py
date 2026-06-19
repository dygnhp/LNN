"""P4 — 펄스 공유 후 ρ 재확인 (§4). 공유로 ρ↑해도 ρ<1 정규화로 수렴 보존.

① coupling↑ 가 결합계 ρ 를 미는지 측정. ② 고이득+공유로 ρ>1 → 자유전개 발산,
이득 사영(ρ<1)으로 수렴 회복. 작은 격자(R=3, M=2).
"""

from __future__ import annotations

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import _common as C  # noqa: E402

from lnn.area import pick_cells  # noqa: E402
from lnn.cluster import DEFAULT_HP, _make_area  # noqa: E402
from lnn.dynamics import buffer_length, make_step_constants, step  # noqa: E402
from lnn.readout import wavelet  # noqa: E402
from lnn_block2.rp_pulse_coupling import measure_rho_coupled  # noqa: E402
from lnn_block2.spectral import _area_tau_gcell  # noqa: E402


def _free_coupled_growth(areas, geo, coupling, G0, drive, P, n_steps=140):
    """공통시계 결합계 임펄스 후 자유전개 성장비(버퍼 노름). ρ>1 이면 증가."""
    M = len(areas)
    L = buffer_length(areas[0].tau_max)
    scs = []
    for a in areas:
        tau, _g = _area_tau_gcell(a, geo, G0)
        scs.append(make_step_constants(geo, tau, jnp.full((geo.N,), G0), L))
    w = wavelet(P)
    bufs = jnp.zeros((M, geo.E, L))
    norms = []
    for t in range(n_steps):
        inj = jnp.zeros((geo.N,))
        if t < P:
            for cc in drive:
                inj = inj.at[cc].add(float(w[t]))
        nb = jnp.stack([step(bufs[m], inj, scs[m])[0] for m in range(M)], 0)
        if coupling > 0.0:
            nb = (1 - coupling) * nb + coupling * jnp.mean(nb, 0, keepdims=True)
        bufs = nb
        norms.append(float(jnp.linalg.norm(bufs)))
    norms = np.asarray(norms)
    return norms[-1] / (norms[P] + 1e-12)


def run(seed=0, R=3, M=2, rho_target=0.95):
    print("[P4] 펄스 공유 후 ρ 재확인 (coupling↑ → ρ↑, ρ<1 사영으로 수렴 보존)")
    geo = C.make_geometry(R)
    keys = jax.random.split(jax.random.PRNGKey(seed), M)
    areas = tuple(_make_area(geo, keys[m], "processor", pick_cells(geo, 1),
                             pick_cells(geo, 3), dict(DEFAULT_HP)) for m in range(M))
    drive = list(pick_cells(geo, 1))
    P = 8

    # ① coupling 이 ρ 를 미는가 (이득=1)
    rho_by_coupling = {c: round(float(measure_rho_coupled(areas, geo, c, gain_override=1.0)), 4)
                       for c in (0.0, 0.3, 0.6)}
    print(f"    ρ vs coupling (G=1): {rho_by_coupling}")

    # ② 고이득+공유 → ρ>1 발산, 사영 → 수렴
    coupling = 0.5
    G_hi = 1.6
    rho_hi = float(measure_rho_coupled(areas, geo, coupling, gain_override=G_hi))
    growth_hi = _free_coupled_growth(areas, geo, coupling, G_hi, drive, P)
    G_reg = G_hi
    rho_reg = rho_hi
    for _ in range(10):
        if rho_reg <= rho_target:
            break
        G_reg *= 0.9
        rho_reg = float(measure_rho_coupled(areas, geo, coupling, gain_override=G_reg))
    growth_reg = _free_coupled_growth(areas, geo, coupling, G_reg, drive, P)

    diverged = bool(growth_hi > 2.0 or not np.isfinite(growth_hi))
    converged = bool(growth_reg < 1.0 and np.isfinite(growth_reg))
    print(f"    고이득 G={G_hi} coupling={coupling}: ρ={rho_hi:.3f} 성장={growth_hi:.2e} "
          f"({'발산' if diverged else '유계'})")
    print(f"    사영 G={G_reg:.3f}: ρ={rho_reg:.3f} 성장={growth_reg:.2e} "
          f"({'수렴' if converged else '미수렴'})")
    passed = bool(diverged and converged)
    return dict(exp="P4", rho_by_coupling=rho_by_coupling, rho_hi=rho_hi, rho_reg=rho_reg,
                growth_hi=float(growth_hi), growth_reg=float(growth_reg),
                coupling_raises_rho=bool(rho_by_coupling[0.6] >= rho_by_coupling[0.0]),
                diverged_unreg=diverged, converged_reg=converged, passed=passed)


if __name__ == "__main__":
    run()
