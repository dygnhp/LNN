"""E4 — 루프 과제 ρ 안정성 (§8). ρ<1 정규화로 수렴 보존.

이득을 키우면(G>1) 한-스텝 전파 ρ>1 → 임펄스 후 자유전개에서 진폭이 지수 발산.
ρ<1 사영(이득 축소)으로 ρ<1 강제 → 자유전개가 감쇠(수렴 보존). 작은 격자(R=3)에서 측정.
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
from lnn_block2.spectral import spectral_radius, step_jacobian  # noqa: E402


def _free_evolution_norms(geo, tau, G0, drive_cells, P, n_steps=140):
    """임펄스(P스텝) 후 자유전개의 버퍼 노름 궤적. ρ>1 이면 증가, ρ<1 이면 감쇠."""
    L = buffer_length(4.0)
    sc = make_step_constants(geo, tau, jnp.full((geo.N,), G0), L)
    w = wavelet(P)
    buf = jnp.zeros((geo.E, L))
    norms = []
    for t in range(n_steps):
        inj = jnp.zeros((geo.N,))
        if t < P:
            for c in drive_cells:
                inj = inj.at[c].add(float(w[t]))
        buf, _o = step(buf, inj, sc)
        norms.append(float(jnp.linalg.norm(buf)))
    return np.asarray(norms)


def run(seed=0, R=3, rho_target=0.95):
    print("[E4] 루프 ρ 안정성 — 고이득(ρ>1) 발산 vs ρ<1 사영 수렴")
    geo = C.make_geometry(R)
    area = _make_area(geo, jax.random.PRNGKey(seed), "processor",
                      pick_cells(geo, 1), pick_cells(geo, 3), dict(DEFAULT_HP))
    tau = jax.lax.stop_gradient(area.edge_tau(geo))
    drive = list(pick_cells(geo, 1))
    P = 8

    # 고이득: ρ>1 유도
    G_hi = 1.6
    rho_hi = spectral_radius(step_jacobian(area, geo, gain_override=G_hi))
    norms_hi = _free_evolution_norms(geo, tau, G_hi, drive, P)
    growth_hi = norms_hi[-1] / (norms_hi[P] + 1e-12)

    # ρ<1 사영: 이득을 줄여 일양 ρ<target 도달까지 반복(ρ-G 비선형이라 1회로 안 되면 추가 축소)
    G_reg = G_hi * (rho_target / rho_hi)
    rho_reg = spectral_radius(step_jacobian(area, geo, gain_override=G_reg))
    for _ in range(8):
        if rho_reg <= rho_target:
            break
        G_reg *= 0.9
        rho_reg = spectral_radius(step_jacobian(area, geo, gain_override=G_reg))
    norms_reg = _free_evolution_norms(geo, tau, G_reg, drive, P)
    growth_reg = norms_reg[-1] / (norms_reg[P] + 1e-12)

    diverged = bool(growth_hi > 2.0 or not np.isfinite(growth_hi))
    converged = bool(growth_reg < 1.0 and np.isfinite(growth_reg))
    print(f"    고이득 G={G_hi}: rho={rho_hi:.3f}  자유전개 성장비={growth_hi:.2e} "
          f"({'발산' if diverged else '유계'})")
    print(f"    사영 G={G_reg:.3f}: rho={rho_reg:.3f}  자유전개 성장비={growth_reg:.2e} "
          f"({'수렴' if converged else '미수렴'})")
    passed = bool(diverged and converged)
    return dict(exp="E4", rho_unreg=float(rho_hi), rho_reg=float(rho_reg),
                growth_unreg=float(growth_hi), growth_reg=float(growth_reg),
                diverged_unreg=diverged, converged_reg=converged,
                rho_target=rho_target, passed=passed)


if __name__ == "__main__":
    run()
