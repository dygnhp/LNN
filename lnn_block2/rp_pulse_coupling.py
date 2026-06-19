"""작업 2 — RP 펄스 수준 공유 완성 (E2 의 'u-수준 근사' 처방, §2).

Phase 1 RP 는 경계 벡터 u 만 공유(늦은 융합). Phase 2 RP 는 **펄스 버퍼를 공통 시계 위에서
직접 공유**(이른 융합·진짜 간섭): 매 스텝 각 Area 의 버퍼를 부분 혼합한다. ARIS step 은 그대로
재사용(코어 불변) — 스텝 사이에 버퍼를 섞는 것만 신규.

위험(§2.2): ① 시간 정렬 — 펄스가 코히어런스 안에서 정렬돼야 간섭 유의(공통 시계로 정렬).
② 해석 가능성 — 펄스 혼합은 u 분해를 흐림 → 융합 후 Area별 u 를 따로 readout 해 진단.
③ ρ↑ — Area 간 되먹임 강화 → §spectral ρ<1 정규화가 더 중요(P4).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from lnn.dynamics import buffer_length, make_step_constants, step
from lnn.readout import matched_filter, signed_readout, wavelet
from lnn_block2.spectral import _area_tau_gcell, spectral_radius


def _scs(areas, geo):
    return [a.step_constants(geo) for a in areas]


def rp_rollout_single(scs, E, L, geo_N, inj_list_TN, coupling):
    """공통 시계 다중 Area 롤아웃(단일 예시). inj_list:[M][T,N] → outs:[T, M, N].

    각 스텝: Area별 ARIS step → 버퍼 부분 혼합(coupling). coupling=0 이면 IP(독립).
    """
    M = len(scs)
    inj_perstep = jnp.transpose(jnp.stack(inj_list_TN, 0), (1, 0, 2))   # [T, M, N]

    def stepfn(bufs, inj_t):                                            # bufs:[M,E,L], inj_t:[M,N]
        nb, outs = [], []
        for m in range(M):
            b2, a = step(bufs[m], inj_t[m], scs[m])
            nb.append(b2)
            outs.append(a)
        nb = jnp.stack(nb, 0)
        if coupling > 0.0 and M > 1:                                   # 펄스 수준 공유(공통 시계)
            mean_b = jnp.mean(nb, axis=0, keepdims=True)
            nb = (1.0 - coupling) * nb + coupling * mean_b
        return nb, jnp.stack(outs, 0)                                  # [M,N]

    buf0 = jnp.zeros((M, E, L))
    _, outs = jax.lax.scan(stepfn, buf0, inj_perstep)                  # [T, M, N]
    return outs


class RPPulseCoupling(eqx.Module):
    """펄스 버퍼 공유 다중 Area. coupling=0 → IP(독립), >0 → RP(펄스 수준 간섭)."""

    areas: tuple
    coupling: float = eqx.field(static=True)
    shared_clock: bool = eqx.field(static=True)
    out_cells: tuple = eqx.field(static=True)
    P: int = eqx.field(static=True)

    def run(self, geo, inj_BTN, windows, inj_list_BTN=None):
        """inj:[B,T,N](공통) 또는 inj_list_BTN(Area별) → u_concat:[B, M·C]. 융합 후 Area별 u 보존."""
        scs = _scs(self.areas, geo)
        L = scs[0].L
        out_arr = jnp.asarray(self.out_cells)
        w = wavelet(self.P)
        M = len(self.areas)

        def single(args):
            if inj_list_BTN is None:
                inj_b = args
                inj_list = [inj_b] * M
            else:
                inj_list = list(args)
            outs = rp_rollout_single(scs, geo.E, L, geo.N, inj_list, self.coupling)  # [T,M,N]
            us = []
            for m in range(M):
                o = outs[:, m, :][:, out_arr]                       # [T, C]
                c = matched_filter(o, w)
                us.append(signed_readout(c, windows[m]))            # [C]
            return jnp.concatenate(us)                              # [M·C]

        if inj_list_BTN is None:
            return jax.vmap(single)(inj_BTN)                        # [B, M·C]
        return jax.vmap(single)(tuple(inj_list_BTN))

    def make_windows(self, geo):
        return [a.tmin_window(geo) for a in self.areas]


# ── P4: 펄스 공유 결합 연산자의 ρ (스택 버퍼 야코비안) ──────────────────────────
def coupled_step_jacobian(areas, geo, coupling, gain_override=None):
    """M-Area 공통시계 1스텝의 스택버퍼([M·E·L]) 야코비안. 작은 격자에서 ρ 측정용."""
    M = len(areas)
    L = buffer_length(areas[0].tau_max)
    scs = []
    for a in areas:
        tau, g = _area_tau_gcell(a, geo, gain_override)
        scs.append(make_step_constants(geo, tau, g, L))
    inj0 = jnp.zeros((geo.N,))

    def f(flat):
        bufs = flat.reshape(M, geo.E, L)
        nb = jnp.stack([step(bufs[m], inj0, scs[m])[0] for m in range(M)], 0)
        if coupling > 0.0 and M > 1:
            nb = (1.0 - coupling) * nb + coupling * jnp.mean(nb, axis=0, keepdims=True)
        return nb.reshape(-1)

    return jax.jacrev(f)(jnp.zeros(M * geo.E * L))


def measure_rho_coupled(areas, geo, coupling, gain_override=None):
    """공통시계 결합계의 ρ(스펙트럼 반경). 펄스 공유가 ρ 를 미는지 측정(P4)."""
    return spectral_radius(coupled_step_jacobian(areas, geo, coupling, gain_override))
