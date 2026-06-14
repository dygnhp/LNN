"""§4.1 Area — 격자 + 지형/이득장 + 입출력 Cell + forward.

Area 는 학습 가능한 파라미터(지형 ``terrain_h``, 이득 ``gain_a``)만 leaf 로 들고,
**격자 기하량(geometry)은 forward 인자**로 받는다 — 기하는 학습 불변 상수이므로
미분 대상에서 빼기 위함(geometry leaf 화 방지). 지형 RBF 중심/폭(``terrain_c``,
``terrain_sigma``)·이득장 중심/폭은 Block I 에서 고정(학습 그룹 label 로 동결).

τ 테이블·이득장은 지형이 롤아웃 내내 고정이므로 forward 당 1회 계산하고(§3.2 ①),
배치는 rollout 만 ``vmap`` 한다(sc 는 예시 무관).

역할 유형(role)은 readout 모드가 아니라 **그래프 위치 표식**일 뿐 — 엔진은 공유.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from . import fields, readout
from .delay import clamp_tau, tobler_f
from .dynamics import buffer_length, make_step_constants, rollout


class Area(eqx.Module):
    # 학습 가능
    terrain_h: jax.Array
    gain_a: jax.Array
    # 고정(동결 그룹) — 지형/이득장 RBF 위치·폭
    terrain_c: jax.Array
    terrain_sigma: jax.Array
    gain_d: jax.Array
    gain_sigma: jax.Array
    # 정적 설정
    role: str = eqx.field(static=True)
    gen_cells: tuple = eqx.field(static=True)
    out_cells: tuple = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    tau_base: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    s_star: float = eqx.field(static=True)
    tau_min: float = eqx.field(static=True)
    tau_max: float = eqx.field(static=True)

    def edge_tau(self, geo):
        """변별 지연 τ:[E] — s=∇T(m)·ê → Tobler f → clamp_τ (§3.3)."""
        s = jnp.sum(
            fields.terrain_grad(geo.edge_mid, self.terrain_h, self.terrain_c, self.terrain_sigma)
            * geo.edge_hat,
            axis=-1,
        )
        f = tobler_f(s, self.tau_base, self.gamma, self.s_star)
        return clamp_tau(f, self.tau_min, self.tau_max)

    def step_constants(self, geo):
        tau = self.edge_tau(geo)
        g_cell = fields.gain_value(geo.pos, self.gain_a, self.gain_d, self.gain_sigma)
        return make_step_constants(geo, tau, g_cell, buffer_length(self.tau_max))

    def forward(self, geo, inject_BTN, window):
        """배치 forward. inject_BTN:[B,T,N], window:[valid] bool → (c_BTS, u_BC).

        c_BTS: 출력 셀별 정합필터 응답, u_BC: 부호 있는 readout(경계 통신 벡터 u, §3.7).
        """
        sc = self.step_constants(geo)
        out_cells = jnp.asarray(self.out_cells)
        w = readout.wavelet(self.P)

        def single(inject_TN):
            o = rollout(sc, inject_TN, out_cells, geo.N)
            c_TS = readout.matched_filter(o, w)
            u = readout.signed_readout(c_TS, window)
            return c_TS, u

        return jax.vmap(single)(inject_BTN)

    def tmin_window(self, geo, sources=None):
        """Dijkstra t_min(§3.6 ※1) 로 생성한 적분 윈도 마스크(호스트, stop_gradient).

        학습 중 재사용할 수 있도록 forward 밖에서 1회 계산한다.
        """
        tau = jax.lax.stop_gradient(self.edge_tau(geo))
        src = list(self.gen_cells) if sources is None else list(sources)
        tmins = readout.dijkstra_tmin(geo, tau, src, list(self.out_cells))
        valid = self.n_steps - self.P + 1
        return readout.window_mask(valid, float(tmins.min()), self.P)


def pick_cells(geo, count, ring_frac=0.75, phase=0.0, include_center=True):
    """격자에서 잘 분산된 ``count`` 개 셀 인덱스를 고른다(생성기/출력 단자 배치용).

    중심(옵션) + 반지름 ring_frac·R 부근 링에서 각도 균등 샘플. ``phase`` 로 링을 회전해
    서로 다른 단자 집합(인코더/버스/디코더)이 겹치지 않게 한다. 결정론적.
    """
    import numpy as np

    pos = np.asarray(geo.pos)
    center = pos.mean(0)
    rad = np.linalg.norm(pos - center, axis=1)
    rmax = rad.max()
    chosen = []
    if include_center and count >= 1:
        chosen.append(int(np.argmin(rad)))  # 중심
    target_r = ring_frac * rmax
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + phase
    ang = np.arctan2(pos[:, 1] - center[1], pos[:, 0] - center[0])
    for a in angles:
        if len(chosen) >= count:
            break
        score = (rad - target_r) ** 2 + 3.0 * (np.angle(np.exp(1j * (ang - a)))) ** 2
        order = np.argsort(score)
        for idx in order:
            if int(idx) not in chosen:
                chosen.append(int(idx))
                break
    # 부족하면 채움
    i = 0
    while len(chosen) < count:
        if i not in chosen:
            chosen.append(i)
        i += 1
    return tuple(chosen[:count])
