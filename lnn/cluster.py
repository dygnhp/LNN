"""§4.2 Cluster — Area 들의 직렬 DAG + 경계 재생(§3.7, 유일한 비선형).

초판은 직렬: encoder → processor(×L) → decoder. 그래프 일반화는 인터페이스만.
Area 경계에서 통신 벡터 u 에 재생을 적용한다:

    φ(u) = tanh(u / u0)                      # 부호 보존 압축(유일한 비선형성)
    다음 Area 재방사 진폭 = g_k · φ(u_k)     # g_k: 채널별 대각 이득(밀집행렬 금지)

# TODO(잔차 스트림 도파로): Cluster 관통 저지연 skip 경로. 초판 미구현.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .area import Area, pick_cells
from .readout import wavelet


def build_pulse_injection(amps_BC, gen_cells, n_cells, n_steps, P, t0=0):
    """채널 진폭 amps:[B,C] → 주입 텐서 [B, T, N].

    채널 k 는 gen_cells[k] 에서 amps[:,k]·wavelet 을 [t0, t0+P) 스텝에 방사.
    """
    B = amps_BC.shape[0]
    w = wavelet(P)  # [P]
    inj = jnp.zeros((B, n_steps, n_cells), amps_BC.dtype)
    for k, cell in enumerate(gen_cells):
        contrib = amps_BC[:, k][:, None] * w[None, :]      # [B, P]
        inj = inj.at[:, t0:t0 + P, cell].add(contrib)
    return inj


class Cluster(eqx.Module):
    areas: tuple                      # (encoder, *processors, decoder) — Area 들
    diag_gains: jax.Array             # [n_boundary, C] 채널별 대각 이득 g_k (학습)
    u0: float = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    P: int = eqx.field(static=True)
    bus_cells: tuple = eqx.field(static=True)

    def forward(self, geo, enc_inject_BTN, windows):
        """enc_inject_BTN:[B,T,N] → (c_final_BTS, u_final_BC). windows: Area별 마스크 list."""
        c, u = self.areas[0].forward(geo, enc_inject_BTN, windows[0])
        for k in range(1, len(self.areas)):
            amp = self.diag_gains[k - 1] * jnp.tanh(u / self.u0)   # 경계 재생 φ + g_k
            inj = build_pulse_injection(amp, self.areas[k].gen_cells, geo.N, self.n_steps, self.P)
            c, u = self.areas[k].forward(geo, inj, windows[k])
        return c, u

    def make_windows(self, geo):
        """Area별 적분 윈도 마스크(Dijkstra t_min, stop_gradient) — forward 밖 1회 계산."""
        return [a.tmin_window(geo) for a in self.areas]


def _make_area(geo, key, role, gen_cells, out_cells, hp):
    """단일 Area 생성: 지형 RBF 중심/폭 고정, h 작은 난수, 이득 a=0(초기 G≈1)."""
    kh, kc = jax.random.split(key)
    K = hp["n_hills"]
    pos = np.asarray(geo.pos)
    lo, hi = pos.min(0), pos.max(0)
    # RBF 중심: 격자 범위에 결정론적으로 흩뿌림
    cc = np.asarray(jax.random.uniform(kc, (K, 2)) ) * (hi - lo) + lo
    terrain_c = jnp.asarray(cc, jnp.float32)
    terrain_sigma = jnp.full((K,), hp["sigma_t"], jnp.float32)
    terrain_h = 0.1 * jax.random.normal(kh, (K,))
    gain_d = jnp.asarray(cc, jnp.float32)              # 이득장 중심 = 지형 중심 재사용
    gain_sigma = jnp.full((K,), hp["sigma_g"], jnp.float32)
    gain_a = jnp.zeros((K,))
    return Area(
        terrain_h=terrain_h,
        gain_a=gain_a,
        terrain_c=terrain_c,
        terrain_sigma=terrain_sigma,
        gain_d=gain_d,
        gain_sigma=gain_sigma,
        role=role,
        gen_cells=tuple(int(x) for x in gen_cells),
        out_cells=tuple(int(x) for x in out_cells),
        n_steps=hp["n_steps"],
        P=hp["P"],
        tau_base=hp["tau_base"],
        gamma=hp["gamma"],
        s_star=hp["s_star"],
        tau_min=hp["tau_min"],
        tau_max=hp["tau_max"],
    )


DEFAULT_HP = dict(
    n_hills=8,
    sigma_t=3.0,
    sigma_g=3.0,
    n_steps=70,
    P=8,
    tau_base=1.2,
    gamma=1.0,
    s_star=-0.2,
    tau_min=1.0,
    tau_max=4.0,
    n_channels=8,
    n_proc=1,
    u0=1.0,
)


def build_serial_cluster(geo, enc_gen_cells, dec_out_cells, key, hp=None):
    """encoder → processor(×L) → decoder 직렬 Cluster 구성.

    - encoder: gen=enc_gen_cells(과제 입력), out=bus(C)
    - processor: gen=bus, out=bus
    - decoder: gen=bus, out=dec_out_cells(과제 출력)
    """
    hp = {**DEFAULT_HP, **(hp or {})}
    C = hp["n_channels"]
    bus = pick_cells(geo, C, phase=1.3)   # 인코더/디코더 단자와 링을 회전해 분리
    keys = jax.random.split(key, hp["n_proc"] + 2)

    areas = [_make_area(geo, keys[0], "encoder", enc_gen_cells, bus, hp)]
    for i in range(hp["n_proc"]):
        areas.append(_make_area(geo, keys[1 + i], "processor", bus, bus, hp))
    areas.append(_make_area(geo, keys[-1], "decoder", bus, dec_out_cells, hp))

    n_boundary = len(areas) - 1
    diag_gains = jnp.ones((n_boundary, C))
    return Cluster(
        areas=tuple(areas),
        diag_gains=diag_gains,
        u0=hp["u0"],
        n_steps=hp["n_steps"],
        P=hp["P"],
        bus_cells=bus,
    )
