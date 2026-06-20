"""Phase 5 §3.3 — 방식 C: Recursive Amplitude Loop (시간 순환, 파라미터 공유).

단일 처리 Area 를 출력→입력 피드백으로 n회 반복 통과시켜 깊이를 만든다(방식 A의 L개 독립
Area 와 같은 'φ 통과 횟수'를 1개 공유 지형으로). 한 루프: x_{n+1} = φ(A x_n).

필수 조건 ①(§1.4): **루프 바디에 φ 가 있어야 깊이가 생긴다.** φ 없이 선형 반복하면
x_n = A^n x_0 — 여전히 하나의 선형 변환(LTI 합성)이라 깊이 0. ``phi`` 인자로 강제·검증.
필수 조건 ②: 루프가 nilpotent(DAG)를 깨 ρ 1급 → φ=tanh 가 진폭을 유계화(자연 정규화);
순수 선형 루프는 ρ>1 에서 발산(E4 재현). ``linear=True`` 로 그 발산을 단위 테스트.

BPTT: 루프는 시간 전개 → jax.checkpoint(remat) 로 스텝 메모리 절감.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from lnn.cluster import build_pulse_injection


class RecursiveAmplitudeLoop(eqx.Module):
    """단일 Area 시간 순환 깊이. encoder→[area→φ→재주입]×n_loops→head."""

    encoder: object
    area: object               # 공유 처리 Area (gen=loop_cells, out=feat_cells)
    head: object
    loop_cells: tuple = eqx.field(static=True)   # 재주입 단자(C개)
    n_loops: int = eqx.field(static=True)
    u0: float = eqx.field(static=True)

    def run_u(self, geo, X, window, phi=jnp.tanh, gain=1.0):
        """n_loops 회 순환 후 최종 u:[B, C]. phi=tanh(기본)/identity(선형 RAL 테스트)."""
        inj = self.encoder.encode(X)                       # 초기 입력(이미지)
        u = None
        for _ in range(self.n_loops):
            _c, u = self.area.forward(geo, inj, window)    # [B, C]
            amp = gain * phi(u / self.u0)                  # 경계 재생(φ 필수)
            inj = build_pulse_injection(amp, self.loop_cells, geo.N,
                                        self.area.n_steps, self.area.P)
        return u

    def forward(self, geo, X, window):
        return jax.vmap(self.head)(self.run_u(geo, X, window))

    def make_window(self, geo):
        return self.area.tmin_window(geo)


def build_ral_classifier(geo, gen_cells, feat_cells, n_classes, n_loops, key, hp, u0=1.0):
    """RAL 분류기 조립. 단일 Area: gen=loop(feat) 단자, out=feat 단자(자기 순환)."""
    from lnn.cluster import _make_area
    from lnn.encodings import ImageEncoder
    ka, kh = jax.random.split(key)
    area = _make_area(geo, ka, "processor", feat_cells, feat_cells, hp)   # gen=out=feat(공유 순환)
    enc = ImageEncoder(gen_cells=gen_cells, P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    head = eqx.nn.Linear(len(feat_cells), n_classes, key=kh)
    return RecursiveAmplitudeLoop(encoder=enc, area=area, head=head,
                                  loop_cells=tuple(int(x) for x in feat_cells),
                                  n_loops=n_loops, u0=u0)


def free_loop_growth(ral, geo, X, window, phi, gain, n=None):
    """루프 진폭 성장비(‖u_last‖/‖u_1‖). 선형(phi=identity)+gain>1 이면 발산, tanh 면 유계."""
    n = n or ral.n_loops
    inj = ral.encoder.encode(X)
    norms = []
    for _ in range(n):
        _c, u = ral.area.forward(geo, inj, window)
        norms.append(float(jnp.linalg.norm(u)))
        amp = gain * phi(u / ral.u0)
        inj = build_pulse_injection(amp, ral.loop_cells, geo.N, ral.area.n_steps, ral.area.P)
    return norms[-1] / (norms[0] + 1e-9), norms
