"""Phase 5 §3 — 방식 A: 비선형 깊이(경계 φ 통과 횟수) = processor Area 수 L 확장.

다섯 축(용량·구조·통합·차원·공간)은 모두 선형 매질 안의 다이얼이었다. LNN 은 Area 내부가
선형(LTI)이고 비선형은 경계 φ=tanh 뿐 → **L=비선형 깊이**가 MLP 의 '층 수'에 대응. L 을
키워(φ 더 자주 통과) ~0.56 이 움직이는지(H-DEPTH).

ARIS·KEI 불변: ``build_serial_cluster`` 의 n_proc=L 노출(encoder→processor×L→decoder, φ가 L+1회).
깊이의 대가(세 기둥① 경로 분해 손실)를 ``path_decomp_residual`` 로 함께 계측(능력-해석 trade-off).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from lnn.cluster import build_pulse_injection, build_serial_cluster


def build_depth_classifier(geo, gen_cells, feat_cells, n_classes, L, key, hp):
    """방식 A: (encoder, cluster(n_proc=L), head). 경계 φ 가 L+1 회 — 비선형 깊이 L."""
    from lnn.encodings import ImageEncoder
    k1, k2 = jax.random.split(key)
    hp = {**hp, "n_proc": L}
    cluster = build_serial_cluster(geo, gen_cells, feat_cells, k1, hp)
    enc = ImageEncoder(gen_cells=gen_cells, P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    head = eqx.nn.Linear(len(feat_cells), n_classes, key=k2)
    return enc, cluster, head


def depth_run_u(cluster, geo, inj_BTN, windows, phi):
    """Cluster.forward 를 φ 교체 가능하게 복제 — 최종 u 반환(경로분해 잔차 계측용).

    phi=jnp.tanh 이면 실제 모델과 동일, phi=identity 이면 완전 선형(경로-합) 근사.
    """
    _c, u = cluster.areas[0].forward(geo, inj_BTN, windows[0])
    for k in range(1, len(cluster.areas)):
        amp = cluster.diag_gains[k - 1] * phi(u / cluster.u0)
        inj = build_pulse_injection(amp, cluster.areas[k].gen_cells, geo.N,
                                    cluster.n_steps, cluster.P)
        _c, u = cluster.areas[k].forward(geo, inj, windows[k])
    return u


def path_decomp_residual(cluster, geo, inj_BTN, windows):
    """세 기둥① 손실 = 전체(φ=tanh) 출력이 선형 경로-합(φ=identity)에서 벗어난 상대량.

    L=1(얕음)이면 ≈0(거의 선형, 분해 가능), L↑·신호 비선형 engage↑로 증가 = 깊이가
    해석 가능성을 깎는 양. 상대 잔차 = ‖u_tanh − u_linear‖ / ‖u_tanh‖ (배치 평균).
    """
    u_nl = depth_run_u(cluster, geo, inj_BTN, windows, jnp.tanh)
    u_lin = depth_run_u(cluster, geo, inj_BTN, windows, lambda z: z)
    num = jnp.linalg.norm(u_nl - u_lin, axis=-1)
    den = jnp.linalg.norm(u_nl, axis=-1) + 1e-9
    return float(jnp.mean(num / den))
