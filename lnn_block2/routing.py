"""§4 동적 라우팅 — ε-전개 self-gating.

Block I 은 정적 라우팅(고정 경로). 내용 의존 라우팅 = 입력 펄스가 어느 Area 로 갈지를
내용이 정함(QKV 의 동적 Query). 이 순간 self-gating 으로 내부 선형성이 처음 깨진다.

ε-전개: R(x) = R_0 + ε·R_1(x) + O(ε²). ε=0 이면 정적(Block I 과 동일), ε↑ 로 동적성 증가.
ε 차수까지 경로 분해(해석 가능성) 유지. ε 는 0부터 점진 증가(어닐링) — 한꺼번에 키우면
진동 손실 지형에서 발산(§0.4).

세 기둥 ① 감쇠(§4.2): ε↑ 시 LTI 정확 함수화가 깨지는 양 = ‖동적 − 선형중첩‖ 으로 측정.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp


class DynamicRouter(eqx.Module):
    """내용 의존 게이팅을 ε-전개로 약한 비선형 근사."""

    epsilon: float = eqx.field(static=True)

    def gates(self, inj_BTN, n_areas):
        """Area별 라우팅 게이트 [B, n_areas]. ε=0 → 모두 1(정적). ε>0 → self-gating.

        self-gating: 게이트가 펄스 자신의 에너지에 의존(R_0=1 + ε·(정규화 에너지 편차)).
        """
        B = inj_BTN.shape[0]
        if self.epsilon == 0.0 or n_areas <= 1:
            return jnp.ones((B, n_areas))
        energy = jnp.sum(inj_BTN ** 2, axis=(1, 2))               # [B] 펄스 에너지(self)
        e = energy / (jnp.mean(energy) + 1e-9)                    # 정규화
        # R_1: Area 인덱스별 위상으로 내용 의존 변조(self-gating). ε 차수 비자명.
        phases = jnp.cos(jnp.arange(n_areas)[None, :] * 0.7 + e[:, None])  # [B,n_areas]
        return 1.0 + self.epsilon * (phases - jnp.mean(phases, axis=1, keepdims=True))

    def route(self, inj_BTN, n_areas):
        """게이트로 변조한 Area별 주입 list (각 [B,T,N]). ε=0 이면 모두 동일(정적)."""
        g = self.gates(inj_BTN, n_areas)                          # [B,n_areas]
        return [inj_BTN * g[:, i][:, None, None] for i in range(n_areas)]


def pathway_decomposition_decay(router, inj_BTN, n_areas):
    """세 기둥 ① 감쇠 지표: 라우팅 게이트의 정적(=1)으로부터의 평균 편차.

    ε=0 이면 게이트가 모두 1 → 0(완전 LTI 경로분해 가능). ε↑ 로 게이트가 내용 의존
    분산되며 증가 = 경로 분해(해석 가능성) 손실량. (게이트 합이 보존돼 중첩 차로는 0 이 되므로
    Area별 편차로 측정한다.)
    """
    g = router.gates(inj_BTN, n_areas)            # [B, n_areas]
    return float(jnp.mean(jnp.abs(g - 1.0)))
