"""§6 그래디언트 검증 — fractional-delay 사슬의 유한차분 대조.

합격 기준: 해석적 그래디언트(``jax.grad``)와 중심 유한차분의 **상대오차 < 1e-3**.

테스트 대상은 사양이 지목한 **fractional-delay 사슬**:
    θ → 경사 s·θ → Tobler f → clamp_τ → delay_idx=τ−1 → (floor, frac) → 선형보간 deposit → loss.
floor 인덱스는 미분 불가(계단)지만 frac(=w0,w1) 로만 그래디언트가 흐른다(§3.5). FD 스텝이
정수 경계를 넘지 않도록 frac 을 ~0.5 부근에 두고 작은 eps 를 쓴다.
"""

import jax

jax.config.update("jax_enable_x64", True)  # FD 대조는 float64 (float32 반올림이 1e-3 임계를 넘음)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from lnn import delay  # noqa: E402


def _deposit_loss(theta, s_e, target, dp, L):
    """fractional-delay 사슬 끝의 스칼라 손실."""
    tau = delay.edge_delay_from_slope(s_e * theta, dp, law="tobler", clamp=True)
    delay_idx = tau - 1.0
    i0 = jnp.floor(delay_idx).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, L - 2)
    frac = delay_idx - i0
    i1 = i0 + 1
    E = s_e.shape[0]
    buf = jnp.zeros((E, L))
    ar = jnp.arange(E)
    buf = buf.at[ar, i0].add(1.0 - frac)   # 단위 펄스의 선형보간 deposit
    buf = buf.at[ar, i1].add(frac)
    return jnp.sum(target * buf)


def test_fractional_delay_grad_fd():
    rng = np.random.default_rng(1)
    E = 12
    L = 7
    # frac 이 0/1 경계에서 멀도록 경사를 조정(아래서 검증).
    s_e = jnp.asarray(rng.uniform(-0.5, 0.5, E))
    target = jnp.asarray(rng.uniform(-1, 1, (E, L)))
    dp = dict(tau_base=2.4, gamma=0.8, s_star=-0.2, tau_min=1.0, tau_max=5.0)
    theta0 = 1.0

    # frac 이 경계에서 충분히 떨어졌는지 확인(테스트 안정성).
    tau = delay.edge_delay_from_slope(s_e * theta0, dp, law="tobler", clamp=True)
    frac = (tau - 1.0) - jnp.floor(tau - 1.0)
    assert float(jnp.min(jnp.minimum(frac, 1 - frac))) > 0.05

    grad_fn = jax.grad(_deposit_loss)
    g = float(grad_fn(theta0, s_e, target, dp, L))

    eps = 1e-4
    fp = float(_deposit_loss(theta0 + eps, s_e, target, dp, L))
    fm = float(_deposit_loss(theta0 - eps, s_e, target, dp, L))
    fd = (fp - fm) / (2 * eps)

    rel = abs(g - fd) / (abs(fd) + 1e-8)
    assert rel < 1e-3, f"relative error {rel:.2e} (analytic={g:.6f}, fd={fd:.6f})"


def test_clamp_tau_grad_fd():
    """clamp_τ(softplus 하한) 단독의 유한차분 대조."""
    def f(u):
        return jnp.sum(delay.clamp_tau(u, tau_min=1.0, tau_max=5.0))

    u = jnp.asarray([1.5, 2.0, 0.5, 3.2])
    g = jax.grad(f)(u)
    eps = 1e-4
    fd = np.zeros(4)
    for i in range(4):
        up = u.at[i].add(eps)
        um = u.at[i].add(-eps)
        fd[i] = (float(f(up)) - float(f(um))) / (2 * eps)
    rel = np.abs(np.asarray(g) - fd) / (np.abs(fd) + 1e-8)
    assert np.max(rel) < 1e-3
