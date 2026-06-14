"""§3.1 격자 — 축좌표(axial) 육각 격자.

격자 기하량(셀 좌표, 이웃 테이블, 변 중점·단위방향)은 **학습 불변**이므로 한 번
계산해 정적 상수로 둔다(§3.2 ③). 변(face) 차원은 동역학에서 ``vmap``/벡터화로 처리.

지연은 셀이 아니라 **여섯 변의 속성**이다(유한체적 프레임, §2). 따라서 핵심 산출물은
방향 있는 변(directed edge) `e = i*6 + d` 단위의 테이블이다:

- ``nbr[i, d]``      : 셀 i에서 방향 d의 이웃 셀 인덱스(경계는 -1, 흡수).
- ``e_in[i, d]``     : 셀 i로 방향 d에서 **들어오는** 변의 인덱스(= 이웃 j의 역방향 변 j→i).
- ``edge_mid[e]``    : 변 e의 중점 m = (p_i + p_j)/2.
- ``edge_hat[e]``    : 변 e의 단위방향 ê = (p_j − p_i)/L.
- ``edge_len``       : 변 길이 L(정육각 격자에서 모든 변이 동일).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

# §3.1 방향 6개. 역방향 인덱스는 (d+3) % 6.
DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
SQRT3 = float(np.sqrt(3.0))


class Geometry(NamedTuple):
    R: int
    N: int                      # 셀 수
    E: int                      # 방향 있는 변 수 = N*6
    cells: tuple                # ((q, r), ...) 정적 메타
    pos: jnp.ndarray            # [N, 2] 픽셀좌표
    nbr: jnp.ndarray            # [N, 6] int, 경계 -1
    nbr_exists: jnp.ndarray     # [N, 6] bool
    e_in: jnp.ndarray           # [N, 6] int, 들어오는 변 인덱스 (경계는 0, 마스크로 무효화)
    edge_mid: jnp.ndarray       # [E, 2]
    edge_hat: jnp.ndarray       # [E, 2]
    edge_len: float


def axial_to_pixel(q: float, r: float) -> tuple[float, float]:
    """§3.1: x = √3·(q + r/2),  y = 1.5·r."""
    return SQRT3 * (q + r / 2.0), 1.5 * r


def build_geometry(R: int) -> Geometry:
    """반지름 R의 육각 디스크 격자를 구성한다 (|q|,|r|,|q+r| ≤ R)."""
    cells = [
        (q, r)
        for q in range(-R, R + 1)
        for r in range(-R, R + 1)
        if abs(q) <= R and abs(r) <= R and abs(q + r) <= R
    ]
    index = {c: i for i, c in enumerate(cells)}
    N = len(cells)

    pos = np.zeros((N, 2), dtype=np.float64)
    for i, (q, r) in enumerate(cells):
        pos[i] = axial_to_pixel(q, r)

    nbr = -np.ones((N, 6), dtype=np.int32)
    for i, (q, r) in enumerate(cells):
        for d, (dq, dr) in enumerate(DIRS):
            nc = (q + dq, r + dr)
            if nc in index:
                nbr[i, d] = index[nc]

    nbr_exists = nbr >= 0

    # 들어오는 변: 셀 i에 방향 d에서 들어오는 신호는 이웃 j = nbr[i, d]가
    # 자신의 역방향 변 (d+3)%6 로 내보낸 것 → e_in[i, d] = j*6 + (d+3)%6.
    e_in = np.zeros((N, 6), dtype=np.int32)
    for i in range(N):
        for d in range(6):
            j = nbr[i, d]
            if j >= 0:
                e_in[i, d] = j * 6 + ((d + 3) % 6)
            else:
                e_in[i, d] = 0  # 더미; nbr_exists 마스크로 무효화

    # 변 중점·단위방향. 경계 변은 이웃 좌표 대신 DIR 픽셀 오프셋으로 가상 끝점을 둔다
    # (동역학에서 emit이 0으로 마스킹되므로 값 자체는 무해, 인덱싱만 유효하면 됨).
    dir_pix = np.array([axial_to_pixel(dq, dr) for dq, dr in DIRS], dtype=np.float64)
    edge_len = float(np.linalg.norm(dir_pix[0]))  # 모든 변 동일 길이

    E = N * 6
    edge_mid = np.zeros((E, 2), dtype=np.float64)
    edge_hat = np.zeros((E, 2), dtype=np.float64)
    for i in range(N):
        for d in range(6):
            e = i * 6 + d
            j = nbr[i, d]
            p_j = pos[j] if j >= 0 else pos[i] + dir_pix[d]
            edge_mid[e] = (pos[i] + p_j) / 2.0
            vec = p_j - pos[i]
            edge_hat[e] = vec / (np.linalg.norm(vec) + 1e-12)

    return Geometry(
        R=R,
        N=N,
        E=E,
        cells=tuple(cells),
        pos=jnp.asarray(pos, dtype=jnp.float32),
        nbr=jnp.asarray(nbr, dtype=jnp.int32),
        nbr_exists=jnp.asarray(nbr_exists),
        e_in=jnp.asarray(e_in, dtype=jnp.int32),
        edge_mid=jnp.asarray(edge_mid, dtype=jnp.float32),
        edge_hat=jnp.asarray(edge_hat, dtype=jnp.float32),
        edge_len=edge_len,
    )


def cell_index(geo: Geometry, q: int, r: int) -> int:
    """축좌표 (q, r)의 셀 인덱스. 없으면 -1."""
    try:
        return geo.cells.index((q, r))
    except ValueError:
        return -1
