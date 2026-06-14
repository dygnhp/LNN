"""§3.1 격자 구성 검증."""

import numpy as np

from lnn.geometry import DIRS, build_geometry, cell_index


def test_cell_count():
    # 반지름 R 육각 디스크의 셀 수 = 1 + 3R(R+1).
    for R in (1, 2, 3, 5):
        geo = build_geometry(R)
        assert geo.N == 1 + 3 * R * (R + 1)
        assert geo.E == geo.N * 6


def test_neighbor_symmetry():
    geo = build_geometry(4)
    nbr = np.asarray(geo.nbr)
    for i in range(geo.N):
        for d in range(6):
            j = nbr[i, d]
            if j >= 0:
                # j 에서 역방향 (d+3)%6 의 이웃은 다시 i.
                assert nbr[j, (d + 3) % 6] == i


def test_incoming_edge_table():
    geo = build_geometry(3)
    nbr = np.asarray(geo.nbr)
    e_in = np.asarray(geo.e_in)
    for i in range(geo.N):
        for d in range(6):
            j = nbr[i, d]
            if j >= 0:
                assert e_in[i, d] == j * 6 + ((d + 3) % 6)


def test_edge_geometry():
    geo = build_geometry(4)
    pos = np.asarray(geo.pos)
    nbr = np.asarray(geo.nbr)
    ehat = np.asarray(geo.edge_hat)
    # 단위방향은 노름 1.
    assert np.allclose(np.linalg.norm(ehat, axis=1), 1.0, atol=1e-4)
    # 모든 변 길이 동일.
    assert geo.edge_len > 0
    # ê 가 실제 i→j 방향과 정렬.
    for i in range(geo.N):
        for d in range(6):
            j = nbr[i, d]
            if j >= 0:
                vec = pos[j] - pos[i]
                vec = vec / np.linalg.norm(vec)
                assert np.allclose(vec, ehat[i * 6 + d], atol=1e-4)


def test_cell_index():
    geo = build_geometry(5)
    assert cell_index(geo, 0, 0) >= 0
    assert cell_index(geo, 100, 100) == -1
    for d, (dq, dr) in enumerate(DIRS):
        assert cell_index(geo, dq, dr) >= 0
