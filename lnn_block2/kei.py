"""§7 KEI 모델 조립 — ARIS 코어 + Block II 껍질.

입구(FreqEncoder/ImageEncoder) → 동적 라우팅(DynamicRouter) → 다중 Area(MultiAreaCluster,
ARIS dynamics 재사용) → 출구(단일-wavelet u 또는 FreqBankReadout) → decode.

KEIImage: 이미지 분류용(E2/E3) — 인코더+라우터+다중 Area+head. (경로 A 빈칸채우기 E1 은
FreqEncoder+FreqBankReadout 을 e1 에서 직접 배선 — 출력 raw 펄스에 주파수 뱅크를 적용.)
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .multi_area import MultiAreaCluster, build_multi_area
from .routing import DynamicRouter


class KEIImage(eqx.Module):
    encoder: object            # ImageEncoder (Block I)
    router: DynamicRouter
    multi: MultiAreaCluster
    head: object               # eqx.nn.Linear(n_areas·C -> n_classes)

    def forward(self, geo, X, windows):
        inj = self.encoder.encode(X)                          # [B,T,N]
        routed = self.router.route(inj, len(self.multi.areas))  # Area별 변조 주입
        u = self.multi.run_list(geo, routed, windows)         # [B, n_areas·C]
        return jax.vmap(self.head)(u)

    def make_windows(self, geo):
        return self.multi.make_windows(geo)


def build_kei_image(geo, pix_cells, feat_cells, n_areas, mode, coupling, epsilon,
                    n_classes, key, hp):
    """KEIImage 조립. 각 처리 Area: gen=pixel, out=feature(C). head: n_areas·C → classes."""
    km, kh = jax.random.split(key)
    multi = build_multi_area(geo, pix_cells, feat_cells, n_areas, mode, coupling, km, hp)
    C = len(feat_cells)
    from lnn.encodings import ImageEncoder
    enc = ImageEncoder(gen_cells=pix_cells, P=hp["P"], n_steps=hp["n_steps"], n_cells=geo.N)
    head = eqx.nn.Linear(n_areas * C, n_classes, key=kh)
    return KEIImage(encoder=enc, router=DynamicRouter(epsilon=epsilon), multi=multi, head=head)
