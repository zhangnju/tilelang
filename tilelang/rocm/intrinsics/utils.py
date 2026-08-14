from typing import Literal

from .mfma_layout import (
    thread_id_shared_access_64x4_to_16x16_layout_C_n_m,
    thread_id_shared_access_64x16_to_32x32_layout_C_m_n,
    thread_id_shared_access_64x16_to_32x32_bf16_layout_C_m_n,
)
from .mfma_layout import make_mfma_swizzle_layout  # noqa: F401


def mfma_store_index_map(thread_id, local_id):
    return thread_id_shared_access_64x4_to_16x16_layout_C_n_m(thread_id, local_id)


def mfma_store_index_map_32x32(thread_id, local_id):
    return thread_id_shared_access_64x16_to_32x32_layout_C_m_n(thread_id, local_id)


def mfma_store_index_map_32x32_bf16(thread_id, local_id):
    # v_mfma_f32_32x32x16_bf16 outputs C N-major (lane holds N, reg holds M).
    # TileLang's store-map convention returns (row_on_store_axis, col) in the
    # same (n, m) order the 16x16 path uses (..._C_n_m); our layout fn returns
    # (m, n), so swap to (n, m) to place each register at logical C[m, n].
    m, n = thread_id_shared_access_64x16_to_32x32_bf16_layout_C_m_n(thread_id, local_id)
    return n, m


def get_mma_micro_size(dtype: Literal["float16", "int8"]):
    micro_size_x = micro_size_y = 16
    micro_size_k = 16
    if dtype in {"float8_e4m3", "float8_e5m2", "int8"}:
        micro_size_k = 32
    return micro_size_x, micro_size_y, micro_size_k
