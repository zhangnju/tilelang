#pragma once

#include "common.h"

using f32 = float;
// using f16 = _Float16;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

using index_t = u32;

using ck_tile::int32x4_t;

struct __attribute__((packed)) buffer_resource {
  const void *ptr;
  uint32_t range;
  uint32_t config;
};

CK_TILE_DEVICE int32x4_t make_wave_buffer_resource(const void *ptr,
                                                   uint32_t size = 0xffffffff) {
  buffer_resource res{ptr, size, CK_TILE_BUFFER_RESOURCE_3RD_DWORD};
  int32x4_t r = __builtin_bit_cast(int32x4_t, res);
  r.x = __builtin_amdgcn_readfirstlane(r.x);
  r.y = __builtin_amdgcn_readfirstlane(r.y);
  r.z = __builtin_amdgcn_readfirstlane(r.z);
  r.w = __builtin_amdgcn_readfirstlane(r.w);
  return r;
}

__device__ void init_m0(uint32_t m0_value) {
  asm volatile("s_mov_b32 m0, %0" : : "s"(m0_value) : "memory");
}

__device__ void inc_m0(uint32_t m0_inc) {
  asm volatile("s_add_u32 m0, %0, m0" : : "n"(m0_inc) : "memory");
}

namespace tl {

// AMDGPU automatically commit memory fence
TL_DEVICE void cp_async_commit() {}

// Global Memory only fence
__device__ void async_gld_fence(index_t cnt) {
  asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

// Global Memory and Shared Memory fence
__device__ void async_gld_sld_fence(index_t cnt) {
  asm volatile("s_waitcnt lgkmcnt(%0)" : : "n"(cnt) : "memory");
}

__device__ void wave_barrier() { asm volatile("s_barrier" : : : "memory"); }

template <int N = 0> TL_DEVICE void cp_async_wait() {
  async_gld_fence(N);
  // or
  // async_gld_sld_fence(N);
}

template <bool pre_nop = false>
CK_TILE_DEVICE void async_buffer_load_dword_v(void *smem, int32x4_t rsrc,
                                              index_t voffset) {
  auto const lds_ptr_sgpr =
      __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(smem)));
  asm volatile("s_mov_b32 m0, %0; \n\t"
               "buffer_load_dword %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
               "v"(voffset), "s"(rsrc)
               : "memory");
}

// -----------------------------------------------------------------------------
// Async global->VGPR load (Triton-style, for latency-bound serial loops).
//
// Unlike the `buffer_load ... lds` path above (which lands directly in LDS and
// whose completion is a shared-memory hazard forcing an early __syncthreads),
// these load a 128-bit tile into a REGISTER vector via `buffer_load_dwordx4`
// WITHOUT `lds`. LLVM leaves the `s_waitcnt vmcnt` where the value is first
// used, so an early-issued load overlaps with the intervening compute and the
// caller stores the register to shared only when ready (deferred ds_write).
// This is exactly how Triton hides the per-step w/k loads on the scan chain:
// the load latency is covered by the two loop-carried MFMAs instead of stalling
// on a shared-memory barrier. Microbench (gfx950, 128-step serial chain):
// buffer_load->VGPR->deferred-shared is ~1.19x faster than sync load->shared.
//
// `buffer_load_to_reg_x4` returns a 16-byte register tile loaded from
// (wave_base + voffset[lane]). The value is only realized (s_waitcnt vmcnt) at
// first use, so issue it EARLY (prefetch next iter) and consume/store it LATE;
// the intervening MFMAs cover the load latency. `wave_base` must be wave-uniform
// (readfirstlane of lane-0's pointer) and `voffset` the per-lane byte delta, so
// arbitrary per-lane strides work (same addressing as cp_async_gs_linear).
// Verified on gfx950: the emitted buffer_load_dwordx4 (no lds) has its waitcnt
// deferred past the loop's compute, hiding the load on the serial scan chain.
CK_TILE_DEVICE int32x4_t buffer_load_to_reg_x4(const void *wave_base,
                                               index_t voffset) {
  auto rsrc = __builtin_amdgcn_make_buffer_rsrc(
      const_cast<void *>(wave_base), /*stride=*/0,
      /*num_records=*/0xffffffffu, /*flags=*/CK_TILE_BUFFER_RESOURCE_3RD_DWORD);
  return __builtin_amdgcn_raw_buffer_load_b128(rsrc, voffset, 0, 0);
}

// Store a previously-loaded register tile into LDS (the deferred ds_write half).
CK_TILE_DEVICE void store_reg_x4_to_lds(void *lds_ptr, int32x4_t val) {
  *reinterpret_cast<int32x4_t *>(lds_ptr) = val;
}

// gfx950 (CDNA4 / MI350): 128-bit direct-to-LDS async load.
// buffer_load_dwordx4 ... lds bypasses VGPRs entirely, giving 4x the
// bandwidth of the 32-bit path and overlapping with MFMA computation.
#if defined(__gfx950__)
CK_TILE_DEVICE void async_buffer_load_dwordx4_v(void *smem, int32x4_t rsrc,
                                                index_t voffset) {
  auto const lds_ptr_sgpr =
      __builtin_amdgcn_readfirstlane((reinterpret_cast<uintptr_t>(smem)));
  asm volatile(
      "s_mov_b32 m0, %0; \n\t"
      "buffer_load_dwordx4 %1, %2, 0 offen lds;\n\t" ::"s"(lds_ptr_sgpr),
      "v"(voffset), "s"(rsrc)
      : "memory");
}
#endif // __gfx950__

template <int N>
TL_DEVICE void cp_async_gs(void *lds_base_ptr, void const *global_base_ptr) {
  if constexpr (N == 16) {
    // NOTE: the gfx950 buffer_load_dwordx4 ... lds path requires the per-thread
    // LDS destination address to be exactly base + threadIdx.x * 16 bytes
    // (i.e. no swizzling).  The pipeline-planning pass generates swizzled LDS
    // layouts that violate this constraint, so we always use the synchronous
    // uint4 store here.  The T.async_copy path (called with a known-linear LDS
    // layout) uses cp_async_gs_linear below for the fast direct-to-LDS path.
    *(uint4 *)lds_base_ptr = *(const uint4 *)global_base_ptr;
  } else if constexpr (N == 8) {
    *(uint2 *)lds_base_ptr = *(const uint2 *)global_base_ptr;
  } else if constexpr (N == 4) {
    async_buffer_load_dword_v(
        lds_base_ptr,
        make_wave_buffer_resource(((const int32_t *)global_base_ptr) -
                                  threadIdx.x),
        threadIdx.x * N /*assume 4 bytes*/);
  }
}

// Direct global->LDS variant: bypasses VGPRs via buffer_load_dwordx4 ... lds
// (gfx950).  REQUIRES a linear, un-swizzled LDS destination (lane i writes the
// i-th 16B slot) — only valid on the T.async_copy(linear_lds) path.  Falls back
// to the synchronous store on non-gfx950 or unsupported N.
//
// The `buffer_load ... offen lds` instruction reads, per lane, from
// (wave_scalar_base + voffset[lane]) in global and appends the results into LDS
// in ascending lane order.  So `voffset` may encode ANY per-lane global offset
// (including a non-unit source stride from a sliced tensor), while the LDS side
// stays contiguous.  Earlier code hardcoded voffset = threadIdx.x*N and derived
// the base as (global_base_ptr - threadIdx.x*N), which is only correct when the
// source is contiguous (adjacent lanes N bytes apart).  For a strided source
// (e.g. U0[b, t0:t0+BT, h, v0:v0+bV]) that assumption breaks and the wave base
// (taken via readfirstlane) is garbage.  Instead pick lane-0's pointer as the
// wave base and compute the true per-lane byte offset from it.
// Broadcast a 64-bit pointer from lane 0 to all lanes. readfirstlane operates on
// 32-bit values, so a 64-bit pointer must be broadcast in two halves — doing a
// single 32-bit readfirstlane on a pointer truncates the high word and yields a
// wild address.
CK_TILE_DEVICE const char *wave_readfirstlane_ptr(const char *p) {
  uint64_t u = reinterpret_cast<uint64_t>(p);
  uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u));
  uint32_t hi =
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(u >> 32));
  return reinterpret_cast<const char *>((static_cast<uint64_t>(hi) << 32) | lo);
}

// Direct global->LDS variant: bypasses VGPRs via buffer_load_dwordx4 ... lds
// (gfx950).  REQUIRES a linear, un-swizzled LDS destination (lane i writes the
// i-th 16B slot) — only valid on the T.async_copy(linear_lds) path.  Falls back
// to the synchronous store on non-gfx950 or unsupported N.
//
// The `buffer_load ... offen lds` instruction reads, per lane, from
// (wave_scalar_base + voffset[lane]) in global and appends the results into LDS
// in ascending lane order.  So `voffset` may encode ANY per-lane global offset
// (including a non-unit source stride from a sliced tensor), while the LDS side
// stays contiguous.  Earlier code hardcoded voffset = threadIdx.x*N and derived
// the base as (global_base_ptr - threadIdx.x*N), which is only correct when the
// source is contiguous (adjacent lanes N bytes apart).  For a strided source
// (e.g. U0[b, t0:t0+BT, h, v0:v0+bV]) that assumption breaks.  Instead take
// lane-0's pointer as the wave base and compute the true per-lane byte offset.
template <int N>
TL_DEVICE void cp_async_gs_linear(void *lds_base_ptr,
                                  void const *global_base_ptr) {
#if defined(__gfx950__)
  if constexpr (N == 16 || N == 4) {
    const char *lane_ptr = reinterpret_cast<const char *>(global_base_ptr);
    // Wave-uniform base = lane 0's global pointer (correct 64-bit broadcast).
    const char *wave_base = wave_readfirstlane_ptr(lane_ptr);
    // True per-lane byte offset from the wave base (handles arbitrary stride;
    // reduces to threadIdx.x*N for a contiguous source).
    index_t voffset = static_cast<index_t>(lane_ptr - wave_base);
    if constexpr (N == 16) {
      async_buffer_load_dwordx4_v(lds_base_ptr,
                                  make_wave_buffer_resource(wave_base), voffset);
    } else {
      async_buffer_load_dword_v(lds_base_ptr,
                                make_wave_buffer_resource(wave_base), voffset);
    }
  } else {
    cp_async_gs<N>(lds_base_ptr, global_base_ptr);
  }
#else
  cp_async_gs<N>(lds_base_ptr, global_base_ptr);
#endif
}

// Swizzled-destination direct global->LDS async copy (gfx950).
//
// NOT CURRENTLY ENABLED — kept as scaffolding. The generator that would select
// this (rocm/op/copy.cc DstLdsLayoutIsSwizzled) is hardcoded off because
// TileLang's 2D GEMM XOR swizzle scatters one wavefront's chunks across a
// NON-CONTIGUOUS LDS span (measured span 544 for a 64-lane wave), while
// `buffer_load ... lds` can only write a contiguous per-wave block. Source-side
// shuffling (below) cannot fix a non-contiguous destination. This code path is
// only correct for a swizzle whose per-wave chunk set is contiguous.
//
// The hardware `buffer_load ... lds` writes LDS CONTIGUOUSLY in ascending lane
// order (lane L -> wave_base + L*N bytes, m0 base is readfirstlane'd — per-lane
// LDS pointers are IGNORED).  So the swizzle must be moved entirely to the
// SOURCE side: lane L must issue the global offset of the element whose swizzled
// destination is contiguous slot L.  Mirrors Triton's emitSwizzledLaneOffsets +
// ds_permute.
//
// `chunk` = this lane's swizzled LDS chunk index (= dst_swizzled_offset / N-elems
// = which contiguous slot this lane's own element belongs in), supplied by
// codegen from the store layout.  We SCATTER each lane's own global voffset to
// its swizzled chunk with ds_permute (out[chunk[L]] = my_voffset[L]); the result
// is exactly the voffset that contiguous slot L must load.  ds_permute address
// is byte-based (chunk << 2).
//
// lds_wave_base = readfirstlane(lds_base_ptr) is the wave's slot-0 LDS address;
// HW adds lane*N to it for the contiguous write.
template <int N>
TL_DEVICE void cp_async_gs_swizzled(void *lds_base_ptr,
                                    void const *global_base_ptr, int chunk) {
#if defined(__gfx950__)
  if constexpr (N == 16 || N == 4) {
    const char *lane_ptr = reinterpret_cast<const char *>(global_base_ptr);
    const char *wave_base = wave_readfirstlane_ptr(lane_ptr);
    index_t my_voffset = static_cast<index_t>(lane_ptr - wave_base);
    // Scatter my_voffset to my swizzled chunk: out[chunk] = my_voffset.
    index_t voffset = __builtin_amdgcn_ds_permute(chunk << 2, my_voffset);
    const char *lds_ptr = reinterpret_cast<const char *>(lds_base_ptr);
    void *lds_write_base =
        const_cast<char *>(wave_readfirstlane_ptr(lds_ptr));
    if constexpr (N == 16) {
      async_buffer_load_dwordx4_v(lds_write_base,
                                  make_wave_buffer_resource(wave_base), voffset);
    } else {
      async_buffer_load_dword_v(lds_write_base,
                                make_wave_buffer_resource(wave_base), voffset);
    }
  } else {
    cp_async_gs<N>(lds_base_ptr, global_base_ptr);
  }
#else
  cp_async_gs<N>(lds_base_ptr, global_base_ptr);
#endif
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void *lds_base_ptr,
                                       void const *global_base_ptr, bool cond) {
  if constexpr (N == 16) {
    // NOTE: same reasoning as cp_async_gs above — the pipeline-planning pass
    // generates swizzled LDS layouts that are incompatible with the gfx950
    // buffer_load_dwordx4 ... lds path (which requires a linear, un-swizzled
    // destination).  Use the synchronous uint4 path here as well so that
    // cp_async_wait (which issues s_waitcnt vmcnt) correctly synchronises the
    // data before the consumer MFMA reads it.
    *(uint4 *)lds_base_ptr =
        cond ? *(const uint4 *)global_base_ptr : make_uint4(0, 0, 0, 0);
  } else if constexpr (N == 8) {
    *(uint2 *)lds_base_ptr =
        cond ? *(const uint2 *)global_base_ptr : make_uint2(0, 0);
  } else {
    if (cond) {
      async_buffer_load_dword_v(
          lds_base_ptr,
          make_wave_buffer_resource(((const int32_t *)global_base_ptr) -
                                    threadIdx.x),
          threadIdx.x * N /*assume 4 bytes*/);
    } else {
      *(u32 *)lds_base_ptr = 0;
    }
  }
}

} // namespace tl
