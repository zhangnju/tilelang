/*!
 * \file tl/rocm/op/copy.cc
 * \brief ROCm implementation for tl.copy lowering.
 */

#include "op/copy.h"
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>

#include "op/builtin.h"
#include "op/utils.h"
#include "rocm/target_utils.h"
#include "rocm/transform/async_copy_injector.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace rocm {

namespace {

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

} // namespace

enum class CopyInst : uint8_t {
  kNormal = 0,
  kCPAsync = 1,
};

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    SelectInst(op, layout_args.target, layout_args.layout_map,
               layout_args.analyzer);
    return op.InferSIMTLayout(layout_args, level);
  }

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer) {
    if (GetIsAsyncCopy(op) || GetNoImplicitAsyncCommitWait(op)) {
      bool cp_async_supported =
          CheckCPAsyncCopy(op, target, layout_map, analyzer);
      ICHECK(cp_async_supported)
          << "Explicit async copy semantics require ROCm async copy lowering, "
             "but constraints were not satisfied. Got src="
          << op.src->name << " (scope=" << op.src.scope()
          << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
          << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype
          << ").";
      return CopyInst::kCPAsync;
    }
    return CopyInst::kNormal;
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    auto copy_inst =
        SelectInst(op, lower_args.target, lower_args.layout_map, analyzer);
    if (copy_inst == CopyInst::kCPAsync) {
      return LowerCPAsync(op, lower_args, analyzer);
    }
    if (copy_inst == CopyInst::kNormal) {
      return LowerNormalCopy(op, lower_args, analyzer);
    }
    LOG(FATAL) << "Unsupported ROCm copy inst " << static_cast<int>(copy_inst);
  }

private:
  static Stmt LowerCPAsync(const CopyNode &op, const LowerArgs &lower_args,
                           arith::Analyzer *analyzer) {
    using namespace tvm::transform;

    PassContext pass_ctx = PassContext::Current();
    bool enable_async_copy =
        pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
    bool no_implicit_commit_wait = GetNoImplicitAsyncCommitWait(op);
    bool explicit_async_semantics =
        no_implicit_commit_wait || GetIsAsyncCopy(op);
    if (!enable_async_copy && !explicit_async_semantics) {
      return LowerNormalCopy(op, lower_args, analyzer);
    }

    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);

    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           lower_args.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    Stmt lowered_loop = LowerParallelLoop(
        par_op->GetRoot(), loop_layout, lower_args.thread_index, analyzer,
        lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
        /*parallel_loop=*/true,
        /*should_vectorize=*/true, par_op->LoopLayoutRequiresPaddingGuard());

    // linear_lds is enabled either explicitly (T.async_copy(linear_lds=True)) or
    // automatically when the auto-pipeliner produced a copy whose dst LDS layout
    // is already linear (non-swizzled). The latter lets the normal T.copy +
    // T.Pipelined path take the direct global->LDS async route without a manual
    // annotation.
    bool linear_lds = GetBoolAnnotation(op, "linear_lds") ||
                      DstLdsLayoutIsLinear(op, lower_args.layout_map);
    bool swizzled_lds = DstLdsLayoutIsSwizzled(op, lower_args.layout_map);
    auto inject_result =
        InjectROCmAsyncCopy(lowered_loop, /*async_without_async_commit_wait=*/
                            no_implicit_commit_wait || GetIsAsyncCopy(op),
                            /*linear_lds=*/linear_lds,
                            /*swizzled_lds=*/swizzled_lds);
    Stmt async_copy_loop = inject_result.stmt;
    if (!inject_result.injected_rocm_async_copy) {
      DLOG(WARNING) << "ROCm async-copy rewrite miss for copy src="
                    << op.src->name << " (scope=" << op.src.scope()
                    << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
                    << " (scope=" << op.dst.scope()
                    << ", dtype=" << op.dst->dtype
                    << "), no_implicit_async_commit_wait="
                    << no_implicit_commit_wait
                    << ", is_async_copy=" << GetIsAsyncCopy(op);
      if (no_implicit_commit_wait) {
        DLOG(WARNING)
            << "Pipeline-managed async copy fallback to normal copy because "
               "ROCm async-copy rewrite found no eligible global->shared "
               "store.";
        return lowered_loop;
      }
      if (explicit_async_semantics) {
        LOG(FATAL) << "Explicit async copy semantics require ROCm async-copy "
                      "lowering, "
                      "but no eligible global->shared store was rewritten.";
      }
      DLOG(WARNING)
          << "Fallback to normal copy because ROCm async-copy rewrite "
             "found no eligible global->shared store.";
      return LowerNormalCopy(op, lower_args, analyzer);
    }
    if (no_implicit_commit_wait) {
      return async_copy_loop;
    }
    if (GetIsAsyncCopy(op)) {
      Stmt commit_group =
          Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
      return SeqStmt({async_copy_loop, commit_group});
    }
    return async_copy_loop;
  }

  static bool CheckCPAsyncCopyPreconditions(const CopyNode &op) {
    if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
      return false;
    }
    if (op.src->dtype != op.dst->dtype) {
      return false;
    }
    return true;
  }

  // Decide whether the direct global->LDS (buffer_load ... lds) async path is
  // legal for this copy without an explicit T.async_copy(linear_lds=True).  It
  // requires the destination shared buffer to have a NON-swizzled (linear)
  // layout, because the wave `buffer_load ... lds` writes LDS contiguously per
  // lane.  We only opt in when the dst buffer actually has a layout entry and
  // DetectSwizzleMode reports None (linear); otherwise stay on the safe
  // synchronous path.
  static bool DstLdsLayoutIsLinear(const CopyNode &op,
                                   const LayoutMap &layout_map) {
    if (!IsSharedBuffer(op.dst)) {
      return false;
    }
    auto it = layout_map.find(op.dst);
    if (it == layout_map.end()) {
      return false;
    }
    Layout dst_layout = (*it).second.as<Layout>().value_or(Layout());
    if (!dst_layout.defined()) {
      return false;
    }
    return DetectSwizzleMode(dst_layout, op.dst) == SwizzleMode::None();
  }

  // Whether the dst shared buffer has a (bank-conflict-avoiding) XOR swizzle
  // layout that we could realize on the direct global->LDS async path via a
  // source-side reindex (cp_async_gs_swizzled).
  //
  // DISABLED (returns false): the gfx950 `buffer_load ... lds` instruction can
  // only write a CONTIGUOUS per-wave LDS block, but TileLang's 2D GEMM XOR
  // swizzle scatters one wavefront's 64 chunks across a non-contiguous span
  // (measured span = 544 chunks for a 64-lane wave due to the (tx&15)>>3 * 4096
  // term). No amount of source-side ds_permute/ds_bpermute can fix that — the
  // destination block itself is non-contiguous, which the direct-to-LDS HW does
  // not support (Triton requires a contiguous per-wave shared block; TileLang's
  // swizzle does not provide one). The scaffolding below (swizzled_lds
  // annotation, cp_async_gs_swizzled template) is kept for the record but not
  // enabled. Realizing swizzled async would require a swizzle layout whose
  // per-wave chunk set is contiguous.
  static bool DstLdsLayoutIsSwizzled(const CopyNode & /*op*/,
                                     const LayoutMap & /*layout_map*/) {
    return false;
  }

  static bool CheckCPAsyncCopy(const CopyNode &op, Target target,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) {
    if (!TargetRocmHasAsyncCopy(target)) {
      return false;
    }
    return CheckCPAsyncCopyPreconditions(op);
  }
};

} // namespace rocm

namespace {

bool MatchROCmCopyTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmCopy() {
  RegisterCopyImpl(CopyImpl{
      "rocm.Copy",
      MatchROCmCopyTarget,
      100,
      rocm::Copy::InferLayout,
      rocm::Copy::Lower,
  });
  return true;
}

const bool rocm_copy_registered = RegisterROCmCopy();

} // namespace

} // namespace tl
} // namespace tvm
