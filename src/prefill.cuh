// Batched prefill (tensor-core GEMM path). Stub: filled in after the decode path is validated.
#pragma once
#include "engine.h"
namespace le {
struct PrefillBufs { int dummy = 0; };
template <class Alloc> inline void prefill_alloc(PrefillBufs&, const Config&, int, Alloc) {}
}  // namespace le
