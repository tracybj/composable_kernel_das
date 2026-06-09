// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, , Inc. All rights reserved.
//
// Hygon DCU: data_cache_prefetch is NOT available.
// Upstream uses __builtin_amdgcn_global_prefetch (gfx1250 only).
// DCU architectures (gfx928/gfx936/gfx938) have no global prefetch builtin.

#pragma once

namespace ck {

// global_prefetch: not supported on DCU.
// Requires gfx1250+ global cache prefetch hardware.

} // namespace ck
