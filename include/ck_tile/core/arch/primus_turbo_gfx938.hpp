// SPDX-License-Identifier: MIT
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#if defined(PRIMUS_TURBO_GFX938) || defined(__gfx938__)
#define CK_TILE_PRIMUS_TURBO_GFX938_TARGET 1
#else
#define CK_TILE_PRIMUS_TURBO_GFX938_TARGET 0
#endif

#ifndef CK_TILE_ARCH_SUPPORTS_FP4
#define CK_TILE_ARCH_SUPPORTS_FP4 (!CK_TILE_PRIMUS_TURBO_GFX938_TARGET)
#endif

#ifndef CK_TILE_ARCH_SUPPORTS_INT4
#define CK_TILE_ARCH_SUPPORTS_INT4 (!CK_TILE_PRIMUS_TURBO_GFX938_TARGET)
#endif

#ifndef CK_TILE_ARCH_SUPPORTS_MXFP8
#define CK_TILE_ARCH_SUPPORTS_MXFP8 (!CK_TILE_PRIMUS_TURBO_GFX938_TARGET)
#endif

#ifndef CK_TILE_ARCH_FORCE_OCP_FP8
#define CK_TILE_ARCH_FORCE_OCP_FP8 CK_TILE_PRIMUS_TURBO_GFX938_TARGET
#endif

namespace ck_tile::primus_turbo {

constexpr bool is_gfx938_target() { return CK_TILE_PRIMUS_TURBO_GFX938_TARGET != 0; }
constexpr bool arch_supports_fp4() { return CK_TILE_ARCH_SUPPORTS_FP4 != 0; }
constexpr bool arch_supports_int4() { return CK_TILE_ARCH_SUPPORTS_INT4 != 0; }
constexpr bool arch_supports_mxfp8() { return CK_TILE_ARCH_SUPPORTS_MXFP8 != 0; }
constexpr bool arch_uses_ocp_fp8() { return CK_TILE_ARCH_FORCE_OCP_FP8 != 0; }

} // namespace ck_tile::primus_turbo
