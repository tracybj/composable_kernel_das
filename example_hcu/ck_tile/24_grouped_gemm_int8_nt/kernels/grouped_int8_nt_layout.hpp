// SPDX-License-Identifier: MIT
#pragma once

#include "ck_tile/core.hpp"

namespace ck_tile {

// INT8 expert weights are stored as [K/64, N/16, 4, 16, 16].
// One wave packs one N16/K64 tile, with one contiguous 16-byte vector per lane.
struct Int8NtPackedWeightLayout
{
    static constexpr index_t NTile       = 16;
    static constexpr index_t KTile       = 64;
    static constexpr index_t VectorBytes = 16;
    static constexpr index_t WaveSize    = 64;
    static constexpr index_t KGroups     = KTile / VectorBytes;
    static constexpr index_t TileBytes   = NTile * KTile;

    template <class Index>
    CK_TILE_HOST_DEVICE static constexpr Index
    vector_offset(Index k_tile, Index n_tiles, Index n_tile, Index lane)
    {
        return ((k_tile * n_tiles + n_tile) * WaveSize + lane) * VectorBytes;
    }
};

} // namespace ck_tile
