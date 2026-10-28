// SPDX-License-Identifier: MIT
#include "grouped_int8_nt_tile.hpp"
#include "grouped_int8_nt_direct.hpp"
#include <hip/hip_runtime.h>
namespace {
template <class P>
int launch(const void* args, int groups, int m, int n, hipStream_t stream)
{
    dim3 grid(P::MFirst ? (m + P::BM - 1) / P::BM : (n + P::BN - 1) / P::BN,
                    P::MFirst ? (n + P::BN - 1) / P::BN : (m + P::BM - 1) / P::BM,
                    groups);
    if constexpr(P::MGroup>0){
        const int mt=(m+P::BM-1)/P::BM,nt=(n+P::BN-1)/P::BN;
        grid=dim3(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,groups);
    }
    ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                           ck_tile::make_kernel<P::Threads, P::MinBlocks>(
                               ck_tile::GroupedInt8NtTileKernel<P>{},
                               grid,
                               dim3(P::Threads),
                               0,
                               static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError() == hipSuccess ? 0 : -100;
}
} // namespace
extern "C" int
ck_tile_grouped_int8_nt_launch(unsigned id, const void* args, int g, int m, int n, void* stream)
{
    if(!args || g <= 0 || g > 65535 || m <= 0 || n <= 0)
        return -1;
    auto s = reinterpret_cast<hipStream_t>(stream);
    switch(id)
    {
    case 6004: {
        dim3 grid((n + 63) / 64, (m + 15) / 16, g);
        ck_tile::launch_kernel(
            ck_tile::stream_config{s, false},
            ck_tile::make_kernel<256, 1>(ck_tile::GroupedInt8NtDirectKernel<>{},
                                         grid,
                                         dim3(256),
                                         0,
                                         static_cast<const ck_tile::GemmTransKernelArg*>(args)));
        return hipGetLastError() == hipSuccess ? 0 : -100;
    }
    case 6005: return launch<ck_tile::GroupedInt8NtTileProblem<32, 128>>(args, g, m, n, s);
    case 6006: return launch<ck_tile::GroupedInt8NtTileProblem<64, 128>>(args, g, m, n, s);
    case 6007: return launch<ck_tile::GroupedInt8NtTileProblem<128, 128>>(args, g, m, n, s);
    case 6008:
        return launch<ck_tile::GroupedInt8NtTileProblem<128, 256, 64, 2, 4>>(args, g, m, n, s);
    case 6035:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, true, true, true, 1, false>>(
            args, g, m, n, s);
    case 6041:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, true, true, false, 1, true>>(
            args, g, m, n, s);
    case 6042:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<128, 256, 64, 2, 4, true, true, true, false, 2, true>>(
            args, g, m, n, s);
    case 6043:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, false, true, false, 1, true>>(
            args, g, m, n, s);
    case 6044:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, false, true, false, 1, true>>(
            args, g, m, n, s);
    case 6045:return launch<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,2,true,true,true,false,1,true,8>>(args,g,m,n,s);
    case 6046:return launch<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,2,true,true,true,false,1,true,4>>(args,g,m,n,s);
    case 6047:return launch<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,2,true,true,true,false,1,true,16>>(args,g,m,n,s);
    case 6048:return launch<ck_tile::GroupedInt8NtTileProblem<256,256,64,4,2,true,true,true,false,1,true,1>>(args,g,m,n,s);
    case 6049:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,64,2,2,true,true,true,true,1,false,8>>(args,g,m,n,s);
    case 6050:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,64,2,2,true,false,true,false,1,true,8>>(args,g,m,n,s);
    default: return -5;
    }
}
