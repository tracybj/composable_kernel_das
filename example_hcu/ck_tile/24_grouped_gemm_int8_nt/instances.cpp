// SPDX-License-Identifier: MIT
#include "grouped_int8_nt_tile.hpp"
#include "grouped_int8_nt_wasp_kernel.hpp"
#include "grouped_int8_nt_direct.hpp"
#include "grouped_int8_nt_direct_epilogue.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include <hip/hip_runtime.h>
namespace {
template <class P, class Epilogue = ck_tile::GroupedInt8NtTileEpilogue<P>>
int launch(const void* args, int groups, int m, int n, hipStream_t stream)
{
    dim3 grid(P::MFirst ? (m + P::BM - 1) / P::BM : (n + P::BN - 1) / P::BN,
              P::MFirst ? (n + P::BN - 1) / P::BN : (m + P::BM - 1) / P::BM,
              groups);
    if constexpr(P::MGroup > 0)
    {
        const int mt = (m + P::BM - 1) / P::BM, nt = (n + P::BN - 1) / P::BN;
        grid = dim3(((mt + P::MGroup - 1) / P::MGroup) * P::MGroup * nt, 1, groups);
    }
    ck_tile::launch_kernel(ck_tile::stream_config{stream, false},
                           ck_tile::make_kernel<P::Threads, P::MinBlocks>(
                               ck_tile::GroupedInt8NtTileKernel<P, Epilogue>{},
                               grid,
                               dim3(P::Threads),
                               0,
                               static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError() == hipSuccess ? 0 : -100;
}
template<int Batch,bool Transpose>
int launch_wasp(const void* args,int groups,int m,int n,hipStream_t stream)
{
    using P=ck_tile::GroupedInt8NtWaspProblem<Transpose,8,4,Batch,true,true>;
    const int mm=Transpose?n:m,nn=Transpose?m:n;
    const int mt=(mm+255)/256,nt=(nn+255)/256;
    dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,groups);
    ck_tile::launch_kernel(ck_tile::stream_config{stream,false},
        ck_tile::make_kernel<P::Threads,1>(ck_tile::GroupedInt8NtWaspKernel<P>{},
            grid,dim3(P::Threads),0,static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError()==hipSuccess?0:-100;
}
} // namespace
extern "C" int
ck_tile_grouped_int8_nt_launch(unsigned instance_id, const void* args, int g, int m, int n, void* stream)
{
    if(!args || g <= 0 || g > 65535 || m <= 0 || n <= 0)
        return -1;
    auto s = reinterpret_cast<hipStream_t>(stream);
    // The public selector chooses this precompiled instance from M/N/K/G,
    // layout and alignment. instance_id is an ABI kernel handle, not a test-case ID.
    using Instance = ck_tile::GroupedGemmInstanceId;
    switch(static_cast<Instance>(instance_id))
    {
    case Instance::int8_nt_direct_m16: {
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
    case Instance::int8_nt_m32: return launch<ck_tile::GroupedInt8NtTileProblem<32, 128>>(args, g, m, n, s);
    case Instance::int8_nt_m64: return launch<ck_tile::GroupedInt8NtTileProblem<64, 128>>(args, g, m, n, s);
    case Instance::int8_nt_m128: return launch<ck_tile::GroupedInt8NtTileProblem<128, 128>>(args, g, m, n, s);
    case Instance::int8_nt_m128_n256:
        return launch<ck_tile::GroupedInt8NtTileProblem<128, 256, 64, 2, 4>>(args, g, m, n, s);
    case Instance::int8_nt_m128_aligned:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, true, true, true, 1, false>>(
            args, g, m, n, s);
    case Instance::int8_nt_m256_n256_aligned:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, true, true, false, 1, true>>(
            args, g, m, n, s);
    case Instance::int8_nt_m128_n256_aligned:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<128, 256, 64, 2, 4, true, true, true, false, 2, true>>(
            args, g, m, n, s);
    case Instance::int8_nt_m256_n256_padded:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, false, true, false, 1, true>>(
            args, g, m, n, s);
    case Instance::int8_nt_m128_padded:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, false, true, false, 1, true>>(
            args, g, m, n, s);
    case Instance::int8_nt_m256_n256_ngroup:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<256, 256, 64, 4, 2, true, true, true, false, 1, true, 1>>(
            args, g, m, n, s);
    case Instance::int8_nt_m128_mgroup8:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, true, true, true, 1, false, 8>>(
            args, g, m, n, s);
    case Instance::int8_nt_m128_padded_mgroup8:
        return launch<
            ck_tile::
                GroupedInt8NtTileProblem<128, 128, 64, 2, 2, true, false, true, false, 1, true, 8>>(
            args, g, m, n, s);
    case Instance::int8_nt_m16_n256_aligned:
        return launch<
            ck_tile::GroupedInt8NtTileProblem<16, 256, 64, 1, 4, false, true, true, true, 1, true>>(
            args, g, m, n, s);
    case Instance::int8_nt_small_m_n64_k128:
        return launch<ck_tile::GroupedInt8NtTileProblem<16, 64, 128, 1, 2,
            false, false, true, true, 1, true>>(args, g, m, n, s);
    case Instance::int8_nt_small_m_n64_k256:
        return launch<ck_tile::GroupedInt8NtTileProblem<16, 64, 256, 1, 2,
            false, false, true, true, 1, true>>(args, g, m, n, s);
    case Instance::int8_nt_small_m_n128_k256:
        return launch<ck_tile::GroupedInt8NtTileProblem<16, 128, 256, 1, 4,
            false, false, true, true, 1, true>>(args, g, m, n, s);
    case Instance::int8_nt_small_g_n32_k512:
        return launch<ck_tile::GroupedInt8NtTileProblem<16, 32, 512, 1, 2,
            false, false, true, true, 1, true>>(args, g, m, n, s);
    case Instance::int8_nt_m32_n64_k128:
        return launch<ck_tile::GroupedInt8NtTileProblem<32,64,128,1,2,
            false, false, true, true, 1, true>>(args, g, m, n, s);
    case Instance::int8_nt_m32_n64_k256:
        return launch<ck_tile::GroupedInt8NtTileProblem<32,64,256,1,2,
            false, false, true, true, 1, true>>(args, g, m, n, s);
    case Instance::int8_nt_m32_n128_k256:
        return launch<ck_tile::GroupedInt8NtTileProblem<32,128,256,2,2,
            false, false, true, true, 1, true>>(args, g, m, n, s);
    case Instance::int8_nt_m64_n128_k128:
        return launch<ck_tile::GroupedInt8NtTileProblem<64,128,128,2,2,
            false, false, true, true, 1, true>>(args, g, m, n, s);
    case Instance::int8_nt_m64_n128_k256:
        return launch<ck_tile::GroupedInt8NtTileProblem<64,128,256,2,2,
            false, false, true, true, 1, true>>(args, g, m, n, s);
    case Instance::int8_nt_small_m_n32_k512_direct: {
        using P = ck_tile::GroupedInt8NtTileProblem<16,32,512,1,2,
            false,false,true,true,1,true>;
        return launch<P,ck_tile::GroupedInt8NtDirectEpilogue<P>>(args,g,m,n,s);
    }
    case Instance::int8_nt_wasp_a4_transposed: return launch_wasp<4,true>(args,g,m,n,s);
    case Instance::int8_nt_wasp_a8_transposed: return launch_wasp<8,true>(args,g,m,n,s);
    case Instance::int8_nt_wasp_a4: return launch_wasp<4,false>(args,g,m,n,s);
    default: return -5;
    }
}
