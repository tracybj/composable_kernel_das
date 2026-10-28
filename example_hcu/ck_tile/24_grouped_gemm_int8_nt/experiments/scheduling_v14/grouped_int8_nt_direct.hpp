// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
namespace ck_tile {
struct GroupedInt8NtDirectProblem
{
    using WarpMmac                     = WarpGemmAttributeMmacImplI8I8I32M16N16K32;
    static constexpr index_t MPerBlock = 16, NPerBlock = 64, BlockSize = 256;
};
// Each wave consumes a physical 16-row x K32 slice of both NT inputs.
// Eight consecutive int8 values per lane match CK's MMAC operand contract;
// no LDS copy or CTA barrier is needed. K32 tail loads are zero-filled.
template <class Problem = GroupedInt8NtDirectProblem>
struct GroupedInt8NtDirectPipeline
{
    using Mmac = typename Problem::WarpMmac;
    using Acc  = typename Mmac::CVecType;
    CK_TILE_DEVICE Acc operator()(const int8_t* a,
                                  const int8_t* b,
                                  index_t m,
                                  index_t n,
                                  index_t k,
                                  index_t sa,
                                  index_t sb,
                                  index_t mr,
                                  index_t nr) const
    {
        Acc c{0};
        const index_t lane = get_thread_id() % 64;
        const index_t ko   = (lane / 16) * 8;
        for(index_t kk = 0; kk < k; kk += 32)
        {
            typename Mmac::AVecType av{0}, bv{0};
            if(mr < m && kk + ko + 8 <= k)
                av = *reinterpret_cast<const typename Mmac::AVecType*>(
                    a + static_cast<long long>(mr) * sa + kk + ko);
            if(nr < n && kk + ko + 8 <= k)
                bv = *reinterpret_cast<const typename Mmac::BVecType*>(
                    b + static_cast<long long>(nr) * sb + kk + ko);
            Mmac{}(c, av, bv);
        }
        return c;
    }
};
template <class Problem = GroupedInt8NtDirectProblem>
struct GroupedInt8NtDirectKernel
{
    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const
    {
        const auto& a      = args[blockIdx.z].group_karg;
        const index_t lane = get_thread_id() % 64, wave = get_thread_id() / 64;
        const index_t mr = blockIdx.y * Problem::MPerBlock + lane % 16;
        const index_t nb = blockIdx.x * Problem::NPerBlock + wave * 16;
        if(a.M <= 0 || a.N <= 0 || a.K <= 0)
            return;
        auto acc = GroupedInt8NtDirectPipeline<Problem>{}(static_cast<const int8_t*>(a.as_ptr[0]),
                                                          static_cast<const int8_t*>(a.bs_ptr[0]),
                                                          a.M,
                                                          a.N,
                                                          a.K,
                                                          a.stride_As[0],
                                                          a.stride_Bs[0],
                                                          mr,
                                                          nb + lane % 16);
        if(mr < a.M)
        {
            static_for<0, 4, 1>{}([&](auto i) {
                const index_t col = nb + lane / 16 + i * 4;
                if(col < a.N)
                    static_cast<int32_t*>(a.e_ptr)[static_cast<long long>(mr) * a.stride_E + col] =
                        acc[decltype(i)::value];
            });
        }
    }
};
} // namespace ck_tile
