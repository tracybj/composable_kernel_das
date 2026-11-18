// SPDX-License-Identifier: MIT
#pragma once
#include "grouped_int8_nt_problem.hpp"
#include "grouped_int8_nt_scaled_epilogue.hpp"
#include "../include/grouped_int8_nt_k_parallel_args.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
namespace ck_tile {
template <int Waves_, int Cols_>
struct Int8NtKParallelProblem
{
    using Mmac                            = WarpGemmAttributeMmacImplI8I8I32M16N16K32;
    using InputVector                     = ext_vector_t<int32_t, 4>;
    static constexpr int WaveSize         = 64;
    static constexpr int InputVectorBytes = sizeof(InputVector);
    static constexpr int MmacRows         = Mmac::kM;
    static constexpr int AccElements      = vector_traits<typename Mmac::CVecType>::vector_size;
    static constexpr int Waves = Waves_, Cols = Cols_, Threads = Waves * WaveSize;
    static constexpr int KSpan = Threads * InputVectorBytes;
    static_assert(Waves == 4 || Waves == 8);
};
// Deliberately contiguous physical K per lane. The MMAC logical rows/columns
// represent K subpartitions; only the diagonal contributes to the dot product.
template <class P, int Rows>
struct Int8NtKParallelPipeline
{
    using V   = typename P::InputVector;
    using H   = typename P::Mmac::AVecType;
    using Acc = typename P::Mmac::CVecType;
    CK_TILE_DEVICE void
    operator()(const Int8NtKParallelArgs& a, int nb, Acc (&c)[Rows][P::Cols]) const
    {
        int tid = get_thread_id();
        for(int kb = 0; kb < a.k; kb += P::KSpan)
        {
            const int ko = kb + tid * P::InputVectorBytes;
            V av[Rows]{}, bv[P::Cols]{};
            static_for<0, Rows, 1>{}([&](auto r) {
                if(ko < a.k)
                    av[r] = *reinterpret_cast<const V*>(
                        a.a + static_cast<long long>(r) * a.stride_a + ko);
            });
            static_for<0, P::Cols, 1>{}([&](auto j) {
                if(ko < a.k && nb + j < a.n)
                    bv[j] = *reinterpret_cast<const V*>(
                        a.b + static_cast<long long>(nb + j) * a.stride_b + ko);
            });
            static_for<0, 2, 1>{}([&](auto h) {
                static_for<0, Rows, 1>{}([&](auto r) {
                    H aa{av[r][h * 2], av[r][h * 2 + 1]};
                    static_for<0, P::Cols, 1>{}([&](auto j) {
                        H bb{bv[j][h * 2], bv[j][h * 2 + 1]};
                        typename P::Mmac{}(c[r][j], aa, bb);
                    });
                });
            });
        }
    }
};
template <class P, int Rows, bool BFloat>
struct Int8NtKParallelEpilogue
{
    using Acc = typename P::Mmac::CVecType;
    CK_TILE_DEVICE void operator()(const Int8NtKParallelArgs& a,
                                   int nb,
                                   const Acc (&c)[Rows][P::Cols],
                                   int32_t* partial) const
    {
        const int tid = get_thread_id(), lane = tid % P::WaveSize, wave = tid / P::WaveSize;
        static_for<0, Rows, 1>{}([&](auto r) {
            static_for<0, P::Cols, 1>{}([&](auto j) {
                int32_t sum = 0;
                static_for<0, P::AccElements, 1>{}([&](auto v) {
                    constexpr int vi = decltype(v)::value;
                    if(lane % P::MmacRows == lane / P::MmacRows + vi * P::AccElements)
                        sum = c[r][j][vi];
                });
                for(int distance = P::WaveSize / 2; distance; distance /= 2)
                    sum += __shfl_down(sum, distance, P::WaveSize);
                if(lane == 0)
                    partial[(r * P::Cols + j) * P::Waves + wave] = sum;
            });
        });
        block_sync_lds();
        if(tid < Rows * P::Cols)
        {
            const int r = tid / P::Cols, col = nb + tid % P::Cols;
            int32_t sum = 0;
            for(int w = 0; w < P::Waves; ++w)
                sum += partial[tid * P::Waves + w];
            if(col < a.n)
            {
                const float sa = a.scale_a ? a.scale_a[r * a.scale_a_stride] : 1.f,
                            sb = a.scale_b ? a.scale_b[col * a.scale_b_stride] : 1.f;
                static_cast<uint16_t*>(a.c)[r * a.stride_c + col] =
                    Int8NtDequantConvert<BFloat>::run(sum, sa * sb);
            }
        }
    }
};
template <int Rows, bool BFloat>
struct GroupedInt8NtScaledKParallelKernel
{
    CK_TILE_DEVICE static void run_group(Int8NtKParallelArgs a)
    {
        using P      = Int8NtKParallelProblem<8, 4>;
        const int nb = blockIdx.x * P::Cols;
        typename P::Mmac::CVecType c[Rows][P::Cols]{};
        __shared__ int32_t partial[Rows * P::Cols * P::Waves];
        Int8NtKParallelPipeline<P, Rows>{}(a, nb, c);
        Int8NtKParallelEpilogue<P, Rows, BFloat>{}(a, nb, c, partial);
    }

    CK_TILE_DEVICE void operator()(const Int8NtKParallelArgs* args) const
    {
        run_group(args[blockIdx.z]);
    }

    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args,
                                   const Int8NtScaleArgs* scales) const
    {
        const auto& arg   = args[blockIdx.z].group_karg;
        const auto& scale = scales[blockIdx.z];
        const Int8NtKParallelArgs group{static_cast<const int8_t*>(arg.as_ptr[0]),
                                        static_cast<const int8_t*>(arg.bs_ptr[0]),
                                        arg.e_ptr,
                                        scale.a,
                                        scale.b,
                                        arg.M,
                                        arg.N,
                                        arg.K,
                                        arg.stride_As[0],
                                        arg.stride_Bs[0],
                                        arg.stride_E,
                                        scale.a_stride,
                                        scale.b_stride};
        run_group(group);
    }
};
} // namespace ck_tile
