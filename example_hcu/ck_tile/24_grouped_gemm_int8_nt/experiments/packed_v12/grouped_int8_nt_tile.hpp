// SPDX-License-Identifier: MIT

#pragma once

#include "grouped_int8_nt_problem.hpp"



namespace ck_tile {

template<index_t BM_,index_t BN_,index_t BK_=64,index_t WM_=2,index_t WN_=2,bool MFirst_=false,bool WideLds_=false,bool Aligned_=false,bool DoubleLds_=false,bool Peeled_=false,bool Fence_=false,index_t Pad_=16,bool BatchedEpi_=false,index_t MinBlocks_=1,bool CopyArg_=false>

struct GroupedInt8NtTileProblem

{

    using Mmac=WarpGemmAttributeMmacImplI8I8I32M16N16K32;

    using CopyVector=ext_vector_t<int32_t,4>;

    static constexpr index_t BM=BM_,BN=BN_,BK=BK_,WM=WM_,WN=WN_;

    static constexpr bool MFirst=MFirst_;

    static constexpr bool WideLds=WideLds_;

    static constexpr bool Aligned=Aligned_,DoubleLds=DoubleLds_;

    static constexpr bool Peeled=Peeled_,Fence=Fence_;

    static constexpr index_t Pad=Pad_;

    static constexpr bool BatchedEpi=BatchedEpi_,CopyArg=CopyArg_;

    static constexpr index_t MinBlocks=MinBlocks_;

    static_assert(!WideLds || BK==64);

    static constexpr index_t Threads=WM*WN*64,MR=BM/(WM*16),NR=BN/(WN*16);

    static_assert(BM%(WM*16)==0 && BN%(WN*16)==0 && BK%32==0);

};



template<class P>

struct GroupedInt8NtTilePolicy

{

    // 16 bytes of row padding separate the banks used by adjacent MMAC rows.

    // Native packed vectors keep each int8 fragment in two dword registers.

    static constexpr index_t OperandPitch=P::BK+P::Pad;

    static constexpr index_t BPitchOffset=P::BM*OperandPitch;

    static constexpr index_t CStripeRows=P::BatchedEpi?P::WM*16:16,CPitch=P::BatchedEpi?P::BN:P::BN+4;

    static constexpr index_t OperandBytes=(P::BM+P::BN)*OperandPitch;

    static constexpr index_t CStripeBytes=CStripeRows*CPitch*4;

    static constexpr index_t OperandAllocation=OperandBytes*(P::DoubleLds?2:1);

    static constexpr index_t SmemBytes=OperandAllocation>CStripeBytes?OperandAllocation:CStripeBytes;

    static constexpr index_t AVectors=(P::BM*P::BK/16+P::Threads-1)/P::Threads;

    static constexpr index_t BVectors=(P::BN*P::BK/16+P::Threads-1)/P::Threads;

};



template<class P,class Policy=GroupedInt8NtTilePolicy<P>>

struct GroupedInt8NtTilePipeline

{

    using Mmac=typename P::Mmac;

    using Copy=typename P::CopyVector;

    using Acc=typename Mmac::CVecType;

    struct GlobalTile{Copy a[Policy::AVectors],b[Policy::BVectors];};



    CK_TILE_DEVICE GlobalTile Load(const UniversalGemmKernelArgs<>& arg,index_t mb,index_t nb,index_t kb) const

    {

        GlobalTile v{};const index_t tid=get_thread_id();

        static_for<0,Policy::AVectors,1>{}([&](auto i){

            constexpr index_t j=decltype(i)::value;

            const index_t vec=tid+j*P::Threads,row=vec/(P::BK/16),ko=(vec%(P::BK/16))*16;

            if(row<P::BM && (P::Aligned || (mb+row<arg.M && kb+ko+16<=arg.K)))

                v.a[j]=*reinterpret_cast<const Copy*>(static_cast<const int8_t*>(arg.as_ptr[0])+static_cast<long long>(mb+row)*arg.stride_As[0]+kb+ko);

        });

        static_for<0,Policy::BVectors,1>{}([&](auto i){

            constexpr index_t j=decltype(i)::value;

            const index_t vec=tid+j*P::Threads,row=vec/(P::BK/16),ko=(vec%(P::BK/16))*16;

            if(row<P::BN && (P::Aligned || (nb+row<arg.N && kb+ko+16<=arg.K)))

                v.b[j]=*reinterpret_cast<const Copy*>(static_cast<const int8_t*>(arg.bs_ptr[0])+static_cast<long long>(nb+row)*arg.stride_Bs[0]+kb+ko);

        });

        return v;

    }

    CK_TILE_DEVICE void Stage(const GlobalTile& v,int8_t* smem) const

    {

        const index_t tid=get_thread_id();

        static_for<0,Policy::AVectors,1>{}([&](auto i){

            constexpr index_t j=decltype(i)::value;

            const index_t vec=tid+j*P::Threads,row=vec/(P::BK/16),ko=(vec%(P::BK/16))*16;

            if(row<P::BM){

              if constexpr(P::WideLds){

                const auto t=v.a[j];

                *reinterpret_cast<typename Mmac::AVecType*>(smem+row*Policy::OperandPitch+(ko%32)/8*16+(ko/32)*8)=__builtin_shufflevector(t,t,0,1);

                *reinterpret_cast<typename Mmac::AVecType*>(smem+row*Policy::OperandPitch+((ko+8)%32)/8*16+((ko+8)/32)*8)=__builtin_shufflevector(t,t,2,3);

              }else *reinterpret_cast<Copy*>(smem+row*Policy::OperandPitch+ko)=v.a[j];

            }

        });

        static_for<0,Policy::BVectors,1>{}([&](auto i){

            constexpr index_t j=decltype(i)::value;

            const index_t vec=tid+j*P::Threads,row=vec/(P::BK/16),ko=(vec%(P::BK/16))*16;

            if(row<P::BN){

              if constexpr(P::WideLds){

                const auto t=v.b[j];

                *reinterpret_cast<typename Mmac::BVecType*>(smem+Policy::BPitchOffset+row*Policy::OperandPitch+(ko%32)/8*16+(ko/32)*8)=__builtin_shufflevector(t,t,0,1);

                *reinterpret_cast<typename Mmac::BVecType*>(smem+Policy::BPitchOffset+row*Policy::OperandPitch+((ko+8)%32)/8*16+((ko+8)/32)*8)=__builtin_shufflevector(t,t,2,3);

              }else *reinterpret_cast<Copy*>(smem+Policy::BPitchOffset+row*Policy::OperandPitch+ko)=v.b[j];

            }

        });

    }

    CK_TILE_DEVICE void Compute(const int8_t* smem,Acc (&c)[P::MR][P::NR]) const

    {

        const index_t lane=get_thread_id()%64,wave=get_thread_id()/64;

        const index_t wm=wave/P::WN,wn=wave%P::WN,ko=(lane/16)*8;

      if constexpr(P::WideLds){

        Copy av[P::MR],bv[P::NR];

        static_for<0,P::MR,1>{}([&](auto i){constexpr index_t mi=decltype(i)::value;

            const index_t row=wm*(P::BM/P::WM)+mi*16+lane%16;

            av[mi]=*reinterpret_cast<const Copy*>(smem+row*Policy::OperandPitch+(lane/16)*16);

        });

        static_for<0,P::NR,1>{}([&](auto j){constexpr index_t ni=decltype(j)::value;

            const index_t row=wn*(P::BN/P::WN)+ni*16+lane%16;

            bv[ni]=*reinterpret_cast<const Copy*>(smem+Policy::BPitchOffset+row*Policy::OperandPitch+(lane/16)*16);

        });

        static_for<0,P::MR,1>{}([&](auto i){static_for<0,P::NR,1>{}([&](auto j){

            constexpr index_t mi=decltype(i)::value,ni=decltype(j)::value;

            Mmac{}(c[mi][ni],__builtin_shufflevector(av[mi],av[mi],0,1),__builtin_shufflevector(bv[ni],bv[ni],0,1));

        });});

        static_for<0,P::MR,1>{}([&](auto i){static_for<0,P::NR,1>{}([&](auto j){

            constexpr index_t mi=decltype(i)::value,ni=decltype(j)::value;

            Mmac{}(c[mi][ni],__builtin_shufflevector(av[mi],av[mi],2,3),__builtin_shufflevector(bv[ni],bv[ni],2,3));

        });});

      }else{

        static_for<0,P::BK/32,1>{}([&](auto kk){

            typename Mmac::AVecType av[P::MR];typename Mmac::BVecType bv[P::NR];

            static_for<0,P::MR,1>{}([&](auto i){constexpr index_t mi=decltype(i)::value;

                const index_t row=wm*(P::BM/P::WM)+mi*16+lane%16;

                av[mi]=*reinterpret_cast<const typename Mmac::AVecType*>(smem+row*Policy::OperandPitch+kk*32+ko);

            });

            static_for<0,P::NR,1>{}([&](auto j){constexpr index_t ni=decltype(j)::value;

                const index_t row=wn*(P::BN/P::WN)+ni*16+lane%16;

                bv[ni]=*reinterpret_cast<const typename Mmac::BVecType*>(smem+Policy::BPitchOffset+row*Policy::OperandPitch+kk*32+ko);

            });

            static_for<0,P::MR,1>{}([&](auto i){static_for<0,P::NR,1>{}([&](auto j){

                Mmac{}(c[decltype(i)::value][decltype(j)::value],av[decltype(i)::value],bv[decltype(j)::value]);

            });});

        });

      }

    }

    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& arg,index_t mb,index_t nb,int8_t* smem,Acc (&c)[P::MR][P::NR]) const

    {

        auto current=Load(arg,mb,nb,0);Stage(current,smem);block_sync_lds();

        // The next global tile stays packed in VGPR while MMAC consumes LDS.

        // All waves finish reading before Stage overwrites this single buffer.

      if constexpr(P::Peeled){

        index_t stage=0;

        for(index_t kb=0;kb+P::BK<arg.K;kb+=P::BK){

            auto next=Load(arg,mb,nb,kb+P::BK);

            if constexpr(P::Fence)__builtin_amdgcn_sched_barrier(0);

            Compute(smem+stage*Policy::OperandBytes,c);

            if constexpr(P::Fence)__builtin_amdgcn_sched_barrier(0);

            if constexpr(P::DoubleLds)stage^=1;

            else block_sync_lds();

            Stage(next,smem+stage*Policy::OperandBytes);block_sync_lds();

        }

        Compute(smem+stage*Policy::OperandBytes,c);

      }else{

        for(index_t kb=0;kb<arg.K;kb+=P::BK){

            GlobalTile next{};

            if(kb+P::BK<arg.K)next=Load(arg,mb,nb,kb+P::BK);

            int8_t* current_smem=smem;

            if constexpr(P::DoubleLds)current_smem+=((kb/P::BK)%2)*Policy::OperandBytes;

            Compute(current_smem,c);

            if(kb+P::BK<arg.K){

                int8_t* next_smem=smem;

                if constexpr(P::DoubleLds)next_smem+=(((kb/P::BK)%2)^1)*Policy::OperandBytes;

                else block_sync_lds();

                Stage(next,next_smem);block_sync_lds();

            }

        }

      }

    }

};



template<class P,class Policy=GroupedInt8NtTilePolicy<P>>

struct GroupedInt8NtTileEpilogue

{

    using Acc=typename P::Mmac::CVecType;

    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& arg,index_t mb,index_t nb,int8_t* smem,const Acc (&c)[P::MR][P::NR]) const

    {

        const index_t tid=get_thread_id(),lane=tid%64,wave=tid/64;

        const index_t wm=wave/P::WN,wn=wave%P::WN;auto* tmp=reinterpret_cast<int32_t*>(smem);

        block_sync_lds();

        static_for<0,P::MR,1>{}([&](auto i){

            constexpr index_t mi=decltype(i)::value;

          if constexpr(P::BatchedEpi){

            const index_t row=wm*16+lane%16;

            static_for<0,P::NR,1>{}([&](auto j){static_for<0,4,1>{}([&](auto v){

                constexpr index_t ni=decltype(j)::value,vi=decltype(v)::value;

                const index_t col=wn*(P::BN/P::WN)+ni*16+lane/16+vi*4;

                tmp[row*Policy::CPitch+(col^((row%8)*4))]=c[mi][ni][vi];

            });});

            block_sync_lds();

            for(index_t vec=tid;vec<Policy::CStripeRows*P::BN/4;vec+=P::Threads){

                const index_t r=vec/(P::BN/4),col=(vec%(P::BN/4))*4;

                const index_t outrow=mb+(r/16)*(P::BM/P::WM)+mi*16+r%16;

                if(P::Aligned || (outrow<arg.M && nb+col+4<=arg.N))

                    *reinterpret_cast<typename P::CopyVector*>(static_cast<int32_t*>(arg.e_ptr)+static_cast<long long>(outrow)*arg.stride_E+nb+col)=

                        *reinterpret_cast<const typename P::CopyVector*>(tmp+r*Policy::CPitch+(col^((r%8)*4)));

            }

            block_sync_lds();

          }else{

          static_for<0,P::WM,1>{}([&](auto wmi){

            constexpr index_t output_wave_m=decltype(wmi)::value;

            if(wm==output_wave_m){

            static_for<0,P::NR,1>{}([&](auto j){static_for<0,4,1>{}([&](auto v){

                constexpr index_t ni=decltype(j)::value,vi=decltype(v)::value;

                const index_t col=wn*(P::BN/P::WN)+ni*16+lane/16+vi*4;

                tmp[(lane%16)*Policy::CPitch+col]=c[mi][ni][vi];

            });});

            }

            block_sync_lds();

            for(index_t vec=tid;vec<Policy::CStripeRows*P::BN/4;vec+=P::Threads){

                const index_t row=vec/(P::BN/4),col=(vec%(P::BN/4))*4;

                const index_t outrow=mb+output_wave_m*(P::BM/P::WM)+mi*16+row;

                if(P::Aligned || (outrow<arg.M && nb+col+4<=arg.N))

                    *reinterpret_cast<typename P::CopyVector*>(static_cast<int32_t*>(arg.e_ptr)+static_cast<long long>(outrow)*arg.stride_E+nb+col)=

                        *reinterpret_cast<const typename P::CopyVector*>(tmp+row*Policy::CPitch+col);

            }

            block_sync_lds();

          });

          }

        });

    }

};



template<class P>

struct GroupedInt8NtTileKernel

{

    CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args) const

    {

        const index_t tid=get_thread_id();

        __builtin_assume(tid<P::Threads);

        using ArgStorage=std::conditional_t<P::CopyArg,UniversalGemmKernelArgs<>,const UniversalGemmKernelArgs<>&>;

        const ArgStorage a=args[blockIdx.z].group_karg;

        const index_t mb=(P::MFirst?blockIdx.x:blockIdx.y)*P::BM;

        const index_t nb=(P::MFirst?blockIdx.y:blockIdx.x)*P::BN;

        if constexpr(!P::Aligned)

            if(a.M<=0 || a.N<=0 || a.K<=0 || mb>=a.M || nb>=a.N)return;

        using Policy=GroupedInt8NtTilePolicy<P>;

        __shared__ __align__(16) int8_t smem[Policy::SmemBytes];

        typename P::Mmac::CVecType c[P::MR][P::NR]{};

        GroupedInt8NtTilePipeline<P>{}(a,mb,nb,smem,c);

        GroupedInt8NtTileEpilogue<P>{}(a,mb,nb,smem,c);

    }

};

} // namespace ck_tile
