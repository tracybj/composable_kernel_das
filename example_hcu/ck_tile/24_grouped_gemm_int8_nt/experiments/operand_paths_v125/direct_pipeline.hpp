// SPDX-License-Identifier: MIT
#pragma once
#include "../../grouped_int8_nt_pipeline.hpp"

namespace ck_tile {
template<class Base,bool ALds_,bool BLds_>
struct GroupedInt8NtOperandProblem : Base {
    static constexpr bool ALds=ALds_,BLds=BLds_;
};

// Keep the normal Cooperative problem/policy and code generation untouched.
// Only experimental problem types replace operand storage;C exchange is shared
// with the existing epilogue and remains part of the LDS allocation.
template<class Base,bool A,bool B>
struct GroupedInt8NtTilePolicy<GroupedInt8NtOperandProblem<Base,A,B>>
    : GroupedInt8NtTilePolicy<Base> {
    using P=GroupedInt8NtOperandProblem<Base,A,B>;
    using Parent=GroupedInt8NtTilePolicy<Base>;
    static constexpr index_t BPitchOffset=A?P::BM*Parent::OperandPitch:0;
    static constexpr index_t OperandBytes=(A?P::BM*Parent::OperandPitch:0)+(B?P::BN*Parent::OperandPitch:0);
    static constexpr index_t OperandAllocation=OperandBytes;
    static constexpr index_t SmemBytes=OperandBytes>Parent::CStripeBytes?OperandBytes:Parent::CStripeBytes;
};

// Same load-next/compute-current Cooperative schedule. Direct operands load
// the exact MMAC 8-byte lane fragment;this first prototype trades narrower
// VMEM requests and cross-wave duplication for fewer LDS operations/barriers.
template<class Base,bool A,bool B,class Policy>
struct GroupedInt8NtTilePipeline<GroupedInt8NtOperandProblem<Base,A,B>,Policy> {
    using P=GroupedInt8NtOperandProblem<Base,A,B>;
    using Mmac=typename P::Mmac;using Copy=typename P::CopyVector;
    using AVec=typename Mmac::AVecType;using BVec=typename Mmac::BVecType;using Acc=typename Mmac::CVecType;
    struct GlobalTile {
        Copy a[A?Policy::AVectors:1],b[B?Policy::BVectors:1];
        AVec ad[A?1:P::MR][P::BK/32];
        BVec bd[B?1:P::NR][P::BK/32];
    };
    CK_TILE_DEVICE GlobalTile Load(const UniversalGemmKernelArgs<>& arg,index_t mb,index_t nb,index_t kb) const {
        GlobalTile v{};const index_t tid=get_thread_id(),lane=tid%64,wave=tid/64,wm=wave/P::WN,wn=wave%P::WN;
        if constexpr(A) {
            static_for<0,Policy::AVectors,1>{}([&](auto i){
                constexpr index_t j=decltype(i)::value;const index_t vec=tid+j*P::Threads,row=vec/(P::BK/16),ko=(vec%(P::BK/16))*16;
                if(row<P::BM && (P::Aligned || (mb+row<arg.M && kb+ko+16<=arg.K)))
                    v.a[j]=*reinterpret_cast<const Copy*>(static_cast<const int8_t*>(arg.as_ptr[0])+static_cast<long long>(mb+row)*arg.stride_As[0]+kb+ko);
            });
        } else {
            static_for<0,P::MR,1>{}([&](auto i){static_for<0,P::BK/32,1>{}([&](auto k){
                constexpr index_t mi=decltype(i)::value,ki=decltype(k)::value;
                const index_t row=mb+wm*(P::BM/P::WM)+mi*16+lane%16,ko=ki*32+(lane/16)*8;
                if(P::Aligned || (row<arg.M && kb+ko+8<=arg.K))
                    v.ad[mi][ki]=*reinterpret_cast<const AVec*>(static_cast<const int8_t*>(arg.as_ptr[0])+static_cast<long long>(row)*arg.stride_As[0]+kb+ko);
            });});
        }
        if constexpr(B) {
            static_for<0,Policy::BVectors,1>{}([&](auto i){
                constexpr index_t j=decltype(i)::value;const index_t vec=tid+j*P::Threads,row=vec/(P::BK/16),ko=(vec%(P::BK/16))*16;
                if(row<P::BN && (P::Aligned || (nb+row<arg.N && kb+ko+16<=arg.K)))
                    v.b[j]=*reinterpret_cast<const Copy*>(static_cast<const int8_t*>(arg.bs_ptr[0])+static_cast<long long>(nb+row)*arg.stride_Bs[0]+kb+ko);
            });
        } else {
            static_for<0,P::NR,1>{}([&](auto j){static_for<0,P::BK/32,1>{}([&](auto k){
                constexpr index_t ni=decltype(j)::value,ki=decltype(k)::value;
                const index_t row=nb+wn*(P::BN/P::WN)+ni*16+lane%16,ko=ki*32+(lane/16)*8;
                if(P::Aligned || (row<arg.N && kb+ko+8<=arg.K))
                    v.bd[ni][ki]=*reinterpret_cast<const BVec*>(static_cast<const int8_t*>(arg.bs_ptr[0])+static_cast<long long>(row)*arg.stride_Bs[0]+kb+ko);
            });});
        }
        return v;
    }
    CK_TILE_DEVICE void Stage(const GlobalTile& v,int8_t* smem) const {
        const index_t tid=get_thread_id();
        if constexpr(A)static_for<0,Policy::AVectors,1>{}([&](auto i){
            constexpr index_t j=decltype(i)::value;const index_t vec=tid+j*P::Threads,row=vec/(P::BK/16),ko=(vec%(P::BK/16))*16;
            if(row<P::BM)*reinterpret_cast<Copy*>(smem+row*Policy::OperandPitch+ko)=v.a[j];
        });
        if constexpr(B)static_for<0,Policy::BVectors,1>{}([&](auto i){
            constexpr index_t j=decltype(i)::value;const index_t vec=tid+j*P::Threads,row=vec/(P::BK/16),ko=(vec%(P::BK/16))*16;
            if(row<P::BN)*reinterpret_cast<Copy*>(smem+Policy::BPitchOffset+row*Policy::OperandPitch+ko)=v.b[j];
        });
    }
    CK_TILE_DEVICE void Compute(const int8_t* smem,Acc (&c)[P::MR][P::NR],const GlobalTile& current) const {
        const index_t lane=get_thread_id()%64,wave=get_thread_id()/64,wm=wave/P::WN,wn=wave%P::WN,ko=(lane/16)*8;
        static_for<0,P::BK/32,1>{}([&](auto k){
            constexpr index_t kk=decltype(k)::value;AVec av[P::MR];BVec bv[P::NR];
            static_for<0,P::MR,1>{}([&](auto i){
                constexpr index_t mi=decltype(i)::value;
                if constexpr(A){const index_t row=wm*(P::BM/P::WM)+mi*16+lane%16;av[mi]=*reinterpret_cast<const AVec*>(smem+row*Policy::OperandPitch+kk*32+ko);}
                else av[mi]=current.ad[mi][kk];
            });
            static_for<0,P::NR,1>{}([&](auto j){
                constexpr index_t ni=decltype(j)::value;
                if constexpr(B){const index_t row=wn*(P::BN/P::WN)+ni*16+lane%16;bv[ni]=*reinterpret_cast<const BVec*>(smem+Policy::BPitchOffset+row*Policy::OperandPitch+kk*32+ko);}
                else bv[ni]=current.bd[ni][kk];
            });
            static_for<0,P::MR,1>{}([&](auto i){static_for<0,P::NR,1>{}([&](auto j){Mmac{}(c[decltype(i)::value][decltype(j)::value],av[decltype(i)::value],bv[decltype(j)::value]);});});
        });
    }
    CK_TILE_DEVICE void operator()(const UniversalGemmKernelArgs<>& arg,index_t mb,index_t nb,int8_t* smem,Acc (&c)[P::MR][P::NR]) const {
        auto current=Load(arg,mb,nb,0);Stage(current,smem);if constexpr(A || B)block_sync_lds();
        for(index_t kb=0;kb+P::BK<arg.K;kb+=P::BK){
            auto next=Load(arg,mb,nb,kb+P::BK);
            __builtin_amdgcn_sched_barrier(0);Compute(smem,c,current);__builtin_amdgcn_sched_barrier(0);
            // All shared reads finish before overwrite;the following barrier
            // publishes the next A/B stage. No cross-wave operand dependency
            // remains when both operands are private registers.
            if constexpr(A || B)block_sync_lds();Stage(next,smem);if constexpr(A || B)block_sync_lds();
            current=next;
        }
        Compute(smem,c,current);
    }
};
} // namespace ck_tile
