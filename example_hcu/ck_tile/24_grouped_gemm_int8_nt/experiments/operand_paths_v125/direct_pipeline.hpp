// SPDX-License-Identifier: MIT
#pragma once
#include "../../grouped_int8_nt_pipeline.hpp"

namespace ck_tile {
template<class Base,bool ALds_,bool BLds_,bool Shuffle_=false,bool Deferred_=false>
struct GroupedInt8NtOperandProblem : Base {
    static constexpr bool ALds=ALds_,BLds=BLds_,Shuffle=Shuffle_,Deferred=Deferred_;
};

// Keep the normal Cooperative problem/policy and code generation untouched.
// Only experimental problem types replace operand storage;C exchange is shared
// with the existing epilogue and remains part of the LDS allocation.
template<class Base,bool A,bool B,bool S,bool D>
struct GroupedInt8NtTilePolicy<GroupedInt8NtOperandProblem<Base,A,B,S,D>>
    : GroupedInt8NtTilePolicy<Base> {
    using P=GroupedInt8NtOperandProblem<Base,A,B,S,D>;
    using Parent=GroupedInt8NtTilePolicy<Base>;
    static constexpr index_t BPitchOffset=A?P::BM*Parent::OperandPitch:0;
    static constexpr index_t OperandBytes=(A?P::BM*Parent::OperandPitch:0)+(B?P::BN*Parent::OperandPitch:0);
    static constexpr index_t OperandAllocation=OperandBytes;
    static constexpr index_t SmemBytes=OperandBytes>Parent::CStripeBytes?OperandBytes:Parent::CStripeBytes;
};

// Same load-next/compute-current Cooperative schedule. Direct operands load
// the exact MMAC 8-byte lane fragment;this first prototype trades narrower
// VMEM requests and cross-wave duplication for fewer LDS operations/barriers.
template<class Base,bool A,bool B,bool S,bool D,class Policy>
struct GroupedInt8NtTilePipeline<GroupedInt8NtOperandProblem<Base,A,B,S,D>,Policy> {
    using P=GroupedInt8NtOperandProblem<Base,A,B,S,D>;
    using Mmac=typename P::Mmac;using Copy=typename P::CopyVector;
    using AVec=typename Mmac::AVecType;using BVec=typename Mmac::BVecType;using Acc=typename Mmac::CVecType;
    struct GlobalTile {
        Copy a[A?Policy::AVectors:1],b[B?Policy::BVectors:1];
        AVec ad[A?1:P::MR][P::BK/32];
        BVec bd[B?1:P::NR][P::BK/32];
    };
    CK_TILE_DEVICE Copy Permute(Copy x) const {
        const index_t lane=get_thread_id()%64,q=lane/16;
        int32_t r0=x[0],r1=x[1],r2=x[2],r3=x[3];
        const bool p0=q&1,p1=q&2;
        int32_t t0=__builtin_amdgcn_ds_bpermute((lane^16)*4,p0?r0:r2);
        int32_t t1=__builtin_amdgcn_ds_bpermute((lane^16)*4,p0?r1:r3);
        r0=p0?t0:r0;r1=p0?t1:r1;r2=p0?r2:t0;r3=p0?r3:t1;
        t0=__builtin_amdgcn_ds_bpermute((lane^32)*4,p1?r0:r2);
        t1=__builtin_amdgcn_ds_bpermute((lane^32)*4,p1?r1:r3);
        r0=p1?t0:r0;r1=p1?t1:r1;r2=p1?r2:t0;r3=p1?r3:t1;
        return Copy{r0,r1,r2,r3};
    }
    template<class Vec,index_t Rows>
    CK_TILE_DEVICE void PrepareDirect(Vec (&v)[Rows][P::BK/32]) const {
        static_for<0,Rows,1>{}([&](auto i){static_for<0,P::BK/64,1>{}([&](auto j){
            constexpr index_t ri=decltype(i)::value,ki=decltype(j)::value;
            const auto lo=bit_cast<ext_vector_t<int32_t,2>>(v[ri][ki*2]);
            const auto hi=bit_cast<ext_vector_t<int32_t,2>>(v[ri][ki*2+1]);
            const auto x=Permute(Copy{lo[0],lo[1],hi[0],hi[1]});
            v[ri][ki*2]=bit_cast<Vec>(ext_vector_t<int32_t,2>{x[0],x[1]});
            v[ri][ki*2+1]=bit_cast<Vec>(ext_vector_t<int32_t,2>{x[2],x[3]});
        });});
    }
    CK_TILE_DEVICE void Prepare(GlobalTile& v) const {
        if constexpr(S && D && !A)PrepareDirect<AVec,P::MR>(v.ad);
        if constexpr(S && D && !B)PrepareDirect<BVec,P::NR>(v.bd);
    }
    // Four lane quads load contiguous16B chunks of one64B K stripe.
    // Two butterfly exchanges move the fragment into MMAC's8B/lane layout.
    // ds_bpermute uses the wave exchange network,not an addressable LDS buffer.
    template<class Vec,index_t Rows>
    CK_TILE_DEVICE void LoadShuffle(const int8_t* ptr,long long stride,index_t limit,
                                    index_t k,index_t rowbase,index_t kb,
                                    Vec (&v)[Rows][P::BK/32]) const {
        const index_t lane=get_thread_id()%64,q=lane/16;
        static_for<0,Rows,1>{}([&](auto i){
            constexpr index_t ri=decltype(i)::value;
            const index_t row=rowbase+ri*16+lane%16;
            static_for<0,P::BK/64,1>{}([&](auto j){
                constexpr index_t ki=decltype(j)::value;
                const index_t ko=ki*64+q*16;
                Copy x{};
                if(P::Aligned || (row<limit && kb+ko+16<=k))
                    x=*reinterpret_cast<const Copy*>(ptr+static_cast<long long>(row)*stride+kb+ko);
                if constexpr(!D)x=Permute(x);
                v[ri][ki*2]=bit_cast<Vec>(ext_vector_t<int32_t,2>{x[0],x[1]});
                v[ri][ki*2+1]=bit_cast<Vec>(ext_vector_t<int32_t,2>{x[2],x[3]});
            });
        });
    }
    CK_TILE_DEVICE GlobalTile Load(const UniversalGemmKernelArgs<>& arg,index_t mb,index_t nb,index_t kb) const {
        GlobalTile v{};const index_t tid=get_thread_id(),lane=tid%64,wave=tid/64,wm=wave/P::WN,wn=wave%P::WN;
        if constexpr(A) {
            static_for<0,Policy::AVectors,1>{}([&](auto i){
                constexpr index_t j=decltype(i)::value;const index_t vec=tid+j*P::Threads,row=vec/(P::BK/16),ko=(vec%(P::BK/16))*16;
                if(row<P::BM && (P::Aligned || (mb+row<arg.M && kb+ko+16<=arg.K)))
                    v.a[j]=*reinterpret_cast<const Copy*>(static_cast<const int8_t*>(arg.as_ptr[0])+static_cast<long long>(mb+row)*arg.stride_As[0]+kb+ko);
            });
        } else if constexpr(S) {
            LoadShuffle<AVec,P::MR>(static_cast<const int8_t*>(arg.as_ptr[0]),arg.stride_As[0],arg.M,arg.K,mb+wm*(P::BM/P::WM),kb,v.ad);
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
        } else if constexpr(S) {
            LoadShuffle<BVec,P::NR>(static_cast<const int8_t*>(arg.bs_ptr[0]),arg.stride_Bs[0],arg.N,arg.K,nb+wn*(P::BN/P::WN),kb,v.bd);
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
        auto current=Load(arg,mb,nb,0);Prepare(current);Stage(current,smem);if constexpr(A || B)block_sync_lds();
        for(index_t kb=0;kb+P::BK<arg.K;kb+=P::BK){
            auto next=Load(arg,mb,nb,kb+P::BK);
            __builtin_amdgcn_sched_barrier(0);Compute(smem,c,current);__builtin_amdgcn_sched_barrier(0);
            // All shared reads finish before overwrite;the following barrier
            // publishes the next A/B stage. No cross-wave operand dependency
            // remains when both operands are private registers.
            if constexpr(A || B)block_sync_lds();Prepare(next);Stage(next,smem);if constexpr(A || B)block_sync_lds();
            current=next;
        }
        Compute(smem,c,current);
    }
};
} // namespace ck_tile
