// SPDX-License-Identifier: MIT
#pragma once
#include "../grouped_int8_nt_problem.hpp"
#include "scaled_epilogue.hpp"
namespace ck_tile {
struct Int8NtKParallelArgs {
 const int8_t* a;const int8_t* b;void* c;const float* sa;const float* sb;
 int m,n,k,as,bs,cs,sas,sbs;
};
template<int Waves_,int Cols_> struct Int8NtKParallelProblem {
 using Mmac=WarpGemmAttributeMmacImplI8I8I32M16N16K32;
 static constexpr int Waves=Waves_,Cols=Cols_,Threads=Waves*64;
 static constexpr int KSpan=Threads*16;
 static_assert(Waves==4 || Waves==8);
};
// Deliberately contiguous physical K per lane. The MMAC logical rows/columns
// represent K subpartitions; only the diagonal contributes to the dot product.
template<class P,int Rows> struct Int8NtKParallelPipeline {
 using V=ext_vector_t<int32_t,4>;using H=typename P::Mmac::AVecType;
 using Acc=typename P::Mmac::CVecType;
 CK_TILE_DEVICE void operator()(const Int8NtKParallelArgs& a,int nb,Acc (&c)[Rows][P::Cols]) const {
  int tid=get_thread_id();
  for(int kb=0;kb<a.k;kb+=P::KSpan){
   const int ko=kb+tid*16;V av[Rows]{},bv[P::Cols]{};
   static_for<0,Rows,1>{}([&](auto r){if(ko<a.k)av[r]=*reinterpret_cast<const V*>(a.a+static_cast<long long>(r)*a.as+ko);});
   static_for<0,P::Cols,1>{}([&](auto j){if(ko<a.k && nb+j<a.n)bv[j]=*reinterpret_cast<const V*>(a.b+static_cast<long long>(nb+j)*a.bs+ko);});
   static_for<0,2,1>{}([&](auto h){
    static_for<0,Rows,1>{}([&](auto r){
     H aa{av[r][h*2],av[r][h*2+1]};
     static_for<0,P::Cols,1>{}([&](auto j){H bb{bv[j][h*2],bv[j][h*2+1]};typename P::Mmac{}(c[r][j],aa,bb);});
    });
   });
  }
 }
};
template<class P,int Rows,int Output> struct Int8NtKParallelEpilogue {
 using Acc=typename P::Mmac::CVecType;
 CK_TILE_DEVICE void operator()(const Int8NtKParallelArgs& a,int nb,const Acc (&c)[Rows][P::Cols],int32_t* partial) const {
  const int tid=get_thread_id(),lane=tid%64,wave=tid/64;
  static_for<0,Rows,1>{}([&](auto r){static_for<0,P::Cols,1>{}([&](auto j){
   int32_t sum=0;
   static_for<0,4,1>{}([&](auto v){constexpr int vi=decltype(v)::value;if(lane%16==lane/16+vi*4)sum=c[r][j][vi];});
   for(int d=32;d;d/=2)sum+=__shfl_down(sum,d,64);
   if(lane==0)partial[(r*P::Cols+j)*P::Waves+wave]=sum;
  });});
  block_sync_lds();
  if(tid<Rows*P::Cols){
   const int r=tid/P::Cols,col=nb+tid%P::Cols;
   int32_t sum=0;for(int w=0;w<P::Waves;++w)sum+=partial[tid*P::Waves+w];
   if(col<a.n){
    if constexpr(Output==0)static_cast<int32_t*>(a.c)[r*a.cs+col]=sum;
    else {
     const float sa=a.sa?a.sa[r*a.sas]:1.f,sb=a.sb?a.sb[col*a.sbs]:1.f;
     static_cast<uint16_t*>(a.c)[r*a.cs+col]=Int8NtDequantConvert<Output==1>::run(sum,sa*sb);
    }
   }
  }
 }
};
// Uses the existing cached descriptors and borrowed scale pointers. No plan
// allocation or pointer transformation is added to the timed run path.
template<int Rows,bool BFloat> struct GroupedInt8NtScaledKParallelKernel {
 CK_TILE_DEVICE void operator()(const GemmTransKernelArg* args,const Int8NtScaleArgs* scales) const {
  using P=Int8NtKParallelProblem<8,4>;
  const auto o=args[blockIdx.z].group_karg;const auto s=scales[blockIdx.z];
  if(o.M!=Rows)return;
  const Int8NtKParallelArgs a{static_cast<const int8_t*>(o.as_ptr[0]),static_cast<const int8_t*>(o.bs_ptr[0]),o.e_ptr,s.a,s.b,o.M,o.N,o.K,o.stride_As[0],o.stride_Bs[0],o.stride_E,s.a_stride,s.b_stride};
  typename P::Mmac::CVecType c[Rows][P::Cols]{};
  __shared__ int32_t partial[Rows*P::Cols*P::Waves];
  Int8NtKParallelPipeline<P,Rows>{}(a,blockIdx.x*P::Cols,c);
  Int8NtKParallelEpilogue<P,Rows,BFloat?1:2>{}(a,blockIdx.x*P::Cols,c,partial);
 }
};
}
