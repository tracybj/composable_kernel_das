// SPDX-License-Identifier: MIT
// Independent CK-Tile implementation of four cooperative waves and N-fragment reuse.
// Uses existing own/CK raw VMEM and INT8 MMAC helpers; no DeepGEMM source copied.
#pragma once
#include "../../grouped_int8_nt_wasp_pipeline.hpp"
namespace ck_tile {
struct Int8NReuseAccTile {
 static constexpr index_t kMIters=8,kNIters=4;
 grouped_int8_nt_wasp_detail::frag_c_t v[kMIters*kNIters];
};
template<class Problem,index_t AReadBatch=1,index_t RecordCap=0>
struct GemmPipelineInt8NReuse : GemmPipelineAgBgCrCompV3<Problem,GemmPipelineAgBgCrCompV4DefaultPolicy> {
 using Shape=remove_cvref_t<typename Problem::BlockGemmShape>;
 using AccTile=Int8NReuseAccTile;
 using Stage=grouped_int8_nt_wasp_detail::stage16_t;
 using Rsrc=grouped_int8_nt_wasp_detail::buffer_rsrc_t;
 static constexpr index_t BlockSize=256,kStageK=64,kAStageBytes=8192,kWaspLdsBytes=16384;
 static_assert(Shape::kM==128 && Shape::kN==256 && Shape::kK==64 && Shape::NumWarps==4);
 static_assert(AReadBatch==1 || AReadBatch==4);
 CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize(){return kWaspLdsBytes;}
 struct ASource {Rsrc rsrc;uint32_t v0,v1;};
 struct BPairs {uint64_t p[4];};
 CK_TILE_DEVICE static uint32_t record_stride(uint32_t pitch){
  if constexpr(RecordCap==0)return pitch;
  uint32_t stride=1;static_for<0,11,1>{}([&](auto i){constexpr uint32_t w=1u<<decltype(i)::value;if(pitch>=w)stride=w;});return stride;
 }
 CK_TILE_DEVICE ASource make_a_source(const int8_t* a,index_t pitch,index_t wave) const {
  using namespace grouped_int8_nt_wasp_detail;uint32_t lane=get_lane_id(),row=lane>>2,chunk=lane&3;
  uint32_t k=(chunk^(row&3))*16;
  return {wasp_make_buffer_rsrc(a,pitch,128u*pitch,false),(wave*32+row)*pitch+k,(wave*32+16+row)*pitch+k};
 }
 CK_TILE_DEVICE void produce(const ASource& s,index_t stage,index_t wave) const {
  uint32_t m0=(stage&1)*kAStageBytes+wave*2048,off=stage*kStageK;
  asm volatile("s_mov_b32 m0, %0\n\tbuffer_load_dwordx4 %1, %3, %4 offen lds\n\ts_add_u32 m0, m0, 0x400\n\tbuffer_load_dwordx4 %2, %3, %4 offen lds\n"::"s"(m0),"v"(s.v0),"v"(s.v1),"s"(s.rsrc),"s"(off):"memory");
 }
 CK_TILE_DEVICE static void load_b_one(Rsrc rsrc,uint64_t pair,uint32_t off,Stage& b){
  asm volatile("buffer_load_dwordx4 %0, %1, %2, %3 idxen offen\n":"=&v"(b.d4):"v"(pair),"s"(rsrc),"s"(off):"memory");
 }
 CK_TILE_DEVICE void load_b(Rsrc rsrc,const BPairs& p,index_t stage,Stage (&b)[4]) const {
  uint32_t off=stage*kStageK;static_for<0,4,1>{}([&](auto nn){constexpr index_t n=decltype(nn)::value;
   load_b_one(rsrc,p.p[n],off,b[n]);});
 }
 template<index_t Offset> CK_TILE_DEVICE static void read_a(uint32_t base,Stage& a){
  asm volatile("ds_read_b128 %0, %1 offset:%c2":"=&v"(a.d4):"v"(base),"n"(Offset):"memory");
 }
 template<index_t Slot> CK_TILE_DEVICE void consume(uint32_t base,const Stage (&b)[4],AccTile& acc) const {
  using namespace grouped_int8_nt_wasp_detail;Stage a[2][AReadBatch];
  static_for<0,AReadBatch,1>{}([&](auto ii){constexpr index_t i=decltype(ii)::value;read_a<Slot+i*1024>(base,a[0][i]);});
  static_for<0,8/AReadBatch,1>{}([&](auto gg){constexpr index_t g=decltype(gg)::value,s=g&1,first=g*AReadBatch;
   asm volatile("s_waitcnt lgkmcnt(0)":::"memory");
   if constexpr(g+1<8/AReadBatch)static_for<0,AReadBatch,1>{}([&](auto ii){constexpr index_t i=decltype(ii)::value;read_a<Slot+(first+AReadBatch+i)*1024>(base,a[s^1][i]);});
   static_for<0,2,1>{}([&](auto kk){constexpr index_t k=decltype(kk)::value;
    static_for<0,AReadBatch,1>{}([&](auto ii){constexpr index_t i=decltype(ii)::value;
     static_for<0,4,1>{}([&](auto nn){constexpr index_t n=decltype(nn)::value;
      acc.v[(first+i)*4+n]=mmac_i8(a[s][i].frag[k],b[n].frag[k],acc.v[(first+i)*4+n]);});});});
  });
 }
 template<index_t Cur> CK_TILE_DEVICE void stage_body(index_t stage,index_t loops,uint32_t base,const ASource& src,index_t wave,Rsrc br,const BPairs& pairs,Stage (&b)[2][4],AccTile& acc) const {
  // Current A/B are complete at entry. Prefetch only into the released opposite slot.
  if(stage+1<loops){produce(src,stage+1,wave);load_b(br,pairs,stage+1,b[Cur^1]);}
  consume<Cur*kAStageBytes>(base,b[Cur],acc);
  // Complete both next-stage VMEM operands, then publish/release LDS collectively.
  __builtin_amdgcn_s_waitcnt(0x0f70);block_sync_lds();
 }
 CK_TILE_DEVICE void run_raw_into(const int8_t* a,index_t lda,const int8_t* b,index_t ldb,index_t b_rows,index_t loops,AccTile& acc,void* smem) const {
  using namespace grouped_int8_nt_wasp_detail;const index_t wave=get_warp_id();const uint32_t lane=get_lane_id(),row=lane&15,col=lane>>4;
  acc=AccTile{};const auto src=make_a_source(a,lda,wave);
  uint32_t abase=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem))+row*64+(col^(row&3))*16;
  uint32_t rec=record_stride(ldb),q=ldb/rec,rem=ldb%rec;BPairs pairs;
  static_for<0,4,1>{}([&](auto nn){constexpr index_t n=decltype(nn)::value;uint32_t nr=wave*64+n*16+row;
   pairs.p[n]=(static_cast<uint64_t>(nr*rem+col*16)<<32)|(nr*q);});
  Rsrc br=wasp_make_buffer_rsrc(b,rec,static_cast<uint32_t>(b_rows)*ldb,true);Stage bv[2][4];
  block_sync_lds();produce(src,0,wave);load_b(br,pairs,0,bv[0]);__builtin_amdgcn_s_waitcnt(0x0f70);block_sync_lds();
  for(index_t s=0;s<loops;s+=2){stage_body<0>(s,loops,abase,src,wave,br,pairs,bv,acc);if(s+1<loops)stage_body<1>(s+1,loops,abase,src,wave,br,pairs,bv,acc);}
 }
};
}
