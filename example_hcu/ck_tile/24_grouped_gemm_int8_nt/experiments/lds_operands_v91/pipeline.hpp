// SPDX-License-Identifier: MIT
// Raw-NT experiment: cooperative A+B staging into swizzled LDS; no external packing.
#pragma once
#include "../../grouped_int8_nt_wasp_pipeline.hpp"
namespace ck_tile {
struct Int8LdsOperandsAccTile {
 static constexpr index_t kMIters=16,kNIters=2;
 grouped_int8_nt_wasp_detail::frag_c_t v[kMIters*kNIters];
};
template<class Problem,index_t AReadBatch=1>
struct GemmPipelineInt8LdsOperands : GemmPipelineAgBgCrCompV3<Problem,GemmPipelineAgBgCrCompV4DefaultPolicy> {
 using Shape=remove_cvref_t<typename Problem::BlockGemmShape>;using AccTile=Int8LdsOperandsAccTile;
 using Stage=grouped_int8_nt_wasp_detail::stage16_t;using Rsrc=grouped_int8_nt_wasp_detail::buffer_rsrc_t;
 static constexpr index_t BlockSize=512,kStageK=64,kAStageBytes=16384,kStageBytes=32768,kWaspLdsBytes=65536;
 static_assert(Shape::kM==256 && Shape::kN==256 && Shape::kK==64 && Shape::NumWarps==8);
 static_assert(AReadBatch==1 || AReadBatch==4);
 CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize(){return kWaspLdsBytes;}
 struct Source {Rsrc rsrc;uint32_t v0,v1;};
 CK_TILE_DEVICE Source make_source(const int8_t* ptr,index_t pitch,index_t wave) const {
  using namespace grouped_int8_nt_wasp_detail;uint32_t lane=get_lane_id(),row=lane>>2,chunk=lane&3;
  uint32_t k=(chunk^(row&3))*16;
  return {wasp_make_buffer_rsrc(ptr,0,256u*pitch,false),(wave*32+row)*pitch+k,(wave*32+16+row)*pitch+k};
 }
 template<bool IsB> CK_TILE_DEVICE void produce(const Source& s,index_t stage,index_t wave) const {
  uint32_t m0=(stage&1)*kStageBytes+(IsB?kAStageBytes:0)+wave*2048,off=stage*kStageK;
  asm volatile("s_mov_b32 m0, %0\n\tbuffer_load_dwordx4 %1, %3, %4 offen lds\n\ts_add_u32 m0, m0, 0x400\n\tbuffer_load_dwordx4 %2, %3, %4 offen lds\n"::"s"(m0),"v"(s.v0),"v"(s.v1),"s"(s.rsrc),"s"(off):"memory");
 }
 template<index_t Offset> CK_TILE_DEVICE static void read_lds(uint32_t base,Stage& a){
  static_assert(Offset<=65535);
  asm volatile("ds_read_b128 %0, %1 offset:%c2":"=&v"(a.d4):"v"(base),"n"(Offset):"memory");
 }
 template<index_t Slot> CK_TILE_DEVICE void consume(uint32_t abase,uint32_t bbase,AccTile& acc) const {
  using namespace grouped_int8_nt_wasp_detail;Stage a[2][AReadBatch],b[2];
  read_lds<Slot>(bbase,b[0]);read_lds<Slot+1024>(bbase,b[1]);
  static_for<0,AReadBatch,1>{}([&](auto ii){constexpr index_t i=decltype(ii)::value;read_lds<Slot+i*1024>(abase,a[0][i]);});
  static_for<0,16/AReadBatch,1>{}([&](auto gg){constexpr index_t g=decltype(gg)::value,s=g&1,first=g*AReadBatch;
   asm volatile("s_waitcnt lgkmcnt(0)":::"memory");
   if constexpr(g+1<16/AReadBatch)static_for<0,AReadBatch,1>{}([&](auto ii){constexpr index_t i=decltype(ii)::value;read_lds<Slot+(first+AReadBatch+i)*1024>(abase,a[s^1][i]);});
   static_for<0,2,1>{}([&](auto kk){constexpr index_t k=decltype(kk)::value;
    static_for<0,AReadBatch,1>{}([&](auto ii){constexpr index_t i=decltype(ii)::value;
     static_for<0,2,1>{}([&](auto nn){constexpr index_t n=decltype(nn)::value;
      acc.v[(first+i)*2+n]=mmac_i8(a[s][i].frag[k],b[n].frag[k],acc.v[(first+i)*2+n]);});});});
  });
 }
 template<index_t Cur> CK_TILE_DEVICE void stage_body(index_t stage,index_t loops,uint32_t abase,uint32_t bbase,const Source& as,const Source& bs,index_t wave,AccTile& acc) const {
  // Current A/B are complete at entry. Opposite LDS slot has been released collectively.
  if(stage+1<loops){produce<false>(as,stage+1,wave);produce<true>(bs,stage+1,wave);}
  consume<Cur*kStageBytes>(abase,bbase,acc);
  __builtin_amdgcn_s_waitcnt(0x0f70);block_sync_lds();
 }
 CK_TILE_DEVICE void run_raw_into(const int8_t* a,index_t lda,const int8_t* b,index_t ldb,index_t b_rows,index_t loops,AccTile& acc,void* smem) const {
  const index_t wave=get_warp_id();const uint32_t lane=get_lane_id(),row=lane&15,col=lane>>4;
  acc=AccTile{};const auto as=make_source(a,lda,wave),bs=make_source(b,ldb,wave);(void)b_rows;
  uint32_t base=static_cast<uint32_t>(reinterpret_cast<uintptr_t>(smem));
  uint32_t abase=base+row*64+(col^(row&3))*16,bbase=base+kAStageBytes+(wave*32+row)*64+(col^(row&3))*16;
  block_sync_lds();produce<false>(as,0,wave);produce<true>(bs,0,wave);__builtin_amdgcn_s_waitcnt(0x0f70);block_sync_lds();
  for(index_t s=0;s<loops;s+=2){stage_body<0>(s,loops,abase,bbase,as,bs,wave,acc);if(s+1<loops)stage_body<1>(s+1,loops,abase,bbase,as,bs,wave,acc);}
 }
};
}
