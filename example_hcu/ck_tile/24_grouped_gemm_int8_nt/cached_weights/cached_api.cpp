// SPDX-License-Identifier: MIT
#include "ck_tile_int8_nt_cached.h"
#include "pack_kernel.hpp"
#include "selector.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_selector.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"
#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int ck_tile_int8_cached_launch_private(
 unsigned,unsigned,const void*,int,int,int,hipStream_t);
extern "C" int ck_tile_int8_cached_pack_private(
 const ck_tile::Int8CachedWeightPackArgs*,int,int,int,hipStream_t);

namespace {
thread_local std::string last_error;
struct ApiError : std::runtime_error {
 int status;
 ApiError(int s,const char* message):std::runtime_error(message),status(s){}
};
void require(bool ok,int status,const char* message){if(!ok)throw ApiError(status,message);}
void hip_check(hipError_t error){if(error!=hipSuccess)throw ApiError(CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1,hipGetErrorString(error));}
template<class F> int api(F f){
 try{last_error.clear();f();return CK_TILE_INT8_CACHED_SUCCESS_V1;}
 catch(const ApiError& e){last_error=e.what();return e.status;}
 catch(const std::exception& e){last_error=e.what();return CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;}
 catch(...){last_error="unknown INT8 cached API failure";return CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1;}
}
template<class T> void header(const T* value){
 require(value!=nullptr,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"null descriptor");
 require(value->abi_version==1,CK_TILE_INT8_CACHED_BAD_VERSION_V1,"unsupported cached ABI version");
 require(value->struct_size==sizeof(T),CK_TILE_INT8_CACHED_BAD_STRUCT_SIZE_V1,"wrong cached descriptor size");
 require(value->reserved[0]==0 && value->reserved[1]==0,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"nonzero reserved descriptor field");
}
bool aligned(const void* p,size_t alignment){return (reinterpret_cast<uintptr_t>(p)&(alignment-1))==0;}
void current_device_is(int expected){int device=-1;hip_check(hipGetDevice(&device));require(device==expected,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"cached handle belongs to another current device");}
using KArg=ck_tile::GemmTransKernelArg;
}

struct ck_tile_int8_nt_cached_weights_impl {
 int device=-1,g=0,n=0,k=0;
 bool packable=false,queued=false,ready_recorded=false;
 std::vector<ck_tile_int8_nt_cached_weight_desc_v1> raw;
 int8_t* packed=nullptr;size_t packed_bytes=0;
 ck_tile::Int8CachedWeightPackArgs* host_pack_args=nullptr;
 ck_tile::Int8CachedWeightPackArgs* device_pack_args=nullptr;
 hipEvent_t ready=nullptr;hipStream_t prepare_stream=nullptr;
 std::atomic<unsigned> plans{0};
 ~ck_tile_int8_nt_cached_weights_impl(){
  if(ready_recorded)(void)hipEventSynchronize(ready);
  else if(queued)(void)hipStreamSynchronize(prepare_stream);
  if(device_pack_args)(void)hipFree(device_pack_args);
  if(host_pack_args)(void)hipHostFree(host_pack_args);
  if(packed)(void)hipFree(packed);
  if(ready)(void)hipEventDestroy(ready);
 }
};
struct ck_tile_int8_nt_cached_plan_impl {
 ck_tile_int8_nt_cached_weights_impl* weights;
 KArg* args=nullptr;
 int g=0,max_m=0,launch_m=1,n=0,k=0;
 unsigned route=0,legacy_instance=0;
 std::vector<hipStream_t> streams;
 std::mutex mutex;
 explicit ck_tile_int8_nt_cached_plan_impl(ck_tile_int8_nt_cached_weights_impl* w):weights(w){weights->plans.fetch_add(1);}
 ~ck_tile_int8_nt_cached_plan_impl(){
  for(auto stream:streams)(void)hipStreamSynchronize(stream);
  if(args)(void)hipFree(args);
  weights->plans.fetch_sub(1);
 }
};

extern "C" CK_TILE_INT8_CACHED_API const char* ck_tile_int8_nt_cached_last_error_v1(){return last_error.c_str();}
extern "C" CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_weights_create_v1(
 const ck_tile_int8_nt_cached_weight_desc_v1* source,int32_t g,void* opaque_stream,
 ck_tile_int8_nt_cached_weights_handle_v1* result){return api([&]{
 require(result!=nullptr,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"null weights result");*result=nullptr;
 require(source && g>0 && g<=65535,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"invalid weights/group count");
 int device=-1;hip_check(hipGetDevice(&device));hipDeviceProp_t prop{};hip_check(hipGetDeviceProperties(&prop,device));
 require(std::strncmp(prop.gcnArchName,"gfx936",6)==0,CK_TILE_INT8_CACHED_UNSUPPORTED_ARCH_V1,"INT8 cached kernels require gfx936");
 for(int i=0;i<g;++i){
  const auto& w=source[i];header(&w);
  require(w.reserved0==0 && w.data && aligned(w.data,16) && w.n>0 && w.n<=1048576 && w.n%4==0 && w.k>=256 && w.k<=131056 && w.k%16==0 && w.row_stride>=w.k && w.row_stride%16==0,
          CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"invalid signed INT8 NT weight descriptor/alignment");
  require(w.n==source[0].n && w.k==source[0].k,CK_TILE_INT8_CACHED_UNSUPPORTED_V1,"v1 cached weights require common N/K");
  require(int64_t(w.n)*w.row_stride<=std::numeric_limits<int32_t>::max(),CK_TILE_INT8_CACHED_UNSUPPORTED_V1,"weight span exceeds INT8 buffer domain");
 }
 auto w=std::make_unique<ck_tile_int8_nt_cached_weights_impl>();w->device=device;w->g=g;w->n=source[0].n;w->k=source[0].k;w->raw.assign(source,source+g);
 // The v1 policy never consumes packed weights belowK512 or outside this
 // native N/K domain;avoid allocating an unused copy for those families.
 w->packable=w->n%256==0 && w->k%128==0 && w->k>=512 && w->k<16384;w->prepare_stream=reinterpret_cast<hipStream_t>(opaque_stream);
 hip_check(hipEventCreateWithFlags(&w->ready,hipEventDisableTiming));
 if(w->packable){
  const uint64_t per_group=uint64_t(w->n)*w->k;
  require(per_group<=std::numeric_limits<size_t>::max()/uint64_t(g),CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"packed byte size overflow");
  w->packed_bytes=static_cast<size_t>(per_group*g);hip_check(hipMalloc(reinterpret_cast<void**>(&w->packed),w->packed_bytes));
  const size_t arg_bytes=sizeof(ck_tile::Int8CachedWeightPackArgs)*g;
  hip_check(hipHostMalloc(reinterpret_cast<void**>(&w->host_pack_args),arg_bytes,hipHostMallocDefault));
  hip_check(hipMalloc(reinterpret_cast<void**>(&w->device_pack_args),arg_bytes));
  for(int i=0;i<g;++i)w->host_pack_args[i]={source[i].data,w->packed+uint64_t(i)*per_group,w->n,w->k,source[i].row_stride};
  w->queued=true;hip_check(hipMemcpyAsync(w->device_pack_args,w->host_pack_args,arg_bytes,hipMemcpyHostToDevice,w->prepare_stream));
  require(ck_tile_int8_cached_pack_private(w->device_pack_args,g,w->n,w->k,w->prepare_stream)==0,CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1,"INT8 weight packing launch failed");
 }
 w->queued=true;hip_check(hipEventRecord(w->ready,w->prepare_stream));w->ready_recorded=true;*result=w.release();
 });}
extern "C" CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_weights_query_v1(
 ck_tile_int8_nt_cached_weights_handle_v1 w,ck_tile_int8_nt_cached_weights_info_v1* info){return api([&]{
 require(w,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"null weights handle");header(info);current_device_is(w->device);
 ck_tile_int8_nt_cached_weights_info_v1 value{};value.struct_size=sizeof(value);value.abi_version=1;
 value.group_count=w->g;value.n=w->n;value.k=w->k;value.format=w->packable?1:0;value.packed_device_data=w->packed;value.packed_bytes=w->packed_bytes;*info=value;
 });}
extern "C" CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_weights_destroy_v1(
 ck_tile_int8_nt_cached_weights_handle_v1 w){return api([&]{
 if(!w)return;current_device_is(w->device);require(w->plans.load()==0,CK_TILE_INT8_CACHED_BUSY_V1,"cached weights still have live plans");delete w;
 });}

extern "C" CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_plan_create_v1(
 ck_tile_int8_nt_cached_weights_handle_v1 w,const ck_tile_int8_nt_cached_group_desc_v1* groups,
 int32_t g,ck_tile_int8_nt_cached_plan_handle_v1* result){return api([&]{
 require(result!=nullptr,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"null plan result");*result=nullptr;
 require(w && groups && g==w->g,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"invalid cached plan/group count");current_device_is(w->device);
 int max_m=0;int64_t total_m=0;bool uniform=true,empty=false,native=true,raw_original=true,raw_transposed=true;
 for(int i=0;i<g;++i){
  const auto& a=groups[i];header(&a);
  require(a.reserved0==0 && a.m>=0 && a.m<=1048560 && a.a_row_stride>=w->k && a.a_row_stride%16==0 && a.c_row_stride>=w->n && a.c_row_stride%4==0 &&
          (a.m==0 || (a.a && a.c && aligned(a.a,16) && aligned(a.c,16))),CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"invalid INT8 activation/INT32 output descriptor");
  require(int64_t(a.m)*a.a_row_stride<=std::numeric_limits<int32_t>::max() && int64_t(a.m)*a.c_row_stride<=std::numeric_limits<int32_t>::max(),CK_TILE_INT8_CACHED_UNSUPPORTED_V1,"plan matrix span exceeds supported domain");
  uniform&=a.m==groups[0].m;empty|=a.m==0;max_m=std::max(max_m,a.m);total_m+=a.m;
  native&=a.m>0 && (a.m%256==0 || a.m==128 || a.m==64) && int64_t(256)*a.a_row_stride<=std::numeric_limits<int32_t>::max();
  raw_original&=w->raw[i].row_stride<16384;
  raw_transposed&=a.a_row_stride<16384;
 }
 ck_tile::GroupedGemmProblem old{};old.arch=ck_tile::GroupedGemmArch::gfx936;old.data_type=ck_tile::GroupedGemmDataType::int8_int32;old.layout=ck_tile::GroupedGemmLayout::nt;old.group_count=g;
 old.m=uniform?ck_tile::GroupedGemmDimensionSummary::Common(std::max(groups[0].m,1)):ck_tile::GroupedGemmDimensionSummary::DeviceLengthsWithCapacity(std::max(total_m,int64_t(1)));
 old.n=ck_tile::GroupedGemmDimensionSummary::Common(w->n);old.k=ck_tile::GroupedGemmDimensionSummary::Common(w->k);old.may_have_empty_groups=empty;old.group_lengths_are_ragged=!uniform;
 const auto legacy=ck_tile::select_grouped_gemm_candidate(old);require(legacy.supported(),CK_TILE_INT8_CACHED_UNSUPPORTED_V1,"unsupported legacy INT8 NT shape summary");
 auto p=std::make_unique<ck_tile_int8_nt_cached_plan_impl>(w);p->g=g;p->max_m=max_m;p->launch_m=std::max(max_m,1);p->n=w->n;p->k=w->k;p->legacy_instance=static_cast<unsigned>(legacy.instance_id);
 const bool native_domain=uniform && !empty && native && w->n%256==0 && w->k%128==0 && w->k<16384;
 p->route=native_domain?ck_tile::int8_cached_route(groups[0].m,w->n,w->k,g,w->packable):0;
 if(((p->route==1 || p->route==2) && !raw_original) || ((p->route==3 || p->route==4) && !raw_transposed))p->route=0;
 std::vector<KArg> args;args.reserve(g);const uint64_t per_group=uint64_t(w->n)*w->k;
 for(int i=0;i<g;++i){
  const auto& a=groups[i];const bool packed=p->route==CK_TILE_INT8_CACHED_PACKED_WEIGHT_V1 ||
   p->route==CK_TILE_INT8_CACHED_PACKED_M128_N256_A2_V1 ||
   p->route==CK_TILE_INT8_CACHED_PACKED_M128_N128_A4_V1 ||
   p->route==CK_TILE_INT8_CACHED_PACKED_M64_N256_A2_V1 ||
   p->route==CK_TILE_INT8_CACHED_PACKED_M64_N64_A2_V1;
  const void* b=packed?w->packed+uint64_t(i)*per_group:w->raw[i].data;
  ck_tile::UniversalGemmKernelArgs<1,1,0> arg{{a.a},{b},{},a.c,a.m,w->n,w->k,{a.a_row_stride},{packed?w->k:w->raw[i].row_stride},{},a.c_row_stride,1};args.emplace_back(std::move(arg));
 }
 hip_check(hipMalloc(reinterpret_cast<void**>(&p->args),sizeof(KArg)*g));hip_check(hipMemcpy(p->args,args.data(),sizeof(KArg)*g,hipMemcpyHostToDevice));*result=p.release();
 });}
extern "C" CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_plan_query_v1(
 ck_tile_int8_nt_cached_plan_handle_v1 p,ck_tile_int8_nt_cached_plan_info_v1* info){return api([&]{
 require(p,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"null plan handle");header(info);current_device_is(p->weights->device);
 ck_tile_int8_nt_cached_plan_info_v1 value{};value.struct_size=sizeof(value);value.abi_version=1;value.group_count=p->g;value.max_m=p->max_m;value.n=p->n;value.k=p->k;value.route=p->route;value.legacy_instance=p->legacy_instance;value.packed_weight_bytes=p->weights->packed_bytes;*info=value;
 });}
extern "C" CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_run_v1(
 ck_tile_int8_nt_cached_plan_handle_v1 p,void* opaque_stream){return api([&]{
 require(p,CK_TILE_INT8_CACHED_INVALID_ARGUMENT_V1,"null plan handle");current_device_is(p->weights->device);std::lock_guard<std::mutex> lock(p->mutex);
 const auto stream=reinterpret_cast<hipStream_t>(opaque_stream);
 if(std::find(p->streams.begin(),p->streams.end(),stream)==p->streams.end()){
  hip_check(hipStreamWaitEvent(stream,p->weights->ready,0));p->streams.push_back(stream);
 }
 require(ck_tile_int8_cached_launch_private(p->route,p->legacy_instance,p->args,p->g,p->launch_m,p->n,stream)==0,CK_TILE_INT8_CACHED_INTERNAL_ERROR_V1,"INT8 cached plan launch failed");
 });}
extern "C" CK_TILE_INT8_CACHED_API int ck_tile_int8_nt_cached_plan_destroy_v1(
 ck_tile_int8_nt_cached_plan_handle_v1 p){return api([&]{if(!p)return;current_device_is(p->weights->device);delete p;});}
