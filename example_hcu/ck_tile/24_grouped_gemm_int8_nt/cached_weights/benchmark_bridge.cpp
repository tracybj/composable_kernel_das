// SPDX-License-Identifier: MIT
// Test/benchmark adapter only;the production API is ck_tile_int8_nt_cached.h.
#include "ck_tile_int8_nt_cached.h"
#include <hip/hip_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>
namespace {
thread_local std::string error;
void check(hipError_t s){if(s!=hipSuccess)throw std::runtime_error(hipGetErrorString(s));}
struct Benchmark {
 ck_tile_int8_nt_cached_plan_handle_v1 plan=nullptr;
 hipEvent_t start=nullptr,stop=nullptr;
 ~Benchmark(){if(start)(void)hipEventDestroy(start);if(stop)(void)hipEventDestroy(stop);}
 void run(void* stream){if(ck_tile_int8_nt_cached_run_v1(plan,stream))throw std::runtime_error(ck_tile_int8_nt_cached_last_error_v1());}
};
}
extern "C" CK_TILE_INT8_CACHED_API const char* cached_i8_bench_error(){return error.c_str();}
extern "C" CK_TILE_INT8_CACHED_API void* cached_i8_bench_create(ck_tile_int8_nt_cached_plan_handle_v1 plan){
 try{
  if(!plan)throw std::runtime_error("null cached plan");
  auto p=std::make_unique<Benchmark>();p->plan=plan;check(hipEventCreate(&p->start));check(hipEventCreate(&p->stop));return p.release();
 }catch(const std::exception& e){error=e.what();return nullptr;}
}
extern "C" CK_TILE_INT8_CACHED_API float cached_i8_bench_run(void* context,int warm,int repeat,void* opaque_stream){
 try{
  if(!context || warm<0 || repeat<1)throw std::runtime_error("invalid cached benchmark arguments");
  auto& p=*static_cast<Benchmark*>(context);auto stream=reinterpret_cast<hipStream_t>(opaque_stream);
  for(int i=0;i<warm;++i)p.run(opaque_stream);
  check(hipEventRecord(p.start,stream));for(int i=0;i<repeat;++i)p.run(opaque_stream);check(hipEventRecord(p.stop,stream));check(hipEventSynchronize(p.stop));
  float ms=0;check(hipEventElapsedTime(&ms,p.start,p.stop));return ms/repeat;
 }catch(const std::exception& e){error=e.what();return -100;}
}
extern "C" CK_TILE_INT8_CACHED_API void cached_i8_bench_destroy(void* context){delete static_cast<Benchmark*>(context);}
