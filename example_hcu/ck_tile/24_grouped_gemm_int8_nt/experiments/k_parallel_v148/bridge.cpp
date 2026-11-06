// SPDX-License-Identifier: MIT
// Isolated experimental adapter. No changes to production cached ABI/selector.
#include "kernel.hpp"
#include <hip/hip_runtime.h>
using Args=ck_tile::Int8NtKParallelArgs;
struct Plan {Args* args;int m,n,g,variant,output;};
template<int W,int N,int M,int O> void launch(Plan* p,hipStream_t s){
 using P=ck_tile::Int8NtKParallelProblem<W,N>;
 ck_tile::launch_kernel(ck_tile::stream_config{s,false},ck_tile::make_kernel<P::Threads,1>(ck_tile::Int8NtKParallelKernel<P,M,O>{},dim3((p->n+N-1)/N,1,p->g),dim3(P::Threads),0,p->args));
}
template<int W,int N> void select(Plan* p,hipStream_t s){
 if(p->m==1){if(p->output==0)launch<W,N,1,0>(p,s);else if(p->output==1)launch<W,N,1,1>(p,s);else launch<W,N,1,2>(p,s);}
 else {if(p->output==0)launch<W,N,2,0>(p,s);else if(p->output==1)launch<W,N,2,1>(p,s);else launch<W,N,2,2>(p,s);}
}
void run(Plan* p,hipStream_t s){switch(p->variant){case 0:select<8,8>(p,s);break;case 1:select<8,4>(p,s);break;case 2:select<4,8>(p,s);break;default:select<4,4>(p,s);}}
extern "C" void* kp_create(const Args* a,int g,int v,int out){
 if(g<1 || v<0 || v>3 || out<0 || out>2)return nullptr;
 for(int i=0;i<g;++i)if((a[i].m!=1 && a[i].m!=2)||a[i].m!=a[0].m||a[i].n!=a[0].n||a[i].k!=a[0].k||a[i].k<=0||a[i].k%16||a[i].as%16||a[i].bs%16||a[i].as<a[i].k||a[i].bs<a[i].k||a[i].cs<a[i].n)return nullptr;
 auto p=new Plan{nullptr,a[0].m,a[0].n,g,v,out};
 if(hipMalloc(&p->args,g*sizeof(Args))!=hipSuccess){delete p;return nullptr;}
 if(hipMemcpy(p->args,a,g*sizeof(Args),hipMemcpyHostToDevice)!=hipSuccess){hipFree(p->args);delete p;return nullptr;}return p;
}
extern "C" float kp_bench(void* h,int warm,int repeat,void* stream){
 if(!h || warm<0 || repeat<1)return -1;auto p=static_cast<Plan*>(h);auto s=reinterpret_cast<hipStream_t>(stream);
 hipEvent_t a,b;if(hipEventCreate(&a)!=hipSuccess)return -2;if(hipEventCreate(&b)!=hipSuccess){hipEventDestroy(a);return -2;}
 for(int i=0;i<warm;++i)run(p,s);
 hipEventRecord(a,s);for(int i=0;i<repeat;++i)run(p,s);hipEventRecord(b,s);
 auto status=hipEventSynchronize(b);float ms=-3;if(status==hipSuccess)hipEventElapsedTime(&ms,a,b);
 hipEventDestroy(a);hipEventDestroy(b);return ms<0?ms:ms/repeat;
}
extern "C" void kp_destroy(void* p){if(p){auto q=static_cast<Plan*>(p);hipFree(q->args);delete q;}}
