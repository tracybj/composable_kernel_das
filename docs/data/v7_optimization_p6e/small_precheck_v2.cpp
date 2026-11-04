// Variant argument: 32=K32, 16=K16 drain; both logical N tile128.
// Experimental mapping precheck only; no performance data from this program.
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "grouped_gemm.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_k32_device_args(void*,int,std::uint32_t,hipStream_t,std::uint64_t,std::int32_t);
extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_drain_device_args(void*,int,std::uint32_t,hipStream_t,std::uint64_t,std::int32_t);
extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_square128_device_args(void*,int,std::uint32_t,hipStream_t,std::uint64_t,std::int32_t);
extern "C" int ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_k32n64_device_args(void*,int,std::uint32_t,hipStream_t,std::uint64_t,std::int32_t);
static void check(hipError_t s) { if(s!=hipSuccess) {std::fprintf(stderr,"HIP %s\n",hipGetErrorString(s));std::exit(3);} }
static std::uint16_t bf16(float x) {std::uint32_t v;std::memcpy(&v,&x,4);v+=0x7fff+((v>>16)&1);return static_cast<std::uint16_t>(v>>16);}
int main(int argc,char** argv)
{
    if(argc!=6) return 2;
    const int variant=std::atoi(argv[1]),m=std::atoi(argv[2]),n=std::atoi(argv[3]),k=std::atoi(argv[4]),g=std::atoi(argv[5]);
    std::vector<std::uint16_t> a(static_cast<std::size_t>(g)*m*k),b(static_cast<std::size_t>(g)*k*n),d(static_cast<std::size_t>(g)*m*n+32,0x5aa5);
    for(int group=0;group<g;++group)
    {
        for(int row=0;row<m;++row) for(int red=0;red<k;++red)
            a[(static_cast<std::size_t>(group)*m+row)*k+red]=bf16(static_cast<float>((group*3+row*5+red*7)%17-8)/8);
        for(int red=0;red<k;++red) for(int col=0;col<n;++col)
            b[(static_cast<std::size_t>(group)*k+red)*n+col]=bf16(static_cast<float>((group*7+red*11+col*3)%17-8)/8);
    }
    void *da=nullptr,*db=nullptr,*dd=nullptr,*dp=nullptr;
    check(hipMalloc(&da,a.size()*2));check(hipMalloc(&db,b.size()*2));check(hipMalloc(&dd,d.size()*2));
    check(hipMemcpy(da,a.data(),a.size()*2,hipMemcpyHostToDevice));check(hipMemcpy(db,b.data(),b.size()*2,hipMemcpyHostToDevice));check(hipMemcpy(dd,d.data(),d.size()*2,hipMemcpyHostToDevice));
    std::vector<ck_tile::GemmTransKernelArg> args;
    args.reserve(static_cast<std::size_t>(g));
    for(int group=0;group<g;++group)
    {
        args.emplace_back(ck_tile::UniversalGemmKernelArgs<>{
            {static_cast<std::uint16_t*>(da)+static_cast<std::size_t>(group)*m*k},
            {static_cast<std::uint16_t*>(db)+static_cast<std::size_t>(group)*k*n}, {},
            static_cast<std::uint16_t*>(dd)+16+static_cast<std::size_t>(group)*m*n,
            m,n,k,{k},{n},{},n,1});
    }
    check(hipMalloc(&dp,args.size()*sizeof(args[0])));check(hipMemcpy(dp,args.data(),args.size()*sizeof(args[0]),hipMemcpyHostToDevice));
    const int tm=variant==128?128:64, tn=variant==64?64:128;
    const int tpg=((m+tm-1)/tm)*((n+tn-1)/tn);
    const int rc=variant==128 ? ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_square128_device_args(dp,g,0,nullptr,static_cast<std::uint64_t>(tpg)*g,g>16?tpg:0)
                 : variant==64 ? ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_k32n64_device_args(dp,g,0,nullptr,static_cast<std::uint64_t>(tpg)*g,g>16?tpg:0)
                 : variant==32 ? ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_k32_device_args(dp,g,0,nullptr,static_cast<std::uint64_t>(tpg)*g,g>16?tpg:0)
                         : ck_tile_hcu_grouped_gemm_v7_bf16_nn_small_m_drain_device_args(dp,g,0,nullptr,static_cast<std::uint64_t>(tpg)*g,g>16?tpg:0);
    if(rc!=0) return 4;
    check(hipDeviceSynchronize());check(hipMemcpy(d.data(),dd,d.size()*2,hipMemcpyDeviceToHost));
    int bad=0;
    for(int group=0;group<g;++group) for(int row=0;row<m;++row) for(int col=0;col<n;++col)
    {
        int sum=0;for(int red=0;red<k;++red) sum+=((group*3+row*5+red*7)%17-8)*((group*7+red*11+col*3)%17-8);
        const auto expected=bf16(static_cast<float>(sum)/64);
        const auto actual=d[16+(static_cast<std::size_t>(group)*m+row)*n+col];
        if(actual!=expected) {if(bad<12)std::printf("BAD g=%d m=%d n=%d got=%04x expected=%04x\n",group,row,col,actual,expected);++bad;}
    }
    int guards=0;for(int i=0;i<16;++i) guards+=(d[i]!=0x5aa5)+(d[d.size()-16+i]!=0x5aa5);
    std::printf("PRECHECK variant=%d M=%d N=%d K=%d G=%d bad=%d guards=%d\n",variant,m,n,k,g,bad,guards);
    check(hipFree(dp));check(hipFree(dd));check(hipFree(db));check(hipFree(da));
    return bad||guards?1:0;
}
