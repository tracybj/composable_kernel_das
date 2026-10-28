// SPDX-License-Identifier: MIT
// Hardware-layout probe; not a production GEMM or timing result.
#include "ck_tile/core.hpp"
#include "ck_tile/core/arch/hcu_buffer_addressing.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include <hip/hip_runtime.h>
namespace ck_tile {
struct DirectLdsCopyProblem { static constexpr index_t Threads=64,Elements=256; };
template<class P=DirectLdsCopyProblem>
struct DirectLdsCopyProbeKernel
{
    CK_TILE_DEVICE void operator()(const uint32_t* src,uint32_t* dst,int mode) const
    {
        const index_t tid=get_thread_id(),row=tid/4,vec=tid%4;
        __shared__ __align__(16) uint32_t smem[P::Elements];
        for(index_t i=tid;i<P::Elements;i+=P::Threads) smem[i]=0x5a5a5a5a;
        block_sync_lds();
        const auto resource=make_wave_buffer_resource(src,P::Elements*sizeof(uint32_t));
        const index_t source_vec=mode==1?row*4+(vec^(row%4)):tid;
        hcu_async_buffer_load_asm<uint32_t,4>(smem+tid*4,resource,source_vec*16,0,mode!=2 || tid%2==0);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        block_sync_lds();
        const index_t physical_vec=mode==1?row*4+(vec^(row%4)):tid;
        static_for<0,4,1>{}([&](auto i){dst[tid*4+i]=smem[physical_vec*4+i];});
    }
};
}
extern "C" int direct_lds_copy_probe(const void* src,void* dst,int mode)
{
    ck_tile::launch_kernel(ck_tile::stream_config{nullptr,false},
        ck_tile::make_kernel<64,1>(ck_tile::DirectLdsCopyProbeKernel<>{},dim3(1),dim3(64),0,
            static_cast<const uint32_t*>(src),static_cast<uint32_t*>(dst),mode));
    const auto status=hipDeviceSynchronize();
    return status==hipSuccess?0:static_cast<int>(status);
}
