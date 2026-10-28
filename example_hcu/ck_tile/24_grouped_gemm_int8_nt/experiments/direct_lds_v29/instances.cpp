// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
#include "../direct_lds_v28/kernel.hpp"
#include "kernel.hpp"
namespace {
enum class BankInstance:unsigned {old_ping=6702,shift1_ping=6802,shift2_ping=6902,
    shift3_ping=7002,shift2_merged=6903,shift2_producers=6904,shift2_group8=6905};
template<class P,template<class> class Kernel>
int launch_bank(const void* args,int g,int m,int n,hipStream_t stream)
{
    const int mt=(m+P::BM-1)/P::BM,nt=(n+P::BN-1)/P::BN;
    dim3 grid(((mt+P::MGroup-1)/P::MGroup)*P::MGroup*nt,1,g);
    ck_tile::launch_kernel(ck_tile::stream_config{stream,false},
        ck_tile::make_kernel<P::Threads,1>(Kernel<P>{},grid,dim3(P::Threads),0,
            static_cast<const ck_tile::GemmTransKernelArg*>(args)));
    return hipGetLastError()==hipSuccess?0:-100;
}
}
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned instance_id,const void* args,int g,int m,int n,void* stream)
{
    if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
    auto s=reinterpret_cast<hipStream_t>(stream);
    switch(static_cast<BankInstance>(instance_id))
    {
        case BankInstance::old_ping:
            return launch_bank<ck_tile::GroupedInt8NtD2LProblem<true,false>,ck_tile::GroupedInt8NtD2LKernel>(args,g,m,n,s);
        case BankInstance::shift1_ping:
            return launch_bank<ck_tile::GroupedInt8NtD2LBankProblem<true,false,0,1,1>,ck_tile::GroupedInt8NtD2LBankKernel>(args,g,m,n,s);
        case BankInstance::shift2_ping:
            return launch_bank<ck_tile::GroupedInt8NtD2LBankProblem<true,false,0,1,2>,ck_tile::GroupedInt8NtD2LBankKernel>(args,g,m,n,s);
        case BankInstance::shift3_ping:
            return launch_bank<ck_tile::GroupedInt8NtD2LBankProblem<true,false,0,1,3>,ck_tile::GroupedInt8NtD2LBankKernel>(args,g,m,n,s);
        case BankInstance::shift2_merged:
            return launch_bank<ck_tile::GroupedInt8NtD2LBankProblem<true,true,0,1,2>,ck_tile::GroupedInt8NtD2LBankKernel>(args,g,m,n,s);
        case BankInstance::shift2_producers:
            return launch_bank<ck_tile::GroupedInt8NtD2LBankProblem<true,true,4,1,2>,ck_tile::GroupedInt8NtD2LBankKernel>(args,g,m,n,s);
        case BankInstance::shift2_group8:
            return launch_bank<ck_tile::GroupedInt8NtD2LBankProblem<true,true,4,8,2>,ck_tile::GroupedInt8NtD2LBankKernel>(args,g,m,n,s);
        default:return baseline_launch(instance_id,args,g,m,n,stream);
    }
}
