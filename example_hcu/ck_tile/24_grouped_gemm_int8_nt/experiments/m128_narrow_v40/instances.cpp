// SPDX-License-Identifier: MIT
#define ck_tile_grouped_int8_nt_launch baseline_launch
#include "../../instances.cpp"
#undef ck_tile_grouped_int8_nt_launch
extern "C" int ck_tile_grouped_int8_nt_launch(unsigned id,const void* args,int g,int m,int n,void* stream)
{
 if(!args || g<=0 || g>65535 || m<=0 || n<=0)return -1;
 auto s=reinterpret_cast<hipStream_t>(stream);
 switch(id){
 case 8501:return launch<ck_tile::GroupedInt8NtTileProblem<128,64,128,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8502:return launch<ck_tile::GroupedInt8NtTileProblem<128,64,256,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8503:return launch<ck_tile::GroupedInt8NtTileProblem<128,64,256,4,1,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8504:return launch<ck_tile::GroupedInt8NtTileProblem<128,32,256,4,1,false,false,true,true,1,true>>(args,g,m,n,s);
 case 8505:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,128,4,1,false,false,true,true,1,true>>(args,g,m,n,s);
 case 7703:return launch<ck_tile::GroupedInt8NtTileProblem<128,128,128,2,2,false,false,true,true,1,true>>(args,g,m,n,s);
 default:return baseline_launch(id,args,g,m,n,stream);
 }
}
