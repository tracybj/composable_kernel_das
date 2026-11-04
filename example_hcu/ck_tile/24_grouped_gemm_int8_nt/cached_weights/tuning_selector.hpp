// SPDX-License-Identifier: MIT
#pragma once
#include <cstdlib>
#include <stdexcept>

namespace ck_tile {
// Built only with CK_INT8_CACHED_TUNING=ON. These environment controls select
// a precompiled instance for measurement, never a workload/case ID. Production
// builds do not include this override and retain the normal dimension policy.
inline unsigned int8_tuning_uint(const char* value,unsigned maximum) {
    char* end=nullptr;
    const unsigned long parsed=std::strtoul(value,&end,10);
    if(value==end || *end || parsed>maximum)
        throw std::invalid_argument("invalid INT8 tuning instance override");
    return static_cast<unsigned>(parsed);
}

inline void int8_tuning_override(bool uniform,bool empty,bool native,bool packed,
                                bool raw_original,bool raw_transposed,int m,int n,int k,
                                unsigned& route,unsigned& legacy) {
    const char* route_value=std::getenv("CK_INT8_TUNE_ROUTE");
    const char* legacy_value=std::getenv("CK_INT8_TUNE_LEGACY");
    if(!route_value && !legacy_value)return;
    if(!uniform || empty)throw std::invalid_argument("tuning requires nonempty uniform groups");
    route=route_value?int8_tuning_uint(route_value,17):0;
    if(legacy_value) {
        if(route)throw std::invalid_argument("legacy tuning requires route0");
        legacy=int8_tuning_uint(legacy_value,10000);
        int bm=1,bn=4,bk=16;
        bool aligned=false;
        switch(legacy) {
        case 6005:case 6006:case 6007:case 6008:
        case 6043:case 6044:case 6050:
        case 6401:case 6403:case 6404:case 6505:
        case 7501:case 7502:case 7504:case 7603:case 7604:case 8734:break;
        case 9001:case 9002:case 9003:case 9004:case 9005:case 9006:
        case 9007:case 9008:case 9009:case 9010:case 9011:case 9012:break;
        case 6035:case 6049:bm=128;bn=128;bk=64;aligned=true;break;
        case 6041:case 6048:bm=256;bn=256;bk=64;aligned=true;break;
        case 6042:bm=128;bn=256;bk=64;aligned=true;break;
        case 6061:bm=16;bn=256;bk=64;aligned=true;break;
        default:throw std::invalid_argument("unlisted cooperative tuning instance");
        }
        if(aligned && (m%bm || n%bn || k%bk))
            throw std::invalid_argument("aligned tuning instance has a partial tile");
    }
    if(!route)return;
    if(!native)throw std::invalid_argument("WASP tuning requires the native domain");
    const int bm=route==6 || route==7 || route>=13?128:route==8 || route==9?64:route==10 || route==11?32:256;
    if(m%bm)throw std::invalid_argument("WASP tuning requires complete M tiles");
    if((route==1 || route==2) && !raw_original)
        throw std::invalid_argument("raw WASP weight stride is unsupported");
    if((route==3 || route==4) && !raw_transposed)
        throw std::invalid_argument("transposed WASP activation stride is unsupported");
    if(route>=5 && !packed)throw std::invalid_argument("packed tuning requires prepared weights");
}
} // namespace ck_tile
