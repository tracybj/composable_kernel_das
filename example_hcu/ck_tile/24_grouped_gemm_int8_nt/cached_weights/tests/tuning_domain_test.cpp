// CPU-only safety gates for experimental FullNK and grid-order instances.
#include "tuning_selector.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main(){
    unsigned checks=0;
    auto check=[&](const char* value,int m,int n,int k,bool native,bool packed,bool accept,
                   bool uniform=true,bool empty=false){
        setenv("CK_INT8_TUNE_ROUTE",value,1);unsetenv("CK_INT8_TUNE_LEGACY");
        unsigned route=0,legacy=0;bool accepted=true;
        try{ck_tile::int8_tuning_override(uniform,empty,native,packed,true,true,m,n,k,route,legacy);}
        catch(const std::invalid_argument&){accepted=false;}
        assert(accepted==accept);++checks;
    };
    check("36",1,4096,7168,false,true,true);
    check("36",1,4096,768,false,true,false);
    check("32",17,512,768,false,true,true);
    check("37",17,512,768,false,true,true);
    check("39",1,4096,768,false,true,false);
    check("38",129,512,8192,false,true,true);
    check("36",1,4096,7168,false,false,false);
    check("36",1,260,512,false,true,false);
    check("38",1,260,512,false,true,false);
    check("48",128,1792,1536,true,true,false);
    check("48",256,1792,1536,true,true,true);
    check("55",256,1792,1536,true,true,true);
    check("55",256,1792,1536,false,true,false);
    check("43",256,4096,7168,true,true,true);
    check("40",1,4096,7168,false,true,true);
    check("40",17,512,768,false,true,false);
    check("41",17,512,768,false,true,true);
    check("42",17,512,768,false,false,false);
    check("40",1,65536,65536,false,true,false);
    check("44",256,4096,7168,true,true,false);
    check("56",256,4096,7168,true,true,true);
    check("58",256,256,512,true,true,true);
    check("58",128,5120,2304,true,true,false);
    check("60",256,1792,1536,true,true,true);
    check("60",256,1792,1536,true,false,false);
    check("61",256,4096,7168,true,true,false);
    check("22",256,4096,7168,true,true,true);
    check("23",128,4096,7168,true,true,false);
    check("23",256,4096,7168,true,false,false);
    check("24",256,4096,7168,true,true,false);
    check("-1",256,4096,7168,true,true,false);
    check("36x",256,4096,7168,true,true,false);
    check("36",1,4096,7168,false,true,false,false,false);
    check("36",1,4096,7168,false,true,false,true,true);
    unsetenv("CK_INT8_TUNE_ROUTE");
    std::cout<<"TUNING_FULLNK_GRID_DOMAIN_PASS checks="<<checks<<"\n";
}
