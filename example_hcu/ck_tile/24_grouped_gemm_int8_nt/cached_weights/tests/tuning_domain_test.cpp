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
    check("43",256,4096,7168,true,true,false);
    check("56",256,4096,7168,true,true,false);
    check("-1",256,4096,7168,true,true,false);
    check("36x",256,4096,7168,true,true,false);
    check("36",1,4096,7168,false,true,false,false,false);
    check("36",1,4096,7168,false,true,false,true,true);
    unsetenv("CK_INT8_TUNE_ROUTE");
    std::cout<<"TUNING_FULLNK_GRID_DOMAIN_PASS checks="<<checks<<"\n";
}
