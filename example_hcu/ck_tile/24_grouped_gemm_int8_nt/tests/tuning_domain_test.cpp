// CPU-only safety gates for experimental FullNK and grid-order instances.
#include "../instances/grouped_int8_nt_tuning_selector.hpp"
#include <cassert>
#include <cstdlib>
#include <iostream>

int main()
{
    unsigned checks = 0;
    auto check      = [&](const char* value,
                     int m,
                     int n,
                     int k,
                     bool native,
                     bool packed,
                     bool accept,
                     bool uniform = true,
                     bool empty   = false) {
        setenv("CK_INT8_TUNE_ROUTE", value, 1);
        unsetenv("CK_INT8_TUNE_LEGACY");
        unsigned route = 0, legacy = 0;
        bool accepted = true;
        try
        {
            ck_tile::int8_tuning_override(
                uniform, empty, native, packed, true, true, m, n, k, route, legacy);
        }
        catch(const std::invalid_argument&)
        {
            accepted = false;
        }
        assert(accepted == accept);
        ++checks;
    };
    check("36", 1, 4096, 7168, false, true, true);
    check("36", 1, 4096, 768, false, true, false);
    check("32", 17, 512, 768, false, true, true);
    check("37", 17, 512, 768, false, true, true);
    check("39", 1, 4096, 768, false, true, false);
    check("38", 129, 512, 8192, false, true, true);
    check("36", 1, 4096, 7168, false, false, false);
    check("36", 1, 260, 512, false, true, false);
    check("38", 1, 260, 512, false, true, false);
    check("48", 128, 1792, 1536, true, true, false);
    check("48", 256, 1792, 1536, true, true, true);
    check("55", 256, 1792, 1536, true, true, true);
    check("55", 256, 1792, 1536, false, true, false);
    check("43", 256, 4096, 7168, true, true, true);
    check("40", 1, 4096, 7168, false, true, true);
    check("40", 17, 512, 768, false, true, false);
    check("41", 17, 512, 768, false, true, true);
    check("42", 17, 512, 768, false, false, false);
    check("40", 1, 65536, 65536, false, true, false);
    check("44", 256, 4096, 7168, true, true, false);
    check("56", 256, 4096, 7168, true, true, true);
    check("58", 256, 256, 512, true, true, true);
    check("58", 128, 5120, 2304, true, true, false);
    check("60", 256, 1792, 1536, true, true, true);
    check("60", 256, 1792, 1536, true, false, false);
    check("22", 256, 4096, 7168, true, true, true);
    check("23", 128, 4096, 7168, true, true, false);
    check("23", 256, 4096, 7168, true, false, false);
    check("24", 256, 4096, 7168, true, true, false);
    check("61", 256, 4096, 7168, true, true, true);
    check("61", 128, 4096, 7168, true, true, true);
    check("61", 64, 4096, 7168, true, true, false);
    check("62", 128, 4096, 7168, true, false, false);
    check("67", 128, 1792, 640, true, true, true);
    check("68", 256, 4096, 7168, true, true, true);
    check("69", 256, 5120, 2304, true, true, true);
    check("70", 256, 7168, 2048, true, true, true);
    check("71", 256, 6144, 1536, true, true, true);
    check("68", 128, 4096, 7168, true, true, false);
    check("69", 256, 4096, 7168, true, false, false);
    check("70", 256, 4096, 7168, false, true, false);
    check("72", 256, 4096, 7168, true, true, true);
    check("73", 256, 5120, 2304, true, true, true);
    check("74", 128, 4096, 7168, true, true, false);
    check("75", 256, 6144, 2048, true, false, false);
    check("76", 256, 4096, 7168, true, true, true);
    check("77", 256, 5120, 2304, true, true, true);
    check("78", 128, 4096, 7168, true, true, false);
    check("79", 256, 6144, 2048, true, false, false);
    check("80", 256, 4096, 7168, true, true, false);
    check("-1", 256, 4096, 7168, true, true, false);
    check("36x", 256, 4096, 7168, true, true, false);
    check("36", 1, 4096, 7168, false, true, false, false, false);
    check("36", 1, 4096, 7168, false, true, false, true, true);
    auto private_check = [&](const char* id, int m, int n, int k, bool accept) {
        setenv("CK_INT8_TUNE_ROUTE", "0", 1);
        setenv("CK_INT8_TUNE_LEGACY", id, 1);
        unsigned route = 0, legacy = 0;
        bool accepted = true;
        try
        {
            ck_tile::int8_tuning_override(
                true, false, false, false, true, true, m, n, k, route, legacy);
        }
        catch(const std::invalid_argument&)
        {
            accepted = false;
        }
        assert(accepted == accept);
        ++checks;
    };
    private_check("9061", 1, 6144, 7168, true);
    private_check("9062", 2, 4096, 8192, true);
    private_check("9063", 3, 512, 640, false);
    private_check("9061", 3, 512, 640, true);
    private_check("9064", 16, 6144, 7168, true);
    private_check("9061", 17, 6144, 7168, false);
    private_check("9061", 1, 260, 512, false);
    private_check("9061", 1, 256, 528, false);
    private_check("9060", 1, 256, 512, false);
    private_check("9071", 1, 260, 528, true);
    private_check("9071", 16, 6144, 7168, true);
    private_check("9071", 17, 6144, 7168, false);
    unsetenv("CK_INT8_TUNE_LEGACY");
    check("80", 1, 4096, 6144, false, true, true);
    check("81", 2, 6144, 7168, false, true, true);
    check("82", 2, 512, 768, false, true, false);
    check("83", 2, 512, 768, false, true, true);
    check("80", 2, 512, 640, false, true, true);
    check("83", 2, 512, 640, false, true, false);
    check("80", 1, 4096, 6144, false, false, false);
    unsetenv("CK_INT8_TUNE_ROUTE");
    std::cout << "TUNING_FULLNK_GRID_DOMAIN_PASS checks=" << checks << "\n";
}
