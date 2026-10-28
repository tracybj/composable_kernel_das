// SPDX-License-Identifier: MIT
// INT8 extension of the existing device-args benchmark. CPU samples plus
// full CK/BLAS comparison complement benchmark.py's full exact FP32 reference.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
struct Desc
{
    const void* a;
    const void* b;
    void* c;
    int kb, m, n, k, sa, sb, sc, nd;
    const void* ds;
    const void* sds;
};
extern "C" void* int8_create(const Desc*, int);
extern "C" float int8_run(void*, int, int);
extern "C" void int8_destroy(void*);
extern "C" const char* int8_error();
extern "C" const char* int8_name(void*);
extern "C" unsigned int8_instance(void*);
extern "C" void* blas_create(int, const Desc*);
extern "C" int blas_count(void*);
extern "C" int blas_select(void*, int);
extern "C" float blas_run(void*, int, int);
extern "C" void blas_destroy(void*);
extern "C" const char* blas_error();
extern "C" const char* blas_info(void*);
extern "C" const char* blas_loaded_path();
namespace {
void hs(hipError_t x)
{
    if(x != hipSuccess)
        throw std::runtime_error(hipGetErrorString(x));
}
struct DeviceBuffer
{
    void* p = nullptr;
    explicit DeviceBuffer(size_t n) { hs(hipMalloc(&p, n)); }
    ~DeviceBuffer()
    {
        if(p)
            (void)hipFree(p);
    }
};
__host__ __device__ int8_t value_at(uint64_t i, uint32_t seed)
{
    uint32_t x = uint32_t(i) ^ uint32_t(i >> 32) * 0x9e3779b9u ^ seed;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return int8_t(int(x % 17) - 8);
}
__global__ void init_input(int8_t* p, uint64_t n, uint32_t seed)
{
    for(uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
        i += uint64_t(gridDim.x) * blockDim.x)
        p[i] = value_at(i, seed);
}
__global__ void init_output(int32_t* p, uint64_t n, uint64_t stride, int32_t sentinel)
{
    for(uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
        i += uint64_t(gridDim.x) * blockDim.x)
    {
        const uint64_t j = i % stride;
        p[i]             = (j < 16 || j >= stride - 16) ? 117 : sentinel;
    }
}
__global__ void
compare_output(const int32_t* a, const int32_t* b, uint64_t n, uint64_t stride, unsigned* errors)
{
    for(uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
        i += uint64_t(gridDim.x) * blockDim.x)
    {
        const uint64_t j = i % stride;
        const bool guard = j < 16 || j >= stride - 16;
        if(guard ? (a[i] != 117 || b[i] != 117) : (a[i] != b[i]))
            atomicExch(errors, 1u);
    }
}
int blocks(uint64_t n) { return int(std::min<uint64_t>((n + 255) / 256, 8192)); }
double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
} // namespace
int grouped_gemm_int8_nt_benchmark_main(int argc, char** argv)
{
    try
    {
        std::map<std::string, std::string> opt;
        for(int i = 1; i < argc; ++i)
        {
            std::string s = argv[i];
            const auto at = s.find('=');
            if(at != std::string::npos)
                opt[s.substr(0, at)] = s.substr(at + 1);
            else if(s.rfind("--", 0) == 0 && i + 1 < argc)
                opt[s] = argv[++i];
        }
        auto number = [&](const char* key, int fallback) {
            return opt.count(key) ? std::stoi(opt[key]) : fallback;
        };
        const int m = number("--m", 256), n = number("--n", 4096), k = number("--k", 4096),
                  g    = number("--groups", 8);
        const int warm = number("--warmup", 10), repeat = number("--repeat", 50),
                  samples   = number("--samples", 3);
        const int requested = number("--blas-index", -1);
        const bool timed    = number("--time", 1) != 0;
        if(m < 1 || n < 1 || k < 1 || g < 1 || warm < 0 || repeat < 1 || samples < 1 ||
           k * 64LL >= (1LL << 24))
            throw std::runtime_error("Invalid INT8 benchmark shape/timing");
        if(opt.count("--layout") && opt["--layout"] != "NT")
            throw std::runtime_error("INT8 implementation supports NT only");
        if(opt.count("--group-lengths"))
            throw std::runtime_error("Use public-ABI boundary tests for INT8 ragged inputs");
        const auto loaded = std::filesystem::canonical(blas_loaded_path());
        if(opt.count("--blas-root") &&
           loaded != std::filesystem::canonical(std::filesystem::path(opt["--blas-root"]) /
                                                "lib/libhipblaslt.so.0.10"))
            throw std::runtime_error("Linked BLAS differs from requested root");
        const uint64_t ac = uint64_t(g) * m * k, bc = uint64_t(g) * n * k,
                       stride = uint64_t(m) * n + 32, cc = uint64_t(g) * stride;
        DeviceBuffer a(ac), b(bc), c(cc * 4), d(cc * 4), errors(4);
        constexpr uint32_t seed_a = 20261026, seed_b = 0x841376a5;
        init_input<<<blocks(ac), 256>>>(static_cast<int8_t*>(a.p), ac, seed_a);
        init_input<<<blocks(bc), 256>>>(static_cast<int8_t*>(b.p), bc, seed_b);
        init_output<<<blocks(cc), 256>>>(static_cast<int32_t*>(c.p), cc, stride, 123456789);
        init_output<<<blocks(cc), 256>>>(static_cast<int32_t*>(d.p), cc, stride, -123456789);
        hs(hipGetLastError());
        hs(hipDeviceSynchronize());
        std::vector<Desc> ck_desc(g), lt_desc(g);
        for(int i = 0; i < g; ++i)
        {
            ck_desc[i]   = {static_cast<int8_t*>(a.p) + uint64_t(i) * m * k,
                            static_cast<int8_t*>(b.p) + uint64_t(i) * n * k,
                            static_cast<int32_t*>(c.p) + uint64_t(i) * stride + 16,
                            1,
                            m,
                            n,
                            k,
                            k,
                            k,
                            n,
                            0,
                            nullptr,
                            nullptr};
            lt_desc[i]   = ck_desc[i];
            lt_desc[i].c = static_cast<int32_t*>(d.p) + uint64_t(i) * stride + 16;
        }
        void* ck = int8_create(ck_desc.data(), g);
        if(!ck)
            throw std::runtime_error(int8_error());
        void* lt = blas_create(g, lt_desc.data());
        if(!lt)
        {
            int8_destroy(ck);
            throw std::runtime_error(blas_error());
        }
        try
        {
            if(int8_run(ck, 0, 1) <= 0)
                throw std::runtime_error(int8_error());
            // Independent exact CPU checks with counter-based reproducible inputs.
            for(int sample = 0; sample < 128; ++sample)
            {
                const int e      = sample == 0   ? 0
                                   : sample == 1 ? g - 1
                                                 : int((uint64_t(sample) * 104729) % g);
                const int row    = int((uint64_t(sample) * 8191) % m),
                          col    = int((uint64_t(sample) * 131071) % n);
                int32_t expected = 0;
                for(int q = 0; q < k; ++q)
                    expected += int(value_at((uint64_t(e) * m + row) * k + q, seed_a)) *
                                int(value_at((uint64_t(e) * n + col) * k + q, seed_b));
                int32_t actual;
                hs(hipMemcpy(&actual,
                             static_cast<int32_t*>(c.p) + uint64_t(e) * stride + 16 +
                                 uint64_t(row) * n + col,
                             4,
                             hipMemcpyDeviceToHost));
                if(actual != expected)
                    throw std::runtime_error("Exact CPU sample mismatch");
            }
            auto equal = [&]() {
                hs(hipMemset(errors.p, 0, 4));
                compare_output<<<blocks(cc), 256>>>(static_cast<int32_t*>(c.p),
                                                    static_cast<int32_t*>(d.p),
                                                    cc,
                                                    stride,
                                                    static_cast<unsigned*>(errors.p));
                unsigned count;
                hs(hipMemcpy(&count, errors.p, 4, hipMemcpyDeviceToHost));
                return count == 0;
            };
            std::vector<std::pair<double, int>> valid;
            for(int i = 0; i < blas_count(lt); ++i)
            {
                if(requested >= 0 && i != requested)
                    continue;
                if(blas_select(lt, i))
                    continue;
                init_output<<<blocks(cc), 256>>>(
                    static_cast<int32_t*>(d.p), cc, stride, -123456789);
                const float ms = blas_run(lt, 2, 5);
                if(ms > 0 && equal())
                    valid.emplace_back(ms, i);
            }
            if(valid.empty())
                throw std::runtime_error("No numerically valid BLAS algorithm");
            std::sort(valid.begin(), valid.end());
            valid.resize(std::min<size_t>(3, valid.size()));
            std::vector<std::pair<double, int>> refined;
            for(auto [_, i] : valid)
            {
                if(blas_select(lt, i))
                    continue;
                std::vector<double> ts;
                for(int s = 0; s < samples; ++s)
                {
                    const double ms = blas_run(lt, warm, repeat);
                    if(ms <= 0)
                        throw std::runtime_error(blas_error());
                    ts.push_back(ms);
                }
                if(equal())
                    refined.emplace_back(median(ts), i);
            }
            if(refined.empty())
                throw std::runtime_error("BLAS refine validation failed");
            std::sort(refined.begin(), refined.end());
            const auto best = refined.front();
            if(blas_select(lt, best.second))
                throw std::runtime_error(blas_error());
            std::vector<double> ck_ms, lt_ms;
            for(int s = 0; s < (timed ? samples : 1); ++s)
            {
                const double x = int8_run(ck, timed ? warm : 0, timed ? repeat : 1);
                const double y = blas_run(lt, timed ? warm : 0, timed ? repeat : 1);
                if(x <= 0 || y <= 0)
                    throw std::runtime_error("Timed launch failed");
                ck_ms.push_back(x);
                lt_ms.push_back(y);
            }
            if(!equal())
                throw std::runtime_error("Full output/guard mismatch");
            const double x = median(ck_ms), y = median(lt_ms), ops = 2.0 * g * m * n * k;
            std::cout.precision(10);
            std::cout << "INT8_CONFIG M=" << m << " N=" << n << " K=" << k << " E=" << g
                      << " instance_id=" << int8_instance(ck) << " instance_name=" << int8_name(ck)
                      << " blas=" << loaded << " algorithm=" << blas_info(lt) << '\n';
            std::cout << "INT8_PERF ck_ms=" << x << " ck_tops=" << ops / (x * 1e9)
                      << " blas_ms=" << y << " blas_tops=" << ops / (y * 1e9) << " R=" << y / x
                      << '\n';
            std::cout << "SUMMARY result=PASS cpu_exact_samples=128 full_backend_comparison=PASS "
                         "output_guards=PASS\n";
        }
        catch(...)
        {
            blas_destroy(lt);
            int8_destroy(ck);
            throw;
        }
        blas_destroy(lt);
        int8_destroy(ck);
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "SUMMARY result=FAIL reason=\"" << e.what() << "\"\n";
        return 1;
    }
}
