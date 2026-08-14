// Copyright (c) 2026 Hygon Info Technologies Ltd.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "ck_tile/ops/gemm/grouped_gemm_device_args.h"
#include "grouped_gemm.hpp"

namespace {

struct Options
{
    int m                   = 1024;
    int n                   = 1024;
    int k                   = 1024;
    int group_count         = 4;
    int warmup              = 10;
    int samples             = 30;
    int correctness_repeats = 3;
    std::uint32_t num_cu    = 0;
    std::string dtype       = "fp16";
    std::string layout      = "NT";
    std::string backend     = "both";
    std::string blas_library;
    bool validate           = true;
    bool include_args_build = true;
};

[[noreturn]] void print_usage_and_exit(const char* program, int code)
{
    std::ostream& os = code == 0 ? std::cout : std::cerr;
    os << "用法: " << program << " [选项]\n"
       << "  --m=<int> --n=<int> --k=<int> --groups=<int>\n"
       << "  --dtype=fp16|bf16 --layout=NT|NN|TN\n"
       << "  --backend=ck|blas|both --blas-lib=/path/to/libhipblas.so\n"
       << "  --warmup=10 --samples=30 --correctness-repeats=3\n"
       << "  --num-cu=0 --validate=0|1 --include-args-build=0|1\n"
       << "\n"
       << "说明: --blas-lib 为空时依次读取 CK_GROUPED_GEMM_BLAS_LIBRARY、"
          "libhipblas.so、libhipblas.so.2。\n"
       << "      CK 性能默认包含一次 GPU device-args 构造，以贴近 Primus 调用；"
          "设置 --include-args-build=0 可只测最终 grouped GEMM launch。\n";
    std::exit(code);
}

std::pair<std::string, std::string> split_option(const std::string& argument)
{
    std::size_t begin = 0;
    while(begin < argument.size() && argument[begin] == '-')
        ++begin;
    const auto equal = argument.find('=', begin);
    if(equal == std::string::npos)
        return {argument.substr(begin), {}};
    return {argument.substr(begin, equal - begin), argument.substr(equal + 1)};
}

int parse_positive_or_zero(const std::string& name, const std::string& value)
{
    if(value.empty())
        throw std::runtime_error("选项 --" + name + " 缺少值");
    std::size_t consumed = 0;
    const long parsed    = std::stol(value, &consumed);
    if(consumed != value.size() || parsed < 0 || parsed > std::numeric_limits<int>::max())
        throw std::runtime_error("选项 --" + name + " 的值无效: " + value);
    return static_cast<int>(parsed);
}

Options parse_options(int argc, char* argv[])
{
    Options options;
    for(int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if(argument == "-h" || argument == "--help")
            print_usage_and_exit(argv[0], 0);
        const auto [name, value] = split_option(argument);
        if(name == "m")
            options.m = parse_positive_or_zero(name, value);
        else if(name == "n")
            options.n = parse_positive_or_zero(name, value);
        else if(name == "k")
            options.k = parse_positive_or_zero(name, value);
        else if(name == "groups" || name == "group-count" || name == "group_count")
            options.group_count = parse_positive_or_zero(name, value);
        else if(name == "warmup")
            options.warmup = parse_positive_or_zero(name, value);
        else if(name == "samples" || name == "repeat")
            options.samples = parse_positive_or_zero(name, value);
        else if(name == "correctness-repeats")
            options.correctness_repeats = parse_positive_or_zero(name, value);
        else if(name == "num-cu")
            options.num_cu = static_cast<std::uint32_t>(parse_positive_or_zero(name, value));
        else if(name == "dtype")
            options.dtype = value;
        else if(name == "layout")
            options.layout = value;
        else if(name == "backend")
            options.backend = value;
        else if(name == "blas-lib")
            options.blas_library = value;
        else if(name == "validate")
            options.validate = parse_positive_or_zero(name, value) != 0;
        else if(name == "include-args-build")
            options.include_args_build = parse_positive_or_zero(name, value) != 0;
        else
            throw std::runtime_error("未知选项: " + argument);
    }

    const auto lower = [](unsigned char value) { return static_cast<char>(std::tolower(value)); };
    const auto upper = [](unsigned char value) { return static_cast<char>(std::toupper(value)); };
    std::transform(options.dtype.begin(), options.dtype.end(), options.dtype.begin(), lower);
    std::transform(options.layout.begin(), options.layout.end(), options.layout.begin(), upper);
    std::transform(options.backend.begin(), options.backend.end(), options.backend.begin(), lower);
    if(options.m <= 0 || options.n <= 0 || options.k <= 0 || options.group_count <= 0)
        throw std::runtime_error("M/N/K/group count 必须为正数");
    if(options.samples <= 0 || options.correctness_repeats <= 0)
        throw std::runtime_error("samples 和 correctness-repeats 必须为正数");
    if(options.dtype != "fp16" && options.dtype != "bf16")
        throw std::runtime_error("dtype 只支持 fp16 或 bf16");
    if(options.layout != "NT" && options.layout != "NN" && options.layout != "TN")
        throw std::runtime_error("layout 只支持 NT、NN 或 TN");
    if(options.backend != "ck" && options.backend != "blas" && options.backend != "both")
        throw std::runtime_error("backend 只支持 ck、blas 或 both");
    return options;
}

void hip_check(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
}

void hipblas_check(hipblasStatus_t status, const char* operation)
{
    if(status != HIPBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed, status=" +
                                 std::to_string(static_cast<int>(status)));
}

int selector_layout(const std::string& layout)
{
    if(layout == "NT")
        return CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1;
    if(layout == "NN")
        return CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NN_V1;
    return CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1;
}

bool a_is_column_major(const std::string& layout) { return layout == "TN"; }
bool b_is_column_major(const std::string& layout) { return layout == "NT"; }

ck_tile_hcu_grouped_gemm_dimension_v1 common_dimension(std::int64_t extent)
{
    return {CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1, 0, extent};
}

template <typename T>
T make_a_value(int group, int row, int reduction)
{
    const float value = static_cast<float>((group * 7 + row * 3 + reduction * 5) % 11 - 5) /
                        16.0f;
    return ck_tile::type_convert<T>(value);
}

template <typename T>
T make_b_value(int group, int reduction, int column)
{
    const float value =
        static_cast<float>((group * 3 + reduction * 2 + column * 7) % 13 - 6) / 16.0f;
    return ck_tile::type_convert<T>(value);
}

template <typename T>
void initialize_inputs(const Options& options, std::vector<T>& a, std::vector<T>& b)
{
    const std::size_t a_group_elements = static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_group_elements = static_cast<std::size_t>(options.k) * options.n;
    a.resize(a_group_elements * options.group_count);
    b.resize(b_group_elements * options.group_count);
    const bool a_col = a_is_column_major(options.layout);
    const bool b_col = b_is_column_major(options.layout);

    for(int group = 0; group < options.group_count; ++group)
    {
        const std::size_t a_base = static_cast<std::size_t>(group) * a_group_elements;
        const std::size_t b_base = static_cast<std::size_t>(group) * b_group_elements;
        for(int row = 0; row < options.m; ++row)
            for(int reduction = 0; reduction < options.k; ++reduction)
            {
                const std::size_t offset = a_col
                                               ? static_cast<std::size_t>(reduction) * options.m + row
                                               : static_cast<std::size_t>(row) * options.k + reduction;
                a[a_base + offset] = make_a_value<T>(group, row, reduction);
            }
        for(int reduction = 0; reduction < options.k; ++reduction)
            for(int column = 0; column < options.n; ++column)
            {
                const std::size_t offset = b_col
                                               ? static_cast<std::size_t>(column) * options.k + reduction
                                               : static_cast<std::size_t>(reduction) * options.n + column;
                b[b_base + offset] = make_b_value<T>(group, reduction, column);
            }
    }
}

// 关键流程 4：和 Primus 一样，在 GPU 上构造 GemmTransKernelArg 数组。
// selector 若声明 C_TRANSPOSED_VIEW contract，则将 TN 问题改写成 C^T=B^T*A，
// 仅交换 descriptor 解释，不移动 A/B/C 的物理数据。
template <typename T>
__global__ void build_uniform_device_args(ck_tile::GemmTransKernelArg* args,
                                          const T* a,
                                          const T* b,
                                          T* c,
                                          int group_count,
                                          int m,
                                          int n,
                                          int k,
                                          int layout,
                                          bool transposed_output_view)
{
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if(group >= group_count)
        return;

    const std::size_t a_offset = static_cast<std::size_t>(group) * m * k;
    const std::size_t b_offset = static_cast<std::size_t>(group) * k * n;
    const std::size_t c_offset = static_cast<std::size_t>(group) * m * n;
    auto& karg                 = args[group].group_karg;

    if(transposed_output_view)
    {
        const_cast<std::array<const void*, 1>&>(karg.as_ptr)[0] = b + b_offset;
        const_cast<std::array<const void*, 1>&>(karg.bs_ptr)[0] = a + a_offset;
        karg.e_ptr                                                = c + c_offset;
        karg.M                                                    = n;
        karg.N                                                    = m;
        karg.K                                                    = k;
        karg.stride_As[0]                                         = n;
        karg.stride_Bs[0]                                         = m;
        karg.stride_E                                             = n;
        karg.k_batch                                              = 1;
        return;
    }

    const bool a_col = layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_TN_V1;
    const bool b_col = layout == CK_TILE_HCU_GROUPED_GEMM_LAYOUT_NT_V1;
    const_cast<std::array<const void*, 1>&>(karg.as_ptr)[0] = a + a_offset;
    const_cast<std::array<const void*, 1>&>(karg.bs_ptr)[0] = b + b_offset;
    karg.e_ptr                                                = c + c_offset;
    karg.M                                                    = m;
    karg.N                                                    = n;
    karg.K                                                    = k;
    karg.stride_As[0]                                         = a_col ? m : k;
    karg.stride_Bs[0]                                         = b_col ? k : n;
    karg.stride_E                                             = n;
    karg.k_batch                                              = 1;
}

class DynamicHipBlas
{
    using CreateFn = hipblasStatus_t (*)(hipblasHandle_t*);
    using DestroyFn = hipblasStatus_t (*)(hipblasHandle_t);
    using SetStreamFn = hipblasStatus_t (*)(hipblasHandle_t, hipStream_t);
    using GemmExFn = hipblasStatus_t (*)(hipblasHandle_t,
                                         hipblasOperation_t,
                                         hipblasOperation_t,
                                         int,
                                         int,
                                         int,
                                         const void*,
                                         const void*,
                                         hipblasDatatype_t,
                                         int,
                                         const void*,
                                         hipblasDatatype_t,
                                         int,
                                         const void*,
                                         void*,
                                         hipblasDatatype_t,
                                         int,
                                         hipblasDatatype_t,
                                         hipblasGemmAlgo_t);

    void* library_          = nullptr;
    hipblasHandle_t handle_ = nullptr;
    DestroyFn destroy_      = nullptr;
    SetStreamFn set_stream_ = nullptr;
    GemmExFn gemm_ex_       = nullptr;
    std::string loaded_name_;

    template <typename Fn>
    Fn load_symbol(const char* name)
    {
        dlerror();
        void* symbol      = dlsym(library_, name);
        const char* error = dlerror();
        if(error != nullptr || symbol == nullptr)
            throw std::runtime_error(std::string("dlsym(") + name + ") failed: " +
                                     (error == nullptr ? "unknown error" : error));
        return reinterpret_cast<Fn>(symbol);
    }

public:
    explicit DynamicHipBlas(const std::string& requested_path)
    {
        std::vector<std::string> candidates;
        if(!requested_path.empty())
            candidates.push_back(requested_path);
        else if(const char* environment = std::getenv("CK_GROUPED_GEMM_BLAS_LIBRARY"))
            candidates.emplace_back(environment);
        else
            candidates = {"libhipblas.so", "libhipblas.so.2"};

        std::string errors;
        for(const auto& candidate : candidates)
        {
            library_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
            if(library_ != nullptr)
            {
                loaded_name_ = candidate;
                break;
            }
            const char* error = dlerror();
            errors += candidate + ": " + (error == nullptr ? "unknown error" : error) + "; ";
        }
        if(library_ == nullptr)
            throw std::runtime_error("无法加载 hipBLAS: " + errors);

        const auto create = load_symbol<CreateFn>("hipblasCreate");
        destroy_          = load_symbol<DestroyFn>("hipblasDestroy");
        set_stream_       = load_symbol<SetStreamFn>("hipblasSetStream");
        gemm_ex_          = load_symbol<GemmExFn>("hipblasGemmEx");
        hipblas_check(create(&handle_), "hipblasCreate");
    }

    ~DynamicHipBlas()
    {
        if(handle_ != nullptr && destroy_ != nullptr)
            destroy_(handle_);
        if(library_ != nullptr)
            dlclose(library_);
    }

    const std::string& loaded_name() const { return loaded_name_; }

    template <typename T>
    void grouped_gemm(const Options& options,
                      const T* a,
                      const T* b,
                      T* c,
                      hipStream_t stream)
    {
        hipblas_check(set_stream_(handle_, stream), "hipblasSetStream");
        const hipblasDatatype_t data_type =
            std::is_same_v<T, ck_tile::half_t> ? HIPBLAS_R_16F : HIPBLAS_R_16B;
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        const bool a_col  = a_is_column_major(options.layout);
        const bool b_col  = b_is_column_major(options.layout);

        // hipBLAS 使用 column-major。这里直接计算 C^T=B^T*A^T，使输出字节仍是
        // 调用方要求的 row-major C[M,N]，因此不需要额外 transpose kernel。
        const hipblasOperation_t trans_b_logical =
            b_col ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const hipblasOperation_t trans_a_logical =
            a_col ? HIPBLAS_OP_T : HIPBLAS_OP_N;
        const int lda = b_col ? options.k : options.n;
        const int ldb = a_col ? options.m : options.k;
        const std::size_t a_group_elements = static_cast<std::size_t>(options.m) * options.k;
        const std::size_t b_group_elements = static_cast<std::size_t>(options.k) * options.n;
        const std::size_t c_group_elements = static_cast<std::size_t>(options.m) * options.n;

        for(int group = 0; group < options.group_count; ++group)
        {
            hipblas_check(gemm_ex_(handle_,
                                   trans_b_logical,
                                   trans_a_logical,
                                   options.n,
                                   options.m,
                                   options.k,
                                   &alpha,
                                   b + static_cast<std::size_t>(group) * b_group_elements,
                                   data_type,
                                   lda,
                                   a + static_cast<std::size_t>(group) * a_group_elements,
                                   data_type,
                                   ldb,
                                   &beta,
                                   c + static_cast<std::size_t>(group) * c_group_elements,
                                   data_type,
                                   options.n,
                                   HIPBLAS_R_32F,
                                   HIPBLAS_GEMM_DEFAULT),
                          "hipblasGemmEx");
        }
    }
};

struct AccuracyResult
{
    std::size_t checked = 0;
    std::size_t failed  = 0;
    double max_abs      = 0;
    double max_rel      = 0;
};

template <typename T>
float as_float(T value)
{
    return ck_tile::type_convert<float>(value);
}

template <typename T>
AccuracyResult compare_values(const std::vector<T>& actual,
                              const std::vector<T>& expected,
                              float atol,
                              float rtol)
{
    AccuracyResult result;
    if(actual.size() != expected.size())
    {
        result.failed = 1;
        return result;
    }
    result.checked = actual.size();
    for(std::size_t i = 0; i < actual.size(); ++i)
    {
        const double a       = as_float(actual[i]);
        const double e       = as_float(expected[i]);
        const double abs_err = std::abs(a - e);
        const double rel_err = abs_err / std::max(std::abs(e), 1.0e-6);
        result.max_abs       = std::max(result.max_abs, abs_err);
        result.max_rel       = std::max(result.max_rel, rel_err);
        if(abs_err > atol + rtol * std::abs(e))
            ++result.failed;
    }
    return result;
}

template <typename T>
AccuracyResult check_cpu_samples(const Options& options,
                                 const std::vector<T>& a,
                                 const std::vector<T>& b,
                                 const std::vector<T>& c,
                                 float atol,
                                 float rtol)
{
    AccuracyResult result;
    const bool a_col = a_is_column_major(options.layout);
    const bool b_col = b_is_column_major(options.layout);
    const std::size_t a_group_elements = static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_group_elements = static_cast<std::size_t>(options.k) * options.n;
    const std::size_t c_group_elements = static_cast<std::size_t>(options.m) * options.n;
    const std::size_t checks_per_group = std::min<std::size_t>(257, c_group_elements);

    for(int group = 0; group < options.group_count; ++group)
    {
        const std::size_t a_base = static_cast<std::size_t>(group) * a_group_elements;
        const std::size_t b_base = static_cast<std::size_t>(group) * b_group_elements;
        const std::size_t c_base = static_cast<std::size_t>(group) * c_group_elements;
        for(std::size_t check = 0; check < checks_per_group; ++check)
        {
            const std::size_t linear = checks_per_group == 1
                                           ? 0
                                           : check * (c_group_elements - 1) /
                                                 (checks_per_group - 1);
            const int row    = static_cast<int>(linear / options.n);
            const int column = static_cast<int>(linear % options.n);
            float reference  = 0;
            for(int reduction = 0; reduction < options.k; ++reduction)
            {
                const std::size_t a_offset =
                    a_col ? static_cast<std::size_t>(reduction) * options.m + row
                          : static_cast<std::size_t>(row) * options.k + reduction;
                const std::size_t b_offset =
                    b_col ? static_cast<std::size_t>(column) * options.k + reduction
                          : static_cast<std::size_t>(reduction) * options.n + column;
                reference += as_float(a[a_base + a_offset]) * as_float(b[b_base + b_offset]);
            }
            const double actual   = as_float(c[c_base + linear]);
            const double expected = as_float(ck_tile::type_convert<T>(reference));
            const double abs_err  = std::abs(actual - expected);
            const double rel_err  = abs_err / std::max(std::abs(expected), 1.0e-6);
            ++result.checked;
            result.max_abs = std::max(result.max_abs, abs_err);
            result.max_rel = std::max(result.max_rel, rel_err);
            if(abs_err > atol + rtol * std::abs(expected))
                ++result.failed;
        }
    }
    return result;
}

struct TimingStats
{
    double median = 0;
    double mean   = 0;
    double min    = 0;
    double max    = 0;
    double cv     = 0;
};

TimingStats summarize(std::vector<float> values)
{
    if(values.empty())
        return {};
    TimingStats result;
    result.min  = *std::min_element(values.begin(), values.end());
    result.max  = *std::max_element(values.begin(), values.end());
    result.mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double variance = 0;
    for(float value : values)
        variance += (value - result.mean) * (value - result.mean);
    variance /= values.size();
    result.cv = result.mean <= 0 ? 0 : std::sqrt(variance) / result.mean;
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    result.median = values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0
                                           : values[middle];
    return result;
}

class EventTimer
{
    hipEvent_t start_ = nullptr;
    hipEvent_t stop_  = nullptr;

public:
    EventTimer()
    {
        hip_check(hipEventCreate(&start_), "hipEventCreate(start)");
        hip_check(hipEventCreate(&stop_), "hipEventCreate(stop)");
    }
    ~EventTimer()
    {
        if(start_ != nullptr)
            static_cast<void>(hipEventDestroy(start_));
        if(stop_ != nullptr)
            static_cast<void>(hipEventDestroy(stop_));
    }

    float measure(hipStream_t stream, const std::function<void()>& launch)
    {
        hip_check(hipEventRecord(start_, stream), "hipEventRecord(start)");
        launch();
        hip_check(hipEventRecord(stop_, stream), "hipEventRecord(stop)");
        hip_check(hipEventSynchronize(stop_), "hipEventSynchronize(stop)");
        float milliseconds = 0;
        hip_check(hipEventElapsedTime(&milliseconds, start_, stop_), "hipEventElapsedTime");
        return milliseconds;
    }
};

void print_timing(const char* backend,
                  const Options& options,
                  const TimingStats& timing)
{
    const double operations = 2.0 * options.group_count * options.m * options.n * options.k;
    const double tflops     = timing.median <= 0 ? 0 : operations / timing.median / 1.0e9;
    std::cout << "PERF backend=" << backend << " median_ms=" << timing.median
              << " mean_ms=" << timing.mean << " min_ms=" << timing.min
              << " max_ms=" << timing.max << " cv=" << timing.cv
              << " tflops=" << tflops << std::endl;
}

template <typename T>
int run(const Options& options)
{
    const bool run_ck_backend   = options.backend == "ck" || options.backend == "both";
    const bool run_blas_backend = options.backend == "blas" || options.backend == "both";
    const std::size_t a_elements =
        static_cast<std::size_t>(options.group_count) * options.m * options.k;
    const std::size_t b_elements =
        static_cast<std::size_t>(options.group_count) * options.k * options.n;
    const std::size_t c_elements =
        static_cast<std::size_t>(options.group_count) * options.m * options.n;

    std::vector<T> host_a;
    std::vector<T> host_b;
    initialize_inputs(options, host_a, host_b);
    std::vector<T> host_ck(c_elements);
    std::vector<T> host_blas(c_elements);

    ck_tile::DeviceMem device_a(a_elements * sizeof(T));
    ck_tile::DeviceMem device_b(b_elements * sizeof(T));
    ck_tile::DeviceMem device_ck(c_elements * sizeof(T));
    ck_tile::DeviceMem device_blas(c_elements * sizeof(T));
    ck_tile::DeviceMem device_args(static_cast<std::size_t>(options.group_count) *
                                   sizeof(ck_tile::GemmTransKernelArg));
    device_a.ToDevice(host_a.data());
    device_b.ToDevice(host_b.data());
    hipStream_t stream = nullptr;

    // 关键流程 1：把用户参数转换成 consumer-independent Problem。架构不由调用者
    // 传入，而是在 CK C ABI 内通过 get_hcu_target_enum() 检测。
    ck_tile_hcu_grouped_gemm_problem_v1 problem{};
    problem.struct_size   = sizeof(problem);
    problem.abi_version   = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    problem.data_type     = std::is_same_v<T, ck_tile::half_t>
                                ? CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1
                                : CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1;
    problem.layout        = selector_layout(options.layout);
    problem.group_count   = options.group_count;
    problem.m             = common_dimension(options.m);
    problem.n             = common_dimension(options.n);
    problem.k             = common_dimension(options.k);

    // 关键流程 2：pure selector 只返回已经预编译的 stable InstanceId；这里不会
    // 根据 M/N/K 在运行时实例化任何 C++ template。
    ck_tile_hcu_grouped_gemm_selection_v1 selection{};
    selection.struct_size = sizeof(selection);
    selection.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    const int select_status =
        ck_tile_hcu_grouped_gemm_select_device_args_v1(&problem, &selection);
    if(select_status != CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1)
        throw std::runtime_error("CK selector failed, status=" + std::to_string(select_status));

    // 关键流程 3：读取候选 contract。尤其是 BF16 TN 的转置输出视图必须在构造
    // device args 前确定，不能让 consumer 猜测具体 TileConfig。
    ck_tile_hcu_grouped_gemm_candidate_info_v1 candidate{};
    candidate.struct_size = sizeof(candidate);
    candidate.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    const int query_status =
        ck_tile_hcu_grouped_gemm_query_instance_v1(selection.instance_id, &candidate);
    if(query_status != CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1)
        throw std::runtime_error("CK candidate query failed, status=" +
                                 std::to_string(query_status));
    const bool transposed_output_view =
        (candidate.contract_flags &
         CK_TILE_HCU_GROUPED_GEMM_CANDIDATE_CONTRACT_C_TRANSPOSED_VIEW_V1) != 0;

    auto build_args = [&] {
        constexpr int threads = 256;
        const int grids       = (options.group_count + threads - 1) / threads;
        build_uniform_device_args<T><<<grids, threads, 0, stream>>>(
            static_cast<ck_tile::GemmTransKernelArg*>(device_args.GetDeviceBuffer()),
            static_cast<const T*>(device_a.GetDeviceBuffer()),
            static_cast<const T*>(device_b.GetDeviceBuffer()),
            static_cast<T*>(device_ck.GetDeviceBuffer()),
            options.group_count,
            options.m,
            options.n,
            options.k,
            problem.layout,
            transposed_output_view);
        hip_check(hipGetLastError(), "build_uniform_device_args launch");
    };

    ck_tile_hcu_grouped_gemm_launch_v1 ck_launch{};
    ck_launch.struct_size          = sizeof(ck_launch);
    ck_launch.abi_version          = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    ck_launch.device_args          = device_args.GetDeviceBuffer();
    ck_launch.num_cu               = options.num_cu;
    ck_launch.expected_instance_id = selection.instance_id;
    ck_launch.stream               = stream;

    // 关键流程 5：run ABI 会再次执行同一个 pure selector，并校验 expected ID。
    // 随后 registry 按 ID/dtype/layout 调用链接进 ELF 的预编译 kernel specialization。
    auto launch_ck = [&] {
        if(options.include_args_build)
            build_args();
        const int status =
            ck_tile_hcu_grouped_gemm_run_device_args_v1(&problem, &ck_launch, nullptr);
        if(status != CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1)
            throw std::runtime_error("CK device-args launch failed, status=" +
                                     std::to_string(status));
    };
    build_args();

    std::unique_ptr<DynamicHipBlas> blas;
    if(run_blas_backend)
        blas = std::make_unique<DynamicHipBlas>(options.blas_library);
    auto launch_blas = [&] {
        blas->grouped_gemm(options,
                           static_cast<const T*>(device_a.GetDeviceBuffer()),
                           static_cast<const T*>(device_b.GetDeviceBuffer()),
                           static_cast<T*>(device_blas.GetDeviceBuffer()),
                           stream);
    };

    std::cout << "CONFIG arch=" << ck_tile::get_device_name()
              << " dtype=" << options.dtype << " layout=" << options.layout
              << " groups=" << options.group_count << " m=" << options.m
              << " n=" << options.n << " k=" << options.k
              << " warmup=" << options.warmup << " samples=" << options.samples
              << " include_args_build=" << options.include_args_build
              << " num_cu=" << options.num_cu << std::endl;
    std::cout << "SELECT instance_id=" << selection.instance_id
              << " instance_name=" << selection.instance_name
              << " contract_flags=" << candidate.contract_flags << std::endl;
    if(blas)
        std::cout << "BLAS library=" << blas->loaded_name()
                  << " mode=hipblas_sequential_per_group" << std::endl;

    // 关键流程 6：先做 correctness，再允许性能计时。CK 与 BLAS 都和 CPU 抽样
    // reference 比较；both 模式还会逐元素比较 CK/BLAS 输出。
    if(options.validate)
    {
        const float atol = std::is_same_v<T, ck_tile::half_t> ? 0.08f : 0.35f;
        const float rtol = std::is_same_v<T, ck_tile::half_t> ? 0.015f : 0.03f;
        for(int repeat = 0; repeat < options.correctness_repeats; ++repeat)
        {
            if(run_ck_backend)
            {
                launch_ck();
                hip_check(hipStreamSynchronize(stream), "CK correctness synchronize");
                device_ck.FromDevice(host_ck.data());
                const auto cpu = check_cpu_samples(options, host_a, host_b, host_ck, atol, rtol);
                std::cout << "ACCURACY backend=ck repeat=" << repeat
                          << " checked=" << cpu.checked << " failed=" << cpu.failed
                          << " max_abs=" << cpu.max_abs << " max_rel=" << cpu.max_rel
                          << std::endl;
                if(cpu.failed != 0)
                    return 2;
            }
            if(run_blas_backend)
            {
                launch_blas();
                hip_check(hipStreamSynchronize(stream), "BLAS correctness synchronize");
                device_blas.FromDevice(host_blas.data());
                const auto cpu =
                    check_cpu_samples(options, host_a, host_b, host_blas, atol, rtol);
                std::cout << "ACCURACY backend=blas repeat=" << repeat
                          << " checked=" << cpu.checked << " failed=" << cpu.failed
                          << " max_abs=" << cpu.max_abs << " max_rel=" << cpu.max_rel
                          << std::endl;
                if(cpu.failed != 0)
                    return 3;
            }
            if(run_ck_backend && run_blas_backend)
            {
                const auto parity = compare_values(host_ck, host_blas, atol, rtol);
                std::cout << "ACCURACY backend=ck_vs_blas repeat=" << repeat
                          << " checked=" << parity.checked << " failed=" << parity.failed
                          << " max_abs=" << parity.max_abs << " max_rel=" << parity.max_rel
                          << std::endl;
                if(parity.failed != 0)
                    return 4;
            }
        }
    }

    // 关键流程 7：使用 GPU event；both 模式按样本奇偶交替 CK/BLAS 顺序，减少
    // 温度和频率漂移。每个样本独立同步 stop event，最终报告 median 和 CV。
    for(int i = 0; i < options.warmup; ++i)
    {
        if(run_ck_backend)
            launch_ck();
        if(run_blas_backend)
            launch_blas();
    }
    hip_check(hipStreamSynchronize(stream), "warmup synchronize");

    EventTimer timer;
    std::vector<float> ck_samples;
    std::vector<float> blas_samples;
    ck_samples.reserve(options.samples);
    blas_samples.reserve(options.samples);
    for(int sample = 0; sample < options.samples; ++sample)
    {
        const bool ck_first = sample % 2 == 0;
        if(run_ck_backend && (!run_blas_backend || ck_first))
            ck_samples.push_back(timer.measure(stream, launch_ck));
        if(run_blas_backend)
            blas_samples.push_back(timer.measure(stream, launch_blas));
        if(run_ck_backend && run_blas_backend && !ck_first)
            ck_samples.push_back(timer.measure(stream, launch_ck));
    }

    TimingStats ck_timing;
    TimingStats blas_timing;
    if(run_ck_backend)
    {
        ck_timing = summarize(ck_samples);
        print_timing("ck_device_args", options, ck_timing);
    }
    if(run_blas_backend)
    {
        blas_timing = summarize(blas_samples);
        print_timing("hipblas", options, blas_timing);
    }
    if(run_ck_backend && run_blas_backend)
        std::cout << "RATIO ck_over_blas="
                  << (blas_timing.median <= 0 ? 0 : ck_timing.median / blas_timing.median)
                  << std::endl;

    std::cout << "SUMMARY result=PASS" << std::endl;
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    try
    {
        const Options options = parse_options(argc, argv);
        if(options.dtype == "fp16")
            return run<ck_tile::half_t>(options);
        return run<ck_tile::bf16_t>(options);
    }
    catch(const std::exception& error)
    {
        std::cerr << "SUMMARY result=FAIL reason=\"" << error.what() << "\"" << std::endl;
        return 1;
    }
}
