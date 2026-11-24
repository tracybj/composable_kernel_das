// SPDX-License-Identifier: MIT

#include "../include/grouped_gemm_int8_nt_device_args.h"

#include "ck_tile/core.hpp"
#include "ck_tile/host/device_memory.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options
{
    int m = 128, n = 2048, k = 1024, group_count = 4;
    int warmup = 10, samples = 3, gemm_repeat = 50, correctness_repeats = 3;
    std::string dtype = "int8_bf16";
    std::vector<int64_t> group_lengths;
    int64_t variable_capacity = 0;
    bool validate = true, time = true, include_args_build = false, allow_empty_groups = false;
};

int nonnegative(const std::string& value)
{
    std::size_t consumed = 0;
    const auto result = std::stoll(value, &consumed);
    if(consumed != value.size() || result < 0 || result > std::numeric_limits<int>::max())
        throw std::runtime_error("Expected a nonnegative integer: " + value);
    return static_cast<int>(result);
}

Options parse_options(int argc, char** argv)
{
    Options options;
    const std::array int_options{
        std::pair{"m", &options.m},
        std::pair{"n", &options.n},
        std::pair{"k", &options.k},
        std::pair{"groups", &options.group_count},
        std::pair{"warmup", &options.warmup},
        std::pair{"samples", &options.samples},
        std::pair{"gemm-repeat", &options.gemm_repeat},
        std::pair{"correctness-repeats", &options.correctness_repeats}};
    const std::array bool_options{
        std::pair{"validate", &options.validate},
        std::pair{"time", &options.time},
        std::pair{"include-args-build", &options.include_args_build},
        std::pair{"allow-empty-groups", &options.allow_empty_groups}};
    bool explicit_groups = false;
    for(int i = 1; i < argc; ++i)
    {
        const std::string option = argv[i];
        const auto eq = option.find('=');
        if(option.rfind("--", 0) != 0 || eq == std::string::npos)
            throw std::runtime_error("Expected --name=value: " + option);
        const auto name = option.substr(2, eq - 2);
        const auto value = option.substr(eq + 1);
        const auto named = [&name](const auto& entry) { return name == entry.first; };
        if(const auto it = std::find_if(int_options.begin(), int_options.end(), named);
           it != int_options.end())
        {
            *it->second = nonnegative(value);
            explicit_groups |= name == "groups";
        }
        else if(const auto flag = std::find_if(bool_options.begin(), bool_options.end(), named);
                flag != bool_options.end())
        {
            const int enabled = nonnegative(value);
            if(enabled > 1)
                throw std::runtime_error("Expected 0 or 1: " + option);
            *flag->second = enabled == 1;
        }
        else if(name == "dtype")
            options.dtype = value;
        else if(name == "variable-capacity")
            options.variable_capacity = nonnegative(value);
        else if(name == "group-lengths")
        {
            if(value.empty() || value.back() == ',')
                throw std::runtime_error("Expected comma-separated group lengths");
            std::istringstream input(value);
            std::string length;
            options.group_lengths.clear();
            while(std::getline(input, length, ','))
                options.group_lengths.push_back(nonnegative(length));
        }
        else if((name == "layout" && value == "NT") || (name == "backend" && value == "ck") ||
                (name == "num-cu" && value == "0") || (name == "blas-streams" && value == "1"))
        {
            // Accept the CK-only command line emitted by the example 19 CSV driver.
        }
        else
            throw std::runtime_error("Unsupported option: " + option);
    }

    if(!options.group_lengths.empty())
    {
        if(options.group_lengths.size() > 65535 ||
           (explicit_groups && options.group_lengths.size() != std::size_t(options.group_count)))
            throw std::runtime_error("Invalid group count for --group-lengths");
        options.group_count = static_cast<int>(options.group_lengths.size());
    }
    else if(options.variable_capacity != 0)
        throw std::runtime_error("--variable-capacity requires --group-lengths");
    if(options.dtype != "int8_bf16" && options.dtype != "int8_fp16")
        throw std::runtime_error("Supported outputs are int8_bf16 and int8_fp16");
    if(options.m <= 0 || options.n <= 0 || options.k <= 0 || options.group_count <= 0 ||
       options.group_count > 65535 || options.gemm_repeat <= 0 || options.correctness_repeats <= 0 ||
       (options.time && options.samples <= 0))
        throw std::runtime_error("Invalid benchmark dimensions or repetition counts");
    return options;
}

void hip_check(hipError_t status, const char* operation)
{
    if(status != hipSuccess)
        throw std::runtime_error(std::string(operation) + ": " + hipGetErrorString(status));
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

    template <typename Launch>
    float measure(hipStream_t stream, Launch&& launch)
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

// Row-constant inputs give an independent exact integer dot for every element.
__host__ __device__ int8_t scaled_int8_a(std::size_t row) { return int8_t(row % 7 + 1); }
__host__ __device__ int8_t scaled_int8_b(std::size_t col) { return int8_t(int(col % 9) - 4); }
__host__ __device__ float scaled_int8_sa(int group) { return 0.031f + (group % 7) * 0.003f; }
__host__ __device__ float scaled_int8_sb(std::size_t col) { return 0.013f + (col % 7) * 0.002f; }

__global__ void initialize_scaled_int8(int8_t* a, int8_t* b, float* sa, float* sb,
                                       std::size_t a_rows, std::size_t b_rows, int k, int groups)
{
    const auto start = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const auto step  = std::size_t(gridDim.x) * blockDim.x;
    for(auto i = start; i < a_rows * k; i += step)
        a[i] = scaled_int8_a(i / k);
    for(auto i = start; i < b_rows * k; i += step)
        b[i] = scaled_int8_b(i / k);
    for(auto i = start; i < b_rows; i += step)
        sb[i] = scaled_int8_sb(i);
    for(auto i = start; i < std::size_t(groups); i += step)
        sa[i] = scaled_int8_sa(static_cast<int>(i));
}

template <typename T>
__global__ void check_scaled_int8(const T* output, const float* sa, const float* sb,
                                  unsigned* mismatch, const std::int64_t* lengths,
                                  int m, int n, int k, int groups)
{
    const auto stride = std::size_t(m) * n + 32;
    for(auto i = std::size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        i < stride * groups; i += std::size_t(gridDim.x) * blockDim.x)
    {
        const auto group = i / stride;
        const auto local = i % stride;
        T expected = ck_tile::type_convert<T>(0.0f);
        if(local >= 16 && local < 16 + std::size_t(lengths[group]) * n)
        {
            const auto row = group * m + (local - 16) / n;
            const auto col = group * n + (local - 16) % n;
            const auto dot = int(scaled_int8_a(row)) * int(scaled_int8_b(col)) * k;
            expected = ck_tile::type_convert<T>(float(dot) * (sa[group] * sb[col]));
        }
        const auto actual_bits = ck_tile::bit_cast<uint16_t>(output[i]);
        const auto expected_bits = ck_tile::bit_cast<uint16_t>(expected);
        if(actual_bits != expected_bits && ((actual_bits | expected_bits) & 0x7fff) != 0)
            atomicExch(mismatch, 1u);
    }
}

template <typename T>
__global__ void build_scaled_int8_args(ck_tile::GemmTransKernelArg* args,
                                       ck_tile_hcu_grouped_gemm_int8_scale_v2* scales,
                                       const int8_t* a,
                                       const int8_t* b,
                                       T* c,
                                       const float* sa,
                                       const float* sb,
                                       const std::int64_t* lengths,
                                       int groups,
                                       int m,
                                       int n,
                                       int k)
{
    const int group = blockIdx.x * blockDim.x + threadIdx.x;
    if(group >= groups)
        return;
    auto& arg = args[group].group_karg;
    const auto c_stride = std::size_t(m) * n + 32;
    const_cast<std::array<const void*, 1>&>(arg.as_ptr)[0] = a + std::size_t(group) * m * k;
    const_cast<std::array<const void*, 1>&>(arg.bs_ptr)[0] = b + std::size_t(group) * n * k;
    arg.e_ptr = c + group * c_stride + 16;
    arg.M = static_cast<int>(lengths[group]);
    arg.N = n;
    arg.K = k;
    arg.stride_As[0] = k;
    arg.stride_Bs[0] = k;
    arg.stride_E = n;
    arg.k_batch = 1;
    scales[group] = {sa + group, sb + std::size_t(group) * n, 0, 1};
}

void check_int8_device_args(int status, const char* operation)
{
    if(status != CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_SUCCESS_V1)
        throw std::runtime_error(std::string(operation) + ", status=" + std::to_string(status));
}

template <typename T>
int run_scaled_int8(const Options& options)
{
    const int n = options.n, k = options.k, g = options.group_count;
    const bool variable_m = !options.group_lengths.empty();
    const auto lengths = variable_m ? options.group_lengths
                                    : std::vector<int64_t>(g, options.m);
    const int64_t total_m = std::accumulate(lengths.begin(), lengths.end(), int64_t{0});
    const int m = std::max(1, static_cast<int>(*std::max_element(lengths.begin(), lengths.end())));
    const bool ragged = variable_m &&
                        std::any_of(lengths.begin(), lengths.end(),
                                    [m](int64_t length) { return length != m; });
    const bool empty = std::find(lengths.begin(), lengths.end(), 0) != lengths.end();
    const auto capacity = options.variable_capacity == 0 ? std::max<int64_t>(total_m, 1)
                                                          : options.variable_capacity;
    if(capacity < total_m || (empty && !options.allow_empty_groups))
        throw std::runtime_error("Invalid group capacity or missing --allow-empty-groups=1");
    ck_tile_hcu_grouped_gemm_int8_problem_v2 problem{};
    problem.struct_size = sizeof(problem);
    problem.abi_version = CK_TILE_HCU_GROUPED_GEMM_SCALED_ABI_VERSION;
    problem.output_type = std::is_same_v<T, ck_tile::bf16_t>
                              ? CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_BF16_V1
                              : CK_TILE_HCU_GROUPED_GEMM_DATA_TYPE_FP16_V1;
    problem.group_count = g;
    if(ragged)
        problem.problem_flags |= CK_TILE_HCU_GROUPED_GEMM_PROBLEM_GROUP_LENGTHS_RAGGED_V1;
    if(empty)
        problem.problem_flags |= CK_TILE_HCU_GROUPED_GEMM_PROBLEM_MAY_HAVE_EMPTY_GROUPS_V1;
    problem.int8_flags = CK_TILE_HCU_GROUPED_GEMM_INT8_DENSE_CHANNEL_SCALE_V2;
    problem.m = {variable_m ? CK_TILE_HCU_GROUPED_GEMM_DIMENSION_DEVICE_LENGTHS_WITH_CAPACITY_V1
                                : CK_TILE_HCU_GROUPED_GEMM_DIMENSION_COMMON_V1,
                 0,
                 variable_m ? capacity : m};
    problem.n = n;
    problem.k = k;
    problem.max_m = m;
    problem.max_stride_a = k;
    problem.max_stride_b = k;
    problem.max_stride_c = n;

    ck_tile_hcu_grouped_gemm_selection_v1 selection{};
    selection.struct_size = sizeof(selection);
    selection.abi_version = CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_ABI_VERSION;
    check_int8_device_args(
        ck_tile_hcu_grouped_gemm_select_int8_nt_device_args_v2(&problem, &selection), "INT8 select");
    if(ck_tile_hcu_grouped_gemm_int8_nt_device_args_size_v2() != sizeof(ck_tile::GemmTransKernelArg))
        throw std::runtime_error("INT8 device_args_size_v2 disagrees with GemmTransKernelArg");
    if(options.validate)
    {
        auto oversized = selection;
        oversized.struct_size = sizeof(selection) + 16;
        check_int8_device_args(
            ck_tile_hcu_grouped_gemm_select_int8_nt_device_args_v2(&problem, &oversized),
            "INT8 select oversized struct_size");
        if(oversized.struct_size != sizeof(selection) + 16)
            throw std::runtime_error("INT8 select overwrote caller selection struct_size");

        auto undersized = selection;
        undersized.struct_size = sizeof(selection) - 1;
        if(ck_tile_hcu_grouped_gemm_select_int8_nt_device_args_v2(&problem, &undersized) !=
           CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_BAD_STRUCT_SIZE_V1)
            throw std::runtime_error("INT8 select failed to reject undersized selection");
    }
    ck_tile_hcu_grouped_gemm_int8_candidate_info_v2 candidate{};
    candidate.struct_size = sizeof(candidate);
    candidate.abi_version = CK_TILE_HCU_GROUPED_GEMM_SCALED_ABI_VERSION;
    check_int8_device_args(
        ck_tile_hcu_grouped_gemm_query_int8_nt_instance_v2(selection.instance_id, &candidate),
        "INT8 query");
    if(candidate.output_type != problem.output_type ||
       std::string(candidate.canonical_name) != selection.instance_name)
        throw std::runtime_error("INT8 select and query disagree");
    if(candidate.contract_flags & CK_TILE_HCU_GROUPED_GEMM_INT8_PACKED_B_V2)
        throw std::runtime_error("INT8 NT benchmark requires raw B");
    std::cout << "SELECT instance_id=" << selection.instance_id
              << " instance_name=" << selection.instance_name
              << " reason=" << selection.reason << " packed_b=0" << '\n';

    const auto a_rows = std::size_t(g) * m, b_rows = std::size_t(g) * n;
    const auto c_stride = std::size_t(m) * n + 32;
    ck_tile::DeviceMem a(a_rows * k), b(b_rows * k), c(c_stride * g * sizeof(T));
    ck_tile::DeviceMem sa(g * sizeof(float)), sb(b_rows * sizeof(float)), error(sizeof(unsigned));
    ck_tile::DeviceMem device_lengths(g * sizeof(std::int64_t));
    ck_tile::DeviceMem args(g * sizeof(ck_tile::GemmTransKernelArg));
    ck_tile::DeviceMem scales(g * sizeof(ck_tile_hcu_grouped_gemm_int8_scale_v2));
    struct Stream
    {
        hipStream_t value = nullptr;
        ~Stream()
        {
            if(value)
                (void)hipStreamDestroy(value);
        }
    } handle;
    hip_check(hipStreamCreateWithFlags(&handle.value, hipStreamNonBlocking), "create stream");
    const auto stream = handle.value;
    auto* ap = static_cast<int8_t*>(a.GetDeviceBuffer());
    auto* bp = static_cast<int8_t*>(b.GetDeviceBuffer());
    auto* cp = static_cast<T*>(c.GetDeviceBuffer());
    auto* sap = static_cast<float*>(sa.GetDeviceBuffer());
    auto* sbp = static_cast<float*>(sb.GetDeviceBuffer());
    auto* mismatch = static_cast<unsigned*>(error.GetDeviceBuffer());
    initialize_scaled_int8<<<1024, 256, 0, stream>>>(ap, bp, sap, sbp, a_rows, b_rows, k, g);
    hip_check(hipGetLastError(), "initialize scaled INT8 inputs");
    device_lengths.ToDevice(lengths.data());

    const int blocks = (g + 255) / 256;
    auto* gpu_args = static_cast<ck_tile::GemmTransKernelArg*>(args.GetDeviceBuffer());
    auto* gpu_scales = static_cast<ck_tile_hcu_grouped_gemm_int8_scale_v2*>(scales.GetDeviceBuffer());
    auto* gpu_lengths = static_cast<const std::int64_t*>(device_lengths.GetDeviceBuffer());
    const auto build_args = [&] {
        build_scaled_int8_args<<<blocks, 256, 0, stream>>>(
            gpu_args, gpu_scales, ap, bp, cp, sap, sbp, gpu_lengths, g, m, n, k);
        hip_check(hipGetLastError(), "build scaled INT8 args");
    };
    if(!options.include_args_build)
        build_args();

    ck_tile_hcu_grouped_gemm_int8_launch_v2 launch_args{};
    launch_args.struct_size = sizeof(launch_args);
    launch_args.abi_version = CK_TILE_HCU_GROUPED_GEMM_SCALED_ABI_VERSION;
    launch_args.device_args = gpu_args;
    launch_args.device_scales = gpu_scales;
    launch_args.expected_instance_id = selection.instance_id;
    launch_args.stream = stream;
    const auto launch = [&] {
        if(options.include_args_build)
            build_args();
        check_int8_device_args(
            ck_tile_hcu_grouped_gemm_run_int8_nt_device_args_v2(&problem, &launch_args, nullptr),
            "INT8 run");
    };
    if(options.validate)
    {
        auto mismatched_launch = launch_args;
        mismatched_launch.expected_instance_id = UINT32_MAX;
        if(ck_tile_hcu_grouped_gemm_run_int8_nt_device_args_v2(
               &problem, &mismatched_launch, nullptr) !=
           CK_TILE_HCU_GROUPED_GEMM_DEVICE_ARGS_INSTANCE_MISMATCH_V1)
            throw std::runtime_error("INT8 run failed to reject a mismatched instance ID");

        for(int sample = 0; sample < options.correctness_repeats; ++sample)
        {
            // Leave the guards zero and poison the output so missed stores also fail.
            for(int i = 0; i < g; ++i)
            {
                hip_check(hipMemsetAsync(cp + i * c_stride, 0, c_stride * sizeof(T), stream),
                          "zero guards and padding");
                if(lengths[i] > 0)
                    hip_check(hipMemsetAsync(cp + i * c_stride + 16, 0xff,
                                             std::size_t(lengths[i]) * n * sizeof(T), stream),
                              "poison output");
            }
            hip_check(hipMemsetAsync(mismatch, 0, sizeof(unsigned), stream), "reset validation");
            launch();
            check_scaled_int8<<<1024, 256, 0, stream>>>(cp, sap, sbp, mismatch, gpu_lengths, m, n, k, g);
            hip_check(hipGetLastError(), "validate scaled INT8 output");
            unsigned invalid = 0;
            hip_check(hipMemcpyAsync(&invalid, mismatch, sizeof(invalid), hipMemcpyDeviceToHost,
                                     stream), "read validation result");
            hip_check(hipStreamSynchronize(stream), "synchronize validation");
            if(invalid)
                throw std::runtime_error("Scaled INT8 full output or guard mismatch");
        }
    }
    if(options.time)
    {
        EventTimer timer;
        std::vector<float> samples;
        for(int sample = 0; sample < options.samples; ++sample)
        {
            for(int warm = 0; warm < options.warmup; ++warm)
                launch();
            samples.push_back(timer.measure(stream, [&] {
                for(int repeat = 0; repeat < options.gemm_repeat; ++repeat)
                    launch();
            }) / options.gemm_repeat);
        }
        const auto timing = summarize(samples);
        if(timing.min <= 0 || !std::isfinite(timing.mean))
            throw std::runtime_error("Invalid scaled INT8 timing");
        std::cout << std::setprecision(10) << "PERF backend=ck_device_args median_ms=" << timing.median
                  << " mean_ms=" << timing.mean << " min_ms=" << timing.min
                  << " max_ms=" << timing.max << " cv=" << timing.cv
                  << " tops=" << (2.0 * total_m * n * k) / timing.median / 1e9
                  << " throughput_unit=TOPS\n";
    }
    hip_check(hipStreamSynchronize(stream), "finish scaled INT8 benchmark");
    std::cout << "SUMMARY result=PASS full_output=" << (options.validate ? "PASS" : "SKIP") << '\n';
    return 0;
}
} // namespace

int main(int argc, char** argv)
{
    try
    {
        if(argc == 2 && std::string(argv[1]) == "--help")
        {
            std::cout << "Usage: " << argv[0]
                      << " --dtype=int8_bf16|int8_fp16 --m=128 --n=2048 --k=1024 --groups=4\n"
                      << "  --group-lengths=1,0,65 --variable-capacity=128 --allow-empty-groups=1\n"
                      << "  --time=0|1 --validate=0|1 --include-args-build=0|1\n"
                      << "  --warmup=10 --samples=3 --gemm-repeat=50 --correctness-repeats=3\n";
            return 0;
        }
        const auto options = parse_options(argc, argv);
        if(options.dtype == "int8_fp16")
            return run_scaled_int8<ck_tile::half_t>(options);
        return run_scaled_int8<ck_tile::bf16_t>(options);
    }
    catch(const std::exception& error)
    {
        std::cerr << "SUMMARY result=FAIL reason=\"" << error.what() << "\"\n";
        return 1;
    }
}
