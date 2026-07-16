# HCU ck_tile 分组 GEMM

本目录是 HCU 侧 ck_tile 分组 GEMM 路径，和 legacy CK
`DeviceGroupedGemm_*` library/factory 路径不同，也和
`example_hcu/ck_tile/17_fused_moe` 下的 MoE 专用 group GEMM 不同。

## 关注的目标

当前 FP16/BF16 最小 smoke 使用已有 fast-build target：

```bash
cmake --build . --target tile_example_grouped_gemm_fp16_fast -j 16
cmake --build . --target tile_example_grouped_gemm_bf16_fast -j 16
```

显式 smoke target 会构建上述两个 target，并运行 FP16/BF16 的 NT/NN/TN、
fixed-K/variable-K 以及非对齐 M/N/K correctness 检查；最后还会运行一个绕过
example CLI、直接链接现有 C ABI 的 layout/padding consumer：

```bash
cmake --build . --target tile_example_grouped_gemm_fp16_bf16_fast_smoke -j 16
```

等价的直接运行命令：

```bash
./bin/tile_example_grouped_gemm_fp16_fast -prec=fp16 -a_layout=R -b_layout=C -group_count=2 -Ms=128,256 -Ns=128,128 -Ks=128,128 -stride_As=128,128 -stride_Bs=128,128 -stride_Cs=128,128 -validate=1 -warmup=1 -repeat=1
./bin/tile_example_grouped_gemm_fp16_fast -prec=fp16 -a_layout=R -b_layout=R -group_count=2 -Ms=128,256 -Ns=128,128 -Ks=128,128 -stride_As=128,128 -stride_Bs=128,128 -stride_Cs=128,128 -validate=1 -warmup=1 -repeat=1
./bin/tile_example_grouped_gemm_bf16_fast -prec=bf16 -a_layout=R -b_layout=C -group_count=3 -Ms=128,256,192 -Ns=128,128,256 -Ks=128,192,256 -stride_As=128,192,256 -stride_Bs=128,192,256 -stride_Cs=128,128,256 -validate=1 -warmup=1 -repeat=1
./bin/tile_example_grouped_gemm_bf16_fast -prec=bf16 -a_layout=R -b_layout=R -group_count=3 -Ms=128,256,192 -Ns=128,128,256 -Ks=128,192,256 -stride_As=128,192,256 -stride_Bs=128,128,256 -stride_Cs=128,128,256 -validate=1 -warmup=1 -repeat=1
```

也可以通过 CTest 运行这组 17 项 smoke：

```bash
ctest -R 'tile_example_grouped_gemm_.*_fast_smoke' --output-on-failure
```

## 现有 C ABI（框架入口）

现有框架（包括 aiter）继续从本 example 目录包含原有 header：

```cpp
#include "grouped_gemm.hpp"
```

`grouped_gemm.hpp` 保留 dtype 枚举、descriptor、workspace 与 run 声明；
`CK_GROUPED_GEMM_ABI_DEFINED` guard 用于和 aiter 的纯 C ABI 声明共存。
本次不新增另一个 C ABI 接口头文件。

FP16/BF16 实现可通过以下 target 构建为位置无关的静态库：

```bash
cmake --build . --target ck_tile_dcu_grouped_gemm -j 32
```

独立 C ABI correctness consumer 的构建与运行命令为：

```bash
cmake --build . --target tile_example_grouped_gemm_c_api_smoke -j 32
./bin/tile_example_grouped_gemm_c_api_smoke
```

该程序不调用 `run_grouped_gemm_example`，而是分别构造对齐 NN variable-K 和
非对齐 TN descriptor，直接调用 `ck_tile_dcu_grouped_gemm_run` 并验证 FP16/BF16
输出。因此 layout 与 padding 能力不依赖 example CLI；fast CLI 是否开放
RowMajor B/ColumnMajor A 分别由 `GemmConfig::SupportsFastRowMajorB` 和
`GemmConfig::SupportsFastColumnMajorA` 声明。

ck4inductor 的 `ck_tile_grouped_gemm` 模块枚举 FP16/BF16 × NT/NN/TN 六个
consumer contract，并提供 variable-K、padding、最小 K、现有 ABI 符号和 CMake
target 信息。它是 ROCm/DCU Torch backend 的接入材料；未修改的 PyTorch
TorchInductor 仍需要 grouped lowering/template 才会自动把 `aten._grouped_mm`
调度到该 C ABI。

## 当前边界

- FP16/BF16 fast target 覆盖 NT（A 行主序、B 列主序）、NN（A/B 行主序）和
  TN（A 列主序、B 行主序），C 均为行主序。其他 fast dtype 保持原有限制。
- variable-K 表示同一次 grouped launch 中每个 descriptor 的 K 可以不同；
  它不等同于 split-K，且当前所有 group 仍使用同一个 `kbatch`。
- 对齐 shape 保持原 `GemmConfigComputeV4` fast 配置；任一 group 存在 M/N/K
  尾块时，FP16/BF16 example 与 C ABI 自动改用 `kPadM/N/K=true` 配置。
- padding 是 tensor descriptor 的逻辑 right-pad，不要求调用方申请物理 padded
  A/B/C 显存；调用方仍按逻辑 M/N/K 和 stride 分配。workspace 仍只存每组
  `TransKernelArg`，大小不随 padded M/N/K 增长。
- 全 padding 路径为边界安全会降低相应向量化粒度，性能可能不同；后续性能
  调优应分别比较对齐 fast 路径与非对齐 padding 路径。
- 现有 FP16/BF16 fast target 使用 CompV4 persistent double-buffer 路径；
  当前要求 `num_loop = K / K_Tile >= 2`。
- FP16/BF16 的 `K_Tile=64`，因此当前 fast example 的 K 维 baseline
  应满足 `K>=128`。
- `num_loop < 2` 的小 K shape 已在 host-side launch/support check 中显式
  拒绝；若要真正支持，需要补充 small-K fallback。
- 本目标不注册 standard grouped GEMM factory/profiler instance，也不覆盖
  grouped GEMM MultiABD。

## `num_loop >= 2` 限制

`tile_example_grouped_gemm_fp16_fast` 对应的配置是
`GemmConfigComputeV4<fp16>`：`M_Tile=64`、`N_Tile=128`、`K_Tile=64`、
`DoubleSmemBuffer=true`、`Persistent=true`。BF16 fast target 同样使用
`K_Tile=64`。

CompV4 pipeline 的 `PrefetchStages=2`，tail 选择逻辑会把奇数 loop 映射到
`TailNumber::Three`。当前 fast example 以至少 2 个 K tile 作为支持前提：

```text
num_loop = K / K_Tile >= 2
```

对 FP16/BF16 fast target 来说，这等价于：

```text
K >= 2 * 64 = 128
```

因此 README 中的 correctness smoke 只选用 `K>=128` 的 shape。当前 fast path
会在 host 侧拒绝 `num_loop < 2` 的 group；后续如果要把 `K<128` 纳入支持范围，
应先增加专门的小 K 路径。
