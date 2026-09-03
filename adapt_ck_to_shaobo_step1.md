# Composable Kernel 适配 Shaobo（gfx946）Step 1

> 状态：Step 1 已完成；后续新增能力已归档，等待独立项目启动
>
> 首次创建：2026-09-01
>
> 当前阶段：P1 低风险/normalization 队列已完成；P2.1 legacy `05_gemm` FP16 NT、P2.6 GEMM wavelet、P2.11 backward-data mixed CShuffle、P2.15 backward-weight wavelet、P2.19-P2.21 BF16 XDL v3 GEMM、P3.3 DSReadM conv、P3.5b/P4.10/P4.12/P4.14 BF16 grouped GEMM fast/forced-baseline/V3/MLS 已按证据跳过；P2.2-P2.5/P2.22-P2.26 GEMM/fused GEMM、P2.7-P2.8/P2.27-P2.29 legacy grouped-conv forward NHWC/NCHW/mixed/FP8/INT4 CShuffle、P2.9-P2.10 backward-data NHWC/NCHW v2 CShuffle、P2.12-P2.14/P2.16 backward-weight、P2.17 bias+ReLU、P2.18 bias+residual+ReLU grouped-conv forward CShuffle、P3.1-P3.2 CK Tile conv、P3.4 CK Tile Conv3D、P3.5a CK Tile FP16 grouped GEMM、P3.5c1-c3 三种 grouped quant 的 FP16-output、P3.5d1-d3 selector/device-args/C API 已通过；P3.6 `ck_tile/18_moe_quant` scalar-output PASS、vectorized-output SKIP-PMD；P3.7 `ck_tile/17_fused_moe`、P4.1 `ck_tile/01_fmha` basic forward、P4.7 basic backward PASS；P4.2 `ck_tile/15_fused_conv` MLS probe、P4.8 Wasp V3 persistent 子路径均已完成构建/真机回归并按各自 PMD 指令执行缺口跳过运行验收；P4.3a-c `ck_tile/38_block_scale_gemm` 三种 quant 的 FP16-output PASS、BF16-output SKIP-PMD；P4.4 grouped-GEMM production-selector C API PASS；P4.5 gfx936-only logical-K-tail smoke 架构边界已修正并回归通过；P4.6 grouped-GEMM device-args benchmark correctness-only 模式及三架构验收 PASS；P4.9 FP16 forced-baseline、P4.11 FP16 V3 reg-prefetch、P4.13 FP16 MLS grouped-GEMM target PASS；P4.15-P4.16 FP8/BF8 fast 完成构建闭包/架构收口并 `SKIP-PMD`；P4.17-P4.18 INT8/INT4 fast 真机 PASS、`dp4x:disable` PMD 跳过；P4.19 当前 94 个 enabled target 已完成逐项映射闭环；P4.20 未入图路径已分层盘点；P5.1-P5.2 legacy TT/TN v1 与 P5.6 NT v1 wavelet 按 PMD 执行链缺口跳过；P5.3 TT v2、P5.4 TN v2 wavelet 与 P5.5 TT v2 wavelet 均为双真机既有坏例；P5.7-P5.8 FP32 grouped-conv 因 gfx946 无当前实例所需 FP32×8 MMAC 而跳过；P5.9-P5.10/P5.14 NHWC/NCHWC32/NCHWC FP16 grouped-conv 已恢复 enrollment 并三架构 PASS；P5.11 NHWC INT8 grouped-conv 按 PMD `dp4x:disable` 能力边界跳过；P5.12-P5.13 NHWC v2 FP16/wavelet 为三架构共享的既有坏例；P5.15-P5.16 backward-data NHWC/NCHWC32 FP16 已补 gfx946 guard并恢复 enrollment，三架构 PASS；P5.17-P5.18 backward-data NHWC v2 FP16/wavelet 为双真机既有坏例；P5.19-P5.20 backward-data NCHW/NHWC FP32 按 gfx946 FP32×8 MMAC 缺口跳过；P5.21-P5.22 forward bias+ReLU NHWC FP16 non-CShuffle v2/wavelet 为双真机既有坏例；P5.23 bias+ReLU CShuffle mixed 为共享编译坏例，机械修复后双真机 correctness 仍失败；P5.24 bias+ReLU NCHW FP16 CShuffle 已补 gfx946 4B LDS direct-load 闭包并三架构 PASS；P5.25 BatchNorm NHWC inference unique-target enrollment 已三架构 PASS；P5.26 BatchNorm NHWC forward-training unique-target enrollment 已三架构 PASS；P5.27 BatchNorm NHWC backward unique-target enrollment 已三架构 PASS；P5.28 fused-conv base Wasp V2R1 已按 PMD modifier 执行缺口跳过；P5.29 fused-conv TLS V1 已按共享坏 baseline 跳过；P5.30 fused-conv MLS V1 已完成 correctness-only runner 并通过 gfx946/gfx938；P5.31 fused-conv CShuffle Wasp V2R1 已按 PMD modifier 执行缺口跳过；P5.32 fused-conv TLS-Wasp V1 已按共享坏 baseline 跳过；P5.33 fused-conv TLS-Wasp persistent V1 已以 gfx946-only unique target 通过 PMD correctness，双真机架构边界回归通过；P5.34 fused-conv MLS-Wasp persistent V1 已按 PMD DSReadM 执行缺口跳过；P6.1-P6.5 img2col/reduce/permute/smoothquant/topk-softmax、P6.7-P6.10 generic elementwise 四 target、P6.12 basic batched-transpose、P6.15-P6.16 Moe Smoothquant aligned/padded-N 均完成最小闭包并三架构 PASS；P6.6/P6.11 两轮覆盖与优先级审计完成；P6.13 padded non-LDS 与 P6.14 LDS batched-transpose 均因共享 HCU core API/ABI 边界按证据跳过
>
> 最新切片：P6.86 RMSNorm2D N≤128 save-RMS 已完成三架构 correctness 与双真机性能 A/B；本轮深入适配收口
>
> 目标：以 `example_hcu` 为验收入口，逐个完成 gfx946 Perf Model 正确性验证，并在 gfx936、gfx938 做同路径回归。Perf Model 只做功能/正确性验证，不采集、不比较性能。

### 实时进度看板（2026-09-03）

| 工作面 | 已完成 | 进行中 | 后续处理 |
| --- | --- | --- | --- |
| enabled target 基础映射 | 94/94 已分类闭环 | 0 | 无未分类 target |
| 深入适配 | P6.1-P6.86 | 0 | 本轮已收口 |
| 后续能力扩展来源判定 | 5/5 已审计 | 0 | 任务前并非 gfx936/gfx938 完整可用能力，统一转入后续项目清单 |
| 后续独立项目 | 0/6 | 0 | RMSNorm save-RMS 三类更大 shape、LayerNorm save-MV、Add-RMSNorm no-save 缺失 shape、CShuffle LDS correctness harness 均不在本 Step 1 实施 |

Step 1 当前没有进行中切片。后续不得把“模板中存在字段/公共 pipeline 有参数”当成 gfx936/gfx938 的原有可用能力；必须同时具有已注册实例、可达 selector、真实输出和 correctness reference 才算原有能力。

## 1. 硬边界

- 本地目标仓库：`D:\src\AIDev\ck\composable_kernel`
- 本地基线：分支 `main-local`，HEAD `07581109898bb779054f331ca4a0c13e3b1aad29`
- 上游参考仓库：`D:\src\AIDev\ck\composable_kernel_github`，分支 `develop`，HEAD `798ec4fb2baab96f35a4fbe941645983c1de68a5`
- gfx946 目标：profile `Perf-model-10.59.41.48`，容器 `zxw_perf_8500`，host/container 路径均为 `/home/zhangxiaowei/composable_kernel`
- gfx936 回归：BMZ，容器 `zxw_vllm_0810`，路径 `/home/users/xiaowei.zhang/ai/composable_kernel`
- gfx938 回归：TJ-yeleng-176.252，容器 `zxw_vllm_0901`，host 路径 `/home/hg/zhangxiaowei/composable_kernel`，container 路径 `/wksp/composable_kernel`
- 不比较 Perf Model 的 ticks、延迟或吞吐，不以 PMD 结果宣称 Shaobo 真机支持。
- 不做整树 merge，不覆盖现有 HCU 路径；每次只适配一个 example 的最小依赖闭包。
- 不 reset、不清理用户工作区、不提交无关 dirty 文件。
- 若某条 Shaobo 指令在 Perf Model 中存在不可解决的实现问题：保存首个终止错误、最小复现/指令路径与已做 A/B，标记 `SKIP-PMD` 或 `SKIP-COMPILER`，然后继续下一个独立模块。

## 2. 已知工作区状态与保护规则

### 2.1 本地

- `script/cmake-ck-dev.sh` 在任务开始前已被修改：默认 `CK_GPU_TARGETS` 从 `gfx936;gfx938` 改成 `gfx946`。
- `.planning/` 在任务开始前已是 untracked。
- 上述内容视为既有用户改动。每次提交前只 `git add --` 本轮明确修改的路径，并检查 `git diff --cached`；除非后续能够证明脚本改动属于某个 example 的必要最小修复，否则不纳入提交。

### 2.2 远端

- Perf Model 按用户约定不比较 Git 状态；以路径、工具链、实际构建参数和本轮同步文件哈希作为证据。
- gfx936/gfx938 当前 HEAD 均为 `07581109898bb779054f331ca4a0c13e3b1aad29`；`hcu-remote` 会产生 `.hcu-remote/` untracked 目录，不纳入源码判断。
- 全局 `remote_config.json` 中 TJ-yeleng 的容器/路径已过期。本轮使用工作区临时配置 `hygon_tmp/shaobo_remote_config.json`，以用户指定的 `zxw_vllm_0901:/wksp/composable_kernel` 为准，不修改全局配置。

## 3. 从 AITER 与 Perf Model 经验继承的规则

1. `GPU_CHIP=sb` 选择 PMD 模型，不代表编译目标已是 gfx946；CK 配置必须显式使用 `GPU_TARGETS=gfx946`/`CK_GPU_TARGETS=gfx946`，并检查实际编译命令或构建缓存中没有 `--offload-arch=native`。
2. Perf Model 入口显式设置：

   ```bash
   export ROCM_PATH=/opt/rocm
   export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:/opt/hyhal/bin:${PATH}
   export LD_LIBRARY_PATH=/opt/rocm-6.3.3/lib:/opt/rocm/lib:/opt/rocm/pmd:${LD_LIBRARY_PATH:-}
   export PMD_PATH=/opt/rocm/pmd
   export GPU_CHIP=sb
   ```

3. PMD 上的 reference 尽量放在 CPU；`example_hcu` 已带 host reference 时，必须关闭计时并保留完整结果校验。
4. 新 builtin/指令按“编译器接受 -> gfx946 code object -> 目标 symbol ISA -> layout/identity -> example correctness”分层，不从 builtin 名称或源码 guard 直接推断支持。
5. 不泛化 AITER AttnRes 的 `number<3>` workaround，不机械引入 FA3 的 MLS/ABarrier/WDRA 拓扑；只有当前 CK example 的局部证据成立时才改。
6. 错误分类保持分层：配置、源码/API、编译器后端、PMD 执行模型、既有基线分别记录；“发生在 Perf Model”本身不足以证明 PMD 缺陷。

## 4. 验收门禁

每个 example 使用相同门禁，状态只允许 `PASS`、`FAIL`、`PARTIAL`、`NOT RUN`、`BLOCKED`：

- G0 Provenance：本地/远端路径、HEAD/文件状态、容器、编译器、目标架构和 build 目录。
- G1 Static/configure：CMake target、架构 guard、依赖和实际 `GPU_TARGETS=gfx946` 配置一致。
- G2 Focused build：在 Perf Model 容器精确构建该 target；记录首个终止错误。
- G3 gfx946 PMD correctness：`GPU_CHIP=sb`，关闭 timing，运行最小代表 shape，完整 host reference 校验通过。
- G4 Regression：gfx936、gfx938 分别配置/构建/运行同一测试；不以一个架构代替另一个。
- G5 Broader integration：仅当该能力切片包含 profiler/library/JIT/package 时执行；Step 1 默认 deferred。
- G6 Performance：gfx946 Perf Model 明确 `NOT RUN`；若不能用架构条件、目标 ISA/code object 或其他直接证据排除对 gfx936/gfx938 的性能影响，则必须在两台真机上做相同工具链、shape、预热/迭代策略的 baseline/candidate A/B。
- Commit：G1-G4 闭环后，只提交该 example 的最小代码、测试和本文件更新；失败证据也先写入本文，但未闭环的业务改动不混入已通过 example 的提交。

## 5. 优先级与实施队列

优先级依据是依赖风险，而不是目录编号。

### P0：配置、架构识别与基础执行链

- [x] 阅读 AITER Shaobo 经验记录和 Perf Model 试用小结。
- [x] 冻结本地 dirty 状态和三台服务器的实际路径/容器。
- [x] 盘点 `example_hcu` 顶层目录、已启用 `ck_tile` 子目录与 CMake target。
- [x] 在 Perf Model 通过 `script/cmake-ck-dev.sh` 生成独立 gfx946 build tree，并核对 offload arch。

### P1：低风险、非矩阵指令入口

1. **P1.1 `04_tensor_layout_transform`（已完成）**
   - target：`example_tensor_layout_transform_ngchw_ngchwc32_fp16`
   - 首个 shape：`G=1,N=1,K=32,C=32,YX=1x1,HiWi=2x2,stride=1,dilation=1,pad=0`
   - 运行参数：verification=1，integer init=1，timing=0。
   - 价值：覆盖 `ck/ck.hpp`、global load/store、launch 和 CPU reference；不先依赖 MMAC/MLS/atomic。
2. `08_batchnorm`
   - [x] P1.2a forward-training FP16/NCHW example。
   - [x] P1.2b backward FP16/NCHW example。
   - 注意历史上 half host test-data NaN 可能属于 host 基线问题，不能误归因 gfx946。
3. normalization 与低风险 ck_tile 组合算子
   - [x] P1.3 `ck_tile/02_layernorm2d` FP16 forward、两种 reduction 策略。
   - [x] P1.4 `ck_tile/10_rmsnorm2d` FP16 forward。
   - [x] P1.5a `ck_tile/11_add_rmsnorm2d_rdquant` 简单 FP16/INT8 example。
   - [x] P1.5b `ck_tile/11_add_rmsnorm2d_rdquant` 完整 instance runner（vector_N=1 PASS；vector_N>1 `SKIP-PMD`，见证据日志）。
   - [x] P1.6 `ck_tile/13_moe_sorting`。
   - reduction、shuffle、quant/store 分开切片；生产 shape 与小 shape 都要验证 mutation 和完整输出。

### P2：经典 MMAC GEMM/Conv

1. `05_gemm`（先 FP16 NT，再 TN/TT/NN 和 CShuffle）
2. `06_gemm_bias_activation`
3. `01_grouped_conv_fwd_mmac`
4. `02_grouped_conv_bwd_data_mmac`
5. `03_grouped_conv_bwd_weight_mmac`
6. `07_grouped_conv_fwd_bias_activation`
7. `09_grouped_conv_fwd_bias_add_activation`
8. `24_batched_gemm`、`25_gemm_xdl_v3`

每个 MMAC 切片额外检查 builtin、C fragment/layout、CShuffle 和最终 store。FP16/BF16/FP8/INT4 不能互相代证。

### P3：ck_tile 基础与组合算子

1. `ck_tile/03_gemm`
2. `ck_tile/14_conv`
3. `ck_tile/16_conv3d_fwd`
4. `ck_tile/19_grouped_gemm`：FP16/BF16、量化、selector/C API 各自建子矩阵。
5. `ck_tile/18_moe_quant`
6. [x] `ck_tile/17_fused_moe`

### P4：高风险 Shaobo 指令/异步流水

1. [x] `ck_tile/01_fmha` basic forward（split-KV/append-KV/paged-KV/backward 后续独立切片）
2. [x] `ck_tile/15_fused_conv`（MLS probe 构建闭包完成，运行 `SKIP-PMD`）
3. `ck_tile/38_block_scale_gemm`

这些路径涉及 lit/lts MMAC、MLS/BPS、复杂 wait/barrier、FP8/MX 或多阶段流水。只有低层 builtin/layout probe 成立后才运行完整 example；Perf Model 明确缺陷无法局部解决时记录并跳过。

### 当前未进入构建图的目录

`ck_tile/04_img2col`、`05_reduce`、`06_permute`、`09_topk_softmax`、`12_smoothquant` 当前在 `example_hcu/ck_tile/CMakeLists.txt` 中未启用。本 Step 1 先记录为 `DEFER-NOT-ENROLLED`，不把“未启用”写成“不支持”；完成已启用目录后再分别建立启用/适配切片。

## 6. 当前架构矩阵

| Architecture | 环境 | G0 | G1 | G2 | Correctness | Performance | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gfx946 | Perf-model-10.59.41.48 / `zxw_perf_8500` | PASS | PASS | PASS | PASS（PMD，2x2/8x8） | NOT RUN（按要求） | P1.1 PASS；不代表真机结论 |
| gfx936 | BMZ / `zxw_vllm_0810` | PASS | PASS | PASS | PASS（2x2/8x8） | PASS（P1.1 A/B） | P1.1 regression PASS |
| gfx938 | TJ-yeleng / `zxw_vllm_0901` | PASS | PASS | PASS | PASS（2x2/8x8） | PASS（P1.1 A/B） | P1.1 regression PASS |

## 7. 分步进度与证据日志

### Step 0.1：资料与规则冻结 — PASS（2026-09-01）

- 已重点阅读 `D:\src\AIDev\aiter\readme_to_adapt_aiter_to_shaobo.md`。
- 已提取 `D:\AI\knowldege\HCU\raw\perf-model机器使用小结.pdf`；PDF 记录了 `ROCM_PATH`、`PMD_PATH`、`GPU_CHIP=sb`、Shaobo=`gfx946`、`GPU_ARCHS` 和 WDRA trap-handler 等现场注意事项。PDF 文本编码有损，因此只采用与当前容器实测一致的配置事实。
- 结论：PMD correctness、真实硬件 correctness、performance 三类证据严格分离。

### Step 0.2：本地与远端 G0 — PASS（2026-09-01）

- 本地 HEAD：`07581109898bb779054f331ca4a0c13e3b1aad29`；既有 dirty 见第 2 节。
- Perf Model：连接、容器、`/home/zhangxiaowei/composable_kernel`、`/opt/rocm/bin/hipcc`、`/opt/rocm/pmd/gem5.opt` 与 `configs/gpu.py` 均存在。hipcc 报告 HIP 6.3、clang 18。非交互 shell 中 `GPU_CHIP/GPU_ARCHS/PMD_PATH` 均未预置，所以测试脚本必须显式导出。
- gfx936：实际设备 `gfx936`，源码 HEAD 与本地一致，AICC HIP 6.2、clang 18。
- gfx938：按本轮指定的 `zxw_vllm_0901:/wksp/composable_kernel` 成功连接；实际设备 `gfx938`，源码 HEAD 与本地一致，AICC HIP 6.2、clang 18。

### Step 0.3：example 清单与首个切片 — PASS（2026-09-01）

- 顶层已启用目录：`01`-`09`（其中若干 target 注释）、`24_batched_gemm`、`25_gemm_xdl_v3`、`ck_tile`。
- `ck_tile` 已启用：`01_fmha`、`02_layernorm2d`、`03_gemm`、`10_rmsnorm2d`、`11_add_rmsnorm2d_rdquant`、`13_moe_sorting`、`14_conv`、`15_fused_conv`、`16_conv3d_fwd`、`17_fused_moe`、`18_moe_quant`、`19_grouped_gemm`、`38_block_scale_gemm`。
- 首个切片确定为 `04_tensor_layout_transform`，理由见 P1.1。

### Step 1.1：`04_tensor_layout_transform` — PASS（2026-09-01）

计划命令（Perf Model，独立 build tree）：

```bash
export ROCM_PATH=/opt/rocm
export PATH=/opt/rocm/bin:/opt/rocm/llvm/bin:/opt/hyhal/bin:${PATH}
export LD_LIBRARY_PATH=/opt/rocm-6.3.3/lib:/opt/rocm/lib:/opt/rocm/pmd:${LD_LIBRARY_PATH:-}
export PMD_PATH=/opt/rocm/pmd
export GPU_CHIP=sb
export CK_GPU_TARGETS=gfx946
mkdir -p build_shaobo_step1
cd build_shaobo_step1
bash ../script/cmake-ck-dev.sh ..
cmake --build . --target example_tensor_layout_transform_ngchw_ngchwc32_fp16 -j 8
./bin/example_tensor_layout_transform_ngchw_ngchwc32_fp16 \
  1 1 0 2 1 1 32 32 1 1 2 2 1 1 1 1 0 0 0 0
```

注意：最后一个执行参数 `time kernel=0`；日志中的 `Perf:` 字段即使由 example 固定打印，也不作为性能结果记录。

实验记录：

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P1.1-E0 | 在新目录 `build_shaobo_step1`，显式 `GPU_CHIP=sb`、`CK_GPU_TARGETS=gfx946` 运行 `cmake-ck-dev.sh` | FAIL；gfx928/936/938-only instance 被架构过滤后，三个 CMakeLists 仍对不存在的 target 调用 `target_compile_options` | `G1/CMake`，不是 PMD 指令问题。为 `fused_conv2d_{cb,cba,cbra}_gfx928` 的 compile options 增加 `if(TARGET ...)`，重新配置 |
| P1.1-E1 | 应用 `if(TARGET ...)` 后重新配置 | FAIL；所有 HCU instance 都被 gfx946 过滤，聚合静态库 `device_operations` 无 source | 首个切片应把通用 `layout_transform` instance 的架构清单扩到 gfx946，而不是制造空库；增加 `gfx946` 后重配 |
| P1.1-E2 | `layout_transform` 注册 gfx946 后配置、focused build、2x2 shape 运行 | Configure/Build PASS，实际编译和链接均有 `--offload-arch=gfx946`、无 `native`；correctness FAIL，出现 NaN 和 `max err: 9`，但旧 example 错误地返回 0 | 先修复 harness：`check_err=false` 和 unsupported argument 必须传递为进程失败；然后做 shape/host-data A/B |
| P1.1-E3 | 修复 harness 后运行 8x8 shape | FAIL（进程正确返回 1）；错误签名与 2x2 相同，仍为奇数位置 NaN，并重复报告 `ds_read_b64 ... addr is not aligned` | 排除“小于 tile”这一单因。代码推导：FP16 `HPerBlock=32` 的 64-byte 行加 4-byte LDS pad 后成为 68 bytes，使隔行 b64 地址不满足 8-byte 对齐；仅 gfx946 把 pad 改为 8 bytes 做单变量 A/B |
| P1.1-E4 | 仅在 `__gfx946__` 下把 LDS row pad 从 4 bytes 改为 8 bytes，重编 example 后运行 2x2/8x8 | PASS；两组 host reference 均通过，原有 `ds_read_b64 ... addr is not aligned` 消失 | 单变量 A/B 支持当前 gfx946 PMD 路径的根因是 b64 LDS 地址对齐；不外推为 Shaobo 真机性能或完整 ISA 结论 |
| P1.1-E5 | 精确构建 `device_layout_transform_instance` | PASS；6 个 layout-transform instance 源文件均以 `--offload-arch=gfx946` 构建并完成静态库聚合 | G2 focused library closure PASS |
| P1.1-E6 | 同步最终文件到 gfx936/gfx938，分别增量构建并运行 2x2/8x8 | PASS；gfx936 实际命令为 `--offload-arch=gfx936`，gfx938 为 `--offload-arch=gfx938`，四次 correctness 均通过 | G4 两个架构独立回归 PASS |
| P1.1-E7 | gfx936/gfx938 对 `0758110` baseline 与当前 candidate 做同会话交替 A/B；2 shapes，5 次 warmup、50 次内部计时、7 次外层重复 | PASS；gfx936 median：`0.100560→0.096112 ms`（-4.42%）、`0.085168→0.085152 ms`（-0.02%）；gfx938：`0.127263→0.125329 ms`（-1.52%）、`0.118879→0.119023 ms`（+0.12%） | 未观察到性能回退；gfx938 的 +0.12% 归为运行噪声。baseline 与 candidate 使用相同机器、GPU 0、工具链、CMake 脚本、shape 和计时策略 |

P1.1-E0 的首个终止错误：

```text
Cannot specify compile options for target
"device_fused_conv2d_cb_gfx928_instance" which is not built by this project.
```

P1.1-E1 的首个终止错误：

```text
CMake Error at library/src/tensor_operation_instance_hcu/gpu/CMakeLists.txt:56 (add_library):
  No SOURCES given to target: device_operations
```

同时观察到 HIP CMake 的自动设备探测会在 PMD 中启动 `amdgpu-arch` 并发出未设置 `AMDGPU_TARGETS` 的 warning；该 warning 不是本轮 configure 的终止错误。重新配置成功后必须从实际 example 编译命令确认 `--offload-arch=gfx946`，不能用 warning 或 `GPU_TARGETS` cache 单独下结论。

P1.1 最终门禁：

| Gate | 状态 | 证据摘要 |
| --- | --- | --- |
| G0 | PASS | 三台环境、容器、路径、编译器与目标架构均已核对 |
| G1 | PASS | gfx946 独立 build tree 配置成功；cache 与实际命令均指向 gfx946 |
| G2 | PASS | example target 与 `device_layout_transform_instance` 均完成 focused build |
| G3 | PASS | `GPU_CHIP=sb` 下 2x2、8x8 完整 host reference 校验通过；timing=0 |
| G4 | PASS | gfx936、gfx938 各自重编并通过相同两组 shape |
| G5 | NOT RUN | 本切片不包含需要额外 package/JIT/profiler 验收的交付面 |
| G6 | PASS / NOT RUN | gfx936、gfx938 性能 A/B PASS；gfx946 按要求不在 CPU 模拟 Perf Model 测性能 |

本切片修改闭包：通用 layout-transform instance 注册 gfx946；gfx946 专用 LDS 8-byte row pad；example 正确传播校验/unsupported 失败；三个架构过滤后的可选 fused-conv target 增加 CMake 存在性保护。任务开始前的 `script/cmake-ck-dev.sh` 与 `.planning/` 未纳入。

### Step 1.2a：`08_batchnorm` forward-training — PASS（2026-09-01）

- 子切片：`example_batchnorm_forward_training`，FP16、NCHW、multiblock Welford；backward 与其他 dtype 独立闭环。
- 已知风险：历史上集成 CK 的 half host test-data 路径曾在 GPU kernel 前产生 NaN。若复现，必须先检查 host tensor 和 H2D/D2H，不能直接归因 gfx946/PMD。
- 性能规则：gfx946 仅验证正确性；gfx936/gfx938 在正确性通过后，若修改无法由架构条件或等价 code object 直接证明无性能影响，则补做 baseline/candidate A/B。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P1.2a-E0 | gfx946 构建 `example_batchnorm_forward_training` | PASS；实际编译/链接为 `--offload-arch=gfx946` | G2 build observation |
| P1.2a-E1 | PMD 运行 `{N,C,H,W}={2,16,1,8}` | FAIL；`reduce_length <= K_BlockTileSize` 导致 `blkGroupSize=1`，V2 invoker 不 launch，PMD 无 dispatch，输出保持 0 | 既有参数域缺陷；不是 Shaobo 指令失败 |
| P1.2a-E2 | PMD 运行 `{2,16,64,65}` | `IsSupportedArgument=false`；V2 要求末维 W 满足 8 元素 vector | 正确拒绝，不作为 gfx946 failure |
| P1.2a-E3 | PMD、gfx936、gfx938 运行 `{2,16,65,64}` | 三者均发生相近 FP16 数值误差，max error 为 0.035-0.043 | 跨架构既有数值基线限制；不降低容差，不归因 PMD |
| P1.2a-E4 | 三架构运行 example 默认 `{128,16,6,512}`，verification=1、init=1、timing=0 | PASS；gfx946 PMD 有两次 kernel dispatch，gfx936/gfx938 分别以自身 offload arch 构建并通过 | G3/G4 PASS |

门禁结果：G0 PASS，G1 PASS，G2 PASS，G3 PASS，G4 PASS，G5 NOT RUN，G6 NOT RUN。G6 不需要补测的直接依据是该子切片没有业务代码改动，gfx936/gfx938 candidate 与 baseline 相同；PMD 仍不做性能测试。`device_batchnorm_instance` 尚未注册 gfx946，因为现有 CMake target 同时包含 forward/backward/infer 和多个 dtype，在对应能力矩阵闭环前不扩大支持声明。

### Step 1.2b：`08_batchnorm` backward — PASS（2026-09-01）

- target：`example_batchnorm_backward`，首个范围为 FP16/NCHW、example 默认 shape、timing=0。
- 优先检查历史 half host-data NaN：先区分 host 初始化、H2D/D2H、device kernel 和 PMD 指令证据，再决定修复或跳过。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P1.2b-E0 | gfx946 focused build `example_batchnorm_backward` | PASS；实际编译/链接为 `--offload-arch=gfx946` | G2 build observation |
| P1.2b-E1 | PMD 运行 `{128,16,3,1024}`、`haveSavedMeanInvVar=0`、init=3、timing=0 | PASS；由 device 现场计算 mean/invVariance，dx/dscale/dbias 完整校验通过 | G3 runtime observation |
| P1.2b-E2 | PMD 同 shape，`haveSavedMeanInvVar=1` | PASS；saved mean/invVariance 分支完整校验通过 | G3 runtime observation |
| P1.2b-E3 | gfx936/gfx938 各自构建，并分别运行 computed/saved 两条路径 | PASS；四次 correctness 均通过，实际 offload arch 分别为 gfx936/gfx938 | G4 regression observation |

PMD 在运行中报告 `v_mad_u64_u32 ... read vgpr before writing`、`Inst_VOP3__V_DIV_FIXUP_F32 not test` 等 warning，但两条路径均完成 dispatch 并通过完整输出校验，因此当前只记录为 PMD warning，不能据此标记指令不支持。门禁结果：G0-G4 PASS，G5 NOT RUN，G6 NOT RUN；本子切片无业务代码改动，回归 candidate 与 baseline 相同，不存在需要性能 A/B 的不确定代码差异。

### Step 1.3：`ck_tile/02_layernorm2d` — PASS（2026-09-01）

- target：`tile_example_layernorm2d_fwd`；首个切片为 FP16 input/output、无 fused add/quant、`save_mv=0`。
- example 原先把 `stream_config.time_kernel` 硬编码为 `true`，无法满足 Perf Model 禁止性能测试的边界。本切片增加 `-time` 参数（默认 1，保持原有真机行为），`-time=0` 时只 dispatch 一次并明确输出 `timing:off`。
- CMake target 包含 894 个对象，覆盖 FP16/BF16、bias/quant 与多个 N 配置；三个架构均完成整个 target 的构建，不把只编译 runner 当作 focused build 通过。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P1.3-E0 | gfx946 独立 build tree 构建 `tile_example_layernorm2d_fwd` | PASS；完整生成源闭包编译、链接，实际命令为 `--offload-arch=gfx946` | G2 build observation |
| P1.3-E1 | PMD 运行 FP16 `m=2,n=128`、`time=0,warmup=0,repeat=1` | PASS；选择 `layernorm2d_fwd_fp16_4x128_4x1_1x128_1x8_wpr_pn`，单次 dispatch，`timing:off, valid:y` | G3，warp-per-row 路径 |
| P1.3-E2 | PMD 运行 FP16 `m=4,n=1024`、相同 correctness 参数 | PASS；选择 `layernorm2d_fwd_fp16_2x1024_2x2_1x512_1x8_bpr_pn`，单次 dispatch，`timing:off, valid:y` | G3，block-per-row 路径 |
| P1.3-E3 | gfx936 独立完整构建并运行相同两组 shape | PASS；实际 offload arch 为 gfx936，两条路径均 `timing:off, valid:y` | G4 regression observation |
| P1.3-E4 | gfx938 独立完整构建并运行相同两组 shape | PASS；实际 offload arch 为 gfx938，两条路径均 `timing:off, valid:y` | G4 regression observation |

G6 判定为 `NOT RUN`，不是性能通过：本切片唯一代码差异位于 host runner 的参数解析、`stream_config` 计时开关和输出分支；该文件不含 device kernel 或 launch 定义，CMake 将未修改的 `LAYERNORM2D_FWD_GEN_BLOBS` 作为独立 source 编译，默认 `-time=1` 仍保持原行为。因此可以用源码/构建闭包直接排除对 gfx936/gfx938 device kernel 与计时区间的影响，无性能不确定性，不触发真机 A/B。gfx946 PMD 未做任何性能测试。

门禁结果：G0-G4 PASS，G5 NOT RUN，G6 NOT RUN。PMD 运行期间出现通用 source/cacheline warning，但两条路径均完成 dispatch 和 host reference 校验；没有发现可归类为 Shaobo 指令不支持的终止错误。

### Step 1.4：`ck_tile/10_rmsnorm2d` — PASS（2026-09-01）

- target：`tile_example_rmsnorm2d_fwd`；FP16、two-pass、无 fused add/quant，覆盖完整输出校验。
- runner 原先把 timing 硬编码为 `true`。本切片增加 `-time` 参数（默认 1），并在校验输出中明确打印 `timing:on/off`；PMD 统一使用 `-time=0,warmup=0,repeat=1`。
- 为解决该 runner 与 device kernel 模板处于同一 HIP translation unit 带来的性能不确定性，在同步 candidate 前分别保存 gfx936/gfx938 baseline 二进制；重编 candidate 后用 `dccobjdump --extract-elf=all` 提取目标 device ELF 并逐字节比较。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P1.4-E0 | gfx936/gfx938 在同步改动前各自构建并保存 baseline；同步后 gfx946/gfx936/gfx938 各自重编 target | PASS；实际 offload arch 分别为 gfx946/gfx936/gfx938 | G2/G4 build observation |
| P1.4-E1 | PMD 运行 FP16 `m=2,n=128`、`time=0` | PASS；单次 dispatch，完整输出校验为 `timing:off, valid:y` | G3 small-shape correctness |
| P1.4-E2 | PMD 运行 FP16 `m=4,n=1024`、`time=0` | PASS；单次 dispatch，完整输出校验为 `timing:off, valid:y` | G3 larger reduction shape |
| P1.4-E3 | gfx936/gfx938 分别运行相同两组 shape | PASS；四次均为 `timing:off, valid:y` | G4 regression observation |
| P1.4-E4 | 提取并比较 baseline/candidate device ELF | PASS；gfx936 两侧 SHA256 均为 `6aa7a4445cbf2798479f19f34d62d928f6ec22bd943349836882aacd657ea93e`；gfx938 两侧均为 `21efce163f63781c1a94b1923c739cde74da0f65ff260e37828748e4e9e16d0e`；`cmp` 均返回 0 | G6 direct code-object evidence |

G6 判定为 `NOT RUN`，不是性能通过：两个真机架构的 candidate device code object 与 baseline 逐字节相同，且默认 `-time=1` 保持原有 host 行为，已直接排除 kernel 性能变化，不需要再做运行时 A/B。gfx946 PMD 未做性能测试。

PMD 报告 `Inst_VOP3__V_DIV_FIXUP_F32 not test`、`VOP3P__V_MAD_MIXLO_F16 not test` 与 pending cacheline warning，但两组 shape 均完成单次 dispatch 和 host reference 校验，因此记录为 PMD warning，不标记指令不支持。门禁结果：G0-G4 PASS，G5 NOT RUN，G6 NOT RUN。

### Step 1.5a：`ck_tile/11_add_rmsnorm2d_rdquant` simple example — PASS（2026-09-01）

- target：`tile_example_add_rmsnorm2d_rdquant_fwd`；FP16 A/B/Gamma、保存 FP16 add/residual X、RMSNorm、FP32 row scale、INT8 row-dynamic-quant，固定 three-pass pipeline。
- runner 原先把 timing 硬编码为 `true`。本切片增加默认 1 的 `-time` 参数并打印 `timing:on/off`；PMD 使用 `-time=0,warmup=0,repeat=1`。
- host reference 分三段独立检查保存的 X、每行 yscale 与 INT8 qy；不以最终 `valid:y` 替代输出契约盘点。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P1.5a-E0 | gfx936/gfx938 保存 baseline，三架构同步 candidate 后重编 `tile_example_add_rmsnorm2d_rdquant_fwd` | PASS；实际 offload arch 分别为 gfx946/gfx936/gfx938 | G2/G4 build observation |
| P1.5a-E1 | PMD 运行 FP16 `m=4,n=128`、`time=0` | PASS；单次 dispatch，X/yscale/qy 完整校验后 `timing:off, valid:y` | G3 exact block-tile shape |
| P1.5a-E2 | PMD 运行 FP16 `m=4,n=1024`、`time=0` | PASS；单次 dispatch，X/yscale/qy 完整校验后 `timing:off, valid:y` | G3 larger reduction shape |
| P1.5a-E3 | gfx936/gfx938 分别运行相同两组 shape | PASS；四次均为 `timing:off, valid:y` | G4 regression observation |
| P1.5a-E4 | 提取并比较 baseline/candidate device ELF | PASS；gfx936 两侧 SHA256 均为 `31d39dc11f4b2731b7352da6782ae0139f7f1b8ddc865a310fa2b12155f6339e`；gfx938 两侧均为 `7e4541a4871ed4830d623548f7e2ddc451f01dea0fc5df36a5946114218af61b`；`cmp` 均返回 0 | G6 direct code-object evidence |

G6 为 `NOT RUN`：两个真机架构的 device ELF 与 baseline 逐字节相同，已直接排除性能变化；gfx946 PMD 未做性能测试。PMD 的 `Inst_VOP3__V_DIV_FIXUP_F32 not test` 与 pending cacheline warning 未阻止 dispatch 或完整校验，当前不标记为指令不支持。门禁结果：G0-G4 PASS，G5 NOT RUN，G6 NOT RUN。

### Step 1.5b：`ck_tile/11_add_rmsnorm2d_rdquant` instance runner — PARTIAL / SKIP-PMD（2026-09-01）

- target：`tile_add_rmsnorm2d_rdquant_fwd`，覆盖 `instances/*.cpp` 的 FP16/BF16、多个 N 与运行时选择闭包。runner 增加默认值为 1 的 `-time` 参数；`-time=0` 只 dispatch 一次并跳过吞吐计算，PMD 未执行性能测试。
- 三个架构均完整构建 runner、API 与所有 instance translation unit。gfx936/gfx938 的 `n=128`、`n=1024` 均完成 X/yscale/qy 校验；gfx946 PMD 的 scalar store instance 通过，但 vectorized INT8 packing/store instance 出现稳定的数据槽位清零。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P1.5b-E0 | 三架构完整构建 `tile_add_rmsnorm2d_rdquant_fwd` | PASS；分别生成 gfx946/gfx936/gfx938 code object，runner、API 与全部 instance TU 均在构建闭包内 | G2/G4 build observation |
| P1.5b-E1 | gfx936/gfx938 运行 FP16/INT8 `m=4,n=128,save_x=1` 与 `m=4,n=1024,save_x=0`、`time=0` | PASS；两台真机共四次均命中预期 instance 并 `valid:y` | G4 regression correctness |
| P1.5b-E2 | PMD 运行 `m=4,n=128`，命中 `...1x2...` instance | FAIL；qy 奇数槽稳定为 0，例如 out[1]=0/ref=18、out[3]=0/ref=2，错误率 45.5078%，max error 126 | G3 PMD functional failure |
| P1.5b-E3 | PMD 单变量扫描 `n={64,127,128,256,1024}` | `vector_N=1` 的 n64/n127 PASS；`vector_N=2/4/8` 的 n128/n256/n1024 FAIL，错误率分别 45.5078%/22.3633%/22.4854% | 最小能力切片锁定为 vectorized INT8 packing/store，不是 RMSNorm/scale/reference 的通用错误 |
| P1.5b-E4 | 提取 gfx946 `.hip_fatbin` 中 `hipv4-amdgcn-amd-amdhsa--gfx946` code object 并反汇编 | code object SHA256 `228b5a9e5a3c2c4d9190a7c5b683c53bf70eb043418d68d01f0f7bb834af0fda`；失败路径包含 `v_cvt_i32_f32_sdwa ... dst_sel:BYTE_1`、byte merge 与 `buffer_store_short` | ISA evidence；与“高字节槽清零”签名一致，判定为 PMD 的 SDWA byte packing/store 模拟缺口（高置信推断） |

PMD 同时报告 `v_mad_u64_u32 ... read vgpr before writing` 与 `Inst_VOP3__V_DIV_FIXUP_F32 not test`，但这些 warning 也出现在已通过的 vector_N=1/simple example，不能作为本次失败根因。simple example 以 `Vector_N=1` 在相同 n128/n1024 通过，进一步排除了输入、reference 和整体算法问题。

本切片不把 production selector 强制改为 gfx946 `Vector_N=1`：现有标量 instance 已可作为定位 probe，但没有 Shaobo 真机可验证这种选择对真实硬件性能与正确性的影响；仅为绕开 CPU 模拟器而改变正式 kernel 选择会制造未经验证的性能回退。按第 1 节规则，将完整 instance runner 的 vectorized 路径标记 `SKIP-PMD` 并继续后续独立模块。

门禁结果：G0-G2 PASS，G3 PARTIAL（vector_N=1 PASS、vector_N>1 SKIP-PMD），G4 PASS，G5 NOT RUN，G6 NOT RUN。G6 的直接依据是本切片代码差异只位于 host runner 的参数解析、计时开关与输出分支；默认 `-time=1` 不变，device instance/API 均未修改，故可直接排除 gfx936/gfx938 kernel 与计时区间变化，无性能不确定性。gfx946 PMD 未做性能测试。

### Step 1.6：`ck_tile/13_moe_sorting` — PASS（2026-09-01）

- target：`tile_example_moe_sorting`；支持 FP32 weight/INT32 index，覆盖 oneshot、local-expert masking、可选 fused-MoE buffer 清零和 multi-phase workspace 路径。
- runner 原先固定执行计时且默认 repeat=5000。本切片增加默认值为 1 的 `-time` 参数；`-time=0` 使用单次 launch，并明确打印 `timing:off`。默认真机行为保持不变。
- 验证固定 `seed=17`。host reference 检查有效范围内的 sorted token ids、sorted weights、sorted expert ids 和 total tokens after padding；`moe_buf_size>0` 时还逐元素检查 buffer 已清零。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P1.6-E0 | gfx946/gfx936/gfx938 完整构建 target | PASS；`moe_sorting.cpp` 与 `moe_sorting_api.cpp` 分别编译，实际 offload arch 为 gfx946/gfx936/gfx938 | G2/G4 build observation |
| P1.6-E1 | 三架构运行 `t=32,e=8,k=2,unit=32` oneshot | PASS；`mp=0`，total tokens after padding 均为 256，全部 `valid:y` | G3/G4 base routing |
| P1.6-E2 | 三架构运行 `t=128,e=8,k=4,local_eid=0,2,4,6` | PASS；`mp=0`，local-expert masking 后 total 为 288，全部 `valid:y` | G3/G4 masking branch |
| P1.6-E3 | 三架构运行 oneshot 并设置 `moe_buf_size=64` | PASS；sorting 输出与可选 FP32 buffer 清零均通过 reference | G3/G4 mutation branch |
| P1.6-E4 | 先在 gfx936 扫描 t512/t1024/t2048/t4096/t8192，锁定 t1024 为最小代表 multi-phase case；再在 PMD/gfx938 运行 `t=1024,e=8,k=2` | PASS；`mp=1`，PMD 实际产生 P0/P1/P23 三次 kernel dispatch，三架构 total 均为 2176 且 `valid:y` | G3/G4 workspace/scan path |

证据边界：当前 example 分配并传入 `tokens_positions_per_expert`，但没有从 device 拷回或与 host reference 比较。因此 P1.6 PASS 表示当前 `example_hcu` 的 basic acceptance 契约通过，不宣称该额外 API 输出已独立验收。PMD multi-phase 路径报告 `v_mov_b32 ... read sgpr ... before writing` 及若干 `src1_* should not set` warning，但最终三阶段输出与 host reference 一致，当前仅记录 warning，不标记为指令阻塞。

门禁结果：G0-G4 PASS，G5 NOT RUN，G6 NOT RUN。G6 的直接依据是代码差异只在 host runner 的参数解析/计时显示和 README；device kernel/dispatch 实现位于未修改的独立 `moe_sorting_api.cpp` translation unit，且默认 `-time=1` 不变。可直接排除 gfx936/gfx938 kernel 与计时区间变化，无性能不确定性；gfx946 PMD 未做性能测试。

### Step 2.1：legacy `05_gemm` FP16 NT — SKIP-GFX946 / PASS-REGRESSION

- 入口：`example_gemm_mmac_nt_fp16`，A=ColumnMajor、B=RowMajor、C=RowMajor，FP16 输入、FP32 累加/输出。
- runner 修复：`time_kernel=false` 时不再用无意义的 `ave_time` 计算性能，并显示 `timing:off`；`IsSupportedArgument=false` 改为失败返回，避免假阳性。
- gfx936/gfx938 正确性：默认 `M128,N512,K256` 和 `M256,N512,K256` 均 PASS。`M256,N128,K32`、`M256,N256,K64` 虽被当前 `IsSupportedArgument` 接受但两架构基线都输出全零，记录为既有参数域校验缺口，不作为 gfx946 失败证据。
- gfx936/gfx938 性能边界：修改前后提取的 device ELF 完全一致；gfx936 SHA256=`4550d1a211b811cfc473a9e457ff41f85d59d5695bf163f74e38e98375ec8379`，gfx938 SHA256=`fcbf28aa94fbd0b5c44b9f58ef67ad22702525aba942b5eac415c8958069c782`。因此设备指令、资源与计时区间均未变化，不需要另做性能 A/B。
- gfx946 诊断：原始架构 guard 使 kernel/FP16 MMAC/DSREADM 走空路径，结果全零。实验性同时放行三处后，code object SHA256=`34231de0b20cdf47aec5089064a8f54c963b60ed4e038fdc3e04447eb9016c3b`，包含 192 条 `v_mmac_f32_16x16x16_f16`、24 条 `ds_read_m32x16_b16` 和 24 条 `ds_read_m32x16_b16_alt`，A/B MMAC 源寄存器已分离，但两个有效形状仍稳定数值错误（最大误差 6655/6653）。
- 跳过依据：少伯资料明确 gfx946 为 64-bank LDS，支持路径是 `DS_READ_MATRIX_FORMAT`/`TRANS_FORMAT` 及对应少伯 LDS swizzle；legacy `DsReadm` 依赖旧式逐-lane 地址的 `ds_read_m32x16_b16(_alt)` 与旧 LDS 排布，不能通过增加宏 guard 闭合。仓库中的 gfx946 CK Tile 路径也使用带 `element/row/col/alt` 的 `ds_read_matrix_format`，证明需要重构完整 LDS descriptor/layout/读指令链，而非局部放行。
- 处置：撤回所有未通过的 gfx946 device guard 实验，只提交 runner 修复与本证据；该 legacy module 标记 `SKIP-ARCH-LAYOUT`，后续优先验证已有 gfx946 native traits/DS matrix format 的 CK Tile example。

门禁结果：gfx946 G2 build PASS、G3 ISA observation PASS、G4 correctness FAIL；gfx936/gfx938 G4 PASS；G6 以 device ELF 完全一致证明无性能不确定性。该模块不宣称 gfx946 支持。

### Step 2.2：`ck_tile/03_gemm` FP16 RC basic GEMM — PASS-PMD

- target：`tile_example_gemm_basic`；A=RowMajor、B=ColumnMajor、C=RowMajor，FP16 输入/输出、FP32 累加，tile=`128x128x32`。
- correctness-only 入口：`-repeat=0` 时关闭 kernel timer、只 launch 一次并显示 `timing:off`；默认 `repeat=100` 的计时行为保持不变，README 已补充该契约。
- gfx946/gfx936/gfx938 均构建 PASS；三架构分别运行单 block `M128,N128,K32` 与多 block/K-loop `M256,N256,K64`，CPU reference 全部 PASS。
- gfx946 code object SHA256=`b39a01f4b1a8b8e84f6fcbf2ecd29184eb134eefb8e187c8d2b94ca218eaeff9`；指令统计包含 64 条 `v_mmac_f32_16x16x16_f16`、20 条 `ds_read_b128`、16 条 `buffer_store_dwordx2`，不是空 kernel。
- gfx936/gfx938 性能边界：修改前后 device ELF 完全一致；gfx936 SHA256=`29c370986544baabcb2e6a77f91fb4c69752ba3eed9c561f7621ed48c5e4d1b4`，gfx938 SHA256=`12e80169f9afcb408a49ccd77ddde3676bbeb667e6e8507df422896493cea649`。因此设备 ISA/资源不变，无性能不确定性，不额外执行性能 A/B。
- PMD 仍打印若干 `v_mad_u64_u32 ... read vgpr ... before writing` 与 pending-cacheline warning，但两档输出均与 CPU reference 一致；记录为模型 warning，不作为当前阻塞。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 由 device ELF identity 闭合。PMD 全程 `repeat=0`，未测试性能。

### Step 2.3：legacy `05_gemm` FP16 TN v2 — PASS-PMD

- target：`example_gemm_mmac_tn_v2_fp16`；A=RowMajor、B=ColumnMajor、C=RowMajor，FP16 输入/输出、FP32 累加，block tile=`64x64x16`、prefetch=2。
- 与 P2.1 的旧 NT 路径不同，本路径不使用 `DsReadm`/`ds_read_m32x16_b16(_alt)`，而是使用普通 swizzled LDS threadwise load。因此最小适配只把 gfx946 加入该 kernel body 与 `intrin_mmac_f32_16x16x16f16` 的架构门控，没有放行已证明不兼容的旧式 matrix-read 路径。
- gfx946/gfx936/gfx938 均 focused build PASS。三架构分别运行默认 `M128,N512,K256` 与 `M256,N512,K256`，`verification=1,init=1,time_kernel=0`；六次均通过完整 CPU reference 校验并返回 0，输出明确为 `timing:off`。
- gfx946 code object SHA256=`0eaa82db56bc401e9634a141562a460676b20894847e6f72616719983c9f8981`；指令统计包含 24 条 `v_mmac_f32_16x16x16_f16`、24 条 `ds_read_b64`、16 条 `buffer_store_dword` 和 12 条 `buffer_load_dwordx2`，证明实际生成并执行了非空 MMAC/LDS 路径。
- gfx936/gfx938 性能边界：修改前后 device ELF 逐字节一致；gfx936 两侧 SHA256 均为 `e3a2c2e6346b809190701f6458d65a49a0a5542de2e67f42850307c7f7ca17be`，gfx938 两侧均为 `d844246731d7d0a32ecb7d9a92fbe67ebda981ccd31236484b94465a31858988`。因此两个真机架构的 ISA、资源与 kernel 性能均未变化，不需要运行时性能 A/B。
- PMD 运行报告通用 pending-cacheline warning，但两档 shape 都完成 dispatch 和 host reference 校验；当前不标记为指令阻塞。PMD 全程只做 correctness，不采集性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 由 gfx936/gfx938 device ELF identity 闭合。本 example 的适配范围严格限定为已验证的 FP16 TN v2 路径，不据此放宽 TF32、BF16、INT8 或 P2.1 的旧式 `DsReadm` 支持声明。

### Step 2.4：legacy `05_gemm` FP16 TN v2 CShuffle — PASS-PMD

- target：`example_gemm_mmac_tn_v2_fp16_cshuffle`；在 P2.3 的 TN v2 FP16/FP32 MMAC 主计算路径上增加 CShuffle LDS 暂存与向量化全局写回，block tile=`128x128x16`、prefetch=2。
- 最小适配只把 gfx946 加入该 CShuffle kernel body 的架构门控；FP16 MMAC 能力沿用 P2.3 已验证且已提交的 guard，不扩大其他 dtype 或其他 gridwise family。
- gfx946/gfx936/gfx938 focused build 均 PASS。三架构分别运行默认 `M128,N512,K256` 与 `M256,N512,K256`，`verification=1,init=1,time_kernel=0`；六次均通过 CPU reference 校验并返回 0，输出为 `timing:off`。
- gfx946 code object SHA256=`e1fcc0e9eace3fbc584420fbe73a5358bbd858dccc63c4c67a6c655edc6ae4b4`；指令统计包含 96 条 `v_mmac_f32_16x16x16_f16`、128 条 `ds_write_b16`、24 条 `ds_write_b64`、80 条 `ds_read_b64`、32 条 `buffer_store_dwordx2` 和 12 条 `buffer_load_dwordx4`，覆盖实际 MMAC、CShuffle LDS 写读与全局写回链路。
- gfx936/gfx938 性能边界：修改前后 device ELF 逐字节一致；gfx936 两侧 SHA256 均为 `192751d6da16eb9d1ba3c29cde7854b66054cc9a1cc68ce7f127318c60c42523`，gfx938 两侧均为 `eb5bb9bb60852a214d9ddc3835fe6b289aa143bc80517f80949d11bcebbb166b`。已直接排除两个架构的 kernel 性能变化，不执行运行时性能 A/B。
- PMD 仅出现 pending-cacheline warning，两档 shape 均完成 dispatch 和完整校验；当前不存在需要跳过的指令阻塞，且未进行性能测试。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 由 gfx936/gfx938 device ELF identity 闭合。P2.4 只声明该 TN v2 CShuffle example 的已测 FP16 能力。

### Step 2.5：`06_gemm_bias_activation` FP16 bias+ReLU — PASS-PMD

- target：`example_gemm_bias_relu_mmac_fp16`；A=RowMajor、B=ColumnMajor、C=RowMajor，FP16 GEMM、FP16 N 维 bias 与 fused ReLU，FP32 累加，block tile=`64x64x16`、prefetch=2。
- 最小适配把 gfx946 加入 `gridwise_gemm_mmac_tn_mk_nk_mn_v2r1` kernel body 门控；该路径复用 P2.3 已验证的普通 LDS threadwise load 与 FP16 MMAC，不依赖旧式 `DsReadm`。同时修复公共 example runner：unsupported argument 返回失败；`time_kernel=0` 时不计算伪性能并显示 `timing:off`。
- gfx946/gfx936/gfx938 focused build 均 PASS。三架构分别运行默认 `M128,N512,K256` 与 `M256,N512,K256`，`verification=1,init=1,time_kernel=0`；六次 bias+ReLU 完整 CPU reference 校验均通过并返回 0。
- gfx946 code object SHA256=`002066acaa04fcecb1dc8dd2f02b3a24dc72daa506e449453c099c6b99557c20`；指令统计包含 32 条 `v_mmac_f32_16x16x16_f16`、32 条 `v_max_f16_e32`、16 条 `v_add_f16_sdwa`、16 条 `v_add_f16_e32`、32 条 `ds_read_b64` 以及实际 global load/store，证明 fused bias 与 ReLU 路径并非空执行。
- PMD 报告 `Inst_VOP2__V_ADD_F16 not test`、`v_max_f16 is not test` 及 operand/pending-cacheline warning，但两档结果均与 CPU reference 一致；当前记录为 PMD 覆盖告警，不标记为指令阻塞。
- gfx936/gfx938 性能边界：修改前后 device ELF 逐字节一致；gfx936 两侧 SHA256 均为 `7ac54db3d9122df91b4d0e23a880aa876535844c2c882ef26a687cafe8c5a19e`，gfx938 两侧均为 `bc7d1ee87d1bbe3196d4b629dc8a4c48017c6ed365f20d6638f6d79f14bda8d4`。已直接排除两个架构的 kernel 性能变化，不执行运行时性能 A/B。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 由 gfx936/gfx938 device ELF identity 闭合。gfx946 PMD 全程关闭计时，没有性能测试。

### Step 2.6：`06_gemm_bias_activation` FP16 bias+ReLU wavelet — SKIP-BASELINE

- target：`example_gemm_bias_relu_mmac_fp16_wavelet`；实验性把 gfx946 加入 wavelet kernel body guard 后，三个架构均完成 focused build。
- gfx946 PMD 在默认 `M128,N512,K256`、`verification=1,init=1,time_kernel=0` 下完成实际 dispatch，但输出大量保持为 0，CPU reference 校验 FAIL，`max err: 2476`；因此不能声明 Shaobo wavelet 支持。
- 为排除本次 gfx946-only guard 引入跨架构回归，在同步实验前已保存 gfx936/gfx938 baseline。gfx936 baseline 在相同默认 case 上也稳定 FAIL，candidate 与 baseline 错误签名一致，均包含 `out[4]=5/ref=11`、`out[6]=0/ref=44`，`max err: 529`、进程返回 1。
- gfx938 baseline/candidate 均在 main 入口明确拒绝并返回 77：`The wavelet direct-load path is not supported on gfx938 because it produces incorrect column ordering.`；CMake 也把 77 配置为该 test 的 skip code。
- 分类：该 wavelet direct-load example 在本任务开始前已不是 gfx936/gfx938 的共同正确性基线，且源码已明确记录 gfx938 列序错误。当前缺少可用于判定 Shaobo 特有问题的健康真机控制，不能仅凭 PMD 失败重构 wavelet load/math group 同步与列映射。
- 处置：撤回 gfx946 wavelet guard，并把原文件同步还原到三台远端；不保留未验证的 device 代码。本模块标记 `SKIP-BASELINE`，转入下一独立模块。gfx946 PMD 未测试性能；因最终无业务代码差异，gfx936/gfx938 不触发性能 A/B。

门禁结果：G2 build observation PASS；gfx946 G3 FAIL；gfx936 baseline/candidate G4 FAIL；gfx938 G4 SKIP(77)；G6 NOT RUN。该结果不影响 P2.5 非 wavelet 路径的 PASS 结论。

### Step 2.7：`01_grouped_conv_fwd_mmac` NHWC v2 FP16 CShuffle — PASS-PMD

- target：`example_grouped_conv_fwd_mmac_nhwc_v2_fp16_cshuffle`；NHWGC/GKYXC/NHWGK，FP16 输入输出、FP32 累加，默认 `G=N=1,C=K=64,Y=X=3,H=W=8,pad=1`，使用 legacy MMAC v2 与 LDS CShuffle。
- 首次 gfx946 build 成功但运行输出全 0、`max err: 884`。源码定位到公共 `kernel_grouped_conv_fwd_mmac_v2` 的 device guard 只允许 gfx926/928/92a/936/938；gfx946 实际编译的是仅消费参数的空 kernel。最小修复只把 `__gfx946__` 加入该 guard，不改 tile、descriptor、MMAC、CShuffle 或真卡已有分支。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.7-E0 | gfx936/gfx938 修改前默认 shape，`verification=1,init=1,time=0` | 两侧均 PASS；gfx936 binary SHA256=`bff54780d1c54a8880fb57c403c1430f44bfb0ce43ed2aab92cf0c8600fd48da`，gfx938=`4cee54732b06da23a26930f3f1936f49d0e4dca8c2360c5ff194122caae7506a`，日志均为 `5716d371757d0ddd0f2ff6f1cb96e3a58f30a7393fd9a583edd343670a7ff545` | G0 healthy controls PASS |
| P2.7-E1 | gfx946 guard 修复前同 shape | focused build PASS，但输出保持 0、`out[0]=0/ref=-53`、`max err: 884`；binary SHA256=`156f0b117f8ad871f40c53368f8a41258693d36275bb0464798f03400ef3867e` | G3 FAIL；定位为空 guard 分支 |
| P2.7-E2 | guard 加入 gfx946 后重构建并运行 | PASS，完整 CPU reference 无错误、进程返回 0；binary SHA256=`e9893cc5c93e50d1151fb3016fec24d7f027adc2e4d1fa67b46c549ea813ffca`，日志 SHA256=`c3266b35ae15bd78e589a6233f968dc7c32c43c22a2febbeebd6569d7be85fd8` | G1-G3 PASS |
| P2.7-E3 | gfx936/gfx938 candidate correctness | 两侧均 PASS，candidate host binary 与各自 baseline 相同；candidate 日志均为 `5716d371757d0ddd0f2ff6f1cb96e3a58f30a7393fd9a583edd343670a7ff545` | G4 PASS |
| P2.7-E4 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧 SHA256 均为 `bb30ff5e565ae726d4d69ad394bd1fe8a22b5894ba4d1710f094de7302b9dbf9`；gfx938 两侧均为 `78a81aacda15d9c7629175269265121e0e57ca27cfd643a823f04f9962b253ad`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD candidate 只报告一次 pending-cacheline warning，实际 kernel dispatch 和完整 CPU reference 均通过。gfx946 未测试性能；gfx936/gfx938 的预处理分支与 device ELF 均未变化，已直接排除性能变化。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。下一切片为同模块独立 target `example_grouped_conv_fwd_mmac_nchw_v2_fp16_cshuffle`，仍需单独验收布局变换与写回。

### Step 2.8：`01_grouped_conv_fwd_mmac` NCHW v2 FP16 CShuffle — PASS-PMD

- target：`example_grouped_conv_fwd_mmac_nchw_v2_fp16_cshuffle`；NCHW/GKCYX/NKHW，FP16 输入输出、FP32 累加。默认真机 shape 为 `G=1,N=256,K=C=512,Y=X=1,H=W=28`；PMD 正式验收使用满足该实例向量化与 pipeline 约束的最小 shape `G=1,N=4,K=32,C=128,H=W=4`，避免 CPU 模拟 1,605,632 个默认 workgroup，且全程 `time=0`。
- 初始 gfx946 kernel 输出全 0。沿实际实例化链逐层定位出三个独立空路径：NCHW device wrapper 未允许 gfx946；其专用 NN grid kernel 未允许 gfx946；`MmmacInterleave=2` 实际选中的 `intrin_ds_read_m32x16_f16_alt` 仍在 gfx946 上执行 `swallow`。最小修复只为这三处既有 FP16 路径加入 `__gfx946__`，不扩大普通 f16 DSReadM、BF16/INT8/FP8 指令或其他 grid family。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.8-E0 | gfx936/gfx938 修改前默认 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `9641f6e04b80729f4947e0ee292f265ef68eb291f17956bbff868ab4633585c8`、`e2e21462feb01ce79514201675029b51f6ea48611d833ce07a5dfe691602796e`，日志均为 `641bbaee3ae10869e62a9032b9e573560047380c1e27091fae93ab62ac4ecac6` | G0 healthy controls PASS |
| P2.8-E1 | gfx946 修改前默认 shape；随后改用最小支持 shape | 默认 shape 发射 1,605,632 workgroup，因 PMD 运行时间不可接受而人工终止，不作为 correctness 证据；最小 shape 完成 dispatch 但全 0、`max err:342` | G3 FAIL；收敛到 device 空路径而非超时 |
| P2.8-E2 | 仅补 device wrapper，再补 NN grid kernel | 两轮最小 shape仍全 0；补 grid 后日志 SHA256=`8d83d37dd38494874c45401f4731ca5683097ef4fdf3f045bbbc7a310b433a6c` | 逐层 A/B 排除外壳，继续追踪 blockwise primitive |
| P2.8-E3 | 把 gfx946 加到普通 `ds_read_m32x16_f16` 探针 | 重构建后的 host binary SHA256 仍为 `5ec3ff57a65afca8c047f6cae012e323bb04fae8ca8e10e242ad3672925a5e6e`，与改前相同，输出仍全 0 | 否证普通指令入口；撤回无关扩展 |
| P2.8-E4 | 仅补实例实际选择的 `ds_read_m32x16_f16_alt` 并重构建、运行最小 shape | PASS，完整 CPU reference 无错误、进程返回 0；binary SHA256=`8fb99f462349f1476d5c1b8f4478bd866f49e76dd97e4a0a71ec20698b4b9bea`，日志 SHA256=`c9bddb0643ae9ef3f00070d19ffacf7aca057c9e1881838012dd7ffdddaa707b` | G1-G3 PASS；根因为三层 gfx946 enrollment 缺口 |
| P2.8-E5 | gfx936/gfx938 candidate 默认 shape | 两侧完整 CPU reference PASS；candidate host binary分别与各自 baseline 逐字节相同，日志均为 `641bbaee3ae10869e62a9032b9e573560047380c1e27091fae93ab62ac4ecac6` | G4 PASS |
| P2.8-E6 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧 SHA256 均为 `52a4ee141fd27d615c3173ae0e9de44f475233522ae1effbc5db5b60f8eba9df`；gfx938 两侧均为 `ecf82cc7608334255dc820657e122257bcd0e503a4c55f621456e30b304a512b`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD 最终运行完成真实 kernel dispatch 和完整 reference 校验，没有出现需要按规则跳过的指令终止。gfx946 仅做 correctness，未测试性能。三处新增条件对 gfx936/gfx938 的预处理结果均不变，host binary 与 device ELF 又分别逐字节相同，已直接排除两架构的正确性和性能变化，不执行运行时性能 A/B。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。P2.8 完成后立即提交该 example 的三处最小适配与本文更新；下一步继续盘点尚未验收的已启用 example。

### Step 2.9：`02_grouped_conv_bwd_data_mmac` NHWC v2 FP16 CShuffle — PASS-PMD

- target：`example_grouped_conv_bwd_data_mmac_nhwc_v2_fp16_cshuffle`；NHWGK/GKYXC/NHWGC，默认 `G=N=1,K=64,C=16,Y=X=3,H=W=4,pad=1`，FP16 输入输出、FP32 累加。该默认 shape 在 PMD 只发射 256 个 workitem，直接用于 correctness，不采集性能。
- 初始 gfx946 build 成功且 dispatch，但 input-gradient 全 0、`max err:601`。源码显示 backward-data device wrapper 与专用 TT grid kernel均只允许 gfx926/928/92a/936/938；最小修复仅把 gfx946 加入这两个空 guard。该实例 `NmmacInterleave=2` 所需的 FP16 alt DSReadM 已由 P2.8 独立验证，本步不再修改底层 builtin。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.9-E0 | gfx936/gfx938 修改前默认 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `97e6ea272f7ce5ef202c1cf5785f21b0b0125a3dc590cd7c08582fbbeaec08e8`、`197e07664886ba705a42766e996dcb2286443f73d246baf90bb6a1af08a0e7b2`，日志均为 `58a521df09912593c8e4a1016891f6ce4d08b9b2c147d765d2009a24370e99db` | G0 healthy controls PASS |
| P2.9-E1 | gfx946 修改前 focused build 与默认 correctness | build PASS，kernel dispatch 后 gradient 全 0；binary SHA256=`d07c94b3b9bc88b4e45f327ed546d25898867586e127bf116cf747cd96d05764`，日志 SHA256=`8108da8897e8675cbfba2ea47dee8c5643f67e31adf2092c549f6b7ece98e286` | G3 FAIL；对应两层空 guard |
| P2.9-E2 | 两处 guard 加入 gfx946 后重构建并运行 | PASS，完整 CPU reference 无错误、进程返回 0；binary SHA256=`0b3643ee768ebbbf9ae24965aa532384e5bd327fdd252158f90d81f6e6a04e6b`，日志 SHA256=`e9d1f63b1778574a3c36bddf6d62908c2aab4ab5539e209fe563517c6fbf8556` | G1-G3 PASS |
| P2.9-E3 | gfx936/gfx938 candidate 默认 shape | 两侧完整 CPU reference PASS；candidate host binary分别与各自 baseline 逐字节相同，日志均为 `58a521df09912593c8e4a1016891f6ce4d08b9b2c147d765d2009a24370e99db` | G4 PASS |
| P2.9-E4 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧 SHA256 均为 `3be96ec65fd071fc65ee3156d1fa125cdce506ea3d81697d82d57822c6785f65`；gfx938 两侧均为 `57af1a7798917c3b8e467f008193060e40b28b971a3bbb0394e629fbef96f3be`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD candidate 报告四条 `v_mad_u64_u32 ... read vgpr before writing` 与一条 pending-cacheline warning，但同一次真实 dispatch 完成且完整输出与 CPU reference 一致，按既有规则只记录为模型告警。gfx946 未测试性能；gfx936/gfx938 host binary 与 device ELF identity 直接排除性能变化。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。P2.9 独立提交后继续同目录 NCHW v2 CShuffle target。

### Step 2.10：`02_grouped_conv_bwd_data_mmac` NCHW v2 FP16 CShuffle — PASS-PMD

- target：`example_grouped_conv_bwd_data_mmac_nchw_v2_fp16_cshuffle`；NGKHW/GKCYX/NGCHW，`Filter1x1Stride1Pad0`，默认真机 shape 为 `G=N=1,K=256,C=64,H=W=28`。源码明确该 descriptor 只支持 `N=1`；PMD 正式验收使用 `G=N=1,K=C=128,H=W=4`，满足向量化与 grid validity，并只发射一个 256-thread workgroup。
- 初始 gfx946 输出全 0。该 NT blockwise 路径除独立 NCHW device wrapper、NT grid guard 外，同时使用 `MmmacInterleave=2` 的 alt FP16 DSReadM 和 `NmmacInterleave=1` 的普通 FP16 DSReadM。P2.8 已覆盖 alt，本步在逐层 A/B 后只新增普通 `ds_read_m32x16_f16` 的 gfx946 入口，不扩大其他 dtype/format。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.10-E0 | gfx936/gfx938 修改前默认 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `abd544cdb7cba224b90ab20c7224aafb3ce4c7889ea81b5d95f196159c4c93dd`、`661175657a51c2d14941f92d8067425dfd92f3cb03123b4aa2ac50f82e0a51d8`，日志均为 `f8003679db10f97645c9fd608fea5be527d707d3e893b3f22da95a38e4a1c253` | G0 healthy controls PASS |
| P2.10-E1 | gfx946 修改前，正式最小 shape | supported、dispatch 完成但 gradient 全 0、`max err:378`；binary SHA256=`b99a65e5c3d2ce12cf6361bfafd33accebb6ea1ab9ac7401ad14fccfaa72111a`，日志 SHA256=`b79475221f6bb5aeaa13b34881b3e2ee38d42d73fe03f197341d2b6e0170228c` | G3 FAIL |
| P2.10-E2 | 只补 NCHW wrapper 与 NT grid 两层 guard | 仍全 0、`max err:378`；binary SHA256=`a19610d2b887905558debcfc05c593d4b85b312cc1b9ce4f05e2dc129b6354f7`，日志 SHA256=`6ffe47c71a1117dd96ec41f67a032cbe272f4c3ff103e87dbac02ab8aabc10a5` | 外层已进入，继续定位普通 DSReadM 空实现 |
| P2.10-E3 | 普通 `ds_read_m32x16_f16` 加入 gfx946 后重构建、运行 | PASS，完整 CPU reference 无错误、进程返回 0；binary SHA256=`f917e30ee22deb22d19e6b981c30068d15190e4c8279aaf7a1e33bd7fb0d101f`，日志 SHA256=`e2e24dd4f098699bdda82a8833e1041312e44ad03a14794b5f8dba96d7fd78d4` | G1-G3 PASS |
| P2.10-E4 | gfx936/gfx938 candidate 默认 shape | 两侧完整 CPU reference PASS；candidate host binary分别与各自 baseline 逐字节相同，日志均为 `f8003679db10f97645c9fd608fea5be527d707d3e893b3f22da95a38e4a1c253` | G4 PASS |
| P2.10-E5 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧 SHA256 均为 `7db31f2308b5fa4d9a545cf3d4be560017cedba05ad964299633d3eb59a16f9f`；gfx938 两侧均为 `9c8e3d4137f61c81578951642a95172a586e3d8d23124a8c25adb37526e15dfb`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD 在 E2/E3 均报告一次 `ds_write_b128 ... Write index 8192 is larger than allocated LDS 8192` 与 pending-cacheline warning；只有补齐普通 DSReadM 的 E3 通过完整 reference，因此该 LDS 信息当前按模型边界调试告警记录，不为绕过 PMD 改写已在 gfx936/gfx938 正确的 LDS descriptor。gfx946 未测试性能；两真机 host binary/device ELF identity 直接排除性能变化。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。P2.10 独立提交后继续同目录 mixed CShuffle target。

### Step 2.11：`02_grouped_conv_bwd_data_mmac` mixed NHWC→NCHW FP16 CShuffle — SKIP-BASELINE

- target：`example_grouped_conv_bwd_data_mmac_nhwc_v2_fp16_cshuffle_mixed`；NHWGK/GKYXC 输入、NGCHW gradient 输出。CMake 已把返回码 77 注册为 skip；runner 又明确在 gfx938 上返回 77，原因是该 mixed 写回不产生有效输出。
- 修改前健康控制表明该问题不只存在于 gfx938：gfx936 默认 `G=N=1,K=64,C=16,3x3,H=W=4,pad=1` 完成 kernel 后输出为大幅错误值，首项 `-15816/ref=277`，最终 `max err:inf`，进程返回 1。gfx936 baseline binary SHA256=`e8b431efea55603c845d78ba52545f3b94db37aa530c7ac3c4ca91a2f5ae82ad`，日志 SHA256=`4335cbf8c2de1b9bf32923487421d1b7db5093a61a5e976d4d80ff286eaab161`。
- gfx938 按既有契约返回 77，未发射 kernel；baseline binary SHA256=`f8955cb52f87d57dab4af0b1537ebfe5c323e36606f6ea7a5dd9435594a00056`，日志 SHA256=`346c067c1200152e1d4160c2b401f9a45e465068d308d2b57420907260aff8a9`。

分类与处置：该 target 在适配开始前既没有 gfx936 正确性基线，也没有 gfx938 可执行基线；因此不能用 gfx946 PMD 的单侧结果证明 mixed layout/writeback 正确，更不能绕开两个真机证据去放宽 device/grid guard。gfx946 focused build 已通过，但不运行无健康控制的 PMD kernel；不保留、不同步任何 P2.11 device 改动，不做性能测试。

门禁结果：G2 build observation PASS；gfx936 G0/G4 FAIL；gfx938 G0/G4 SKIP(77)；gfx946 G3 NOT RUN；G6 NOT RUN。模块标记 `SKIP-BASELINE`，继续 grouped-conv backward-weight 独立 target。

### Step 2.12：`03_grouped_conv_bwd_weight_mmac` NHWC FP16 — PASS-PMD

- target：`example_grouped_conv_bwd_weight_mmac_nhwc_fp16`；NHWGK/GKYXC/NHWGC，默认真机 shape 为 `G=1,N=128,K=C=128,Y=X=1,H=W=14`。PMD 正式验收使用 `G=1,N=32,K=C=128,Y=X=1,H=W=1`，关闭 timing，只发射一个最小有效 workgroup 并执行完整 CPU reference。
- 初始 gfx946 focused build 与 dispatch 均成功，但 weight-gradient 全 0、`max err:202`。静态闭包表明 `device_grouped_conv_bwd_weight_mmac.hpp` 顶层 kernel body 只允许 gfx926/928/92a/936/938；其调用的 `gridwise_gemm_mmac_bwd_weight.hpp` 没有额外 gfx guard。本步只把 gfx946 加入该顶层入口，不改变 tile、MMAC、LDS descriptor、epilogue 或 launch 参数。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.12-E0 | gfx936/gfx938 修改前默认 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `3520b6bed5526589b1225213c69b395d991d1c7ac5a758702fe297a27e83d0f7`、`1001722c1d7607e76bfd81c6a88d178d16dd636a5050fddc9837d4a1f989a992`，日志均为 `d7c725fe596d9ec7ba7a2944d60b2b69695a700e40c27ef1175a76f2d7e870a2` | G0 healthy controls PASS |
| P2.12-E1 | gfx946 修改前最小 shape | build/dispatch PASS，但输出全 0、`max err:202`；binary SHA256=`f1431a077dcb4cad680f336313a70b7c0594f91a79e88e54df73e1ba69bbf696`，日志 SHA256=`e170d3ced5d2358704cb493e7d3a9e37b384d2ff352a87b063a1d1f53a70338a` | G3 FAIL；命中顶层空 guard |
| P2.12-E2 | 顶层 kernel body guard 加入 gfx946 后重构建、运行 | PASS，真实 dispatch 后完整 CPU reference 无错误、进程返回 0；binary SHA256=`fab079586d9e69f06274780bfb124dff07933368d94ab1e6a81132977fe93631`，日志 SHA256=`caea07c46d5329cd32bb75cce1449e64a7971e90076667edb9a948bd652bdc77` | G1-G3 PASS |
| P2.12-E3 | gfx936/gfx938 candidate 默认 shape | 两侧完整 CPU reference PASS；candidate host binary 分别与各自 baseline 逐字节相同，日志仍均为 `d7c725fe596d9ec7ba7a2944d60b2b69695a700e40c27ef1175a76f2d7e870a2` | G4 PASS |
| P2.12-E4 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧 SHA256 均为 `bf7eba40cca962aef91ffe448d903d7129ae882cb84097d99254fb01607c62da`；gfx938 两侧均为 `0d22bbb53d009a7f0a5a5f57a19f6abe3cab24b5648f65bb59eaa73e875d886b`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD candidate 报告较多 `src1_neg/src1_abs should not set`、未对齐 `ds_write_b128/ds_read_b128` 和一次 `ignore GPUVM APE` 信息，但 kernel 未终止，最终所有输出都通过 reference；按既有规则将其记录为模型告警，不为绕过 PMD 改写已在两台真机正确的 device 路径，也不扩大为 Shaobo 真机指令结论。gfx946 未测试性能；gfx936/gfx938 host binary 与 device ELF identity 已直接排除性能变化。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。P2.12 独立提交后继续同目录 NCHWc32 target；共享 guard 已生效，但必须独立完成该 example 的三架构验收后才能声明支持。

### Step 2.13：`03_grouped_conv_bwd_weight_mmac` NCHWc32 FP16 — PASS-PMD

- target：`example_grouped_conv_bwd_weight_mmac_nchwc32_fp16`；NGKHWk32/GKCYXc32/NGCHWc32，FP16 out/input、FP32 weight 与累加。该 example 复用 P2.12 已提交的 `device_grouped_conv_bwd_weight_mmac.hpp` 顶层 guard，本步没有新增 device 或 runner 改动。
- PMD 使用 `G=1,N=32,K=C=128,Y=X=1,H=W=1`、`verification=1,init=1,time=0`；layout 实际展开为 K/C 外层 4、内层 32，dispatch grid 为 8192、workgroup 为 256。gfx936/gfx938 使用默认 `G=1,N=128,K=C=128,Y=X=1,H=W=14`，同样关闭 timing。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.13-E0 | gfx946 focused build 与最小 NCHWc32 correctness | PASS；实际编译/链接为 `--offload-arch=gfx946`，完整 CPU reference 通过并返回 0；binary SHA256=`f6362572d74a3fce0b7a5b67cf64cda3e68cf4fbdacb8aa59bcfcacc49806a0e`，日志 SHA256=`2b4d84fcf9690fb159fc97e23427e59c5eec92816cf2696289275a19fa33061f` | G1-G3 PASS |
| P2.13-E1 | gfx936 默认 shape correctness 与 code object | PASS；binary SHA256=`d8b7c0b2efd31cfabed67b8cee949c9cf7955e903d7d1a8a42c121eba560ce62`，日志 SHA256=`f81d061231515e09a70e96a9bfac6a0c545c91b66bb901bcad528c9608e2751a`，device ELF SHA256=`89888c1d83c6f8217012c7fa8bd59702ccfaf9b5d12bd57f71d4f9f7cee18ad1` | G4 PASS |
| P2.13-E2 | gfx938 默认 shape correctness 与 code object | PASS；binary SHA256=`b2bf208af0e74899dcd3c968591816c773682ac67bf48e3e5ab0520e10a2a1a2`，日志 SHA256=`f81d061231515e09a70e96a9bfac6a0c545c91b66bb901bcad528c9608e2751a`，device ELF SHA256=`f18d2f6c6d1b697a525d0162a0d00a3ca3e050ee4a7fa0e83ec1ac7dacf49a9e` | G4 PASS |

PMD 仍报告与 P2.12 同类的 source-modifier、未对齐 `ds_write_b128/ds_read_b128` 和 `ignore GPUVM APE` 信息，但 dispatch 完成且全部 reference 正确，继续归类为模型告警。P2.13 自 P2.12 提交后没有任何源码差异；共享条件对 gfx936/gfx938 仍走原有 `defined(__gfx936__)`/`defined(__gfx938__)` 真分支，故 device 执行路径和性能区间无变化，不存在需要运行时 A/B 的不确定性。gfx946 未测试性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（no source delta/direct architecture-condition evidence）。本例独立文档提交后继续 NHWC v2 target。

### Step 2.14：`03_grouped_conv_bwd_weight_mmac` NHWC v2 FP16 — PASS-PMD

- target：`example_grouped_conv_bwd_weight_mmac_nhwc_v2_fp16`；NHWGK/GKYXC/NHWGC，FP16 out/input、FP32 weight 与累加，默认 `G=N=1,K=C=64,Y=X=3,H=W=8,pad=1`。该 v2 实例使用 `gridwise_gemm_mmac_nt_m0k0k1m1_n0k0k1n1_mn_v1`；由于 P2.1 同 family 的 legacy GEMM NT 曾因 Shaobo LDS/DSReadM 布局跳过，本步严格只放开 grouped-conv wrapper 的外层 guard 做单变量验证。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.14-E0 | gfx936/gfx938 修改前默认 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `bf8eea1d622e5d47cea2d029eae8dd3e379a6504f4deb4516a6bc9085627cc32`、`c91b23c774fec7ff52940a2e7e38ed47dd0c3860d93a9cdf723880338fbafd6e`，日志均为 `0461d1831b1939f27a9d8aadfbc0d7a27620a94a5816b6350510b1ae2c111353` | G0 healthy controls PASS |
| P2.14-E1 | gfx946 修改前默认 shape | build/dispatch PASS，但 weight-gradient 全 0、`max err:271`；binary SHA256=`0703d99e8a7c344550066069e874781c09dd6b0515dfe6c2df974a4a26b6f157`，日志 SHA256=`93c5659fb07e5abb0a1db86cbf633612284f52b83cf596ffdbe268e30e6ea45c` | G3 FAIL；顶层空 guard |
| P2.14-E2 | 只把 gfx946 加入 v2 grouped-conv kernel body guard | PASS；默认 3x3 shape 的真实 dispatch 与完整 CPU reference 正确、进程返回 0；binary SHA256=`92de8e88df4d08b6bc7f7fa9e0f7a2c971db677f6ac6ce495266881619b9adf7`，日志 SHA256=`8c2445ad2d7a2b59f74ad40b08c48d3f288bdf5219a54bc9f893dc494082c3a1` | G1-G3 PASS；该具体 descriptor 未复现 P2.1 失败 |
| P2.14-E3 | gfx936/gfx938 candidate 默认 shape | 两侧完整 CPU reference PASS；candidate host binary 分别与各自 baseline 逐字节相同，日志保持不变 | G4 PASS |
| P2.14-E4 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧 SHA256 均为 `88df53a6144d5f11682c401a4f80a34ddfc1517ad455b3cb18f771d00e9aeb00`；gfx938 两侧均为 `9d4fc27005bb15266add597df5c026951e1d388c934f8f503fff8f68879cc689`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD candidate 只报告一次 `ignore GPUVM APE`，未出现指令终止或数值错误。该结果只证明当前 NHWC v2 conv descriptor/shape 的执行闭包，不能反向取消 P2.1 对另一 legacy NT descriptor 的跳过结论。gfx946 全程 `time=0`，未测试性能；gfx936/gfx938 host binary 与 device ELF identity 已直接排除性能变化。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。P2.14 独立提交后继续同目录 NHWC v2 wavelet target。

### Step 2.15：`03_grouped_conv_bwd_weight_mmac` NHWC v2 FP16 wavelet — SKIP-PMD

- target：`example_grouped_conv_bwd_weight_mmac_nhwc_v2_fp16_wavelet`；数据契约和默认 `G=N=1,K=C=64,Y=X=3,H=W=8,pad=1` 与 P2.14 相同，但使用 512-thread wavelet pipeline。与 P2.6 GEMM wavelet 不同，本 target 在 gfx936/gfx938 均有健康 correctness 基线，因此可以用同 shape 的 non-wavelet/wavelet A/B 收敛 PMD 边界。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.15-E0 | gfx936/gfx938 修改前默认 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `b2d4c6e53be67f765bf69825fae390d57ffbb874bb36024c53ae463515ed7766`、`885e26abaa2becb18efdf0c8c72b98807cea075ff35146fa7d962cdc8d14b26b`，日志均为 `0461d1831b1939f27a9d8aadfbc0d7a27620a94a5816b6350510b1ae2c111353` | G0/G4 healthy controls PASS |
| P2.15-E1 | gfx946 修改前默认 shape | build/dispatch PASS，但外层 guard 为空，weight-gradient 全 0、`max err:271`；binary SHA256=`e697ca20013ad4ba90fbe537953491105cc7aeca830d0bde1eb031956b399f86`，日志 SHA256=`a2e5996c2a669e957029402e63ffb8b6e0e892d83f478fb98d2983568dd78f43` | G3 FAIL；建立空-kernel control |
| P2.15-E2 | 仅实验性放开 wavelet grouped wrapper 的 gfx946 guard | dispatch 进入 512-thread wavelet pipeline，但输出全 NaN、reference FAIL；binary SHA256=`7ec9665742a25833b9313f75406ea4dd880bc047ca8de95798011e2d447ed19d`，日志 SHA256=`6a05fd72987551e8d95a6e6cd629d4e32bbd39394308b5235c825107c4935691` | G3 FAIL；wavelet execution-chain PMD gap |
| P2.15-E3 | 与 P2.14 同 shape 非 wavelet control | P2.14 在相同数据契约/shape 的 PMD 完整 reference PASS；差异仅为 wavelet pipeline 与 512-thread 配置 | controlled A/B；不扩大到普通 v2 |
| P2.15-E4 | 撤回实验 guard并同步 PMD source | local/PMD header SHA256 均为 `28ebd6097ac9305de7a10053633f70b011ab7e6f90105d4014267252e00456f1`；最终不保留 device 改动 | restore PASS |

PMD candidate 只显式报告 `ignore GPUVM APE`，不足以把 NaN 指认给某一条 opcode；当前证据仅支持“gfx946 PMD 的该 wavelet pipeline 执行链不能用于正确性验收”，不支持“Shaobo 真机不支持 wavelet”。继续局部调 tile/layout 会超出可由 PMD验证的最小闭包，因此按规则标记 `SKIP-PMD`，保留真实 kernel，不制造模拟器专用回退。

门禁结果：G0-G2 PASS；G3 FAIL/SKIP-PMD；G4 PASS；G5/G6 NOT RUN。最终无业务代码差异，gfx936/gfx938 不存在 candidate 性能变化；gfx946 未测试性能。P2.15 只提交本文证据，随后继续 NCHW v2 target。

### Step 2.16：`03_grouped_conv_bwd_weight_mmac` NCHW v2 FP16 — PASS-PMD

- target：`example_grouped_conv_bwd_weight_mmac_nchw_v2_fp16`；NGKHW/GKCYX/NGCHW、1x1 stride1 pad0，FP16 out/input、FP32 weight 与累加。默认 `N=K=C=256,H=W=28` 的 CPU reference 过大，三架构正式验收统一使用最小健康 shape `G=1,N=2,K=C=128,H=2,W=4`，满足 `H*W` 的 8 元素向量读取和 GEMM reduction K=16。
- 首个 `N=32,H=W=1` 探针在三架构均被 `IsSupportedArgument` 正确拒绝：`H*W=1` 不满足 A/B `SrcScalarPerVector=8`，不计为架构失败。有效 shape 上 gfx936/gfx938 先建立修改前健康控制，gfx946 初始输出全 0；最小适配只把 gfx946 加入 NCHW v2 grouped wrapper 的 kernel body guard。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.16-E0 | 三架构无效 `H*W=1` 参数边界 | 均返回 unsupported；gfx936/gfx938 日志 SHA256=`4f9205436ef24a49ab42adf9381b0f831a3aeacdcb33c0efc089f934dc702d43`，gfx946 为 `4482194918d9e5b593a04892bf71623d40aaabbad132dd53d4268226a8d346bb` | contract control；改用 `H*W=8` |
| P2.16-E1 | gfx936/gfx938 修改前有效最小 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `c8e0f860c4534562753b1ec4948821c27b27dade83f2dca19fee1fd4c2e88830`、`dae1698dc045bcbc1024ea27865390fae50689854940b455c0a9354150dccab2`，日志均为 `2d8d5b840f3ccd89587c068875351071c01f9af440150c89a4422b43f0398a8e` | G0 healthy controls PASS |
| P2.16-E2 | gfx946 修改前有效最小 shape | supported/dispatch PASS，但输出全 0、`max err:152`；binary SHA256=`db403c03ca2d99bea7b87e7e72485c08813ec613b18bb898cafb1c7c5e518f14`，日志 SHA256=`e8837590a6d3813adbd0e3ef33bf6332dc9fec04f8f58e71630412e16985b846` | G3 FAIL；顶层空 guard |
| P2.16-E3 | NCHW v2 grouped wrapper guard 加入 gfx946 | PASS；真实 dispatch 和完整 CPU reference 正确、进程返回 0；binary SHA256=`708f8a908aca3175b97ac4b93f91849ab0f3afc8c4769867ff65e38b97a6a1d6`，日志 SHA256=`0de4341d5c2d652a0f0c28ebbfa5da1a99049d43511b7e43897dcd71cbd1e54b` | G1-G3 PASS |
| P2.16-E4 | gfx936/gfx938 candidate 有效最小 shape | 两侧完整 CPU reference PASS；candidate host binary 分别与各自 baseline 逐字节相同，日志保持不变 | G4 PASS |
| P2.16-E5 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧 SHA256 均为 `02288303da539cbf3fa1c1205cb88dfee420d6650cfa2e460339fd8f128038c8`；gfx938 两侧均为 `8bc272993f697a366ab3b0847f0f84e39105dad6dcb50c57f3667c1826df6c52`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD candidate 只报告一次 `ignore GPUVM APE`，未出现指令终止或数值错误。gfx946 全程关闭 timing，未测试性能；gfx936/gfx938 host binary/device ELF identity 已直接排除性能变化。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。至此 `03_grouped_conv_bwd_weight_mmac` 五个 target 已逐例验收：四个 PASS-PMD、wavelet 一个 SKIP-PMD。P2.16 独立提交后继续 `07_grouped_conv_fwd_bias_activation`。

### Step 2.17：`07_grouped_conv_fwd_bias_activation` NHWC FP16 CShuffle — PASS-PMD

- target：`example_grouped_conv_fwd_bias_relu_mmac_nhwc_fp16_cshuffle`；NHWGC/GKYXC/NHWGK、FP16 input/weight/bias/output、FP32 accumulate、3x3 filter 与 AddRelu epilogue。该 target 只在 `CK_BUILD_UNSTABLE_HCU_EXAMPLES=ON` 时入图，本步在三棵既有 build tree 上显式启用该选项，原 GPU target cache 保持 gfx946/gfx936/gfx938 不变。
- 正式 shape 为 `G=1,N=2,K=C=128,H=W=8,pad=1`。先前 `C=64` 探针被三架构共同拒绝，是因为 `num_k_loop=18` 只够一次 `3x3*NumPrefetch(2)` 外层流水，而当前实现显式要求 main loop；`C=128` 使 `num_k_loop=36`，满足完整 pipeline，同时保持 CPU reference 规模可控。
- 最小适配只把 gfx946 加入 `device_grouped_conv_fwd_bias_activation_mmac_v2r1_cshuffle.hpp` 顶层 device kernel body guard。未修改 descriptor、tile、multistage pipeline、cache swizzle、AddRelu epilogue或 launch 参数。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.17-E0 | gfx936/gfx938 修改前有效最小 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `78adedd9175e226fbb912b98de5f49617c7d9af8b02c9bace9cfd9093a99f563`、`0388e6625323e8505c4e2cafe3a41f43f744e28021906e536f126e40546462eb`，日志均为 `37c498f4dd8d87dd52e5f010b50388eb248a652752f3f74bdf2cb0af6302a948` | G0 healthy controls PASS |
| P2.17-E1 | gfx946 修改前有效最小 shape | focused build/dispatch PASS，但输出全 0、`max err:1286`；binary SHA256=`96c0af62a261e187b93b77f15600c421a6465cf91ffd56afc56e55c70ef3235b`，日志 SHA256=`9c17ba2643c80ed7a525ba9609df8efa4c1cfd9f7d37eabfba156a80f85adb69` | G3 FAIL；顶层空 guard |
| P2.17-E2 | 只把 gfx946 加入 forward bias activation wrapper guard | PASS；实际以 `--offload-arch=gfx946` 重编译，真实 dispatch 与完整 CPU reference 正确、进程返回 0；binary SHA256=`3f2208c48ced797e026271a8212dd2cedd70a26a6de7a60f4ca84429d9692b0a`，日志 SHA256=`b5ea8464c91dd509ae1d3c0a5eb14e9f41d8cef9a6349a197b902e0480b60fd9` | G1-G3 PASS |
| P2.17-E3 | gfx936/gfx938 candidate 同 shape | 两侧完整 CPU reference PASS；candidate host binary 分别与各自 baseline 逐字节相同，日志保持 `37c498f...a948` | G4 PASS |
| P2.17-E4 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧 SHA256 均为 `9520627fd8fa58744a6f48f4d26f06e3933d9fd46f14dbc5c8f2b183cc95bfee`；gfx938 两侧均为 `01640edfb6c87efd707bd8fe6cfaf133989194edb520758c6fee96c2d5531f69`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD candidate 仅报告一次 pending-cacheline warning，未出现指令终止或数值错误。gfx946 全程关闭 timing，未测试性能；gfx936/gfx938 的 host binary 与 device ELF identity 已直接排除 kernel 性能变化，因此不执行运行时性能 A/B。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。P2.17 独立提交后重新盘点尚未验收的已启用 target，继续下一个最小闭包。

### Step 2.18：`09_grouped_conv_fwd_bias_add_activation` NHWC FP16 CShuffle — PASS-PMD

- target：`example_grouped_conv_fwd_bias_add_relu_mmac_nhwc_fp16_cshuffle`；在 P2.17 的 conv+bias+ReLU 基础上增加同输出 shape 的 residual tensor，使用 `AddAddRelu` CDE epilogue。该 target 同样要求 `CK_BUILD_UNSTABLE_HCU_EXAMPLES=ON`；三棵既有 build tree 已在 P2.17 配置，实际编译目标分别保持 gfx946/gfx936/gfx938。
- 正式 correctness shape 沿用 `G=1,N=2,K=C=128,H=W=8,pad=1`。最小适配包含两点：把 gfx946 加入 CDE wrapper 的 device body guard；把该 example 的 KPack 从 4 调整为 8。KPack=8 与 P2.17 已通过的 CD epilogue实例一致，并在本步受控实验中成为 gfx946 正确性的必要条件。
- KPack 会改变 gfx936/gfx938 device code，不能用架构 guard 排除性能影响。因此本步按用户规则在两台真卡上执行 correctness、device ELF 差异确认和默认大 shape 的 7 轮交错性能 A/B，而不是跳过 G6。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.18-E0 | gfx936/gfx938 KPack4 修改前有效最小 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `72290d4a6cfdf158e4d26372c986565c4d826875986980a130238e246d7324bf`、`45f322d965d99425d493e0fe887f765268d2f9c1afff8bb8de66d1a1890e49f9`，日志均为 `2a8c62495c57684a8ebdebdb37dc3a0f602fa82eca127f5be3d28076b6949b44` | G0 healthy controls PASS |
| P2.18-E1 | gfx946 KPack4 修改前 | build/dispatch PASS，但空 guard 使输出全 0、`max err:1281`；binary SHA256=`116af3c8391e9fff2bef57b0ac7360700d05261913cd04992b6338296cb8cdab`，日志 SHA256=`fb68589ce98f1259389c8ec618e85d24d9654b1e307f1ada91a691e8cc20139b` | G3 FAIL；空-kernel control |
| P2.18-E2 | 只放开 gfx946 guard，仍使用 KPack4 | 真实 kernel 执行但 reference FAIL、`max err:271`；binary SHA256=`440105b3b5f6b728d1d0a354392dbc3bc998fd986a28b974420d2b171a24dbab`，日志 SHA256=`e46f136a0709c417d4f6f58f816d57012fcc75c9c8119deab22aaf7df5aba599` | G3 FAIL；继续收敛 KPack |
| P2.18-E3 | KPack4 的 CDE 受控初始化 | residual=0 时 MMAC+bias 仍 FAIL；input/weight/bias=0、只保留 residual+ReLU 时 PASS。诊断 binary SHA256=`32d97e7b4c11f117c29536ec3c94e2ae1524f4c0b369a7b55cc47892c6d6b2eb`，组合日志 SHA256=`21d1868853c3d7a611d1212ea82470fdb9aeb961b8af52214a978a484eb25812`；诊断初始化随后撤回 | controlled A/B；排除 D-tensor load 与 AddAddRelu 本身 |
| P2.18-E4 | 同一 gfx946 target 把 KPack4 改为 KPack8 | PASS；真实 dispatch 与完整 CPU reference 正确、进程返回 0；正式 binary SHA256=`7f19e8fb81ed5bcf3457ea074877537f2ca2af6ba03b172339eccc1fa5f7371d`，日志 SHA256=`389d9036e008c764f0da2969846f0e095fa8d54c531652aed702e2e0b4679bdc` | G1-G3 PASS |
| P2.18-E5 | gfx936/gfx938 KPack8 candidate correctness | 两侧完整 CPU reference PASS；candidate binary SHA256 分别为 `85d87981331ef8821bf264e0f30da0e9983b609c0861e051305efb7480186b13`、`bfafa3bcd8836864523d5b7cdf3a2ce2025762a4f132bc8a4b9f65d0bd76ba9f`，日志均为 `08b1cb7f247e6041ad5e26ac17503895d0dc6d3b3e3fe9328ae519491ae07281` | G4 PASS |
| P2.18-E6 | gfx936/gfx938 baseline/candidate device ELF | gfx936 `3a667f8ccec5e7c068a5d42bd8c87ec67ccdf9eadbfe8ba9731c0dec8700ae08`→`2ab3306dff0800d4dfa2d5749634fb02041825082491b5661bcde3e83eb48262`；gfx938 `e56663f0727066ca24eda61580b37cb537388fa83a22cd69f1cf08cea0e5210a`→`ba9b2047edb9f28876882b9b8fa0daacd96f6d63e2844525a10ba58cdca1a610` | G6 requires runtime A/B |
| P2.18-E7 | 两台真卡默认 `G1,N256,K=C256,3x3,H=W14`，预热后 7 轮交错 A/B | gfx936 中位数 `0.348542`→`0.346319 ms`（-0.64%，日志 `f12af0e2d92a15f5e491468524e0f7f3d0cf2567fad73b5261468f4a3f108606`）；gfx938 `0.462861`→`0.457874 ms`（-1.08%，日志 `3c70d7b6152d35310ddfd10ea6dbde741920b2b6826a17cca0e76e9f2b37c1b5`） | G6 PASS；无性能回退 |

性能测试前两台服务器各 GPU 使用率均为 0%。gfx938 candidate 第 3 轮出现一次 `4.40648 ms` 外部抖动，原值保留在日志中；7 轮中位数对该离群点稳健，且其余 6 次 candidate 均在 `0.450223-0.460047 ms`。gfx946 PMD 全程只做 correctness、关闭 timing；candidate 仅报告 pending-cacheline warning，无指令终止。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 PASS（gfx936/gfx938 runtime performance A/B）。P2.18 独立提交后继续 P2.19 `24_batched_gemm`。

### Step 2.19：`24_batched_gemm` BF16 XDL v3 — BUILD-PASS / SKIP-PMD

- target：`example_hcu_batched_gemm_xdl_bf16_v3`；BF16 A/B/E、FP32 accumulate、row-major A/E 与 column-major B，使用 `DeviceBatchedGemmMultiD_Xdl_CShuffle_V3` 的空 D-tensor、Intrawave v3 配置。本切片只验收基础 batched GEMM；同目录带一个 D tensor 的 Add epilogue留给 P2.20 独立闭环。
- 初始 gfx946 构建在 `MfmaSelector<bhalf_t,16,16>` 处选择 AMD `mfma_f32_16x16x8bf16`，被 HCU 静态断言拒绝。最小构建闭包把 gfx946 的 BF16 16x16 fragment 元数据、selector、support trait 和 intrinsic 分派接到已有 `mfma_f32_16x16x16bf16_1k -> intrin_mmac_f32_16x16x16bf16 -> __builtin_hcu_mmac_f32_16x16x16_bf16` 兼容链，并把 gfx946 加入两个 batched kernel body guard。未修改 tile、pipeline、descriptor、launch 或既有架构分派。
- 正式 correctness shape 为 `M=N=K=128,Batch=2`，验证开启、整数初始化、计时关闭。最终 gfx946 code object 明确包含 128 条 `v_mmac_f32_16x16x16_bf16`，证明不是空 kernel；但 PMD 输出稳定错误，首四个 mismatch 与重复运行一致、`max err:484`。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.19-E0 | gfx936/gfx938 修改前 focused build 与正式 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `ebef8cb2cc1eff1bf203411133208047d47138dbf9f69bffe706caa0de667821`、`5cf24386c2c61d96c6ba268b144e4c59f94a16e744073fef5c0824fdfe0d64e1`，日志均为 `31c89bc326d3def99457d939b1238ef877cd055a6fb9ccb0329135db3b725593` | G0 healthy controls PASS |
| P2.19-E1 | gfx946 修改前 focused build | FAIL；`mfma_is_supported_on_hcu<mfma_f32_16x16x8bf16>` 静态断言终止，未生成本 target gfx946 binary | G1 compile closure missing |
| P2.19-E2 | 补齐 gfx946 BF16 XDL-to-MMAC 兼容链和 batched wrapper guard | build/code object PASS；binary SHA256=`902aab1fb2ea5f98815563593958135d3d90d1d82af633b13171bd8c9a34e36e`，device ELF SHA256=`362d84e8a79d20ba8d5819bea16710b6872856ab3f23a9934229f908b54c072f`，ISA 含 128 条 BF16 MMAC | G1/G2 PASS |
| P2.19-E3 | gfx946 正式 shape 单次 correctness dispatch | FAIL；实际 kernel 完成但 `max err:484`，日志 SHA256=`70b1138bfd128b91e2433cec0b206ae03e7857632c8e89c4f70416bfcd2aae06` | G3 SKIP-PMD |
| P2.19-E4 | gfx936/gfx938 candidate 同 shape | 两侧完整 CPU reference PASS；candidate binary 与日志分别和各自 baseline 逐字节相同 | G4 PASS |
| P2.19-E5 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧均为 `31146bcccb529b17a444ddd61e1d6fdf0413a1c6b653ef5a48170ada00327c46`；gfx938 两侧均为 `0f528b5a81523ef4366d30a30d79e6a5675ff2e43f8fcd62986779949963ddd9`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

诊断期间尝试只改变 CShuffle/output dtype 以隔离 BF16 写回，但 V3 通用 invoker 会编译期实例化未使用的 KBatch AtomicAdd 分支：FP32/FP16 输出分别因缺少 vector atomic、混合 CShuffle 到 BF16 又因未定义的混合 `PassThrough` 而无法形成可运行 A/B；这些诊断源码均已撤回。已有同一 PMD 的 BF16 basic GEMM 健康控制，以及 P3.5b 已收敛的 BF16 grouped CShuffle 失败，说明不能扩大为“gfx946 BF16 MMAC 不支持”；当前证据只足以把本 legacy batched v3 的 BF16 accumulator-to-output/CShuffle 模拟链标记为 `SKIP-PMD`，等待 PMD 更新或 Shaobo 真机复核。

PMD 同时报告 4 组 `src1_neg/src1_abs should not set` 与一次 pending-cacheline warning；尚不足以点名单一 opcode。gfx946 全程未测试性能。gfx936/gfx938 测试前 GPU 使用率均为 0%，且候选 device ELF identity 已直接排除性能变化，因此不执行运行时性能 A/B。

门禁结果：G0-G2 PASS；G3 FAIL/SKIP-PMD；G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。保留 build/ISA 闭包但不声明 gfx946 运行正确；P2.19 独立提交后继续 P2.20 multi-D Add target。

### Step 2.20：`24_batched_gemm` BF16 multi-D Add — BUILD-PASS / SKIP-PMD

- target：`example_hcu_batched_gemm_multi_d_xdl_bf16_v3`；在 P2.19 的 BF16 batched GEMM 上增加一个 row-major BF16 D tensor，并用 `Add` epilogue生成 BF16 E。正式 shape、初始化与 timing contract 均保持 `M=N=K=128,Batch=2,verification=1,init=1,time=0`。
- 本 example 直接复用 P2.19 已提交的 gfx946 BF16 XDL-to-MMAC 和 wrapper guard 闭包，没有新增业务代码。为保持独立 baseline，gfx936/gfx938 先从父提交 `928b666` 提取四个旧头文件快照编译 baseline，保存 binary 后再恢复当前提交 `ee073b2` 的头文件并编译 candidate；没有 reset 或覆盖本地 dirty。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.20-E0 | gfx936/gfx938 父提交 baseline focused build 与正式 shape | 两侧完整 CPU reference PASS；baseline binary SHA256 分别为 `e2daaebeb90528be29adddae4e49de66790043be3678c9d64fcfb8815847c7df`、`294aa7809aa5cb1a859e1a3797d101f167027f8e5969cfa02f22ed6a24d33bbb`，日志均为 `10efa2997686fd295ccb067323f77cbbeecc7d4675e70118b46925aa98bb43ed` | G0 healthy controls PASS |
| P2.20-E1 | gfx946 focused build、正式 shape 单次 correctness dispatch | build/code object PASS、实际 dispatch 完成，但 reference FAIL、`max err:71`；binary SHA256=`564db85d78e8d2adef130ebf3d9f30ab9b6a8b2a41b2eed06fdb1f8cf1236458`，日志 SHA256=`dc338fb642ca3e45e1cdda71f102801057b658d2ab80c5ce3ae86da4cd8810de` | G1/G2 PASS；G3 SKIP-PMD |
| P2.20-E2 | gfx946 device ELF/ISA | device ELF SHA256=`f3ffe8455ba5d6407140b737bec8e4c76d510ea5561d1e52a5d8d8ef062866dc`；含 128 条 `v_mmac_f32_16x16x16_bf16`、32 条 `buffer_load_dwordx2` D-load 与 BF16 global store | 实际 multi-D Add kernel observation |
| P2.20-E3 | gfx936/gfx938 candidate 同 shape | 两侧完整 CPU reference PASS；candidate host binary 和日志分别与各自 baseline 逐字节相同 | G4 PASS |
| P2.20-E4 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧均为 `0c7c4d8d6d62b9fcaf25d36064cc81fd431845981be1d4410ec8da0f162f09e4`；gfx938 两侧均为 `10e2763d440e1951fc7ad5279884bf3ae1e50be2104b35321e79959427d0fc0f`；内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD 失败签名仍包含 4 组 `src1_neg/src1_abs should not set` 和一次 pending-cacheline warning；加入 D load/Add 后错误幅度变化，但没有形成区别于 P2.19 的可定位单一指令证据。由于 P2.19 已确认空 D-tensor 的同一 BF16 accumulator-to-output/CShuffle 链失败，本步不为绕开模拟器而改写已在两台真卡正确的 Add epilogue，按任务规则合并记录为同一 `SKIP-PMD` 子模块。

gfx946 全程未测试性能。gfx936/gfx938 的 host binary 与 device ELF identity 已直接排除 kernel 性能变化，不执行运行时性能 A/B。门禁结果：G0-G2 PASS；G3 FAIL/SKIP-PMD；G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。P2.20 只提交本文证据，随后继续 P2.21 `25_gemm_xdl_v3`。

### Step 2.21：`25_gemm_xdl_v3` BF16 CShuffle 模块 — BUILD-PASS / SKIP-PMD

- 本目录启用五个 target：basic、空 D-tensor multi-D、multi-ABD、Add、Add+ReLU；它们均为 BF16 A/B/output、FP32 accumulate、BF16 CShuffle、Intrawave v3。三种实际 grid wrapper 的 single-LDS/2-LDS kernel body guard 均遗漏 `__gfx946__`。
- 采用 `example_hcu_gemm_xdl_bf16_v3` 作为共享执行链的代表性最小复现，正式 shape 为 `M=N=K=128,KBatch=1,verification=1,init=1,time=0`。未放开门控时 dispatch 后输出全零、`max err:99`，确认当前 gfx946 binary 是空 kernel。
- 诊断仅临时把 basic wrapper 的两个 body guard 加入 `__gfx946__`；P2.19 已提交的 BF16 XDL-to-MMAC 闭包使真实 kernel 成功构建并 dispatch，但 CPU reference 仍失败。诊断门控随后撤回，远端源码和 binary 也重建恢复，避免提交“能够运行但结果错误”的架构声明。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P2.21-E0 | gfx946 原始 guard focused build 与单次 correctness | build/dispatch 完成但输出全零、`max err:99`；binary SHA256=`4bb139cdd86b2924e506014e9a1d054c867907a7addb420ff780c88e2c4380b8` | G1/G2 PASS；确认 empty-kernel guard 缺口 |
| P2.21-E1 | 仅临时放开 basic single-LDS/2-LDS gfx946 guard | 实际 kernel dispatch，reference FAIL、`max err:87`；diagnostic binary SHA256=`53079dfc2d3de1abfc9341fab1e87c4e7e6ba69722bedfcc03bda66d6fbca2ef` | G3 SKIP-PMD |
| P2.21-E2 | gfx946 diagnostic code object/ISA | code object SHA256=`6d2ef168a7687a5a8765ea1ac16797356cc8de64f4aca1c4715259e70623b051`；含 128 条 `v_mmac_f32_16x16x16_bf16`、32 条 `ds_read_b64`、32 条 `ds_read_b128` 与 BF16 stores | 排除空 kernel，命中真实 BF16 MMAC/CShuffle 链 |
| P2.21-E3 | 撤回诊断门控并恢复 PMD 构建物 | source 恢复无 `__gfx946__` guard；重建 binary SHA256 回到 `4bb139cdd86b2924e506014e9a1d054c867907a7addb420ff780c88e2c4380b8` | 不保留未通过正确性的业务改动 |
| P2.21-E4 | gfx936/gfx938 代表 target focused build 与同 shape | 两侧完整 CPU reference PASS；binary SHA256 分别为 `c49357c2ee222353e3324f131ed709726d2701178df0ce1f6beb158abaaa3b9d`、`2db3500c86a8d039e5dc2897bff45e9f6ef01982f681cdd2a8f547cdba801ff2`，日志 SHA256 均为 `47da1957b282d9e7564435887ad21e6e9d396904481a20529305875ac7765dca` | G4 PASS |

门控后的 PMD 失败仍稳定出现 4 组 `src1_neg/src1_abs should not set` 与一次 pending-cacheline warning，和 P2.19/P2.20 的 BF16 XDL v3 失败签名一致。其余四个 target 只是在同一 BF16 accumulator-to-output/CShuffle 执行链上增加 multi-D/multi-ABD/Add/ReLU，且对应 wrapper 具有完全相同的 gfx946 guard 缺口；按“Perf Model 指令支持问题无法解决则记录并跳过模块”的规则，不重复提交四组会运行错误的 guard，也不逐个重复相同 PMD 失败。

最终只有本文证据变化，device 源码恢复到切片前状态；gfx936/gfx938 已完成代表性 correctness 回归，确定不存在候选 kernel 或性能变化，因此无需性能 A/B。gfx946 全程只做 correctness，不测试性能。门禁结果：G0-G2 PASS；G3 FAIL/SKIP-PMD；G4 PASS；G5/G6 NOT RUN。P2.21 模块闭合，下一步重新盘点其余已启用 target。

### Step 2.22：legacy `05_gemm` TT v2 FP16 CShuffle — PASS-PMD

- target：`example_gemm_mmac_tt_v2_fp16_cshuffle`；A/B/C 均为 row-major，FP16 输入输出、FP32 accumulate，block tile=`128x4x32`。该 target 与 P2.4 的 TN v2 CShuffle 是独立数据布局，源码当前已具备 gfx946 kernel body guard，因此本步不新增 device 或 runner 代码。
- 三架构均 focused build PASS，并分别运行 `M128,N512,K256` 与 `M256,N512,K256`，`verification=1,init=1,time=0`；六次完整 CPU reference 校验均通过、输出明确显示 `timing:off`。
- gfx946 binary SHA256=`7d95a41567d3ed51c9de9c25ccd4e9f1a8edea8139d43ad48794ada6b272997c`，日志 SHA256=`8b5336c8df8a88a8d90061aa1f9feb63a7f6f11f4343429bcd8184d89406188e`，code object SHA256=`302a44770ca9744eadcbd16e869108cc8f60ac82d1ecd2ff3fc92fa6fffe4651`。ISA 包含 96 条 `v_mmac_f32_16x16x16_f16`、12 条 `ds_read_m32x16_b16_alt`、24 条 `ds_read_b64`、32 条 `ds_write2st64_b32` 与 32 条 `buffer_store_dwordx2`，证明 TT 数据搬运、MMAC 与 CShuffle 实际执行。
- gfx936 binary SHA256=`11fdf8305fe002eb6b3ecc4d3c780c0e684221f8770ab2d55b2dcfd7f176d3cc`，gfx938=`e252c44f462f41ca449fad5383ea57bfd3e172564e3d0cd23383854e6fc88b6b`；两侧日志 SHA256 均为 `97ff1f4b7d4a3f0756207d258f9195a0c606ee49989cc8601dc2288533f2503b`，正确性回归 PASS。

PMD 两档各仅报告一次 pending-cacheline warning；没有出现 P2.1 NT 路径的错误输出。该结果只证明当前 TT v2 CShuffle 的整体地址/layout/指令组合可用，不把单独存在的 `ds_read_m32x16_b16_alt` 泛化为 P2.1 的旧 NT 双 matrix-read 布局已被支持。

本切片没有源码候选差异，gfx936/gfx938 的 device kernel 和计时路径均未改变，可直接排除性能影响，因此不执行性能 A/B；gfx946 PMD 全程关闭计时。门禁结果：G0-G4 PASS；G5/G6 NOT RUN。下一步继续 P2.23 NN v1 FP16 CShuffle。

### Step 2.23：legacy `05_gemm` NN v1 FP16 CShuffle — PASS-PMD

- target：`example_gemm_mmac_nn_v1_fp16_cshuffle`；A/B/C 均为 column-major，FP16 输入输出、FP32 accumulate，block tile=`4x128x32`。其 grid wrapper 已有 gfx946 guard，本步不修改源码。
- gfx946/gfx936/gfx938 均 focused build PASS，运行 `M128,N512,K256` 与 `M256,N512,K256` 两档，显式使用 column-major stride（A/C stride=M、B stride=K）以及 `verification=1,init=1,time=0`；六次 CPU reference 均通过。
- gfx946 binary SHA256=`023762e2e2b8c31da16c280fe4c471c3a78404545d5f688a7579c2ef77ad9205`，日志 SHA256=`55f7d8dff49777667ac4c405e3d314b4e9d3751abdb6f79b6122255c1acedeb4`，code object SHA256=`9982dccdf092578dabd8e039af7e7cbb952068b319211f625a564dd8f27d5189`。ISA 含 96 条 `v_mmac_f32_16x16x16_f16`、12 条 `ds_read_m32x16_b16_alt`、24 条 `ds_read_b64`、32 条 `ds_write2st64_b32` 与 32 条 `buffer_store_dwordx2`，覆盖 NN column-major load/layout 与 CShuffle 写回。
- gfx936/gfx938 binary SHA256 分别为 `dcb139e626bda24f7d6db363264016cfe6a54f0b7df1db17934eda2a7fcdc5a0`、`cf7426282328e1bae377eb8036c46bc1f6850aa64aecd83c2dce070abaaba81a`；两侧日志 SHA256 均为 `f30d8c4281efd69f22d15c2ccd5e45f5e805cde72e754feec3bcd64bc25a9a69`，G4 PASS。

PMD 两档各出现一次 pending-cacheline warning，但实际输出完整通过校验。与 P2.22 相同，当前证据只覆盖该 target 的整体布局/指令组合，不据单条 matrix-read 指令外推其他 legacy 模块。最终无源码改动，gfx936/gfx938 无性能变化面，不运行性能 A/B；gfx946 全程 timing off。门禁结果：G0-G4 PASS；G5/G6 NOT RUN。下一步继续 P2.24 NT v1 FP16 CShuffle。

### Step 2.24：legacy `05_gemm` NT v1 FP16 CShuffle — PASS-PMD

- target：`example_gemm_mmac_nt_fp16_cshuffle`；A/C column-major、B row-major，FP16 输入输出、FP32 accumulate。专用 `..._nm_v1_cshuffle` grid wrapper 已有 gfx946 guard，本步没有源码改动。
- 三架构 focused build 后运行 `M128,N512,K256` 与 `M256,N512,K256`，stride 分别遵循 A/C column-major 与 B row-major，`verification=1,init=1,time=0`；六次均完整通过 CPU reference。
- gfx946 binary SHA256=`8f36f9cbd64d8ead6efe39643d479bf79df0eb42b2044919384dc2123ec4dd7c`，日志 SHA256=`2d3f2e95e8757009b58c330a9f8bc786cf0dfb93239feefcab638fa752d403de`，code object SHA256=`01536e5d7616562e37f179422fa242f9d973813ad847fa3405f7d120e220cb69`。ISA 含 192 条 `v_mmac_f32_16x16x16_f16`、24 条 `ds_read_m32x16_b16`、24 条对应 `_alt`、32 条 `ds_read_b128` 与 CShuffle stores，证明并非空 kernel。
- gfx936/gfx938 binary SHA256 分别为 `5a311cda4566a76d58e394e47b7f8f07e9072cee9471df9b87a5e78213695436`、`a9b7331e69fed161311f620cd3d916dc2fa1dc34804ff86fa9a24d136bb550c3`；日志 SHA256 均为 `f5b760f0111b0cf743b81cc1f6eae930ea6b18dfa0160728ac2d896cedc07983`，G4 PASS。

PMD 每档报告 16/32 条地址计算类 uninitialized-VGPR warning 与一次 pending-cacheline warning，但输出均正确。P2.1 非 CShuffle NT 的实际 tile/layout 与本 target 不同，且其两档曾稳定错误；因此本步按 example 粒度声明 PASS，不覆盖或删除 P2.1 的 `SKIP-ARCH-LAYOUT`。没有候选源码差异，gfx936/gfx938 不存在性能影响，不做性能 A/B；gfx946 未测试性能。门禁结果：G0-G4 PASS；G5/G6 NOT RUN。下一步继续 P2.25 TN v2 CShuffle NM。

### Step 2.25：legacy `05_gemm` TN v2 FP16 CShuffle NM — PASS-PMD

- target：`example_gemm_mmac_tn_v2_fp16_cshuffle_nm`；A row-major、B/C column-major，FP16 输入输出、FP32 accumulate，block tile=`128x128x16`。初始专用 grid wrapper 的 kernel body guard 只允许 gfx926/928/92a/936/938，gfx946 dispatch 后输出全零、`max err:2472`。
- 最小适配只把 `__gfx946__` 加入 `gridwise_gemm_mmac_tn_mk_nk_nm_v2_cshuffle.hpp` 的 device guard，不修改 tile、descriptor、MMAC、CShuffle、runner 或既有架构分支。修改后 gfx946 在 `M128/256,N512,K256` 两档、`verification=1,init=1,time=0` 均完整通过 CPU reference。
- 基线 empty-kernel binary SHA256=`e7778eb032ba86dd07021107f520604189232a8352f942dcb9a1c2015d794574`、日志 SHA256=`143f1ce6054dc3c910531418783c07a0822262081e29b584fb71564161d3460a`。candidate binary SHA256=`5ae7530573e9d7c74ae366214a5f554d53aadcc930229914fe4ee0a1ebcaff11`，日志 SHA256=`a5d40a594e3bda104606431f6313c621a0f0ab00bf2e2978eeaa3d1da653cbef`，code object SHA256=`b41d6d7b8b35325267e2fa6e7963a8adad4d2daf54495100937edada78696eaa`。ISA 含 96 条 `v_mmac_f32_16x16x16_f16`、128 条 `ds_write_b16`、80 条 `ds_read_b64` 与 32 条 `buffer_store_dwordx2`，排除空执行。
- gfx936/gfx938 同步最终头文件后 focused build 与两档 correctness 均 PASS；binary SHA256 分别为 `a7ef14f1c7d55039cdec2834781a92d79dd7543bb4fbd15e11c8452481886774`、`ef139fbb6ad1a308218a55370c18e011c87939f680890b1f4ed94aa0f5d10064`，日志 SHA256 均为 `19a37ed2ee31bef999bf1b91c0b8d99f26ccfb1ab159c866604b6aa1b4f2f9a7`。

PMD candidate 两档仅报告一次 pending-cacheline warning，未出现指令终止或数值错误。代码差异位于 `defined(__gfx946__)` 条件，gfx936/gfx938 预处理与 device 路径确定不变；结合两台真卡 correctness PASS，可直接排除性能影响，无需性能 A/B。gfx946 PMD 全程关闭 timing。门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（direct architecture condition）。下一步继续 P2.26 TT v2 CShuffle NM。

### Step 2.26：legacy `05_gemm` TT v2 FP16 CShuffle NM — PASS-PMD

- target：`example_gemm_mmac_tt_v2_fp16_cshuffle_nm`；A/B row-major、C column-major，FP16 输入输出、FP32 accumulate。初始专用 wrapper 未包含 gfx946，`M128,N512,K256` 输出全零、`max err:560`。
- 只把 `__gfx946__` 加入 `gridwise_gemm_mmac_tt_mk_n0k0k1n1_nm_v2_cshuffle.hpp` 的 kernel-body guard；修改后 gfx946 在 `M128/256,N512,K256` 两档、`verification=1,init=1,time=0` 均通过完整 CPU reference。
- 基线 binary/log SHA256=`3d3ed8d3c957d6a7472ca1b993f0666c5f3ba3244fad291be02f2f5b1a189d78`/`4e7b69fcb6e152897182c6f081b6c1b4b3372cdf2d4d051e3f55ba2a492bf0eb`；candidate binary/log/code object SHA256=`fcea20707bf3ffea9bf27e560799702d9587609ddb8397ee613aaa53790e9428`/`eaf29bace2c222467bc9b6aab2b86b289818837ae70b608117e8cd5b9591dab4`/`1830994bfa39bb8ee1b05395cd8aa78194997d1ee0cf8df26d15fe8f06789418`。ISA 含 96 条 FP16 MMAC、12 条普通 matrix-read、CShuffle LDS 与 32 条 vector store。
- gfx936/gfx938 同步最终头文件后两档均 PASS；binary SHA256=`d2b12191ae7a4164f92de520a592082eb65455197c889788f186f9b0f88a8c8f`/`a8165e664e6d4953a70dc1328060e339c70190adaddda502d541a5e6c628e439`，日志 SHA256 均为 `a144ea9c0bc7440f60e52d9286978c0d98716097bfd192092e421a3dce63ad29`。

PMD candidate 仅有 pending-cacheline warning。改动严格受 `__gfx946__` 条件隔离，gfx936/gfx938 device 路径确定不变且 correctness 已通过，无性能不确定性，不执行性能 A/B；gfx946 timing off。门禁结果：G0-G4 PASS；G5/G6 NOT RUN。下一步转向尚未验收的 grouped-conv forward mixed CShuffle target。

### Step 2.27：legacy grouped-conv forward mixed-layout FP16 CShuffle — PASS-PMD

- target：`example_grouped_conv_fwd_mmac_nhwc_v2_fp16_cshuffle_mixed`；输入为 `NHWGC`、权重为 `GKYXC`、输出为 `NGKHW`，FP16 输入/输出、FP32 accumulate。初始 `device_grouped_conv_fwd_mmac_v2_cshuffle_mixed.hpp` 的 kernel-body guard 只允许 gfx926/928/92a/936/938，gfx946 dispatch 后输出全零、CPU reference `max err:884`。
- 最小适配只在该 mixed wrapper 的 device guard 中加入 `__gfx946__`，不修改 tensor descriptor、tile、MMAC、CShuffle、runner 或既有架构分支。gfx946 使用默认健康形状 `G=N=1,K=C=64,Y=X=3,Hi=Wi=8,stride=dilation=1,pad=1`，`verification=1,init=1,time=0`，完成真实 dispatch 且完整 CPU reference PASS。
- gfx946 baseline binary/log SHA256=`bea038a4c3904f34fd63668d84835919c87d583a8481d3f963fc09806dd14b04`/`554a5edabd82694a2e19b6efe691a40c7f81d92c0c2b03f605d166b16aa808ce`；candidate binary/log/code object SHA256=`0bcc6e009f2e4b9035af0f44ba1059cecb60b3e5413a102b032e44a7845123db`/`290fa669f6ffb2bf3b22d07b91cb0a82c91de5ea5117424e8ee5b79493c32f73`/`c3eb90d1e2427d61afbf7500d8eaf6c7f6f6ff56a361743dc313b76dc7d49d3c`。ISA 包含 12 条 `v_mmac_f32_16x16x16_f16`、20 条 LDS read、35 条 LDS write，排除空 kernel 并覆盖 mixed output descriptor 与 CShuffle 写回。
- gfx936/gfx938 同步最终 header 后同形状 correctness 均 PASS；两侧 baseline/candidate 宿主 binary 分别保持 `0e36fee6f324094cd2b62400ec4c52cddc4037f0712c6d5853e002f6190fc3b0`、`7cfab6ae911270406a26d9c1916f49efb93085b7d277aff139933d0916f732e3`，日志 SHA256 均为 `0f0223db8a85405b4783a6779d500353ccb379275337f5ba4708a0f99d3bd006`。提取出的 device code object 也逐字节相同：gfx936 两侧均为 `35ab104e14aeb9b1a0872fac1a2adce16751264bada5a01425738d550399f143`，gfx938 两侧均为 `f6f58f95fc7983cb7cfd505a50c4c2bb46269bb645881c4f126ff6c1277b44e6`。

PMD candidate 仅报告一次 pending-cacheline warning，未出现指令终止或数值错误。修改严格受 `defined(__gfx946__)` 隔离，且 gfx936/gfx938 的宿主 binary 与 device code object identity 已直接排除 kernel 性能变化，因此不执行真实性能 A/B；gfx946 全程关闭 timing。门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（direct architecture condition and code-object identity）。下一步继续 P2.28 grouped-conv forward FP8 target。

### Step 2.28：legacy grouped-conv forward FP8 CShuffle — PASS-PMD

- target：`example_grouped_conv_fwd_mmac_nhwc_fp8`；`GNHWC/GKYXC/GNHWK`，FP8×FP8 输入、FP32 accumulate/output，block tile=`128x128x64`。最小健康形状为 `G=1,N=128,K=128,C=64,Y=X=Hi=Wi=1`，可在单个 block 内独立覆盖 FP8 load、MMAC、CShuffle 与 FP32 store。
- 初始 gfx946 外层 `device_grouped_conv_fwd_mmac_cshuffle.hpp` kernel guard 不执行 body；只增加该 guard 后仍输出全 0。逐层追踪确认第二个独立缺口是 `hcu_mmac.hpp` 的 `intrin_mmac_f32_16x16x32f8_f8` 在 gfx946 上执行 `swallow`。最终最小适配仅把 gfx946 加入共享外层 wrapper 和 FP8×FP8 builtin guard，不放开 FP8×BF8、BF8×BF8、BF8×FP8 或 INT4 指令。
- gfx946 初始 baseline binary/log SHA256=`6c94cf4485a8d49bb95e911fafcd7fb81cf0a26b4845f4007f7d88697c005588`/`4c632d3fcc9d8f05489ec83a49970c3de550c4ccf7bbce9952b0e829df39bb8c`，输出全 0、`max err:316`。只放开外层 guard 后仍同样失败；再放开 FP8 builtin 后完整 CPU reference PASS，candidate binary/log/code object SHA256=`d302015ea42d457ad12841330061c42edad670314b1114edf329421d32e971c2`/`fc40920a42b77891bceb99e920306a48dfca1437fe4fbe5d052855dfe27455f0`/`23644eb45714a61ad75dbfe26ba2e21d7ead83fa95c91c47fd8a4557af14f5d2`。ISA 含 96 条 `v_mmac_f32_16x16x32_fp8_fp8`、56 条 LDS read 与 76 条 LDS write，证明 compiler、gfx946 code object、PMD FP8 MMAC 与 CShuffle 全链闭合。
- gfx938 修改前/后同形状均正确，宿主 binary 两侧均为 `176857662b4834760079829154646e7140288a601495d3ab0296641cdf466b44`，日志两侧均为 `1544140a7c94fb8644b8aac9e2eda2f3e3b134703e8cd078783d5f28468e4e0d`，device code object 两侧均为 `2e27558b519787b3a8a69be8857cdf531df0ca5cd202b8347d41a705c4712953`。
- gfx936 的既有 FP8 builtin guard明确不支持该架构；baseline/candidate 均输出全 0、`max err:316`，宿主 binary 两侧均为 `8fb9e3b1de06d028680563115ce2ed2be4de748e21f85f1f3130ad335909b2be`，日志均为 `762b773fabeeb1922c2885cf25cd939019a88720ef303deeea57790a495ff836`，device code object 两侧均为 `f2be9e96923605996e1624afda66fd1b7c0887716c570ceffe77181ffb7b5b9d`。这属于修改前已存在的 gfx936 FP8 架构边界，不归类为 candidate regression。

PMD candidate 仅报告一次 pending-cacheline warning，reference 正确。两处改动都严格受 `__gfx946__` 隔离，gfx936/gfx938 宿主 binary 与 device code object identity 进一步直接排除性能变化，因此不执行真实性能 A/B；gfx946 timing off。门禁结果：G0-G3 PASS；G4 gfx938 correctness PASS、gfx936 architecture-bound no-regression；G5 NOT RUN；G6 NOT RUN（direct architecture condition and code-object identity）。下一步继续 P2.29 INT4，必须独立验证其 `v_mmac_i32_16x16x64_i4` 路径。

### Step 2.29：legacy grouped-conv forward signed INT4 CShuffle — PASS-PMD

- target：`example_grouped_conv_fwd_mmac_nhwc_int4`；`GNHWC/GKYXC/GNHWK`，host 将 channel 维两个 signed INT4 打包到一个字节，device 使用 INT4×INT4→INT32 MMAC 并输出 INT32。验收形状与 P2.28 对齐为 `G=1,N=128,K=128,C=64,Y=X=Hi=Wi=1`，同时覆盖 pack contract、INT4 MMAC、CShuffle 与 INT32 store。
- P2.28 已启用共享外层 wrapper，但初始 `intrin_mmac_i32_16x16x64i4` 的 gfx946 分支仍为 `swallow`，因此 baseline 实际执行搬运/写回后输出全 0、`max err:316`。本步只把 gfx946 加入 signed INT4 builtin guard，不开放 unsigned INT4，也不改变 example、packing、tile 或 descriptor。
- gfx946 baseline binary/log SHA256=`9e23624d28444b6b98aa3cd22d7443cd405b4eab853152414be86d383e0989ed`/`d54f4346a847efbaf65656ee94017b0336a5da0a6443faa59713185b9ade7a70`。candidate 完整 CPU reference PASS，binary/log/code object SHA256=`53aab21226182b75a16b89efe8bbaf2da849a81f84b1d95d29cafa4c4ee658c9`/`66e052ea4fa140f04a814c517c4f2f7219084930cea13a37f954de5ba0c6c948`/`3ee191449f420fc027d166417bb617556fa1c4f18491fb248bc23dceb46ef0a8`；ISA 含 96 条 `v_mmac_i32_16x16x64_i4`、56 条 LDS read 与 76 条 LDS write，证明 compiler、code object、PMD INT4 执行和结果布局均闭合。
- gfx936/gfx938 修改前后同形状均 correctness PASS，日志 SHA256 均为 `3710a897cc8fb564f49f80ea095e97145467365fd8640f454205c473adc342c6`。gfx936 baseline/candidate 宿主 binary 均为 `4a5b2249a443c8a34118aef3f1ab53700724dcf8599a59220a61724454c9d930`、device code object 均为 `53523dfbb677f91d5ac0d6209c21c92e20f086d00dd5ea8ab6463b25bd98d9ac`；gfx938 宿主 binary 均为 `63f5bf5a00973727b59cb06235d9628815a9cc9dd536477f1f3ed0532e6bfc35`、device code object 均为 `52ce9d5bf0bcb067f576c39e45bde5aea4efbdf295244e05f188495e0bfcfd7c`。

PMD candidate 仅报告一次 pending-cacheline warning，reference 正确。改动严格受 `defined(__gfx946__)` 隔离，且 gfx936/gfx938 的 binary/code-object identity 直接排除性能变化，因此不执行真实性能 A/B；gfx946 全程关闭 timing。门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（direct architecture condition and code-object identity）。P2.27-P2.29 grouped-conv forward 补充矩阵闭合，下一步重新盘点剩余 enabled target。

### Step 3.1：`ck_tile/14_conv` FP16 2D forward — PASS-PMD

- target：`tile_example_conv_fwd`；NHWGC input、GKYXC weight、NHWGK output，FP16 输入/输出、FP32 累加，tile=`64x64x16`，包含 CK Tile MMAC pipeline 与 LDS CShuffle epilogue。
- 该 device 路径在当前源码中已具备 gfx946 native 能力，本步不增加架构 guard。runner 原先强制启用计时；新增默认值为 1 的 `-time` 参数，`-time=0` 只执行 correctness launch 并显示 `timing:off`，默认真机行为不变。
- gfx946/gfx936/gfx938 focused build 均 PASS。三架构分别运行 `G=1,N=4,C=K=64,H=W=4,Y=X=3,pad=1` 与 `N=8` 两档；六次完整 host reference 校验均通过并返回 0，PMD 每档只有一次 dispatch。
- gfx946 code object SHA256=`f23a5be4e34308f6ebdefff6310f037cfd50b05c8298ca5d906b1f155e559a70`；指令统计包含 16 条 `v_mmac_f32_16x16x16_f16`、`ds_write_b16(_d16_hi)`、`ds_read2st64_b64`、`ds_write2st64_b64`、`ds_read_b64` 及 global load/store，覆盖实际 convolution-IGEMM 与 CShuffle 链路。
- gfx936/gfx938 性能边界：修改前后 device ELF 逐字节一致；gfx936 两侧 SHA256 均为 `b7e6ef5c9487977df161813633dbdd84e063ea388e5d764436977e4a595aaf4f`，gfx938 两侧均为 `e6d8f493c847afe4b5c1e00d4d67c18e941caafc403db5b10c4c7bf06eb347df`。已直接排除 kernel 性能变化，不执行运行时性能 A/B。
- PMD 仅报告 pending-cacheline warning，不影响两档 dispatch 与 reference 校验；全程 `time=0`，没有性能测试。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 由 gfx936/gfx938 device ELF identity 闭合。本步只声明已测 2D FP16 forward 路径，不外推至 `conv_fwd_dsreadm`、LDS-direct 或 3D/fused conv。

### Step 3.2：`ck_tile/14_conv` FP16 2D forward LDS-direct — PASS-PMD

- target：`tile_example_conv_fwd_lds_direct`；与 P3.1 相同的 NHWGC/GKYXC/NHWGK、FP16/FP32 数据契约，但输入 pipeline 使用 LDS-direct 路径，仍以 MMAC 和 LDS CShuffle 完成计算/写回。
- 该 device 路径在当前源码中已具备 gfx946 native 能力，不增加架构 guard。runner 增加默认值为 1 的 `-time` 参数；`-time=0` 只做一次 correctness launch 并输出 `timing:off`，默认真机行为保持不变。
- gfx946/gfx936/gfx938 focused build 均 PASS。三架构运行与 P3.1 相同的 N=4/N=8 两档 3x3 convolution；六次完整 host reference 校验均通过并返回 0，PMD 每档只有一次 dispatch。
- gfx946 code object SHA256=`b9e2c579c2cd9593219016db492a0e97ced370ddd365c13d0e351925e229acd7`；指令统计包含 16 条 `v_mmac_f32_16x16x16_f16`、`ds_write_b16(_d16_hi)`、`ds_read2st64_b64`、`ds_read_b64` 及 global load/store，证明 LDS-direct variant 实际执行而非空路径。
- gfx936/gfx938 性能边界：修改前后 device ELF 逐字节一致；gfx936 两侧 SHA256 均为 `e65433e3894b23d292dc1770abd07ebd2cbb5d9e189b88dfd1d8d364042666a6`，gfx938 两侧均为 `308bf4b588f74d5fb7da710d6c8afc55fa2fef5de47c2b6079f314fbf817e8ef`。已直接排除 kernel 性能变化，不执行运行时性能 A/B。
- PMD 仅报告 pending-cacheline warning，不影响两档校验；全程 `time=0`，没有性能测试。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 由 gfx936/gfx938 device ELF identity 闭合。本结论不覆盖同目录已在 gfx938 显式 skip(77) 的 `conv_fwd_dsreadm`。

### Step 3.3：`ck_tile/14_conv` FP16 2D forward DSReadM — SKIP-ARCH-CONTRACT

- target：`tile_example_conv_fwd_dsreadm` 使用 `ConvIgemmWrwPipelineDefaultPolicy` 与 `warp_dsreadm_attribute_impl.hpp` 中的旧式 `__builtin_hcu_ds_read_m32x16_f16(_alt)`，不是 P3.1/P3.2 已通过的普通 LDS/LDS-direct 数据通路。
- 当前 example 已在 main 中对 gfx938 明确返回 77，并说明现有 DSReadM NT block-GEMM 的 LDS/distribution contract 会产生 invalid values；CMake 同时把 77 注册为该 test 的 skip code。因此该 target 在适配开始前就不具备用户要求的 gfx936+gfx938 双架构回归闭包。
- P2.1 已对同一旧式 `ds_read_m32x16_b16(_alt)` 能力族完成最小放行实验：gfx946 虽能生成/执行指令，但数值错误，且 Shaobo 需要 64-bank LDS 下完整的 matrix-format descriptor/layout 重构，不能靠局部 guard 解决。
- 处置：不再次放行旧 DSReadM、不修改 runner、不在 PMD 上启动其强制计时路径；本模块标记 `SKIP-ARCH-CONTRACT`，转入不依赖该旧指令族的独立模块。该跳过不影响 P3.1/P3.2 的 PASS。

门禁结果：G0 source audit PASS；G2-G6 NOT RUN / SKIP。最终无代码差异，不涉及 gfx936/gfx938 性能测试，gfx946 PMD 也未测试性能。

### Step 3.4：`ck_tile/16_conv3d_fwd` BF16 1x1x1 forward — PASS-PMD

- target：`tile_example_conv3d_fwd`；NDHWGC input、GKZYXC weight、NDHWGK output，BF16 输入/输出、FP32 累加，固定 `128x128x32` block tile、`NumPrefetch=2`，使用 Wasp MMAC pipeline。
- runner 原先强制启用计时。本步增加默认值为 1 的 `-time` 参数；`-time=0` 只做一次 correctness launch 并输出 `timing:off`，默认真机行为保持不变。
- 首次 gfx946 构建虽然带 `--offload-arch=gfx946`，但公共 `ck_add_arch_vgpr_compile_options()` 漏掉 gfx946，编译报告只有 128 VGPR、64 VGPR spill、196 bytes/lane scratch；PMD 实际 dispatch 后 16384 个输出全部为 NaN。相同 PMD 上独立 BF16 basic GEMM `M128,N128,K64,repeat=0` 校验正确，排除了笼统的 BF16/MMAC 不支持。
- HCU 硬件资料与当前 PMD 配置分别给出 gfx946 最大 512 VGPR、`vreg-file-size=512`。最小修复把 gfx946 纳入现有 `-mllvm -support-512-vgprs=true` 分支；重编后资源变为 173 VGPR、零 spill、零 scratch，PMD 两档 correctness 均恢复正确。该变化是 gfx946 构建资源提示补全，不修改 gfx936/gfx938 的 768-VGPR 分支。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.4-E0 | 三架构 focused build；核对 gfx946 实际命令和资源 | PASS；实际 offload arch 为 gfx946/gfx936/gfx938；gfx946 命令包含 `-support-512-vgprs=true`，资源为 VGPR 173、spill 0、scratch 0 | G1/G2 build observation |
| P3.4-E1 | PMD 运行 `G1,N2,K128,C64,DHW=4x4x4`、1x1x1、stride1/pad0、`time=0` | PASS；M=128、K-loop=2，单次 dispatch，完整 CPU reference 校验通过 | G3 base pipeline |
| P3.4-E2 | PMD 运行 `G1,N4,K128,C128,DHW=4x4x4`、其余相同 | PASS；M=256、K-loop=4，完整 CPU reference 校验通过 | G3 multi-tile/multi-loop |
| P3.4-E3 | gfx936/gfx938 分别运行相同两档 shape | PASS；四次均 `timing:off` 且完整 reference 校验通过 | G4 regression observation |
| P3.4-E4 | 提取 gfx946 device image 并检查 metadata/ISA | PASS；device ELF SHA256=`07b31fe246e6dc21d06cb0bb7bcfff0155e00b1c4c2e753cae7362c0d939f4df`，metadata 为 VGPR 173、spill/scratch 0，ISA 统计包含 256 条 `v_mmac`、64 条 `ds_read`、24 条 `ds_write`、24 条 `buffer_load`、128 条 `buffer_store` | G2 ISA/resource observation |
| P3.4-E5 | 比较 gfx936/gfx938 baseline/candidate device ELF | PASS；gfx936 两侧 SHA256 均为 `44195fb6222a7488f866513596b7a707669af4710ea6bd817d3a63d595041ec4`；gfx938 两侧均为 `80b71a42688b96161225fbf8f5218550d514e1e847d10fcebbc55fde3c65f6d2`；`cmp` 均返回 0 | G6 direct code-object evidence |

shape 边界：`C=32` 只产生一个 K-loop，不满足当前 `NumPrefetch=2` pipeline 的 `IsSupported` 条件；这属于 example 的既有参数契约，不是 gfx946 失败。正式验收使用 C=64/128 覆盖两个和四个 K-loop。

G6 为 `NOT RUN`：两个真机架构的 device ELF 与 baseline 逐字节相同，已直接排除 kernel 性能变化，因此不执行运行时性能 A/B。gfx946 PMD 全程 `time=0`，未测试性能。门禁结果：G0-G4 PASS，G5 NOT RUN，G6 NOT RUN。

### Step 3.5a：`ck_tile/19_grouped_gemm` FP16 fast runner — PASS-PMD

- target：`tile_example_grouped_gemm_fp16_fast`；本切片只验收 FP16 basic/bias/multiple-D runner，不外推至 BF16、量化、selector、device-args 或 C API。
- runner 原先把 `warmup/repeat` 无条件传入 `stream_config` 并打印吞吐；`repeat=0` 在 multi-stream 实验分支还会形成除零。本步把 `repeat=0` 统一定义为“warmup=0、一次 correctness launch、关闭计时”，覆盖 ordinary、persistent、bias、multiple-D 路径；默认 `repeat=100` 的真机 benchmark 行为不变。JSON 关闭计时时保留统计字段并写零值。
- 首轮 focused build 暴露 basic 路径后续 JSON 汇总仍需要 `tflops/gb_per_sec` 的 host 作用域错误；在启动运行验收前已修正并重新完成三架构构建。这是公共 C++ 编译错误，不是 gfx946 指令或 PMD 问题。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.5a-E0 | 三架构 focused build；实际 target=`tile_example_grouped_gemm_fp16_fast` | PASS；实际 offload arch 分别为 gfx946/gfx936/gfx938 | G1/G2 build observation |
| P3.5a-E1 | PMD basic NN fixed-K：group=2，`M=128,256`、`N=128,128`、`K=128,128`，`repeat=0` | PASS；persistent kernel 单次 dispatch，`Timing: off`，完整 CPU reference 校验正确；日志 SHA256=`c09ba5eef23e654478932dc9b63dbdbf08e37ad0ab73836ce5f7f99905096c69` | G3 base path |
| P3.5a-E2 | PMD basic NT variable-K：group=3，`M=128,256,192`、`N=128,128,256`、`K=128,192,256`，`repeat=0` | PASS；persistent kernel 单次 dispatch，完整 CPU reference 校验正确；日志 SHA256=`240b4d38feb123b8192058dfca7568b4bb62740d97dca13c21c3924da44edc26` | G3 layout/variable-K path |
| P3.5a-E3 | PMD RC bias Add 与 multiple-D Add：group=2，`M=128,256`、`N=K=128`，`repeat=0` | PASS；两条 non-persistent epilogue 路径均单次 dispatch、完整 CPU reference 正确；日志 SHA256 分别为 `88e763541211331affbb48c918bc617b1d3478da34265240edcc30316d327c67`、`12e38e6c767249095f9a3b4cf0fc52cdc867d9781abff80df73fe0d97751b93a` | G3 direct stream-config paths |
| P3.5a-E4 | gfx936/gfx938 运行与 E1-E3 相同四条 correctness 路径 | PASS；八次均输出 `Timing: off` 与 `CPU verification result is:correct` | G4 regression observation |
| P3.5a-E5 | 提取 gfx946 device image 并检查 metadata/ISA | PASS；宿主 binary SHA256=`3d777cf638f9df94091e1d733fcf7e53a8e52f21a85876d42783ef30739ff876`，device ELF SHA256=`36be1c97f020d5e8ea15a793810dc02cb2fd9ff4e992e669c642b4bc45319c68`；bundle 含 gfx946 target，ISA 统计包含 `v_mmac=140160`、`ds_read=50974`、`ds_write=22924`、`buffer_load=38852`、`buffer_store=12051` | G2 ISA observation；该 fast binary 含 337 个配置，不能把全集资源统计等同于本次选中 kernel |
| P3.5a-E6 | 比较 gfx936/gfx938 baseline/candidate device ELF | PASS；两侧各只提取出一个目标 ELF。gfx936 两侧 SHA256 均为 `c445b7dbd70855e427e182ec70074cf1ea6eb91fec28c7692e43404c90edafb2`；gfx938 两侧均为 `7cde8730d4874ada88985ecd9178b30f06896ac2a41ad4def64b66cfa71b17b0`；`cmp` 均返回 0 | G6 direct code-object evidence |

PMD 告警边界：NN fixed-K 日志稳定出现 48 条 `v_mad_u64_u32 ... read vgpr ... before writing` 和 4 条 `src1_neg/src1_abs should not set`，但同一次运行完成 dispatch 且全部输出与 CPU reference 一致；NT variable-K 日志对应计数均为 0。因此当前把它记录为 PMD 对特定地址计算路径的调试告警，不标记为指令阻塞，也不跳过模块。

G6 为 `NOT RUN`：gfx936/gfx938 候选设备 ELF 与各自修改前 baseline 逐字节相同，已直接排除 kernel 性能变化；不执行运行时性能 A/B。gfx946 PMD 四条路径均 `repeat=0`，未测试性能。门禁结果：G0-G4 PASS，G5 NOT RUN，G6 NOT RUN。下一切片为 P3.5b BF16 grouped GEMM，仍与量化/selector/C API 分开验收。

### Step 3.5b：`ck_tile/19_grouped_gemm` BF16 fast runner — SKIP-PMD

- target：`tile_example_grouped_gemm_bf16_fast`；沿用 P3.5a 已提交的 `repeat=0` correctness-only runner 语义。本切片没有放宽 gfx946 架构声明，也没有保留未通过校验的 device 代码。
- gfx946 能构建完整 BF16 fast target，实际选择 `gemm_grouped_bf16_64x128x64_8x8x4_0x0x0` V4 kernel。对齐 NN、NT variable-K、bias Add 与 multiple-D Add 均完成真实 PMD dispatch，但 CPU reference 稳定失败。
- gfx936/gfx938 使用同一普通 V4 kernel 和相同矩阵全部正确；因此不是 example 输入、host reference 或该配置跨架构共同失效。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.5b-E0 | 三架构 focused build `tile_example_grouped_gemm_bf16_fast` | PASS；实际 offload arch 为 gfx946/gfx936/gfx938；gfx946 宿主 binary SHA256=`cd2d2ff1d6e1889b76d67b2ae6416fa425df2cb410c8249ad93a7dfc20bc8513`，device ELF SHA256=`a6cf5a7d316a77ead84103c4cf0e8a92d9b65442af6e6237964f481781ab218d` | G2 build/code-object observation |
| P3.5b-E1 | PMD V4 NN single group，`M=N=K=128`、`repeat=0` | FAIL；首错从 flat index 4097（第 32 行附近）出现，1520/16384 错误、错误率 9.277344%、max err 7056；日志 SHA256=`daba349aa76a5ba0079ce022fcc8a15aee5a560198c80fda461e68a632fb7340` | G3 BF16 grouped functional failure |
| P3.5b-E2 | PMD V4 NT single group及 NT variable-K 三组 | FAIL；单组错误率 9.118652%，三组分别约 9.12%/9.04%/8.95%；NT 路径 PMD 未报告 uninitialized-VGPR/source-modifier warning；日志 SHA256=`41798e019b0b4cd2e54d90c602ed4ed7a34636e43d73d15cf91a5cda08aa6da9`、`c86b36121b8668f187f1361b735f649d88f05912f462de29db67ecfd4bf5a595` | 排除 B layout 与 NN 地址告警单因 |
| P3.5b-E3 | PMD non-persistent bias Add 与 multiple-D Add，group=2、`M=128,256,N=K=128` | FAIL；两条路径均无 PMD 指令告警，日志 SHA256=`fe1e5e88799771a395b37d443e2b7fd8c4669bb434be709aa270f4fb804df7d6`、`cf5e4dc26e7fd4cd620fa469f075728c93729fc70001204991d0872dd71be636` | 排除 persistent scheduler 单因 |
| P3.5b-E4 | 检查选中 64x128x64 kernel 的 metadata/resources | 选中 kernel private-segment=0、SGPR/VGPR spill=0；fast binary 中其他未选配置存在 spill，未混作本 shape 根因 | 排除选中 kernel spill/scratch 单因 |
| P3.5b-E5 | PMD 健康控制：`tile_example_gemm_basic -prec=bf16 -m=128 -n=128 -k=128 -repeat=0` | PASS；实际 gfx946 BF16 kernel dispatch 后 CPU verification correct；日志 SHA256=`6577f2ef344c0099c0d7937eb3e3565cfc2b4912bf6af4df4636081948cfa829` | BF16/MMAC 基础能力健康，不作笼统不支持判断 |
| P3.5b-E6 | gfx946-only Compute-V3 单-LDS 探针，替代 V4 双缓冲；NT variable-K 三组 | FAIL；错误率约 13.65%/13.50%/11.92%，无 source-modifier warning；探针 binary SHA256=`c17a6054d160fcd665999a069a36e64d51c013f12cf8da546556df87ab4b92af`，日志 SHA256=`001efa3709adfbeeae7e2c39e583b7f8b3e69ba07fb33124559125d70cc197ed` | 排除 V4 双缓冲/pipeline 单因；探针已撤回 |
| P3.5b-E7 | gfx936/gfx938 运行 NN fixed-K、NT variable-K 同矩阵 | PASS；两台真机均选择同名 V4 kernel 并完整 CPU verification correct；宿主 binary SHA256 分别为 `5e768f18c3282844acd4f8021f3c043d0efdc6081875088c456bb943ca55f406`、`d9ba1499bb79ed42caacbc9094ef846d81951423d9cff03bf26f72ae0a90a1d6` | G4 healthy controls |

根因边界：当前证据把失败收敛到 gfx946 PMD 中 BF16 grouped kernel 共用的 accumulator-to-output/CShuffle 执行链，但尚不足以指认某一条 opcode；FP16 grouped CShuffle 与 BF16 basic GEMM 均已在同一 PMD 正确，不能扩大为“所有 CShuffle”或“所有 BF16/MMAC”不支持。V3/V4、persistent/non-persistent、NN/NT 和无/有 D epilogue 都失败，继续局部调参已不具备可验证的最小修复闭包。

处置：按硬边界标记 `SKIP-PMD`，撤回并同步还原 V3 探针，不提交任何 BF16 device 配置变化；保留 P3.5a 的通用 correctness-only runner 改动。G2 PASS，G3 FAIL/SKIP-PMD，G4 PASS，G5 NOT RUN，G6 NOT RUN。最终无 P3.5b 业务代码差异，gfx936/gfx938 不存在 candidate 性能变化，不触发性能 A/B；gfx946 全程未测试性能。下一切片转入 P3.5c 量化 grouped-GEMM，不让 BF16 PMD 缺口阻塞其他独立模块。

### Step 3.5c1：`ck_tile/19_grouped_gemm` tensorwise quant — PASS-PMD / BF16-output SKIP-PMD

- target：`tile_example_grouped_gemm_quant_tensor`；输入矩阵覆盖 FP8xFP8、BF8xBF8、FP8xBF8、BF8xFP8，输出分别为 FP16/BF16，布局覆盖 NT/NN/TN，group=3，并包含 fixed-K、variable-K 与 M/N/K padding。
- 原 runner 硬编码 `time_kernel=true,warmup=5,repeat=20`，不能在 Perf Model 上运行。本步增加默认不变的 `-time=1,-warmup=5,-repeat=20` 参数；`-time=0` 强制零预热、单次 launch 并输出 `timing:off`。另增加默认开启的 `-bf16_output=1`，只在 PMD 验收时显式设为 0，不屏蔽 gfx938 或未来真机的完整默认矩阵。
- 全矩阵 PMD 探针证明 13 个 FP16-output case 全部正确，12 个 BF16-output case 全部失败；后者与 P3.5b 已收敛的 BF16 grouped accumulator-to-output/CShuffle PMD 缺口一致。本切片只声明已实测 FP16-output quant 路径，并显式跳过 BF16-output 子矩阵。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.5c1-E0 | gfx946/gfx938 focused build tensorwise quant target；gfx936 配置并精确请求同 target | gfx946/gfx938 PASS，实际 offload arch 分别为 gfx946/gfx938；gfx936 configure PASS，但 target 返回 `No rule to make target`，与 CMake 的 `gfx938|gfx946` 专属门控一致 | G1/G2 architecture contract |
| P3.5c1-E1 | PMD 全矩阵 `time=0,bf16_output=1` 探针 | 13 个 FP16-output 全 PASS；12 个 BF16-output 全 FAIL；日志 SHA256=`4bcf8750f6f72bddc08d1f5a53df3c22d415ee1108c3a1cfbb2171d87eecaca4` | 能力切片，不用 FP16 代证 BF16 |
| P3.5c1-E2 | PMD `time=0,warmup=0,repeat=1,bf16_output=0` 正式验收 | PASS；13/13 correctness case 覆盖四种 FP8/BF8 输入组合、NT/NN/TN、variable-K 和 padding；三条布局各输出明确 `BF16 output ... SKIP-PMD`；binary SHA256=`1c8ed56b8d8be2a3957a9b85c14c55cf20b053cf0e9cce92e4c3648648bf2746`，日志 SHA256=`c6559081a7d43bcf269ac85254c304d67625c9e2fdefa029934e665e00cd2d7b` | G3 supported slice PASS |
| P3.5c1-E3 | gfx938 candidate `time=0,warmup=0,repeat=1`、默认 `bf16_output=1` | PASS；25/25 完整矩阵均 `timing:off`，日志 SHA256=`1aff61201511d14ad116ca79f3f1d0dd937a04e864f3df4ee211159eb5e17cc2` | G4 gfx938 full regression |
| P3.5c1-E4 | gfx936 同目录回归：重构建并运行 FP16 grouped fast NN/NT | PASS；quant target 在该架构不入图，目录级健康控制两条均 CPU verification correct | G4 gfx936 architecture-bound regression |
| P3.5c1-E5 | gfx938 baseline/candidate code-object 检查 | 宿主 binary SHA256=`dcbd58e157ff3d83fe71ca302aac58ef01ca01006de9e7b77098459bb2b2d5e3`/`bc16330d89df659c511ec06a81da24c80397b5623aa91a2d396058e829c9083b`；device ELF SHA256=`ccd4d2972b62bdfb8f99b586a76a35b40c362d5986be6518ba29f55094aab516`/`b0f36d428b7b7e93ba4dca475de59bcb824b47a6270264ae87ae87c26536b8dd`，不相同，故触发真实性能 A/B | G6 trigger evidence |
| P3.5c1-E6 | gfx938 baseline/candidate 交替 5 轮；每个 binary 每轮均为 25 case、每 case warmup=5/repeat=20 | PASS；baseline/candidate 的每轮 25-case 时间和均值为 0.646100/0.645795 ms，candidate 相对 -0.0472%，十次运行均 25/25 correctness PASS | G6 performance A/B PASS |

PMD 正式验收日志仍有 136 条地址计算类 uninitialized-VGPR warning 与 4 条 source-modifier warning，但 13 个支持 case 全部完成真实 dispatch 和完整 CPU reference 校验；按既有规则记录告警，不扩大为指令阻塞。gfx946 全程只运行 `time=0`，没有性能测试。

门禁结果：G0-G2 PASS；G3 为 FP16-output PASS、BF16-output SKIP-PMD；G4 gfx938 完整 PASS、gfx936 架构边界及目录级回归 PASS；G5 NOT RUN；G6 gfx938 A/B PASS、gfx936 无该 target 性能面。下一切片为 P3.5c2 row/column-wise quant。

### Step 3.5c2：`ck_tile/19_grouped_gemm` row/column-wise quant — PASS-PMD / BF16-output SKIP-PMD

- target：`tile_example_grouped_gemm_quant_rowcol`；A scale 按 M 行、B scale 按 N 列应用，输入 dtype/layout/shape 矩阵与 P3.5c1 对齐，但 scale load 和乘法数据通路独立验收。
- runner 增加默认不变的 `-time=1,-warmup=5,-repeat=20` 与 `-bf16_output=1`；`time=0` 强制单次 correctness launch，PMD 使用 `bf16_output=0`，gfx938 保持默认全矩阵。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.5c2-E0 | gfx946/gfx938 focused build row/column-wise quant target | PASS；实际 offload arch 分别为 gfx946/gfx938；gfx936 仍受同一 `gfx938|gfx946` target 门控 | G1/G2 architecture contract |
| P3.5c2-E1 | PMD 全矩阵 `time=0,bf16_output=1` 探针 | 13 个 FP16-output 全 PASS；12 个 BF16-output 全 FAIL；binary SHA256=`b2fd817bf74e02dc62d6fdba1acad921ab18d28a2871f1f2178541c926475c5f`，日志 SHA256=`614d9a91c7e6859d6f582491182a0e1d8a5074d7277b9036bba50301fb10a65b` | 复现已知 BF16 grouped CShuffle PMD 边界 |
| P3.5c2-E2 | PMD `time=0,warmup=0,repeat=1,bf16_output=0` 正式验收 | PASS；13/13 FP16-output case 覆盖四种 FP8/BF8 输入组合、三种 layout、variable-K 和 padding；三条 BF16-output 布局明确 SKIP-PMD；binary SHA256=`6b33b6815c97447f36f241ecf6609e221aee2dfb99b0295ee7d98ceef5555a84`，日志 SHA256=`70f21403aa50ed41afdd7bc3aa30f181f03240f88c8b5519dad302221472a90e` | G3 supported slice PASS |
| P3.5c2-E3 | gfx938 candidate `time=0,warmup=0,repeat=1`、默认 BF16-output | PASS；25/25 完整 row/column-wise quant 矩阵均正确；日志 SHA256=`b44d0f045d0204b645f6cdf43c4b8ddc91fe67322d78802b19a89137f38d52e3` | G4 gfx938 full regression |
| P3.5c2-E4 | gfx936 同步最终源后重构建并运行 FP16 grouped fast NN/NT | PASS；quant target 不入图，目录级健康控制两条均 CPU verification correct | G4 gfx936 architecture-bound regression |
| P3.5c2-E5 | gfx938 baseline/candidate device ELF | 宿主 binary SHA256=`b4770fa55059d98b50545c6b450717c70b93d20bd45e196baf3b053ae48888bf`/`980442d589e1888e602ba38928efc1bbbc96a59ef17f48aadb8d6537a078713c`；提取出的唯一 gfx938 device ELF 两侧 SHA256 均为 `a75087973e17c00424faafb514c27e7ed08b1326a700606eba5c6d0245f37863`，逐字节相同 | G6 direct code-object evidence |

PMD 正式验收的 136 条地址计算类 uninitialized-VGPR warning 与 4 条 source-modifier warning 和 P3.5c1 计数相同，未阻止 13 个支持 case 完整校验。G6 为 NOT RUN：gfx938 device ELF identity 已直接排除 kernel 性能变化；gfx936 无该 target 性能面；gfx946 PMD 未测试性能。

门禁结果：G0-G2 PASS；G3 为 FP16-output PASS、BF16-output SKIP-PMD；G4 gfx938 完整 PASS、gfx936 架构边界及目录级回归 PASS；G5 NOT RUN；G6 NOT RUN。下一切片为 P3.5c3 blockwise quant。

### Step 3.5c3：`ck_tile/19_grouped_gemm` blockwise quant — PASS-PMD / BF16-output SKIP-PMD

- target：`tile_example_grouped_gemm_quant_blockwise`；scale 按 block 应用，输入 dtype/layout/shape 矩阵与前两个 quant target 对齐，但 block scale descriptor、dense-BQ variable-K 和对应 load/乘法路径独立验收。
- runner 增加默认不变的 `-time=1,-warmup=5,-repeat=20` 与 `-bf16_output=1`；`time=0` 强制零预热、单次 correctness launch，PMD 使用 `bf16_output=0`，gfx938 保持默认完整矩阵。默认 case 顺序保持原始交错顺序，不改变真机 kernel 实例化布局。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.5c3-E0 | gfx946/gfx938 focused build blockwise quant target | PASS；gfx936 仍受同一 `gfx938|gfx946` target 门控 | G1/G2 architecture contract |
| P3.5c3-E1 | PMD 全矩阵 `time=0,bf16_output=1` 探针 | 13 个 FP16-output 全 PASS；12 个 BF16-output 全 FAIL；binary SHA256=`0f5f886bb7e591e8d0bc0ad76b30665608d1fb656a4741935d50b4da29526fa1`，日志 SHA256=`361be2ddbadaaaa9e543681935d56d731f244d4425bbef1f4798cc63bbec08a1` | 复现已知 BF16 grouped CShuffle PMD 边界 |
| P3.5c3-E2 | PMD `time=0,warmup=0,repeat=1,bf16_output=0` 正式验收 | PASS；13/13 FP16-output case 覆盖四种 FP8/BF8 输入组合、三种 layout、variable-K、dense-BQ 和 padding；三条 BF16-output 布局明确 SKIP-PMD；binary SHA256=`476e7644a7bc4ddfc7851f5dfece4fb9abcf5cd48e0b60716faaa6827bdd7597`，日志 SHA256=`eac358b38f5cb9e3c2ed3e640d57d37a9a36770dfaaa3f4fd406c6e6afcb1af8` | G3 supported slice PASS |
| P3.5c3-E3 | gfx938 candidate `time=0,warmup=0,repeat=1`、默认 BF16-output | PASS；25/25 完整 blockwise quant 矩阵均正确；日志 SHA256=`ce4b07a32556ea7a5427c2f68859962d3f91d6836d5cda1538fc7347b8224ae0` | G4 gfx938 full regression |
| P3.5c3-E4 | gfx936 同步最终源后重构建并运行 FP16 grouped fast NN/NT | PASS；两条均 CPU verification correct；candidate binary SHA256=`9080853fa3d42fac4f83df1b66e476655cbb9eee225a7c4cb88bff3ca4d1613c` | G4 gfx936 architecture-bound regression |
| P3.5c3-E5 | gfx938 baseline/candidate device ELF | 宿主 binary SHA256=`95ff439a421886593ab32f63f0850687919f792d6cf9019638f86401df13f101`/`81efaf5adaf48f2ff104275073f52553430a86b0fbd550d225de9d74eab48387`；提取出的唯一 gfx938 device ELF 两侧 SHA256 均为 `9782d918867ce9ab5a6081f2787268d90963bafd535005f059d57166443d8788`，逐字节相同 | G6 direct code-object evidence |

PMD 正式验收记录 64 条地址计算类 uninitialized-VGPR warning 与 4 条 source-modifier warning，未阻止 13 个支持 case 完整校验。G6 为 NOT RUN：gfx938 device ELF identity 已直接排除 kernel 性能变化；gfx936 无该 target 性能面；gfx946 PMD 未测试性能。

门禁结果：G0-G2 PASS；G3 为 FP16-output PASS、BF16-output SKIP-PMD；G4 gfx938 完整 PASS、gfx936 架构边界及目录级回归 PASS；G5 NOT RUN；G6 NOT RUN。P3.5c 三种 quant scale 路径均完成，下一切片为 P3.5d selector/device-args/C API。

### Step 3.5d1：quant selector/device-args C ABI — PASS-PMD / BF16-output SKIP-PMD

- target：`tile_example_grouped_gemm_quant_selector_test`、`tile_example_grouped_gemm_quant_device_args_c_header_smoke`、`tile_example_grouped_gemm_quant_device_args_smoke`。原 kernel/pipeline 已把 gfx938/gfx946 定义为共同 quant target，P3.5c 也已证明 gfx946 FP16-output 路径可运行；实际缺口是 device-args CMake 只入图 gfx938、selector/runtime arch 只识别 gfx938、公开 capability mask 只声明 gfx938。
- 最小闭包把 gfx946 加入 CMake、selector arch/stable name、运行时 `get_hcu_target_enum()` 映射和 ABI v1 architecture mask；既有 InstanceId 5000/5001、结构体布局、ABI version 均不变。smoke 先通过 `query_instance_v1` 校验公开 mask=`gfx938|gfx946`，再运行 TN variable-K tensorwise/rowwise 两条 FP16-output device-args dispatch。
- BF16-output 的 selector contract 仍可静态查询，但按照 P3.5b/P3.5c 已收敛的 PMD 缺口，不在 gfx946 PMD 上声明 device-launch 通过，也不据此关闭未来 gfx946 真机能力。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.5d1-E0 | 三架构重配置；构建 quant selector、strict C11 header；gfx946/gfx938 构建 device-args provider/smoke | PASS；gfx946 compile 使用 `CK_TILE_HCU_QUANT_GEMM_GFX946=1` 与 `-support-512-vgprs=true`，gfx938 保持原 gfx938 定义；gfx936 device target 仍按 CMake 契约不入图 | G1/G2 architecture contract |
| P3.5d1-E1 | host-only selector matrix及新增 gfx946 stable-ID control | PASS；15 个既有 gfx938 case 全通过，gfx946 control 返回 InstanceId 5000、stable name 前缀 `ck.qgg.gfx946.`，failures=0 | G2 selector contract |
| P3.5d1-E2 | PMD strict C11 header、公开 registry mask、两条 FP16-output device-args correctness | PASS；mask=3 expected=3；tensorwise/rowwise 均返回 InstanceId 5000、stable name 为 gfx946 且 CPU reference 正确；binary SHA256=`b051bb6fc57ccda69ae4a80982853b57bd2575cc5a5efc5eaf667a1e244b2c1e`，日志 SHA256=`aec1b9e15ff4b5d90f1680fe9d3d36f43d48342b9634e347cda3d0b3ac291728` | G3 supported slice PASS |
| P3.5d1-E3 | gfx938 baseline/candidate selector、C11 header、registry query 与两条 device launch | PASS；mask=3 expected=3，selector failures=0，两条 correctness 均通过；baseline/candidate host binary SHA256=`18bd49e796f6908fdc9ebcb888af408d8fca4f713d6d8953e051ba9c9e831469`/`0ab4297938011134e8ee3563b3d378da3238652c7bff7681d9348b50ec874369`，candidate 日志 SHA256=`5c685561898f54f7987f4f105d54a7446150fcfdd8bf9fd236a45f2635728989` | G4 gfx938 regression |
| P3.5d1-E4 | gfx936 selector与strict C11 header；精确请求 device target | PASS；selector failures=0、C11 编译通过；device target 返回 `No rule to make target`，符合 quant provider 仅 gfx938/gfx946 入图的架构契约 | G4 gfx936 architecture-bound regression |
| P3.5d1-E5 | gfx938 baseline/candidate device ELF | 两侧均只提取出一个 gfx938 ELF，SHA256 均为 `9242b388e98c51ded4194edd4db6e6463f6eea5b8da2a56833dc31b992381e06`，逐字节相同 | G6 direct code-object evidence |

PMD 两次 dispatch 各记录 8 条 `v_mad_u64_u32 ... read vgpr ... before writing`，另有一次 pending-cacheline warning；两条均完成真实 dispatch 与完整 CPU reference 校验，因此只记录为模型告警。gfx946 未执行性能测试。G6 为 NOT RUN：gfx938 device ELF identity 已直接排除 kernel 性能变化；gfx936 无该 target 性能面。

门禁结果：G0-G4 PASS（BF16-output device launch 延续 SKIP-PMD）；G5 NOT RUN；G6 NOT RUN。下一切片为 P3.5d2 非量化 selector/device-args/C API，需独立证明 gfx946 应选择何种 provider，不能机械复用 gfx936/gfx938 专用 InstanceId。

### Step 3.5d2：非量化 selector/device-args C ABI — PASS

- target：`tile_example_grouped_gemm_selector_test`、`tile_example_grouped_gemm_device_args_c_header_smoke`、`tile_example_grouped_gemm_device_args_query_test`、`tile_example_grouped_gemm_device_args_smoke`。gfx946 不复用 gfx936 V3/BW 或 gfx938 MLS InstanceId，只启用既有通用 V4 provider：小规模 N 对齐选择 4002，padding 选择 4003；4000/4001 仍保留为大 shape 通用 fallback。
- ABI v1 只在 architecture mask 追加 gfx946 bit，结构体布局、ABI version 和稳定 InstanceId 均不变。通用 ID 4000-4003 的 registry mask 扩展为 `gfx936|gfx938|gfx946`；gfx936 V3/BW、gfx938 MLS 以及只在 gfx936/gfx938 入图的 BF16-TN ID 1003/1004 保持原架构 mask。
- `--pmd-correctness` 是显式正确性模式，只在 Perf Model 跑 64/65 量级的 FP16 NT/NN/TN 三条实际 dispatch，默认 gfx936/gfx938 完整矩阵不变。P3.5b 已记录的 BF16-output PMD 缺口不在本切片重复探测，也不据此关闭未来 gfx946 真机 BF16 contract。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.5d2-E0 | 三架构 focused build、host selector、strict C11 header | PASS；三台均使用各自精确 offload arch；selector 80 cases、registry 18 entries，C11 `-Wall -Wextra -Werror` 通过 | G1/G2 architecture and ABI contract |
| P3.5d2-E1 | gfx946 query/registry 与显式 PMD correctness | PASS；query 选择 `ck.gg.gfx946.fp16.NT.v4.64x128x64.nonpadding`/`NN...full_padding`；FP16 NT/NN/TN 3/3 与 negative cases 全通过，binary SHA256=`c6dc7735092ae3a69f5e977e57319f7cf74d7a6b30a894f6435a574c5ad6ba52`，日志 SHA256=`751a943c68d0aeec4df8b882c2692752537ec02255b61fad326553515b8ee04a` | G3 PASS |
| P3.5d2-E2 | gfx936 baseline/candidate 完整回归 | baseline/candidate 均 21/21 device cases、negative、selector/query PASS；binary SHA256=`3664d760789732ab95eb598d3d5a5c65b26ee2d908081aa9ee9f17d7bc27911b`/`23955c3e13e95ed6dbf40bca1dee31eb3d2a4091625cb99e8a68bc9f6cd14844`，candidate 日志 SHA256=`28200947585e437761609f46c46ef40107df35fef83ce64095d3ced5d98aa405` | G4 gfx936 PASS |
| P3.5d2-E3 | gfx938 baseline/candidate 完整回归 | baseline/candidate 均 13/13 device cases、negative、selector/query PASS；binary SHA256=`992445dde423341d6c7c4843da030f21efe0d890a2bef41691bb6e8e28fb688c`/`302e7b0ca9a5357f72055dd3596b12b770fb8b9b7671e1099722ba9c030c5ce9`，candidate 日志 SHA256=`e1bdba4ee0a470c0455ac7c8bb717301504891276e06ef45796f5549b805a9c0` | G4 gfx938 PASS |
| P3.5d2-E4 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧唯一 device ELF 均为 `25bf9658437fa7277b2326d331a2761a3cfa5aa54648d1b9cf0cc5f2abb6537a`；gfx938 两侧均为 `6309f86d32325cf8fba44621d94b789f833e2ca42b57bcff6d5bc570d1d0a1d2`；内容逐字节相同 | G6 direct code-object evidence |

PMD 的 padding/variable-K dispatch 记录了 `v_mad_u64_u32 ... read vgpr ... before writing` 与 source modifier 告警，另有一次 pending-cacheline warning；三种 layout 的实际输出和 negative contract 均正确，因此按既有规则记录为模型告警。gfx946 未执行性能测试。gfx936/gfx938 的设备 ELF 均未变化，已直接排除 kernel 性能变化，G6 运行时性能 A/B 为 NOT RUN。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。下一切片为 P3.5d3 旧 host-descriptor `tile_example_grouped_gemm_c_api_smoke`，在 Perf Model 只验收 FP16 C ABI，BF16 延续 P3.5b 的 SKIP-PMD。

### Step 3.5d3：非量化旧 host-descriptor C API — PASS

- target：`tile_example_grouped_gemm_c_api_smoke`。新增显式 `--fp16-only` correctness 模式，只跳过 BF16 case；FP16 仍覆盖 NT/NN/TN 的 aligned-packed 与 ragged-strided 共 6 条真实 C ABI dispatch。未知参数返回 2，避免静默改变验收范围。
- 默认无参数行为保持完整 60-case 矩阵：BF16 三种 layout、6 种 group count、3 种 storage mode 共 54 条，外加 FP16 6 条；gfx936/gfx938 回归仍走默认模式。gfx946 PMD 按 P3.5b 已收敛的边界不重复运行 BF16-output，只验证 FP16 C ABI。
- 本步只修改 host smoke 的参数解析与 case 选择，不修改 provider、selector、InstanceId、ABI 结构体或 device kernel。真卡 baseline/candidate device ELF 内容相同，直接排除 kernel 性能变化。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.5d3-E0 | gfx946 focused build 与 `--fp16-only` PMD correctness | PASS；NT/NN/TN 的 aligned/ragged 共 6/6，BF16 cases=0；binary SHA256=`4f181097d37b1f586afca1e142af57dce86a849888180fec2b1fb99edab1d4ec`，日志 SHA256=`36c3f9b538a8da0c5eba3bf753090d17fcfd243feb7348f7943f42deac94da05` | G1-G3 PASS |
| P3.5d3-E1 | gfx936 修改前 baseline 与候选默认完整矩阵 | baseline/candidate 均 BF16 54/54、FP16 6/6；binary SHA256=`c7a005144fbe8fabe348fb9b6c4cddd984f725d13c0577e96a252caf05714294`/`bb149573195dd7e3a0fe914989e96029f8dab17da0057618447ec8f0c37edb6a`，日志 SHA256=`27b7a9f683256446fb756a86f93abe0405d7e0366490da84e7e52e85e3274c4f`/`091ccf70ef767b796891a602153ab1fe647c0a276f067712af0a29b43df81ffb` | G4 gfx936 PASS |
| P3.5d3-E2 | gfx938 修改前 baseline 与候选默认完整矩阵 | baseline/candidate 均 BF16 54/54、FP16 6/6；binary SHA256=`bf63669defce2cf5cb91de852d97100aed02ec8c0c97f30f60916346a9aabfdd`/`77ed3f8c527a4ed2322f3c6d2730318cb91ea2e4bec3e1fa8b8f1c1bf4c4536a`，日志 SHA256=`27b7a9f683256446fb756a86f93abe0405d7e0366490da84e7e52e85e3274c4f`/`091ccf70ef767b796891a602153ab1fe647c0a276f067712af0a29b43df81ffb` | G4 gfx938 PASS |
| P3.5d3-E3 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧唯一 device ELF 均为 `15b59f31b21d9546375cd807bea2ecd38bbf765a9e3e06a3ab6619d5ab4a5c91`；gfx938 两侧均为 `2f76cc50208d8b07b79911cd3c664ca40d12c447473e6eafbfee47979b6520d3`；内容逐字节相同 | G6 direct code-object evidence |

PMD 在 NN/TN 地址计算路径继续报告 `v_mad_u64_u32 ... read vgpr ... before writing`、source modifier 与 pending-cacheline warning，但 6 次 dispatch 均完成且输出与 CPU reference 一致，因此沿用既有分类为模型告警，不标记为指令阻塞。gfx946 未执行性能测试。gfx936/gfx938 的 device ELF identity 已直接排除 kernel 性能变化，运行时性能 A/B 为 NOT RUN。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。`ck_tile/19_grouped_gemm` 的 FP16/BF16、三种 quant、selector/device-args/C API 子矩阵已形成闭包，下一切片为 P3.6 `ck_tile/18_moe_quant`。

### Step 3.6：`ck_tile/18_moe_quant` — scalar-output PASS / vectorized-output SKIP-PMD

- target：`tile_example_moe_quant`。runner 原先强制 warmup/repeat 并打印带宽；新增默认值为 1 的 `-time` 参数，`-time=0` 只发射一次 correctness kernel 并显示 `timing:off`。默认真机行为不变，provider/dispatch/实例和 device pipeline 均未修改。
- PMD 支持边界由受控 A/B 收敛到 int8 输出 `Vector_N`：FP16/BF16 的 `Vector_N=1` 均正确；`Vector_N=2/4/8` 均出现规律性零值。`v_max3_f32`、row scale 与同一 warp-per-row 拓扑同时存在于通过和失败组，因此不能把失败泛化为 max3、跨-warp归约或 BF16 问题。
- gfx936/gfx938 对同一 FP16/BF16、h=64 与 h=1024 的 warp-per-row/cross-warp 实例均正确。PMD 缺口不能用真实硬件无回归的 kernel 重写来规避；按任务规则停止该 vectorized 子路径并标记 `SKIP-PMD`，继续下一独立模块。

实验记录：

| ID | 动作 | 结果 | 分类 |
| --- | --- | --- | --- |
| P3.6-E0 | gfx946 focused build；FP16/BF16 `h=64/255`、`Vector_N=1`、`-time=0` | PASS，4/4 scale 与 qy 均通过 CPU reference；binary SHA256=`5e87f1435318425a6a01c30560e2e4a19bd0edd8b1a6503f14ac7aa24d0736ba`，日志 SHA256=`826e51e557f5849cfca34752eec11ae244fb96b72335bde82039b88eab5d723b` | G1-G3 supported slice PASS |
| P3.6-E1 | PMD vector-width A/B：FP16 h=127/128/255/256/511/512，BF16 h=127/128/255/256 | `Vector_N=1` 全 PASS；`Vector_N=2` 为 63/128（49.2188%）错误，`Vector_N=4` 为 63/256（24.6094%）错误，`Vector_N=8` 为 127/512（24.8047%）错误，错误位置输出为 0；A/B 日志 SHA256=`61312388a54fc86774ef97620a72738f3c3402f3a532a43e41c210453aacc6f9` | G3 vectorized-output SKIP-PMD |
| P3.6-E2 | gfx936 baseline/candidate FP16/BF16、h=64/1024 | 两侧均 4/4 `valid:y`；binary SHA256=`a8aa7595b71b4d58f12da9aa1b5ebfdaed51e5875c8479d6cc78ee38fa6ee23a`/`b6658c58ea33157affadff37285ecf093b6783fee113a7f98b0808b46add5695`，日志 SHA256=`b2b350a56e9d0576ebdc6881633551c782e3fc43e836bab2328602c0014984ff`/`cabf64e6565e47626d17b775563cc3f7382d4e296cc16a0c7673e6c2a39abc82` | G4 gfx936 PASS |
| P3.6-E3 | gfx938 baseline/candidate FP16/BF16、h=64/1024 | 两侧均 4/4 `valid:y`；binary SHA256=`fecdb2341066cdc7d5ccf911e6016db6fddcb72ad00eace6da7430de0d76f456`/`20d41af288b41756832c57781485028a97b4da8e9e0f85d62a5ad257df7ea49e`，日志 SHA256=`2f48f9af97fd90641c249f74ae203353db46115e13e0b3553db6d3de93d34af1`/`30cb0944fd5ed530b4f8278a69d449ef9ec2dd675c65d275a0adedcdfc83d921` | G4 gfx938 PASS |
| P3.6-E4 | gfx936/gfx938 baseline/candidate device ELF | gfx936 两侧唯一 device ELF 均为 `a60c9d263f84f7a70f2b3dbc41fd0abee8b07827027534924e48c307ad87dbe3`；gfx938 两侧均为 `7298a0d287d29f12bbda35d521a3b836e20aa3ad790c6a0b6beabb7e3ceca4ca`；内容逐字节相同 | G6 direct code-object evidence |

PMD 的所有组都报告 `Inst_VOP3__V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，但这些告警在 `Vector_N=1` 正确组同样存在；只有失败的 `Vector_N>=2` 组额外稳定出现 `src1_sext/src1_neg/src1_abs should not set`。当前证据指向 PMD 对该 vectorized int8 qy store/packing 执行链的支持缺口，尚不足以点名单一 opcode；不把它写成 Shaobo 真机不支持。gfx946 全程未执行性能测试。

门禁结果：G0-G2 PASS；G3 scalar-output PASS、vectorized-output SKIP-PMD；G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。本模块不再改写 device kernel，下一切片为 P3.7 `ck_tile/17_fused_moe`。

### Step 3.7：`ck_tile/17_fused_moe` — PASS（2026-09-02）

- 静态闭包：当前 `tile_example_fused_moe` 的 `main.cpp` 实际只开放非量化 FP16 input/weight、FP32 output；默认 `block_m=16` 路径使用两段 128-wide MMAC tile，并由 API 0 覆盖 sorting+MoE GEMM、API 1 覆盖 MoE GEMM。执行链包含 DPP/DS bpermute、MMAC 与 FP32 atomic，未发现需要新增的 gfx946 架构 guard。
- 最小改动：runner 增加默认值为 1 的 `-time` 参数；API 0/1 在 `-time=0` 时各只发射一次 correctness launch、跳过 warmup/repeat 和吞吐计算并输出 `timing:off`。默认真机行为保持不变，device kernel、traits、dispatch 和 launch shape 均未修改。
- 既有参数边界：真实 gfx936 baseline 在 `h=128,i=128` 的 API 0/1 均无法通过 reference；扩大 intermediate 到 1024 仍失败。`h=1024,i=128` 的 API 0/1、GELU/SiLU 则稳定 4/4 正确。该失败在同步本轮改动前已复现，因此正式三架构验收采用最小健康控制 `h=1024,i=128`，不把 `h=128` 误归因于 gfx946 或本次改动。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P3.7-E0 | gfx946 focused build；`t=1,e=1,k=1,h=1024,i=128,bm=16,act=SiLU`，API 1/API 0，`-time=0` | PASS，2/2 `valid:y` 且均显示 `timing:off`；binary SHA256=`69a3b4ab4f35bea889e32b6ee59046e90ad65aa59cd3584433bc6d8541739266`，日志 SHA256=`ec43902451d12c35fd3a16d2cfec8edfad62f9babc3d86bfa8beaa5cd6d2272c` | G1-G3 PASS；PMD correctness only |
| P3.7-E1 | gfx936 baseline/candidate：`t=2,e=2,k=1,h=1024,i=128,bm=16`，GELU/SiLU × API 0/1 | 两侧均 4/4 `valid:y`；binary SHA256=`43bcad8ac7023a1367f9a8af558504dededf069451d830891941e8d72477c230`/`24858396e44e00564308a380ddaf2672a2bf7246f9c184b8cb84fef08adc29ae`，日志 SHA256=`29e5d25167d7b669960e5a31e376ac519c9220ced8f941cccdd8ae4b7a7072ac`/`acd5897b4c0c18f325dfc71d55c264066f36281eefa2e1b627be556c57389829` | G4 gfx936 PASS |
| P3.7-E2 | gfx938 同矩阵 baseline/candidate | 两侧均 4/4 `valid:y`；binary SHA256=`43be9ea9857d23b01d124617c283e570927cda01a14c546cafafa84c11a1b3a6`/`cca9a55f242ce671538ccc63134eeba99124f9ccbd055c733c8f52037c457218`，日志 SHA256=`73de5ebe497f01e3d74628326ebd77d29c6da179cdbbba2679480f5524268b61`/`acd5897b4c0c18f325dfc71d55c264066f36281eefa2e1b627be556c57389829` | G4 gfx938 PASS |
| P3.7-E3 | gfx936/gfx938 baseline/candidate device ELF | 两架构各提取 1 个 device ELF；gfx936 manifest SHA256 两侧均为 `f591862b373b3e428b6b67d044dd0f8d89cfbe663b241c51045b3e539dc7a605`，gfx938 两侧均为 `464d6ab335289ff8bfec59c194c0af04c4d6f49d8e4c00ab2d2de4f678958373`，内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD 的 API 0 sorting 阶段报告一条 pending-cacheline warning，但后续 MoE GEMM dispatch 完成且完整输出与 CPU reference 一致；当前只记录为 PMD warning，不标记为指令阻塞。gfx946 全程未采集或比较性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity 直接排除 gfx936/gfx938 性能变化）。P3 队列完成，下一切片为 P4.1 `ck_tile/01_fmha`。

### Step 4.1：`ck_tile/01_fmha` basic forward — PASS（2026-09-02）

- 范围冻结：只验收默认生成的普通 batch forward API，矩阵为 FP16/BF16 × head dim 64/128，`b=h=1,s=s_k=64`、无 bias/mask/dropout/LSE、row-major V。split-KV、append-KV、paged-KV、group mode、FP8 和 backward 均不混入本切片，后续按独立依赖闭包处理。
- 静态闭包：CMake 的 `gfx9` filter 接受 gfx946，codegen receipt 3 为 gfx946 生成四个 `qr_async` 实例；当前源码已在 async pipeline、reduce 与 custom policy 中把 gfx946 纳入 gfx938 lit/lts C-layout 和 MMAC 路径。本步不新增或扩大架构 guard。
- 最小改动：`example_fmha_fwd.cpp` 增加默认值为 1 的 `-time` 参数；`-time=0` 关闭 stream timing、warmup 和重复计时，只发射一次 correctness kernel。runner 在该模式显式输出 `timing:off`；默认 benchmark 行为保持不变。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.1-E0 | gfx946 focused build；FP16/BF16 × d64/d128，`-time=0` | PASS，4/4 `valid:y` 且显示 `timing:off`；binary SHA256=`dc9e8c76bd9260ebcd8024b2cb826dc07979291bf125c5bdfa1ccf550b060cc3`，日志 SHA256=`52b5bf423be94453281c25688d15ca3b52e4efd3b3075642301e08ec940503d0` | G1-G3 basic forward PASS |
| P4.1-E1 | gfx936 同矩阵 baseline/candidate | 两侧均 4/4 `valid:y`；binary SHA256=`5bc9b2d8ca301fdc02b1b21fed9fed4c12a98eba5d40cda26239c7351612b975`/`ab345c932cf3022f8eb36d30d9432d5d8d4e940211d7e682978ca0596c2d0f46`，日志 SHA256=`447a2cbf80c35482bcd7cc2052b525203746fbd70999b3b193d62c5b121e2f2a`/`b5795a5db20a61ac29696ef6366cbd5338eb8ca30399dc7087111f2f0384607b` | G4 gfx936 PASS |
| P4.1-E2 | gfx938 同矩阵 baseline/candidate | 两侧均 4/4 `valid:y`；binary SHA256=`2e6c3d72a526d6a17a17eca99c4513d4c32a29a64aa23765d6af376d669fc3e2`/`5d7e31a103aca9cf26adb762c228759537390e7b8257dc280ee632f57e1aaf2a`，日志 SHA256=`26451287b40561b404d39d188d6383371e4726dec195dfb23a6f87c9dd7d0fde`/`b5795a5db20a61ac29696ef6366cbd5338eb8ca30399dc7087111f2f0384607b` | G4 gfx938 PASS；运行时 GPU 0 为 100% 占用，不使用其 timing 数据 |
| P4.1-E3 | gfx936/gfx938 baseline/candidate device ELF | 两架构各提取 1 个 device ELF；gfx936 manifest SHA256 两侧均为 `b4e11208a7795af9b021e380056a4b1e276417841a4346b9cedd7c7bca099f00`，gfx938 两侧均为 `979e324030dcb83bb9fbca917649c2abbf61caee271717b9bcc67b2fe7f93141`，内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD 四组均报告 `S_SETREG_IMM32_B32 need to be test`、`V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，但所有 dispatch 均完成且 host reference 正确；这些告警在当前 basic forward 切片不构成指令阻塞。gfx946 未采集或比较性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device ELF identity）。下一切片为 P4.2 `ck_tile/15_fused_conv`。

### Step 4.2：`ck_tile/15_fused_conv` MLS Wasp probe — BUILD-PASS / SKIP-PMD（2026-09-02）

- 范围冻结：当前默认构建图只包含同一源码生成的 FP16/BF16 `tile_example_fused_conv_mls_wasp_{,bf16_}probe`；unstable persistent target 与 CMake 中注释的 TLS/MLS 变体不混入本切片。该 MLS kernel 的架构契约只允许 gfx938/gfx946，gfx936 只做构建和拒绝路径回归。
- 发现并修复两处 gfx946 构建闭包缺口：根 CMake 在 `GPU_TARGETS=gfx946` 时未生成 host-side `CK_ENABLE_HCU_GFX946_INSTANCES`；gfx946 的九个 `mls_ds_traits` 特化暴露 `DsFormatInst`，但公共 block consumer 与 gfx938 对应实现统一读取 `Type`。本步补齐宏并机械统一别名，不改变 dispatcher 参数、MLS atom、tile layout 或指令编码。
- probe 根据构建目标实例化 `HcuArch=gfx946/gfx938`，不再无条件硬编码 gfx938；增加默认值为 1 的 `-time` 参数，`-time=0` 关闭 warmup/repeat、只发射一次 correctness kernel并输出 `timing:off`。默认 `k` 从不满足 `NPerBlock=256` 的 128 修为最小健康值 256；原默认在 gfx938 baseline 也会被 `IsSupportedArgument` 拒绝。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.2-E0 | gfx946 FP16/BF16 focused build；核对实际 arch、资源和 spill | PASS；两 target 均以 `--offload-arch=gfx946` 和 `CK_ENABLE_HCU_GFX946_INSTANCES=1` 编译。FP16 为 SGPR 32、VGPR 98、spill/scratch 0、LDS 49152 bytes；BF16 为 SGPR 33、VGPR 98、spill/scratch 0、LDS 49152 bytes | G1/G2 build PASS |
| P4.2-E1 | gfx946 FP16 默认健康形状 `G1,N64,K256,C256,H=W=2,R=S=1`，`-time=0` | dispatch 完成但 65536/65536 输出均为 NaN；binary SHA256=`5edce81f20ff29c62d2a198ca5a35b4b57bbe0c0c0896dd32f6c640c885a181b`，日志 SHA256=`67620281366bcb64f7795316fd49d495f95dd65b8be325dae7168b4949ad407c` | G3 SKIP-PMD |
| P4.2-E2 | 提取 gfx946 FP16 device ELF 并统计 ISA | device ELF SHA256=`5e94e66873b6ec08636c7d990d642a5a67f25d9217c190e55bb166f3978892cc`；含 `ds_read_matrix_trans_format=64`、`v_mmac_f32_16x16x16_f16=256`、`ds_read_b128=16`、`ds_write2_b32=32` | G2 ISA observation |
| P4.2-E3 | gfx938 FP16 修改前 baseline `-k=256` 与 candidate 默认 `-time=0`；BF16 candidate 默认 `-time=0` | FP16 baseline/candidate 与 BF16 candidate 均完整 reference PASS；FP16 binary SHA256=`52bce3a9c92d9ac582d50168f6d7d115af0ff7946660f3bf17677604c51129dc`/`b79e0c136c1c4e820463d3e3151fc40daf38d25270e035cacbc83cf1e1ae18cf`，BF16 candidate=`f62931ae7494f4fcb27a546b40833bcd16dda4f58b273fde6bed9fec2592b201` | G4 gfx938 PASS |
| P4.2-E4 | gfx936 FP16 baseline/candidate 与 BF16 candidate | 三次均在 `target_enum != HcuArch(gfx938)` 门禁处拒绝，未发射 MLS kernel；FP16 baseline/candidate SHA256=`df3bb8aefd83b905b36ba89a795d0653946f2f2a5026ee7fd5488c0579a6383d`/`28bfa1a497c8778723bf4868c6945be973df913d4c10dd39ff3a353c7b787424`，BF16 candidate=`69e2ebf2db52e674d8fb80b5ff4600d3540026276edf575dd81dd8f5eb77541b` | G4 architecture-bound PASS |

PMD 在首个 kernel dispatch 明确报告 `ds_read_matrix_trans : testing`，这与提取出的 64 条 transpose-format DSReadM 指令以及全 NaN 输出一一对应；同一 FP16/BF16 kernel 拓扑和健康形状在 gfx938 真卡正确。当前证据不足以声明 Shaobo 真机指令错误，且不能为绕过模拟器而改写已在真实 gfx938 正确的 MLS pipeline，因此按任务规则把该共享 MLS-trans 子模块标记为 `SKIP-PMD`，不再重复执行 BF16 PMD 路径。

gfx938 的实际 device kernel 模板仍是原 `HcuArch=gfx938`，gfx946 CMake 宏与 traits 文件在该构建中均不生效；FP16 修改前/后编译报告保持 SGPR 44、VGPR 98、spill/scratch 0、LDS 49152 bytes、Instructions 1714。device 执行体没有不确定变化，不执行性能 A/B；gfx936 无该 kernel 性能面，gfx946 PMD 严禁性能测试。门禁结果：G0-G2 PASS；G3 SKIP-PMD；G4 PASS；G5/G6 NOT RUN。下一切片为 P4.3 `ck_tile/38_block_scale_gemm`。

### Step 4.3a：`ck_tile/38_block_scale_gemm` tensorwise quant — PASS-PMD / BF16-output SKIP-PMD（2026-09-02）

- target：`tile_example_gemm_quant_tensor`，与 P3.5c 的 grouped quant 不同，是单 GEMM tensor-scale runner。默认矩阵覆盖 FP8/BF8 四种输入组合、FP16/BF16 输出、NT/NN/TN layout，以及 M/N/K padding。
- 最小改动：增加默认不变的 `-time=1,-warmup=5,-repeat=20` 与 `-bf16_output=1`；`time=0` 强制零预热、单次 correctness launch并输出 `timing:off`。PMD 使用 `bf16_output=0`，不重复探测 P3.5b/P3.5c 已确认不可靠的 BF16-output CShuffle 模拟路径；gfx938 默认完整矩阵保持原行为。
- 架构边界：本目录 CMake 只为 gfx938/gfx946 入图三个 quant target，gfx936 精确构建返回 `No rule to make target`，符合既有契约，不是回归失败。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.3a-E0 | gfx938 修改前默认完整矩阵 | PASS，11/11；baseline binary SHA256=`36bb3014d0f2d8a92a5529ce24e9d3757a74b53bc14eeef98163b9c2902e3ba5`，日志 SHA256=`bfb15782282a3ac985b63f1638ad0bb26d009ee7a134578d4494ce0251f98816` | G0 baseline PASS |
| P4.3a-E1 | gfx946 focused build；`-time=0 -bf16_output=0` | build PASS；7/7 FP16-output case 正确，覆盖四种 FP8/BF8 输入组合、RRR/CRR layout 与 RCR padded 160x192x80；binary SHA256=`d6795b30ddb9acf805beff5edb8ecaa9cc86a9b8bb5efbac17ae2054501948eb`，日志 SHA256=`51ad6a887242e53463dbcec43a034e187f2f36088ff0ccec1739ccf3c59bc472` | G1-G3 supported slice PASS |
| P4.3a-E2 | gfx938 candidate 默认完整矩阵 | PASS，11/11；candidate binary SHA256=`240d5f19e0363764b175987d15930e267443e1ecf125505bcb221f5755f7ea6f`，日志 SHA256=`6d66fd57aa698711a7b68f9f2ecf16ec1385361e0d02987fb41ad77ca34da878` | G4 gfx938 full regression PASS |
| P4.3a-E3 | gfx936 精确请求同 target | baseline/candidate 构建图均无该 target；CMake 架构门禁未变化，未发射不受支持的 quant kernel | G4 architecture-bound PASS |
| P4.3a-E4 | gfx938 baseline/candidate device ELF | 两侧均只提取一个 gfx938 ELF，SHA256 均为 `f743ba8a4dd3d9e90db3527dd4070a6266846cdd35cfca931ce77d90b3162f77`，内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD 记录了地址计算类 `v_mad_u64_u32 ... read vgpr ... before writing`、少量 source-modifier 和 pending-cacheline warning，但 7 次实际 dispatch 均完成且完整输出与 CPU reference 一致，因此沿用既有规则记为模型告警。gfx946 未执行性能测试。gfx938 device ELF identity 直接排除 kernel 性能变化；gfx936 无该 target 性能面。

门禁结果：G0-G4 PASS（BF16-output 延续 SKIP-PMD）；G5 NOT RUN；G6 NOT RUN（device ELF identity）。下一切片为 P4.3b `tile_example_gemm_quant_rowcol`，完成后独立提交。

### Step 4.3b：`ck_tile/38_block_scale_gemm` row/column-wise quant — PASS-PMD / BF16-output SKIP-PMD（2026-09-02）

- target：`tile_example_gemm_quant_rowcol`；与 tensorwise 独立之处是 A scale 按 M 行、B scale 按 N 列读取并逐元素组合。默认 dtype/layout/padding 矩阵与 P4.3a 对齐。
- runner 使用同一默认不变的 `-time=1,-warmup=5,-repeat=20,-bf16_output=1` contract；PMD 正式验收为 `-time=0 -bf16_output=0`，gfx938 继续运行默认完整矩阵。device pipeline、scale descriptor、kernel args 与模板实例均未改动。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.3b-E0 | gfx938 修改前默认完整矩阵 | PASS，11/11；baseline binary SHA256=`c042a84c73b02aa984d0cd55592046f1454553d1128fdfc5c20e5f1dbbc549cc`，日志 SHA256=`08bd93439d60eaf998125be78a876b2b6188ba62c613477cdcf4fc6a1a6fcf26` | G0 baseline PASS |
| P4.3b-E1 | gfx946 focused build；`-time=0 -bf16_output=0` | build PASS；7/7 FP16-output case 正确，覆盖四种 FP8/BF8 输入组合、RRR/CRR 与 RCR padded；binary SHA256=`e1ee901c5f04c0effa26cc02893a660a54385c7a913716f1fa1a641fcfabf0df`，日志 SHA256=`4b323998319c49785ad04b6787ec8142303af31854f83c2907d3c51e68a28b3e` | G1-G3 supported slice PASS |
| P4.3b-E2 | gfx938 candidate 默认完整矩阵 | PASS，11/11；candidate binary SHA256=`e074d66e500842f04da960afb88c84ac6c4ded1e0db97ea45a5fb92ccd1ecc3c`，日志 SHA256=`4dc9a63bafaa9ec64c6e8ba76075f0e1c4a9479b2cc9997f8480fc7851337036` | G4 gfx938 full regression PASS |
| P4.3b-E3 | gfx936 精确请求同 target | `No rule to make target`；gfx938/gfx946-only CMake 门禁保持不变 | G4 architecture-bound PASS |
| P4.3b-E4 | gfx938 baseline/candidate device ELF | 两侧唯一 gfx938 ELF SHA256 均为 `4f9b13ab59d5ad81baa09ad9a0babd7764a183de77833b591708cebf2a2217bc`，内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD 告警集合与 P4.3a 一致，主要为地址计算类 uninitialized-VGPR、source modifier 与 pending-cacheline warning；7 次独立 scale-load dispatch 全部完成且 CPU reference 正确。gfx946 未执行性能测试；gfx938 device code 未变化，gfx936 无性能面。

门禁结果：G0-G4 PASS（BF16-output 延续 SKIP-PMD）；G5 NOT RUN；G6 NOT RUN（device ELF identity）。下一切片为 P4.3c `tile_example_gemm_quant_blockwise`，完成后独立提交。

### Step 4.3c：`ck_tile/38_block_scale_gemm` blockwise quant — PASS-PMD / BF16-output SKIP-PMD（2026-09-02）

- target：`tile_example_gemm_quant_blockwise`；A scale 为 M×ceil(K/128)，B scale 为 ceil(K/128)×ceil(N/QuantGroupN)，并包含 CRR dense-BQ（`QuantGroupN=1`）与 K=160 padding 两条独立路径。
- runner 增加默认不变的 `-time=1,-warmup=5,-repeat=20,-bf16_output=1`；PMD 使用单次 correctness launch 和 FP16-output 全矩阵。没有修改 ABQuant pipeline、scale descriptor、MMAC/CShuffle 或 device kernel。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.3c-E0 | gfx938 修改前默认完整矩阵 | PASS，11/11；baseline binary SHA256=`2e5d497d0bba325b1190e43ce0e7084ff1e9dd85df6e5aa03b88b227c7479150`，日志 SHA256=`f3f49276409ef5742b808d2e477e79dcf417dc6665a14613889336dcd3d2d4d3` | G0 baseline PASS |
| P4.3c-E1 | gfx946 focused build；`-time=0 -bf16_output=0` | build PASS；7/7 FP16-output case 正确，包含 RCR padded 160x192x160、RRR 与 CRR dense-BQ；binary SHA256=`a857b27b52fd860f52da06eec447313ec05d7f7876930f92af8629e703888207`，日志 SHA256=`6619c74f05b72d5ef359565df50a793217591959e141d80b31e15025a7c5d5d4` | G1-G3 supported slice PASS |
| P4.3c-E2 | gfx938 candidate 默认完整矩阵 | PASS，11/11；candidate binary SHA256=`2bd2765a5ab28057032a7e2d2ddfa292f34eab5d5e5a93aea9874e3080e9ad92`，日志 SHA256=`d943b48eb7fc901b18132e3f3cd36b093ac8c6310c8077a470ea524b056ea017` | G4 gfx938 full regression PASS |
| P4.3c-E3 | gfx936 精确请求同 target | `No rule to make target`；目录架构边界未变化 | G4 architecture-bound PASS |
| P4.3c-E4 | gfx938 baseline/candidate device ELF | 两侧唯一 gfx938 ELF SHA256 均为 `b0169d066cea478da83ab1b9211a08853cc13f6b981cce726605fa53da5a3d4e`，内容逐字节相同 | G6 direct code-object evidence；无需性能 A/B |

PMD 在首条 dispatch 报告 pending-cacheline 与 source-modifier warning，后续 7 条 supported case 均完成，包含多 K-block scale 累加和 dense-BQ 读取，输出与 CPU reference 一致。gfx946 未执行性能测试；gfx938 device code 未变化，gfx936 无该 target 性能面。

门禁结果：G0-G4 PASS（BF16-output 延续 SKIP-PMD）；G5 NOT RUN；G6 NOT RUN（device ELF identity）。P4.3 三种单 GEMM quant scale 路径形成闭包；下一步重新盘点尚未验收的已启用 target，并选择独立高风险切片。

### Step 4.4：`ck_tile/19_grouped_gemm` production-selector C API — PASS（2026-09-02）

- target：`tile_example_grouped_gemm_c_api_selected_smoke`。它与 P3.5d3 使用同一个公共 C ABI 60-case runner，但链接包含生产 selector/provider portfolio 的 `ck_tile_hcu_grouped_gemm`，而不是只含通用 fallback 的 `ck_tile_hcu_grouped_gemm_generic`，因此需要独立证明 gfx946 链接闭包和运行选择。
- 本切片不修改 runner、selector、provider、kernel 或 CMake。gfx946 延续 P3.5b 已确认的 BF16-output PMD 边界，使用既有 `--fp16-only` 模式覆盖 NT/NN/TN 的 aligned-packed 与 ragged-strided 六条真实 dispatch；gfx936/gfx938 运行默认完整矩阵。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.4-E0 | gfx946 focused build；核对实际 build tree/target | PASS；`build_shaobo_step1` 以 gfx946 provider archive 完成链接；binary SHA256=`161187ed19b2eb6446c0633fa63c29303ff5791a2959373d37ef4ef105372066` | G1/G2 PASS |
| P4.4-E1 | gfx946 PMD `--fp16-only` correctness | PASS，FP16 6/6；NT/NN/TN aligned case 选择 `generic_v4`，ragged case 选择 `generic_v4_padding`，全部通过 host FP32 accumulate/reference；日志 SHA256=`40ba19ca05ae093958bb19f3b388ce6a8a09aa1a8f957b61b1e02d6c7a187d8d` | G3 PASS；BF16 延续 SKIP-PMD |
| P4.4-E2 | gfx936 production-selector 默认完整矩阵 | PASS，BF16 54/54、FP16 6/6；binary SHA256=`9b6b44f04386554eed203e58d4da8f3b5dcadf46f1b4d95feae9899af66101d3`，日志 SHA256=`091ccf70ef767b796891a602153ab1fe647c0a276f067712af0a29b43df81ffb` | G4 gfx936 PASS |
| P4.4-E3 | gfx938 production-selector 默认完整矩阵 | PASS，BF16 54/54、FP16 6/6；binary SHA256=`abfb1ecefd80066d01dd2f1f78d592ca6d8890ee3242ec609b44c439cf20e234`，日志 SHA256=`091ccf70ef767b796891a602153ab1fe647c0a276f067712af0a29b43df81ffb` | G4 gfx938 PASS |

PMD 仍报告 grouped-GEMM 已记录的地址计算类 uninitialized-VGPR、source-modifier 与 pending-cacheline warning，但六条支持路径均完成且完整 reference 正确，故不升级为阻塞。本切片是对既有 target 的验证补洞，没有 baseline/candidate 源码差异，也没有改变任何 gfx936/gfx938 device code；因此能直接排除性能影响，G6 不触发。gfx946 PMD 未运行性能测试。

门禁结果：G0-G4 PASS（BF16-output 延续 SKIP-PMD）；G5 NOT RUN；G6 NOT RUN（validation-only，无代码候选）。下一切片为 `tile_example_grouped_gemm_c_api_tn_logical_k_tail_smoke` 的架构契约审计与独立验收。

### Step 4.5：`ck_tile/19_grouped_gemm` gfx936 logical-K-tail smoke 架构收口 — PASS（2026-09-02）

- target：`tile_example_grouped_gemm_c_api_tn_logical_k_tail_smoke`。源码固定验证 gfx936 BF16 TN logical-K-tail selected provider：16 groups、M=N=2048、逻辑 K 总和 32757，并在同一 ELF 中用环境开关回滚到 generic padding 后比较输出 hash/guard。
- 审计确认公共 C ABI 只有当前设备为 gfx936 且 dtype/layout/group/shape/stride 全部满足约束时才选择该路径；gfx938 共享 BF16 TN device-args family但不会命中旧 host-descriptor selected path，gfx946 也没有该 provider。原 CMake 却在所有架构创建专用 target，使 gfx938 只重复三次 generic fallback，并把一个巨大 BF16 workload 暴露给已知不可靠的 gfx946 PMD。
- 最小修复：仅当 `GPU_TARGETS MATCHES "gfx936"` 时创建该 executable；不改 provider manifest、production library sources/definitions、selector、C ABI 或 kernel。README 同步明确 target 的架构边界。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.5-E0 | gfx936/gfx938 修改前 target baseline | 两台均 build/run PASS；gfx936 实际 selected 两次并与 rollback hash 一致，binary SHA256=`1804e4d4c5524faa3f927d176d9709a7b7d0e0d471e885876d9ba35dd3f68d29`、日志 SHA256=`58fdf6d172bcdff350cd2106ead41be5960796b515c734628b6c616f955d61e4`；gfx938 三次均为 generic fallback，binary SHA256=`90ca474010360b8fa05df067511227b11c59f73786f78813e3b85ab3dafd16d5`、日志 SHA256=`d9063143b3ef974fcd58a87611e6b3e233898a1a55c095a1d14a26c4809715c7` | G0 contract evidence |
| P4.5-E1 | gfx946 同步 CMake 后重新生成并精确请求 target | 按预期返回 `No rule to make target`；拒绝日志 SHA256=`599c5ad11031e64b41845424edf65a8b63615454723e4465ae5f421de2ec6c5d` | G1-G3 architecture-bound PASS；未运行 PMD 性能或巨大 BF16 workload |
| P4.5-E2 | gfx936 候选 build/run selected/rollback smoke | PASS；两次不同 input guard 的 selected 输出 hash 相同，rollback 输出 hash 相同；candidate binary SHA256 仍为 `1804e4d4c5524faa3f927d176d9709a7b7d0e0d471e885876d9ba35dd3f68d29`，与 baseline 逐字节一致；日志 SHA256=`7b7b9eae4d28c90ed7bdac3ccbf22df11283a0dae7d2b9b6e09f5fb23f752fed` | G4 gfx936 PASS |
| P4.5-E3 | gfx938 同步 CMake 后重新生成并精确请求 target | 按预期返回 `No rule to make target`；拒绝日志 SHA256=`47cdee4db7090700318c948ff2b4d9cbfcd54351107e518c4b5b98b5a0f899ed` | G4 gfx938 architecture-bound PASS |

该 CMake 变更只移除 gfx946/gfx938 上无 selected path 可验的专用 executable，不改变 `ck_tile_hcu_grouped_gemm` 或 provider archive 的源文件、宏和编译选项；gfx936 target 保持入图且 binary baseline/candidate 完全相同。因此能直接排除 gfx936/gfx938 kernel 性能变化，G6 不触发。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（CMake test-scope only，gfx936 binary identity）。下一切片为 `tile_example_grouped_gemm_device_args_benchmark` 的 correctness-only 模式与三架构验收。

### Step 4.6：`ck_tile/19_grouped_gemm` device-args benchmark correctness-only — PASS（2026-09-02）

- target：`tile_example_grouped_gemm_device_args_benchmark`。它是可参数化的生产 device-args ABI 示例，默认执行 correctness、warmup 与 GPU-event samples；原实现要求 `samples>0`，无法在 Perf Model 上遵守“只验证指令正确性、不测试性能”的硬边界。
- 最小改动：新增默认值为 1 的 `--time`；`--time=0` 强制 `warmup=0,samples=0`，完成 CPU accuracy/guard 检查后输出 `TIMING status=off` 并在创建 `EventTimer` 前返回。默认 `--time=1` 的 benchmark 路径、selector、provider、device-args ABI 和 kernel 全部保持不变。三个 CTest smoke 改用 `--time=0`，使常规 correctness 测试不再混入单样本性能测量。
- 使用说明同步声明该 target 支持 gfx936/gfx938/gfx946，并区分 correctness-only 与正式 benchmark；CSV runner 不传 `--time=0` 时仍保持原性能采样协议。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.6-E0 | gfx936/gfx938 修改前 FP16 NT 128³、group=2、warmup=1/samples=1 baseline | 两台均 CPU accuracy/guard PASS 并输出 PERF；gfx936 binary/log SHA256=`3c32e51ddf0770beccf3c6ce7c316d0eb4c2f95bf7cad960b3c4729c59dd327c`/`d0d621769f39c78a096cc3cde5f4fce93479ef193fa96d3255227f3057643e6a`，gfx938=`9e1bc01098395111be312282a936ee6bbb9eba5a3b77580fbbd057ef3164e136`/`3d96c16212bab3b37e6a95264accf6f72b14f8b2f09006fb88d9103667a74c96` | G0 baseline PASS |
| P4.6-E1 | gfx946 focused build；FP16 NT 64³、group=2、`--time=0` | PASS；选择 InstanceId 4003 `ck.gg.gfx946.fp16.NT.v4.64x128x64.full_padding`，CPU sample `checked=514,failed=0`、guard PASS，明确 `TIMING status=off warmup=0 samples=0`；binary/log SHA256=`9ac0351545b2697ec251e7a65325b0d54daf0e32aad4b3f9b8b33d5eaf588b36`/`58c69cb673367ad37e3a48eda3cbad0c89bfd0149351366f2fab1bf3b4d5809c` | G1-G3 PASS；未测试 PMD 性能 |
| P4.6-E2 | gfx936 candidate `--time=0` 与默认计时路径 | 两种模式均选择 InstanceId 4002、accuracy/guard PASS；correctness/default 日志 SHA256=`2a291fc1eae6c97738fd287d16699a24b7b2d70678a3cc725e904ec9f0ca94f3`/`63b2f4f22bd07ef84f61b3aa022a89c94aa377dc2980b892e800c81fe44c4f1a`；默认路径仍输出 PERF | G4 gfx936 PASS |
| P4.6-E3 | gfx938 candidate `--time=0` 与默认计时路径 | 两种模式均选择 InstanceId 4002、accuracy/guard PASS；correctness/default 日志 SHA256=`47f9726af7558ffaa837f4de5abfc5689926bd5d3003ab3b4db9c475f8d9303d`/`ad2f05f83d7158a661803d927b7ba130f160c1d5571c46f4d66b7fe3968fa897`；默认路径仍输出 PERF | G4 gfx938 PASS |
| P4.6-E4 | CTest enrollment | gfx946 基础 FP16 smoke 1/1 PASS；gfx936/gfx938 的基础 FP16、BF16 ragged-M、BF16 ragged-K 各 3/3 PASS，全部使用 CMake 中的 `--time=0` | G1/G3/G4 PASS |
| P4.6-E5 | gfx936/gfx938 baseline/candidate `.hip_fatbin` | gfx936 两侧 SHA256 均为 `173611416c25d10318c66b733168ba1e35aeda5a51ac470cc36cc6a89f204799`；gfx938 两侧均为 `3f734c17c6c7149dc2a6bfa74bc4dec39440a64463423a14e1d816c10983633d`，逐字节相同 | G6 direct device-payload evidence |

候选 host binary 因新增 CLI/输出分支而变化（gfx936=`5eacaeb94625dd2022231defd2b4fa0a1c1c4b38cd74daa11abc433f2f7e4b0f`，gfx938=`7c4f41a0ea793b9aa07c897f40d89fb76b8d22eaad4e4316033d3e426b3c55a9`），但两架构 device payload identity 已直接排除 kernel 性能变化；默认路径的单次 PERF 运行只作为行为回归，不作为性能结论。故不触发额外性能 A/B，gfx946 PMD 全程未采样性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device payload identity）。`ck_tile/19_grouped_gemm` 先前遗漏的三个 enabled target 已全部形成独立记录；下一步重新盘点其余 enabled target 与文档覆盖关系。

### Step 4.7：`ck_tile/01_fmha` basic backward — PASS（2026-09-02）

- target：`tile_example_fmha_bwd`，由 13 个 gfx-specific generated blob 与 backward runner 组成。本切片冻结为 FP16、batch/head=1、Sq=Sk=64、head-dim=64、B H S D layout、无 bias/mask/dropout、非 deterministic，覆盖 workspace prepare、dQ/dK/dV backward stages 和 CPU reference。
- 缺口：runner 原先把 `stream_config.time_kernel_` 固定为 true，即使 warmup=0/repeat=1 仍会创建 timer并报告性能，不适合 gfx946 PMD。最小改动与 P4.1 forward 对齐：新增默认值为 1 的 `-time`；`-time=0` 设置 `time_kernel=false,warmup=0,repeat=1`，runner 明确输出 `timing:off`。默认 benchmark 行为保持不变。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.7-E0 | gfx936/gfx938 修改前构建并运行最小 backward baseline | 两台均 `valid:y`；gfx936 binary/log SHA256=`fa856797a89ab987f9763a1814d9aafde428ff4cbcd8ae0e88b1f3e3a389ea05`/`0446ab5631ae1d062542ecdb443cd9375e76a23fa87c6b28530d818e41d2df11`，gfx938=`23d099070ca964a8bbfa1c790b5da2032fb73b7f2bef568b760e4549b3e55540`/`629a0ff65db4dab42c4ced8c5bd0d0d227d6f586fad89b1819b1278764105f8f` | G0 healthy controls |
| P4.7-E1 | gfx946 13-blob focused build；`-time=0` PMD correctness | PASS；binary SHA256=`12d00ce0b139ba5fe31a673e061a8537f66768b2bc756716eb91523530a0a6d0`；workspace prepare 后共发射 dispatch id 0-5 六个 backward stage kernel，无 warmup/repeat，最终 `timing:off, valid:y`；日志 SHA256=`5d625e338b34f9ed0cfc389284847836c6bfc7c91beb3b4efd9c3fdd5cc552a4` | G1-G3 PASS；未测试 PMD 性能 |
| P4.7-E2 | gfx936 candidate correctness-only 与默认路径 | 两种模式均 `valid:y`，前者明确 `timing:off`、后者仍输出 ms/TFLOPS/GB/s；correctness/default 日志 SHA256=`e3aff3e84dc6377502ba56e6e0c4cc293705456a3a61044e55c6d2bc64b1fa02`/`1cef750d4429c1dc77ae9936ae8c968dfa6138f3c84216c3b291c7159f14e2cf` | G4 gfx936 PASS |
| P4.7-E3 | gfx938 candidate correctness-only 与默认路径 | 两种模式均 `valid:y`，前者明确 `timing:off`、后者仍输出原性能字段；correctness/default 日志 SHA256=`c2a47d34706d9eec38cf521e9d360f70ebef559f6508e8796022c55a1ed8ef09`/`5a793ebfb301414be6a0b4b9573afc473c64d0882762f845681ac0a0076c8abd` | G4 gfx938 PASS |
| P4.7-E4 | gfx936/gfx938 baseline/candidate `.hip_fatbin` | gfx936 两侧 SHA256 均为 `e05210fb61caaf667c09c988e36223be31cd5e53f7b051d238b06e1d369650fc`；gfx938 两侧均为 `b0e158e99c9ab0e3dad820d268249a77e3527a4734609cdbc4f8c832c60f0770`，逐字节相同 | G6 direct device-payload evidence |

PMD 报告多条标量/向量寄存器 read-before-write 模型 warning，但六个 stage dispatch 均完成且完整 dQ/dK/dV reference 得到 `valid:y`，因此按已建立的规则记为非阻塞模拟告警。候选只改变 host runner 的 timing 控制，且两台真机 device payload identity，直接排除 kernel 性能变化；默认路径单次运行只用于接口回归，不作为性能结论，不触发性能 A/B。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（device payload identity）。下一切片审计 `tile_example_fused_conv` persistent target，区分其与 P4.2 MLS probe 的覆盖关系。

### Step 4.8：`ck_tile/15_fused_conv` Wasp V3 persistent — BUILD-PASS / SKIP-PMD（2026-09-02）

- target：仅在 `CK_BUILD_UNSTABLE_HCU_EXAMPLES=ON` 时生成的 `tile_example_fused_conv`，源码为 `fused_conv_wasp_persistant.cpp`。它使用 Wasp V3 两级 LDS pipeline 与 persistent tile scheduler，不是 P4.2 的 MLS Wasp probe，因此单独验收。
- 形状筛选先在两台真机完成：C=32 被 `IsSupportedArgument` 正确拒绝（两级 pipeline 的 K-loop 不满足约束）；C=64 在 gfx936/gfx938 都产生约 62% 错误，判为 example 自身不支持/不健康的输入而未用于 PMD 归因；最终冻结 `G=1,N=1,K=128,C=256,H=8,W=16,Y=X=1,stride=1,dilation=1,pad=0`，该形状在两台真机完整 reference 正确。
- runner 原先始终 warmup/计时 10 次，并忽略 `run_fused_conv` 的布尔结果，导致即使 reference 正确也返回失败。最小改动新增默认开启的 `-time`；`-time=0` 只发射一次 correctness launch 并输出 `timing:off`；同时把 parse、unsupported 和 reference 结果正确传播到进程退出码。默认 `-time=1` 的性能输出与 launch 次数不变，kernel、tile、pipeline 和调度参数未修改。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.8-E0 | gfx936/gfx938 修改前构建；筛选 persistent 健康形状 | C=32 两台均被 `IsSupportedArgument` 拒绝；C=64 两台均约 62% 错误；C=256 两台 reference 无错误，但旧 host runner 因忽略/反转结果退出 1。C=256 日志 SHA256：gfx936=`5b8d6badfd186f1d0c05d8c7d6f9eead0f092237e6a6c59e9ac807c56ddd1595`，gfx938=`c87ead7ff9be7ca48ce804063518cff79859caaedd72708ea88443d553612b94` | G0 healthy-control selection |
| P4.8-E1 | gfx946 focused build；C=256、`-time=0` PMD correctness | BUILD PASS，binary SHA256=`c83c966516b35adf8b2da8315e3c4156aee83ae033ba02fff9f1fcbaff06a648`；单次 persistent dispatch（grid=`110592x1x1`，workgroup=`768x1x1`）完成但 reference FAIL：10889 errors、66.46118%、max error=0.01009417；日志 SHA256=`f63b0dc78cce31778bf855ebc1a38461aeaeeb254188d316cfa6113762654c52` | G1/G2 PASS，G3 `SKIP-PMD`；未测试 PMD 性能 |
| P4.8-E2 | 提取 gfx946 code object/ISA 并统计执行链 | code object SHA256=`53e18959eabc0e1f12086940e30d13b352b0dc930c1129874fd7ee80f4ed1120`，ISA SHA256=`07bf6e603e83aca5a068849ad2f79934a165b36a111cfede6cb20cecb03849c0`；`v_mmac=128`、`buffer_load.*lds=16`、`ds_read=56`、`ds_write=64`、`s_barrier=34`、`s_waitcnt=43`，16 条关键指令均为 `buffer_load_dwordx4 ... idxen offen lds` | 锁定 Wasp persistent async global-to-LDS PMD 执行边界；不同于 P4.2 的 `ds_read_matrix_trans_format` 缺口 |
| P4.8-E3 | gfx936 candidate correctness-only 与默认路径 | 两种模式均 reference PASS、exit 0；correctness/default 日志 SHA256=`068e5001cbbab80580863aac9e6d31fe75b2d9b94ab7853d70a5adb94649d68d`/`a3e8fab2c6a09bb140a74a9021923c87eda3715ee13e14ff02a2d12409699850`；默认路径仍输出 Perf | G4 gfx936 PASS |
| P4.8-E4 | gfx938 candidate correctness-only 与默认路径 | 两种模式均 reference PASS、exit 0；correctness 日志 SHA256=`068e5001cbbab80580863aac9e6d31fe75b2d9b94ab7853d70a5adb94649d68d`，default=`1e82c25d8934cc64be0dc2d8c7089718f03b051756c7aa96bd8d8893328d6521`；默认路径仍输出 Perf | G4 gfx938 PASS |
| P4.8-E5 | gfx936/gfx938 baseline/candidate `.hip_fatbin` | gfx936 两侧 SHA256 均为 `7c6c0651f842e023223498dfebf5723718943b3062e310be004b4f5fa6264683`；gfx938 两侧均为 `babd2416ee4de1acd7440f11d27547c06d58dd00c7ebd9b9578168e88dc4c262`，逐字节相同 | G6 direct device-payload evidence |

同一健康形状仅在 gfx946 PMD 错误，且 PMD 同时报告 source-modifier/pending-cacheline warning；依照“模拟器指令问题无法解决则记录并跳过”的规则，本步不为模拟器改写真实 Wasp kernel。候选只改变 host runner 的计时/退出码，真机 device payload 完全相同，能够直接排除 gfx936/gfx938 kernel 性能影响；默认路径只做行为回归，不作为性能结论，故不触发额外性能 A/B。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5 NOT RUN；G6 NOT RUN（device payload identity）。下一切片执行 enabled-target/CMake build-graph 与本文验收记录的覆盖闭环审计。

### Step 4.9：`ck_tile/19_grouped_gemm` FP16 forced baseline — PASS（2026-09-02）

- 首次以 PMD 实际 build tree 执行 `cmake --build build_shaobo_step1 --target help`，得到 137 行 build target、其中 94 个 `example_*`/`tile_example_*`/`ckProfiler` target；原始清单 SHA256=`ca7e549a3a3750f17eb37cc6192e7fccdce46ee08fbe4e1bd6f718732649165e`，筛选清单 SHA256=`0a2c3bd15d2ea64201d8f64338a74e92c8c9da05e9680c6b43e01781310a9e02`。聚合 custom target 与已验收 executable 分开归类，不重复当成新 kernel。
- 本切片选择真实缺口 `tile_example_grouped_gemm_fp16_baseline`：它定义 `CK_TILE_GROUPED_GEMM_FORCE_BASELINE`，强制使用 Compute V4/V4-Mpad baseline，不命中 P3.5a production fast runner 的 gfx936/gfx938 大 shape selector，因此需要独立运行证明。
- 正式 case 为 FP16 NN、group=2、`M={128,256},N={128,128},K={128,128}`、三类 stride 均显式设为 128、`validate=1,warmup=0,repeat=0`。预检曾因遗漏 stride 列表触发 runner 回退默认 shape；该次结果不计入正式证据，补齐完整参数闭包后重新执行。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.9-E0 | gfx946 focused build 与正式 case PMD correctness | PASS；binary SHA256=`9f0c24d414717a427c7ecc11354469cbefdb5bcc8c38783de59985bdea435556`；选择 `gemm_grouped_fp16_64x128x64_8x8x4_0x0x0_Persistent_MAll_64`，单次 launch、`Timing: off`、两组 CPU reference correct；日志 SHA256=`3c22b229b6897724e56f8a903a174a4ad894092031982d88c99991e713cf5dab` | G1-G3 PASS；未测试 PMD 性能 |
| P4.9-E1 | 提取 gfx946 target code object/ISA | code object SHA256=`9167c03c4f6b15ec1272e5a1c98ae860d54ffa1629be6a2c091976a5bbb906dc`，ISA SHA256=`9c01cb463bdef1ba69ff0c31e2ad9bc5e99b0525f8d5594261ea873a20b9e416`；整个 baseline binary 含 `v_mmac=15232`、`ds_read=8480`、`ds_write=6596`、`buffer_load=16661`、`buffer_store=2123` | G2 actual gfx946 payload；全集计数不等同于单个已选 kernel |
| P4.9-E2 | gfx936 focused build 与同一 correctness case | PASS；binary SHA256=`ac1f5f48a4390fc98682a6dca904bf71a77dc4d10d4b4d2cf91879fc3329fb58`，单次 launch、两组 CPU reference correct；日志 SHA256=`8d85076dfae1b4303b4b4e42f9fc19268f18a14fe456e881e62c49ab5df19ecd` | G4 gfx936 PASS |
| P4.9-E3 | gfx938 focused build 与同一 correctness case | PASS；binary SHA256=`ddb602a50735b8ad6de17f7796b86af8e8971534eae4cbbd50a1c998ee435e0e`，单次 launch、两组 CPU reference correct；日志 SHA256=`17118d5aa19c65e4181aca0b172a905413dc4c1d449096840745db84d789df21` | G4 gfx938 PASS |

PMD 报告该 Compute V4 路径既有的 `v_mad_u64_u32` read-before-write、source-modifier 与 pending-cacheline warning，但 dispatch 完成且完整 reference 正确，不升级为阻塞。本切片没有源码、编译选项或 binary candidate 变化，只验证一个既存 target；因此可直接排除 gfx936/gfx938 性能影响，G6 不触发。

门禁结果：G0-G4 PASS；G5/G6 NOT RUN（validation-only）。下一切片继续处理 build graph 中尚未独立验收的 grouped-GEMM 专用复现/量化 fast target。

### Step 4.10：`ck_tile/19_grouped_gemm` BF16 forced baseline — BUILD-PASS / SKIP-PMD（2026-09-02）

- target：`tile_example_grouped_gemm_bf16_baseline`，定义 `CK_TILE_GROUPED_GEMM_FORCE_BASELINE` 并强制使用 BF16 Compute V4/V4-Mpad。虽然 P3.5b 已记录 production fast target 的同配置失败，本步仍以实际 executable 独立运行，不用推断代替验收。
- 正式 case 与 P4.9 对齐：BF16 NN、group=2、`M={128,256},N={128,128},K={128,128}`、三类 stride 均为 128、`validate=1,warmup=0,repeat=0`。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.10-E0 | gfx946 focused build 与正式 case PMD correctness | BUILD PASS，binary SHA256=`cccdc16da4067041aeb96a0e05ee2f64f0a6e3500785403b65298e4e4c546a47`；选择 `gemm_grouped_bf16_64x128x64_8x8x4_0x0x0_Persistent_MAll_64`，单次 launch；group0 从 flat index 4097 起错误，1520/16384（9.277344%），group1 错误 3039（9.274292%），最终 reference fail；日志 SHA256=`b15bf08e49b288d78cba5025316e7d3cbba4fac64f019adc0dc3d0423ca66eda` | G1/G2 PASS，G3 `SKIP-PMD`；未测试 PMD 性能 |
| P4.10-E1 | 提取 gfx946 target code object/ISA | code object SHA256=`e9e25c5027b26c663b3589edbaf4a59bb8c5db421cb4360be2e2669a896041bd`，ISA SHA256=`2f8e28c659342a6552579c94c85fca4d6821315dceb14648055a533338c9e242`；整个 binary 含 BF16 `v_mmac=15232`、`ds_read=8480`、`ds_write=6596`、`buffer_load=16648`、`buffer_store=2293` | G2 actual BF16 payload；排除空 kernel |
| P4.10-E2 | gfx936 focused build 与同一 correctness case | PASS；binary SHA256=`8a574a042af47ca619b8fa40835e945b7895e1a5d79e93c81a7ee78456ba2e05`，两组 CPU reference correct；日志 SHA256=`370132ae18a5d0b109c35b35678e968dd2c5102c84fad8fe0c893b575a6647f9` | G4 gfx936 healthy control |
| P4.10-E3 | gfx938 focused build 与同一 correctness case | PASS；binary SHA256=`70765a4201223082ca090656b48e5f9b88e16b2ae13830d17e890ad37ff1c2b7`，两组 CPU reference correct；日志 SHA256=`19e7773b54624e240d9f3dfce37bca1f264831690a51d996d076e6104cb304e4` | G4 gfx938 healthy control |

PMD 的首错位置、group0 错误数和 `src1_neg/src1_abs`、pending-cacheline 告警与 P3.5b 同一 BF16 Compute V4 执行链完全吻合；两台真机同 target/shape 正确。因此本 target 合并归入已建立的 gfx946 PMD BF16 grouped accumulator-to-output/CShuffle 限制，不修改真实 kernel并继续下一模块。本切片没有源码或编译候选变化，能直接排除 gfx936/gfx938 性能影响，不触发性能 A/B。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN（validation-only）。下一切片审计 gfx936 reg-prefetch 与 gfx938 MLS 专用复现 target 是否应继续暴露给 gfx946 build graph。

### Step 4.11：`ck_tile/19_grouped_gemm` FP16 V3 reg-prefetch reproduction — PASS（2026-09-02）

- target：`tile_example_grouped_gemm_fp16_v3_regprefetch_256_w16_group8`，固定 `GemmConfigComputeV3RegPrefetch256W16Group8Aligned`。CMake 注释记录其来源是 gfx936 大 shape 调优，因此最初把“从 gfx946 build graph 移除”作为待证假设；实际三架构 correctness 证据表明该配置不是 gfx936-only，故不进行架构裁剪。
- 正式 case 为 FP16 NT、group=2、每组 `M=N=256,K=128`、显式 packed stride、`validate=1,warmup=0,repeat=0`，触发 256x256x64、8-wave、W16 V3 persistent kernel；这与 P3.5a 的普通 64x128x64 V4 配置不同。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.11-E0 | gfx946 focused build 与 PMD correctness | PASS；binary SHA256=`43b9a75ddc364e73a8edcbbcb7646e7c2e9f7b4b365030f698e4da254b58c39f`；单次 dispatch，grid=`49152x1x1`、workgroup=`1024x1x1`，两组 CPU reference correct；日志 SHA256=`f55a406d17ca5f4d90b01d4b4e74e3aacf7ee94637b4b344b1895cea8dd083ee` | G1-G3 PASS；未测试 PMD 性能 |
| P4.11-E1 | 提取 gfx946 target code object/ISA | code object SHA256=`5b7d8938b996c5eb65d5a5bb0a816736ae28959b220f6c8feae99d3a45ed8e82`，ISA SHA256=`9311eb5f5451053e6824bbba182cb85280e66c5bfe484f0cb564826d42f04568`；整个 target 含 `v_mmac=7168`、`ds_read=2702`、`ds_write=1008`、`buffer_load=3440`、`buffer_store=2748` | G2 actual V3 payload |
| P4.11-E2 | gfx936 focused build 与同一 correctness case | PASS；binary SHA256=`5b4125ad2392165199a7b14e49248952db65a11efeb8cec3f294c41f87aeb8bd`，两组 CPU reference correct；日志 SHA256=`928a7bad3ffafebdfb7004ac020c505dbcfd48a4c852d2715e9bc7a2e9cd6a68` | G4 gfx936 PASS |
| P4.11-E3 | gfx938 focused build 与同一 correctness case | PASS；binary SHA256=`4cbbb974d17100396a0cf27ac8522b13c7aa15daa0ba1e5c4559087f28272ec2`，两组 CPU reference correct；日志 SHA256=`335a57a514792f8f9f18bb3e032f083eca3c4ec553accce9fe1e110da8cc6783` | G4 gfx938 PASS |

PMD 仅报告一次 pending-cacheline warning，单次 V3 dispatch 与完整 reference 均正确。调优 provenance 不能单独证明架构不兼容；在三架构实际通过后保留该 target。该步骤没有源码或 binary candidate 变化，不存在 gfx936/gfx938 性能影响，故不触发性能 A/B；这里也不宣称该配置在 gfx946 有性能收益。

门禁结果：G0-G4 PASS；G5/G6 NOT RUN（validation-only）。下一切片继续 BF16 V3 reg-prefetch target；其 PMD 结论必须独立于 FP16。

### Step 4.12：`ck_tile/19_grouped_gemm` BF16 V3 reg-prefetch reproduction — BUILD-PASS / SKIP-PMD（2026-09-02）

- target：`tile_example_grouped_gemm_bf16_v3_regprefetch_256_w16_group8`，与 P4.11 相同 V3 tile/pipeline但输入输出为 BF16。正式 case 为 BF16 NT、group=2、每组 `M=N=256,K=128`、packed stride、`validate=1,warmup=0,repeat=0`。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.12-E0 | gfx946 focused build 与 PMD correctness | BUILD PASS，binary SHA256=`849c6375d7d24a60f3a6ec31274fe066750f84b7b64b6282b2c2d143b1311b93`；单次 V3 dispatch 后两组分别 8942/65536 错误（13.64441%），首错 flat index 129，max error=21440；日志 SHA256=`8e84643b75dfd0f89ecaf739ba9690fe8ae51e1f73dad1577f89137603bc256f` | G1/G2 PASS，G3 `SKIP-PMD`；未测试 PMD 性能 |
| P4.12-E1 | 提取 gfx946 target code object/ISA | code object SHA256=`bd63d05e61f8c09c2d4d813b4157048e21c782930871863fa05cc96bdcca6c45`，ISA SHA256=`e9f3f9539bc0d49b01f72c9967661e540ddb6f242041bee4900d13d67c4b72fb`；整个 target 含 BF16 `v_mmac=7168`、`ds_read=2702`、`ds_write=1008`、`buffer_load=3436`、`buffer_store=2854` | G2 actual V3 payload；排除空 kernel |
| P4.12-E2 | gfx936 focused build 与同一 correctness case | PASS；binary SHA256=`26381dd88c88c968979db307c087a6f453d49da0dd5ca030cf0d4e5354a258dd`，两组 CPU reference correct；日志 SHA256=`5e4c442feda656ccc2b9891f9a7ccb65a202333ca885229b5c455efd330cf2d5` | G4 gfx936 healthy control |
| P4.12-E3 | gfx938 focused build 与同一 correctness case | PASS；binary SHA256=`3edf2b87322bb2c0c519e6b1a7d8a4c98f91b03136922e72cf42a6168b1ff1ea`，两组 CPU reference correct；日志 SHA256=`f51fe035526014756b65833e28c1ad33267f49bbbfef6ccd6e8e292ed20b71b2` | G4 gfx938 healthy control |

该错误率/首错签名复现 P3.5b-E6 的 gfx946-only BF16 Compute V3 诊断结论，而相邻 FP16 V3 target 已在同一 PMD 正确；因此不能归因于 V3 pipeline 本身，只合并到 BF16 grouped accumulator-to-output/CShuffle PMD 边界。按任务规则不为模拟器改写两台真机正确的 kernel。本切片没有源码或 binary candidate 变化，可直接排除 gfx936/gfx938 性能影响，不触发性能 A/B。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN（validation-only）。下一切片为 FP16 MLS 256x256x32 single-stage reproduction target。

### Step 4.13：`ck_tile/19_grouped_gemm` FP16 MLS single-stage reproduction — PASS / architecture-scope fix（2026-09-02）

- target：`tile_example_grouped_gemm_fp16_mls_256_k32_single_stage`，固定 256x256x32、single-stage MLS、global-load overlap 与 skip-post-MMAC-sync。正式 case 为 FP16 NT、group=2、每组 `M=N=256,K=128`、packed stride、`validate=1,warmup=0,repeat=0`。
- baseline 显示 gfx946 PMD 与 gfx938 reference 正确，而 gfx936 输出几乎全零；该 target 的源码/CMake 注释本就把配置定义为 gfx938 MLS policy，Shaobo 又具备独立 gfx946 MLS traits。最小修复只在单架构 `GPU_TARGETS=gfx938` 或 `gfx946` 时创建本 FP16 target，不改 MLS implementation、config、kernel 或生产 selector。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.13-E0 | 三架构修改前 focused build 与同一 correctness case | gfx946 PASS，binary/log SHA256=`6ccf5be0ba7c555736d8c212e1e33add148711089553edbd2d455a6243423f77`/`560af4b2207324bb996df7d325ac2aeee35aa77e37dd2b807ab90f9307d49b5a`；gfx938 PASS，binary/log=`28d32eabe5c28ec9dfc42931b02288555ef11e1f63029e234c576b0925356210`/`25141d14b0466e3b312ddc250a117d890dda7a38e2d1c7c94c74f0d495f8551d`；gfx936 FAIL，65428/65426 个错误、约 99.8% 输出错误，binary/log=`a1a7fcbcebb1c633f31014b9972344e246f5a2b8c719f896b28b9434b3748909`/`31021826b4fe4d9a1053749ffecf9b548d459e6ddde19b87854a8e0fe6a37226` | G0 architecture boundary |
| P4.13-E1 | gfx946 PMD 实际 dispatch | PASS；`gemm_grouped_fp16_256x256x32_...` 单次 dispatch，grid=`24576x1x1`、workgroup=`512x1x1`，两组 CPU reference correct；PMD 报告 `matrix_load_32x32_b16`/`ds_read_matrix_trans` testing warning和一次 pending-cacheline warning | G1-G3 PASS；未测试 PMD 性能 |
| P4.13-E2 | 提取 gfx946 target code object/ISA | code object SHA256=`fc95439cb6cd8d46b8ca7ec695d298bcf213078c86d21115710488784be6f60e`，ISA SHA256=`bbe40fbcf938e41d50654943bb866e5402ed2e797bad4087406164cc1d430701`；整个 target 含 `v_mmac=10880`、`ds_read_matrix=2040`、`matrix_load=340`、`ds_read=3128`、`ds_write=1088` | G2 actual Shaobo MLS payload |
| P4.13-E3 | 同步 CMake 后 gfx946/gfx938 重生成、构建和运行 | 两侧仍 PASS；binary 分别保持 `6ccf5be0...3f77`、`28d32eab...6210`，与 baseline 逐字节相同；gfx938 日志仍为 `25141d14...551d`，gfx946 candidate 日志 SHA256=`5085eac42ea7290f683c6ff24c85b1bcc713a09c52824b9ef4acd0806417a4fe` | G4 supported-architecture regression PASS |
| P4.13-E4 | gfx936 重生成并精确请求 target | 按预期 `No rule to make target`、build rc=2，target help 中不存在该名字；旧 binary 仅是 build-tree 残留且未再执行 | G4 architecture-bound PASS |

候选只改变 example target 的 CMake enrollment；gfx946/gfx938 支持路径的 binary identity 直接排除 kernel 性能变化，gfx936 仅移除已证错误的 standalone reproduction target，不改变生产 library 或任何可运行 kernel。因此性能影响确定可排除，不触发真机性能 A/B。

门禁结果：G0-G4 PASS；G5/G6 NOT RUN（CMake test-scope only，supported binary identity）。下一切片为 BF16 MLS single-stage target，独立确认 Shaobo 的 MLS 与 BF16 output 组合边界。

### Step 4.14：`ck_tile/19_grouped_gemm` BF16 MLS single-stage reproduction — BUILD-PASS / SKIP-PMD / architecture-scope fix（2026-09-02）

- target：`tile_example_grouped_gemm_bf16_mls_256_k32_single_stage`；配置、shape 与 P4.13 相同，仅把输入输出切换为 BF16。修改前 target 同样无条件生成到 gfx936，故本步在完成三架构基线后把它限制为单架构 gfx938/gfx946；gfx946 虽受当前 PMD BF16 输出限制，仍保留其源码/构建能力，不把模拟器失败误写成真实 Shaobo 不支持。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.14-E0 | gfx946 focused build 与 PMD correctness | BUILD PASS，binary SHA256=`e2e1d7efc259c3214fd9d485494410335be47574665fa4b81f55a84cb51f3f91`；单次 MLS dispatch，输出后半区从 flat index 32768 起错误，两组各 3918/65536（5.978394%），max error=5760；日志 SHA256=`8e9f9eadf3f78b696ac59240940c27e0e803e92e5a3153d2bb38d572f7b2eaa8` | G1/G2 PASS，G3 `SKIP-PMD`；未测试 PMD 性能 |
| P4.14-E1 | 提取 gfx946 target code object/ISA | code object SHA256=`098f78b83f6cfe6e3221df289ad86bb1a1174583eaea668bebe35c3728ad05c7`，ISA SHA256=`e2ab85cc912856139645e1bcd53d9e5ee9043e82fd5a5664a8adc214b1d32432`；整个 target 含 BF16 `v_mmac=10880`、`ds_read_matrix=2040`、`matrix_load=340`、`ds_read=3256`、`ds_write=1088` | G2 actual BF16 MLS payload；排除空 kernel |
| P4.14-E2 | gfx936/gfx938 修改前同一 correctness case | gfx936 FAIL，两个 group 各 62622/65536（95.55359%），binary/log SHA256=`5dc496fdce576a81233c5a23d0c970861c33e175825e0c8df22cb53ddcc51212`/`1efbc5e251235b5bcb19eed98ba220b72ae3ed34e7ece6238eda1312d3d066af`；gfx938 PASS，binary/log=`8fe9d0900f2032cb508fbb34150613004b02944b294b7de25efa0c277b21fd61`/`8ba4229dd4a3d8a80e6b2b7bd56893c9b10fcc32b6035dfc03561ba3db285c51` | G0/G4 architecture boundary |
| P4.14-E3 | 同步 CMake 后 gfx946/gfx938 重生成 | gfx946 target 保留；host binary 因重链变为 `be89127b4f862f2aa45fa2104a01f5723e4e4e7e61d5a22f6ddf9215a356420e`，但提取的 code object 仍为 `098f78b...05c7`、逐字节相同；gfx938 binary/log 均保持 baseline 值并再次 reference PASS | G4 supported-architecture regression PASS；device payload identity |
| P4.14-E4 | gfx936 重生成并精确请求 target | 按预期 `No rule to make target`、build rc=2，target help 中不存在该名字 | G4 architecture-bound PASS |

相邻 FP16 MLS 已在 gfx946 PMD 正确，故本步不能归因于 `matrix_load`/`ds_read_matrix_trans` 整体缺失；错误只合并到已有 BF16 grouped accumulator-to-output/CShuffle PMD 边界。CMake 变更不修改 kernel，gfx946 code object 与 gfx938 binary identity 直接排除支持架构性能变化；gfx936 只移除已证错误的 standalone reproduction target，因此不触发性能 A/B。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN（CMake test-scope only，device payload identity）。下一切片继续 grouped-GEMM FP8/BF8/INT8/INT4 fast target 的逐 dtype 验收。

### Step 4.15：`ck_tile/19_grouped_gemm` FP8 V5 fast — BUILD-PASS / SKIP-PMD / architecture-scope fix（2026-09-02）

- target：`tile_example_grouped_gemm_fp8_fast`，FP8 A/B、FP32 accumulate/output、128x128x128 Compute V5 persistent config。它不同于 P3.5c 的“FP8/BF8 + scale + FP16/BF16 output”quant example，必须独立验收。
- 首次 gfx946 构建失败：普通 `kbatch=1` target 仍运行时实例化 Split-K 的 `atomic_add<float,4>`，而公共 atomic contract 只实现 float1/float2。最小修复对 FP32/INT32 output 的 `splitk=true` 明确抛出 unsupported，不再编译不存在的 vector atomic；`splitk=false` 的 set kernel/template/launch 不变。该闭包也适用于后续 BF8/INT8/INT4 fast target。
- 正式 case 为 FP8 NT、group=2、每组 `M=N=128,K=256`、packed stride、`validate=1,warmup=0,repeat=0,kbatch=1`。K=128 在现有 V5 pipeline 上每 split K-loop 不足，已作为参数预检拒绝，不计架构失败。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.15-E0 | gfx946 修改前 focused build | FAIL；`generic_memory_space_atomic.hpp` 静态断言 `atomic_add<float,4>` 未实现，来自未使用的 Split-K CShuffle 分支 | G1 shared compile closure |
| P4.15-E1 | 裁剪未实现 atomic 实例后 gfx946 focused build/PMD correctness | BUILD PASS，binary SHA256=`dd6a8361d51ca9e3916469199a43bb8856b5267e9ce549eb216cd233aa1428fd`；单次 V5 dispatch 后两组各 935/16384 错误（5.706787%），首错 flat index 8256、max error=40.64813；日志 SHA256=`4a9baf8c8cb827b8e3dfbdebb3263b538695b597de1ccc9d8a6f49dc86b47b33` | G1/G2 PASS，G3 `SKIP-PMD`；未测试 PMD 性能 |
| P4.15-E2 | gfx946 code object/ISA | code object SHA256=`fd9d492c104f665b0bb4affa9bf571d98945c104ed3a9648783ef9d499c0128b`，ISA SHA256=`578dd7f5807e95033bd13a42ec4f5d23fd46db07ddbf574cfdfb7969c2677595`；含 FP8 `v_mmac=604`、`ds_read_b96=252`、`buffer_load=28518`、`buffer_store=22640` | PMD 同时对多条 `ds_read_b96` 报 non-aligned warning，记录为该 V5 模拟执行边界 |
| P4.15-E3 | gfx936/gfx938 candidate 同一 correctness case | gfx936 FAIL，两组各 3772/16384（23.02246%），binary/log SHA256=`e857307c2163cc4432d2514f5724c9d381be07ed4b355df984ce4cfdbd1b7ec2`/`fb821ff42d8845215fff0fe2ec2637ddccee45f732e242382471c257c2bbb807`；gfx938 PASS，binary/log=`969b3638fb105065a3875d73346b416759e80ab84b315f689ff15d22ed5cf4aa`/`23bb37f5db90f48078f93f3343389f8d69e0a06af91a1638c89b553223ed3003` | G4 architecture boundary；gfx938 healthy control |
| P4.15-E4 | gfx938 `K=512,kbatch=2` unsupported contract | 按预期返回 1，明确报 `fp32/int32 grouped_gemm output does not support SplitK vector atomic`；日志 SHA256=`dad1f56a85e00df194d9f94107cdc30f127d41006917947805d01ecdb250fee7` | host contract PASS，无 kernel dispatch |
| P4.15-E5 | CMake 只在单架构 gfx938/gfx946 创建 FP8 fast target | gfx936 重生成后精确请求 target 返回 No rule、target help 不含该名字；gfx938 target/binary 保持并再次 reference PASS；gfx946 target 保持，重链 host binary=`a590a921ce4570e8e03067059d97c49ebc8276d188e077e964f7ae5ff0e9c9b5`，code object SHA 保持 `fd9d492c...128b` | G4/G6 architecture-scope and device-payload evidence |

gfx938 健康控制说明 PMD 错误不是 host reference 或 V5 跨架构共同失败；但 gfx936 也不健康，因此只把 target 从 gfx936 移除，不扩大为通用 FP8 支持。候选不会改变 `kbatch=1` 的 set kernel类型与 launch，gfx946 code object identity；gfx938 同一 correctness path 通过。性能影响可由未选 atomic template 的编译期裁剪和 target enrollment 边界直接排除，不触发性能 A/B。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN（device code identity/compile-time-only closure）。下一切片为 BF8 fast target。

### Step 4.16：`ck_tile/19_grouped_gemm` BF8 fast — SKIP-PMD / architecture-scope fix（2026-09-02）

- target：`tile_example_grouped_gemm_bf8_fast`，BF8 A/B、FP32 accumulate/output；正式 case 与 P4.15 对齐：NT、group=2、每组 `M=N=128,K=256`、packed stride、`validate=1,warmup=0,repeat=0,kbatch=1`。
- P4.15 已完成公共 FP32/INT32 output 的非 Split-K 构建闭包，本步不再改 kernel/dispatch 源码，只验证 BF8 target 并收紧已被 correctness 否定的 gfx936 enrollment。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.16-E0 | gfx946 focused build/PMD correctness | BUILD PASS，修改前 binary SHA256=`6739b3d45bcc6d8a20fb4e3f8c411d17736815fc7c6cfbc8ca9b628545eecffd`；两组各 55/16384 错误（0.3356934%），首错 flat index 10320、max error=40.36877；日志 SHA256=`ebc09b344fd72ecd82405368c99a0eed2bb24c0fbd1bd5ffbca05e9485be146e` | G1/G2 PASS，G3 `SKIP-PMD`；未测试 PMD 性能 |
| P4.16-E1 | gfx946 code object/ISA | code object SHA256=`42acebf6a3b405e970c0023929edd4da303fb5b47e3e05e7ddc06eb33739bfa8`，ISA SHA256=`270b1862b7e9e4456acdc79d2b8e68d2aa4711eaa44ad3e40303f7bc729c6f22`；含 BF8/F8 MMAC 17024 条、`ds_read_b96=162`、`buffer_load=28480`、`buffer_store=22741` | PMD 对该 dispatch 的 `ds_read_b96` 报 non-aligned warning，沿用 V5/V2 模拟执行边界分类 |
| P4.16-E2 | gfx936/gfx938 同一 correctness case | gfx936 FAIL，两组各 272/16384（1.660156%），binary/log SHA256=`22c65f5fa4e58505e2ce8852feaa45af6dbbe03842795a6035217f8704b59ac4`/`1e908ab33a1ba751de1405f3f74f441c7d38d5b734acddbc055825d1fb795a4f`；gfx938 PASS，binary/log=`6b8919ca8d54ce5534ef2237629e61bd92236ef297c61e4b7ca8efb48203d3aa`/`ca64a4986071238ce8c0caf344c1c4223d22041c2cc52126cad2f4aa447ffcf7` | G4 architecture boundary；gfx938 healthy control |
| P4.16-E3 | CMake 只在单架构 gfx938/gfx946 创建 BF8 fast target | gfx936 重生成后精确请求 target 返回 No rule、target help 不含该名字；gfx938 target/binary 与 reference log SHA 均保持；gfx946 target 保持，重链 host binary=`6f9789652bdeef74b597a85de89dc96e2fd4567d263e78293d62b5bd49df4bc4` | G4/G6 architecture-scope evidence |

gfx938 健康控制说明 BF8 算法与 reference contract 正常，gfx946 的少量错误与 `ds_read_b96` PMD 告警相关，按模拟器能力问题跳过；gfx936 的真机失败则通过 CMake 精确移除该 target。唯一候选变更是 target enrollment 条件，gfx938 binary 与运行日志逐字不变，故可直接排除 device 性能影响，不触发性能 A/B。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN（CMake-only enrollment and gfx938 binary identity）。下一切片为 INT8 fast target。

### Step 4.17：`ck_tile/19_grouped_gemm` INT8 fast — real-GPU PASS / SKIP-PMD（2026-09-02）

- target：`tile_example_grouped_gemm_int8_fast`，INT8 A/B、INT32 accumulate/output；正式 case 为 NT、group=2、每组 `M=N=128,K=256`、packed stride、`validate=1,warmup=0,repeat=0,kbatch=1`。
- kernel source 经 `WarpGemmAttributeMmacImplI8I8I32M16N16K32` 调用 `__builtin_hcu_mmac_i32_16x16x32_i8`。本步不修改 kernel 或 target enrollment，只建立真实 GPU 与 PMD 的能力边界。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.17-E0 | gfx946 focused build/PMD correctness | BUILD PASS，binary SHA256=`fc3de30f212bb8fd15c9dcff22ffc9011273d4d96511566cfb7f078b8e8b24bd`；两组各 16380/16384 错误（99.9756%），输出几乎全 0，max error=38562；日志 SHA256=`786b388c4ef018f3f302c365f8d1ca9e8fafee2714c1fb6ac364780d65e475c3` | G1/G2 PASS，G3 `SKIP-PMD`；未测试 PMD 性能 |
| P4.17-E1 | gfx946 capability/ISA evidence | PMD 启动横幅明确为 `deviceid:0x66b7(gfx946), dp4x:disable`；code object/ISA SHA256=`1cb3c3b8437854d534c62f5c4f8ef3d8e2f0e833899961df250d4e417f908d1d`/`cac5f8a4d2fa9fb4b46652aefdd20632d7585e9be41639e43b22650a02e4e11d` | source 明确使用 I8 MMAC，而当前 PMD 关闭整数点积模拟；不尝试用错误 fallback 改写真实 kernel |
| P4.17-E2 | gfx936/gfx938 同一 correctness case | gfx936 PASS，binary/log SHA256=`262a25746fc126fa296bdefc92f6d481047fa7ab5330344b01f5d047de81686c`/`d901fb2167d58a78f1edbef92a02e87459666b29bcda51bfbf5f64352f7aa77c`；gfx938 PASS，binary/log=`fdaff74aa74a80f383e390899f2b2d75c2afd467f62ac74a52617ffbb5be721f`/`d0172ee998fd55caa3d24537a6c4699ced3b584555f40d685a010ab5eb552a88` | G4 PASS；双真机 healthy control |

两种真实架构均逐元素 exact PASS，证明公共 INT8 kernel/reference contract 正常；gfx946 的全零型失败与 PMD 明示 `dp4x:disable` 一致，记录为无法在当前 Perf Model 解决的指令模拟缺口并跳过该 PMD subpath。没有候选源码变更，也就没有 gfx936/gfx938 性能不确定性，不触发性能 A/B。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN（evidence-only）。下一切片为 INT4 fast target。

### Step 4.18：`ck_tile/19_grouped_gemm` INT4 fast — real-GPU PASS / SKIP-PMD（2026-09-02）

- target：`tile_example_grouped_gemm_int4_fast`，packed INT4 A/B、INT32 accumulate/output；正式 case 为 NT、group=2、每组 `M=N=128,K=512`、`stride_A=stride_B=512,stride_C=128`、`validate=1,warmup=0,repeat=0,kbatch=1`。
- `K=256` 在三端一致被现有 pipeline 的 per-split K-loop 参数预检拒绝，因此不计架构失败；提高到 `K=512` 后进入同一 kernel dispatch。source 使用 `__builtin_hcu_mmac_i32_16x16x64_i4`。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P4.18-E0 | gfx946 focused build/PMD correctness | BUILD PASS，binary SHA256=`09511c3ef6172994e2b0bf3c9ada88baf45434f14aee32866c61e785686bb983`；group0 16363/16384 错误（99.8718%）、group1 16370/16384（99.9146%），输出几乎全 0，max error=2093；日志 SHA256=`d579cc5119916d949cd186d9a8cc150819607b28f18362dc8edafe976563e559` | G1/G2 PASS，G3 `SKIP-PMD`；未测试 PMD 性能 |
| P4.18-E1 | gfx946 capability/ISA evidence | PMD 横幅同样明确 `deviceid:0x66b7(gfx946), dp4x:disable`；code object/ISA SHA256=`1561fc3b0586ddb092d1fdd469b98edaa073be4045953ee0893b674e3ed65c5b`/`31030d52c7e7f065b8de4d3e4c5c541d53d15876adfb70363bdd96585771a0af` | source 使用 I4 MMAC，而当前 PMD 关闭整数点积模拟；与 P4.17 同类指令缺口 |
| P4.18-E2 | gfx936/gfx938 同一 correctness case | gfx936 exact PASS，binary/log SHA256=`561d1be65642046238248fd9ee074d289e4dfc4cbbd1b4f5bdadf80394a7c6eb`/`8944d4da7783c56a71a824679c66a9bdbbf2d0f4283ae2121be304a755a3e2f5`；gfx938 exact PASS，binary/log=`e8cc07734b8bb7234023d9c85f77b3196cfffb7318e6263f485474a0e2422f2f`/`5db821491f21a9d3b55ea09f03b924d8a509b212e964e06e497aeb2d8ca0d4b8` | G4 PASS；双真机 healthy control |

两种真实架构均逐元素 exact PASS，证明 packed INT4 kernel/reference contract 正常；gfx946 的全零型失败与 PMD 明示 `dp4x:disable` 一致，记录为无法在当前 Perf Model 解决的整数 MMAC 模拟缺口并跳过该 subpath。没有候选源码变更，也没有 gfx936/gfx938 性能不确定性，不触发性能 A/B。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN（evidence-only）。下一步继续 actual enabled-target coverage diff。

### Step 4.19：gfx946 actual enabled-target coverage closure（2026-09-02）

- 在最新 PMD build tree 重新执行 `cmake --build build_shaobo_step1 --target help`。结果仍为 raw 137 行、筛选后 94 个 `example_*`/`tile_example_*`/`ckProfiler` target；raw/filtered SHA256 仍分别为 `ca7e549a3a3750f17eb37cc6192e7fccdce46ee08fbe4e1bd6f718732649165e`/`0a2c3bd15d2ea64201d8f64338a74e92c8c9da05e9680c6b43e01781310a9e02`，说明 P4.15/P4.16 的 gfx936 scope 修正没有误删 gfx946 target。
- 将 94 个名字与本文精确字符串比较，只有 14 个名字未原样出现；逐项回查 CMake 和既有门禁后，没有新的独立 kernel：
  - 聚合/custom smoke：`example_gemm_mmac_bias_activation`、两个 grouped-conv bias aggregate、batched multi-D smoke、`example_hcu_gemm_xdl_v3`、两个 grouped-GEMM aggregate smoke，共 7 个；其 dependency executable 与对应 correctness 矩阵已在 P2.5/P2.6、P2.17/P2.18、P2.20/P2.21、P3.5a-c 验收。
  - 已验收但正文使用模块名描述：`tile_example_fmha_fwd` 与两个 `tile_example_fused_conv_mls_wasp{,_bf16}_probe`，共 3 个，对应 P4.1/P4.2。
  - P2.21 共享 BF16 XDL v3 module sibling：`multi_d`、`multi_abd`、`add`、`add_relu` 四个 executable target 未逐名重复写入；它们与 basic target 共享已确认失败的 BF16 accumulator-to-output/CShuffle PMD 链和相同 gfx946 guard，已按模块 `SKIP-PMD`。
- 上述分类合计 14 个 target 名；没有把 custom aggregate 重复计算为新 example，也没有用一个支持 dtype 替代另一个 dtype。`ckProfiler` 是构建工具，不属于 `example_hcu` correctness example，保持在 inventory 中但不作为新增 kernel 验收项。

本步为只读 build-graph/文档映射，没有源码、binary 或架构行为变化，故不运行 gfx936/gfx938 correctness 或性能 A/B。门禁结果：enabled-target coverage 94/94 mapped；下一步转向 `example_hcu` 中未进入当前 gfx946 target graph 的源码/CMake 路径，区分“未 enrollment”“注释/禁用”和“无独立 executable”。

### Step 4.20：not-enrolled source/CMake inventory and priority（2026-09-02）

- `example_hcu/CMakeLists.txt` 通过 glob 加入现存全部顶层目录；`example_hcu/ck_tile/CMakeLists.txt` 实际加入 14 个现存子目录。文件中仍注释的 `04_img2col/05_reduce/06_permute/09_topk_softmax/12_smoothquant` 在当前 HCU checkout 并不存在，只存在于 AMD 参考树，属于后续跨树 selective port，不能简单解除注释。
- 对所有现存 `CMakeLists.txt` 扫描注释的 `add_example_executable/add_executable`，共 36 条：`01_grouped_conv_fwd_mmac` 8、`02_grouped_conv_bwd_data_mmac` 6、`05_gemm` 6、`07_grouped_conv_fwd_bias_activation` 4、`08_batchnorm` 3、`ck_tile` 9（basic GEMM mem-pipeline 1、fused-conv 8）。除 `08_batchnorm` 三个 NHWC runner 实际位于同仓 `example/34_batchnorm` 外，其余对应 `.cpp` 均在各自目录存在；没有发现源码完全不存在的声明。
- 优先级按最小依赖与命名冲突排序：
  1. 先逐个启用 `05_gemm` 六个 target；它们有独立 target 名、既有 FP16 MMAC reference 与已验证的同目录控制组。
  2. 再处理 grouped-conv forward/backward-data 和 bias 四类注释 target；按 layout/dtype 独立门禁。
  3. batchnorm NHWC 三个声明与已启用 NCHW target 重名，fused-conv 八个声明也共用同一 target 名；必须先给出独立 target 命名，不能批量取消注释。
  4. 最后按 `04_img2col -> 05_reduce -> 06_permute -> 12_smoothquant -> 09_topk_softmax` 建立 AMD-to-HCU 最小依赖闭包；不把参考树目录整体复制视为适配完成。

本步只更新实施队列，没有 device 源码变化，不运行回归或性能 A/B。下一切片 P5.1 为 `example_gemm_mmac_tt_fp16`，先建立 target enrollment，再在 gfx946/gfx936/gfx938 做 focused build 与 correctness。

### Step 5.1：commented `05_gemm` TT v1 FP16 — diagnostic BUILD-PASS / SKIP-PMD（2026-09-02）

- 临时只启用 `example_gemm_mmac_tt_fp16`，并显式重生成三棵原 build tree；正式 shape 为 `M=128,N=512,K=256`、packed row-major strides、`verification=1,init=1,time=0`。
- 初始 gfx946 binary 可 dispatch 但输出全 0，定位到 `gridwise_gemm_mmac_tt_mk_n0k0k1n1_mn_v1.hpp` 的 device body guard 未含 `__gfx946__`。诊断只把 gfx946 加入该 guard；gfx936/gfx938 的预处理路径不变。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.1-E0 | 三架构临时 enrollment/focused build | PASS；gfx946/gfx936/gfx938 binary SHA256 分别为修改 guard 前 `844a97a4810fea5cbd4f175c4d8d21fb595b49d5aa76cb5d674de897bbdcec5c`、`c205b0c31ca758831a36e02fe2e8055799cd8007ee5938126de2f3e8e2e09202`、`0b06633e6c49e4844683306bf411541abf7d18b086527db731fe26d6190116ea` | G1/G2 diagnostic build PASS |
| P5.1-E1 | gfx946 原 guard correctness | dispatch 后输出全 0、max error=560，日志 SHA256=`1197d3ba756a3f87642bfc6a6874163cb47725ccb3d6f7b08d0d27ecf5d2f63c` | empty-kernel control |
| P5.1-E2 | gfx946 临时放开 guard 后同 shape | 真实 dispatch，但输出稳定为 NaN；candidate binary/log SHA256=`2a529d2ca2ae684fe0c2f01bc1335f22d439be09515d4f5dea29618c44e7abd2`/`adf07aa6b7aeb3c7be3e0d568c82e96fd6a56bca8c7a27b15d9d7663a4d563b8` | G3 `SKIP-PMD` |
| P5.1-E3 | gfx946 第二 shape `M=256,N=512,K=256` | 同样真实 dispatch、输出稳定为 NaN；日志 SHA256=`a826cb53a8d110d5c3c2f61c737dc6d13efee168faee03c14fbaf931adbd33dd` | 排除单一 M/grid 大小偶发问题 |
| P5.1-E4 | gfx946 code object/ISA | code object/ISA SHA256=`9cd4e5062472cd7d3747b08918ac50e78efb716fe207ae72079215ea615d6e0a`/`46355fa4d4dc3c2abcadbfa4088e571e5ef545613a61c7577c4d81cc9b22c45d`；含 24 条 `v_mmac_f32_16x16x16_f16`、6 条 `ds_read_m32x16_b16_alt`、8 条普通 LDS read 与 16 条 global store | 排除空 kernel；命中 TT v1 MMAC/layout 链 |
| P5.1-E5 | gfx936/gfx938 同 shape、guard 前后 | 两架构均完整 reference PASS；各自 binary SHA 在 guard 前后保持 `c205b0c3...9202`/`0b06633e...6ea`，日志均为 `8e59fff37466fe3fd7f3c999fac41aea92a475dedd72f612956ced06b1563f93` | G4 PASS；direct architecture isolation |
| P5.1-E6 | 撤回 enrollment 与 gfx946 guard，三棵 build tree 重生成 | 三端 target help 均不再包含 `example_gemm_mmac_tt_fp16`（`TARGET_PRESENT=0`） | 未保留未通过 correctness 的架构声明 |

该 ISA 中的 FP16 MMAC 与 `ds_read_m32x16_b16_alt` 在其他 gfx946 PMD example 已各自通过，因此不能把问题归因于单一 opcode；当前只把旧 TT v1 的 transposed-C/layout/pipeline 组合标记为 PMD 执行缺口。按任务规则不改写真实 gfx936/gfx938 已正确的 kernel，撤回诊断代码并跳过本 target。最终没有 device 源码差异，真机也不存在性能变化面，不触发性能 A/B；PMD 全程 timing off。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN。下一切片继续 commented `example_gemm_mmac_tn_fp16`。

### Step 5.2：commented `05_gemm` TN v1 FP16 — diagnostic BUILD-PASS / SKIP-PMD（2026-09-02）

- 临时只启用 `example_gemm_mmac_tn_fp16`；正式 shape 为 `M=128,N=512,K=256`、A row-major/B column-major/C row-major、`verification=1,init=1,time=0`。
- gfx946 初始输出全 0，原因是 `gridwise_gemm_mmac_tn_mk_nk_mn_v1.hpp` 的 device body guard 未含 gfx946。只在诊断中放开该架构 guard；gfx936/gfx938 的预处理结果不变。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.2-E0 | 三架构临时 enrollment/focused build | PASS；gfx946/gfx936/gfx938 初始 binary SHA256=`8fb97358f378f354ac4736f1825fb2c0e4d5583bc2afbe1894e9477091894d6a`/`8c83ada286ea3ed84a8cca1fde358ff2086b74d84fc7b0232bfc8cfbd3b0ffe5`/`de0a26e866c17463b08f64af437247205202c9900dbf17ead48f5dc3645f8551` | G1/G2 diagnostic build PASS |
| P5.2-E1 | gfx946 原 guard correctness | dispatch 后输出全 0、max error=2472，日志 SHA256=`0cdb6095f93ca52d888c63a692ccf32bac094b78990dd6f98f1faf4f205e0e75` | empty-kernel control |
| P5.2-E2 | gfx946 临时放开 guard 后同 shape | 真实 dispatch，但输出稳定为 NaN；candidate binary/log SHA256=`cb103e10186def367447029b7d69af00b0da91f0eeb9fb2a6f0ab6e1787ffd02`/`ed795d6cca91bc54a130f6ba9cb9f7db53ff9abd91572deb6bff474847c0187d` | G3 `SKIP-PMD` |
| P5.2-E3 | gfx946 code object/ISA | code object/ISA SHA256=`0fe5008fe5d8b179147353ae9255cba0e536755e56b5f209b92e831dd03a860b`/`44c0aa6b8742adfa9cce0c728c629580b4d6024bc0eba57bfda10055d6a17477`；含 24 条 `v_mmac_f32_16x16x16_f16`、6 条 `ds_read2_b64`、12 条 `ds_read_b64` 与 16 条 store，不含 matrix-read | 排除空 kernel，也排除 P5.1 的 `ds_read_m` 单点解释 |
| P5.2-E4 | gfx936/gfx938 同 shape、guard 前后 | 两架构均完整 reference PASS；binary 分别保持 `8c83ada2...ffe5`/`de0a26e8...8551`，日志均为 `e9ae9a2c15e84ecc1a4e2ef2462b1d849708b30fbfbc53992b6503b0b380bd93` | G4 PASS；direct architecture isolation |
| P5.2-E5 | 撤回 enrollment 与 gfx946 guard，三棵 build tree 重生成 | 三端 target help 均返回 `TARGET_PRESENT=0` | 未保留未通过 correctness 的架构声明 |

TN v1 不使用 `ds_read_m` 仍与 TT v1 一样产生全 NaN，且已启用的 TN v2/CShuffle 路径在同一 PMD 正确，因此不能扩大为 FP16 MMAC 或 TN layout 普遍不支持。当前把问题限定为旧 v1 pipeline/layout 组合的 PMD 执行缺口，撤回诊断并跳过。最终无 device 源码差异，真机无性能变化面，不触发性能 A/B；PMD 未测试性能。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN。下一切片继续 commented `example_gemm_mmac_tt_v2_fp16`。

### Step 5.3：commented `05_gemm` TT v2 FP16 — SKIP-BASELINE（2026-09-02）

- 临时启用 `example_gemm_mmac_tt_v2_fp16`，同时只在对应 v2 wrapper guard 加入 gfx946；正式 shape 使用 runner 默认矩阵 `M=128,N=512,K=256`、packed row-major strides、`verification=1,init=1,time=0`。
- 三端均完成 focused build。gfx946 binary/log SHA256=`2c4462a6cb493a9b3c650aa252ede270b439409bbca0ef7a2aa9d4a9868fb786`/`f8f6530efaa2052d66e244116301a5a4ea07a2be64a95223c0918bdfba832a4b`，真实 dispatch 输出 NaN；但在归因 PMD 前，两个真实架构已先否定该 example baseline。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.3-E0 | gfx936 同一默认 correctness case | FAIL；从 flat index 2048 起稳定错误，max error=869；binary/log SHA256=`43060d564be4dbb52ef00f586cf1a8330584bf3579ddbf92dffadbe4dc48156a`/`7dabcb7fea972ed684bb288de66d1669f49ba365937f775b67ef5838e8a843bf` | G0 healthy-control FAIL |
| P5.3-E1 | gfx938 同一默认 correctness case | FAIL；错误位置、数值和 max error 与 gfx936 完全相同；binary/log SHA256=`e908d31925293e7fef42384e919e609e2d04d05f3da8c69c07f6aab1e8af7705`/`7dabcb7fea972ed684bb288de66d1669f49ba365937f775b67ef5838e8a843bf` | G0 healthy-control FAIL；`SKIP-BASELINE` |
| P5.3-E2 | 撤回 enrollment 与 gfx946 guard并重生成 | 三端 exact target 从 target help 消失；同前缀的 `_cshuffle/_cshuffle_nm` 已启用 target 保持 | 不保留跨架构错误 example |

双真机一致失败证明该长期注释 target 的默认配置本身不健康，不能把 gfx946 输出归为 Shaobo/PMD 缺陷，也不应在本适配中顺带重写跨架构 TT v2 算法。最终源码完全撤回，故无 regression/performance candidate；不做真实性能 A/B，PMD 未测试性能。

门禁结果：G0 FAIL；G1/G2 PASS；G3 非归因；G4 FAIL；overall `SKIP-BASELINE`。下一切片继续 commented wavelet target，仍先以 gfx936/gfx938 健康控制决定是否进入 Shaobo 诊断。

### Step 5.4：commented `05_gemm` TN v2 FP16 wavelet — SKIP-BASELINE（2026-09-02）

- 临时只启用 `example_gemm_mmac_tn_v2_fp16_wavelet`，并仅在对应 wavelet wrapper guard 中临时加入 gfx946；正式用例为 `M=128,N=512,K=256`、A row-major/B column-major/C row-major、strides `256/256/512`、`verification=1,init=1,time=0`。
- 三端 focused build 均成功。gfx946 确实进入 `DeviceGemm_mmac_tn_mk_nk_mn_v2_wavelet` dispatch，但两个真实架构先在完全相同的输出位置和值上失败，故停止 Shaobo 指令归因。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.4-E0 | gfx936 正式 correctness case | FAIL；从 flat index 4 起错误，max error=772；binary/log SHA256=`7c8d7f2a3059862ad3f77ec5390c49d46554c919c9c7b29932ab885576ca6efa`/`9c2699b4125b21f6921c796dba7a3851e5513b15f0ba1a7edee09bc1fc8ee14b` | G0 healthy-control FAIL |
| P5.4-E1 | gfx938 同一正式 case | FAIL；错误位置、错误值与 max error 均和 gfx936 完全相同；binary/log SHA256=`bf8883839fb544c0ec78cc846b6743f8ca8a939004bc4c686cf193e099d7686b`/`9c2699b4125b21f6921c796dba7a3851e5513b15f0ba1a7edee09bc1fc8ee14b` | G0/G4 FAIL；`SKIP-BASELINE` |
| P5.4-E2 | gfx946 诊断 run | 真实 dispatch 后输出 NaN；binary/log SHA256=`c3ae61696f6708feb03a553bb2a45c1f9f66b4b441850b8c0d8663d5afa43350`/`f95761384567ae2aaf52fd138b38a3935bd1848bef4fdc82e7cd0c4cb1edeedd` | 不归因 PMD：健康控制已失败 |
| P5.4-E3 | 撤回 enrollment 与 gfx946 guard并重生成 | 三端 `target help` 均确认 exact target 不存在（`TARGET_PRESENT=0`） | 不保留跨架构错误 example |

gfx936/gfx938 的逐项错误与日志哈希完全相同，证明长期注释的 TN wavelet 默认配置在两个真实架构上具有一致的既有正确性缺陷。本适配不扩展为跨架构 wavelet 算法修复；临时源码将全部撤回。最终无 device/binary 性能候选，因此不做真实性能 A/B；gfx946 Perf Model 全程关闭计时。

门禁结果：G0 FAIL；G1/G2 PASS；G3 非归因；G4 FAIL；overall `SKIP-BASELINE`。三端撤回确认完成；本步单独提交本文，下一切片继续 commented TT v2 wavelet。

### Step 5.5：commented `05_gemm` TT v2 FP16 wavelet — SKIP-BASELINE（2026-09-02）

- 临时只启用 `example_gemm_mmac_tt_v2_fp16_wavelet`，正式健康控制为 `M=128,N=512,K=256`、A/B/C row-major、strides `256/512/512`、`verification=1,init=1,time=0`。
- 遵循真实 GPU 基线优先原则，先在 gfx936/gfx938 focused build/run；两台机器逐项输出完全一致的错误，因此不再放开 gfx946 guard，也不启动 Perf Model，避免把既有跨架构缺陷误记为 Shaobo/PMD 问题。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.5-E0 | gfx936 正式 correctness case | FAIL；从 flat index 2048 起错误，max error=869；binary/log SHA256=`7ea099e616d79dbe48847ef3caba77b0e8051008d77f375ed7813dc0e678884e`/`d0bf514cc35d16d8d01c4bc7a4e4c3fa8bfd7b65116e4137d8b80bc97086400f` | G0 healthy-control FAIL |
| P5.5-E1 | gfx938 同一正式 case | FAIL；错误位置、错误值与 max error 均和 gfx936 完全相同；binary/log SHA256=`e56ad3aad7639d852bfd8dbb129301dc53a7b7038a41d75cb64c39dfc70bc49b`/`d0bf514cc35d16d8d01c4bc7a4e4c3fa8bfd7b65116e4137d8b80bc97086400f` | G0/G4 FAIL；`SKIP-BASELINE` |
| P5.5-E2 | gfx946 PMD | NOT RUN；健康控制已双双失败 | 不作 Shaobo 指令归因 |
| P5.5-E3 | 撤回临时 enrollment并重生成 | gfx936/gfx938 均确认 exact target 不存在（`TARGET_PRESENT=0`）；PMD 从未同步本次 enrollment | 恢复原始注释状态 |

该错误模式也与 P5.3 TT v2 非 wavelet target 相同：均从 flat index 2048 起偏离且 max error=869。当前证据指向 TT v2 系列既有配置/实现问题，而非 gfx946 适配差异；本任务不顺带重写跨架构算法。最终不保留 device 或 build-graph 变化，故无真实性能 A/B；Perf Model 未运行。

门禁结果：G0 FAIL；G1/G2 PASS；G3 NOT RUN；G4 FAIL；overall `SKIP-BASELINE`。双真机撤回确认完成；本步单独提交本文，下一切片继续 commented NT FP16 wavelet。

### Step 5.6：commented `05_gemm` NT v1 FP16 wavelet — diagnostic BUILD-PASS / SKIP-PMD（2026-09-02）

- 临时只启用 `example_gemm_mmac_nt_fp16_wavelet`；正式 shape 为 `M=128,N=512,K=256`、A column-major/B row-major/C row-major、strides `128/512/512`、`verification=1,init=1,time=0`。
- gfx936/gfx938 健康控制先通过，之后只在 `gridwise_gemm_mmac_nt_m0k0k1m1_n0k0k1n1_mn_v1_wavelet.hpp` 的 device guard 临时加入 gfx946。该条件不改变两种真实架构的预处理路径。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.6-E0 | gfx936/gfx938 guard 前健康控制 | 两架构均 reference PASS；binary SHA256=`764a5d2c8e9cbe8a38b58a0459d66d8f3fad7e4752a5304320aeb429c0e04167`/`c41ba2e6c320c058829a3272303ba87b5be1398d82affaf03610d33f720e15dc`，日志 SHA256 均为 `1fa910c3d0eca346809628052cf1c5054ba32d96267b11e80255456827cb6c59` | G0 healthy-control PASS |
| P5.6-E1 | gfx946 guard candidate，正式 shape | 真实 `DeviceGemm_mmac_nt...v1_wavelet` dispatch，输出 NaN；binary/log SHA256=`b6276a420716a3dddc4c84d01d595ce2775cf7f3ecd01c9e9521e283986338ed`/`c8c337f82d8068aad33f28d4e6cbdead94d155faec8cde0e6f97b0a7ace8eb70` | G3 `SKIP-PMD` |
| P5.6-E2 | gfx946 第二 shape `M=256,N=512,K=256`、strides `256/512/512` | 同样真实 dispatch 后输出 NaN；日志 SHA256=`9b8d0e568e9f7796526afffdb8010cc0c136bad2b9df98970951fcbbd218bce5` | 排除单一 M/grid 大小偶发问题 |
| P5.6-E3 | gfx946 code object/ISA | code object/ISA SHA256=`4b0648cb5c374afb81abc0587baf25bd1e95066cf3d48dd68a296c12be5aee28`/`27211488c318686e228635ac64bd34f8d16069b9186804406284098acdb7f6aa`；完整对象含 192 条 `v_mmac_f32_16x16x16_f16`、24 条 `ds_read_m32x16_b16` 与 24 条 `ds_read_m32x16_b16_alt` | 排除 empty kernel；命中 NT wavelet 组合路径 |
| P5.6-E4 | gfx936/gfx938 guard candidate 回归 | 两架构再次 reference PASS；binary/log SHA256 与 P5.6-E0 各自完全一致 | G4 PASS；direct architecture isolation |
| P5.6-E5 | 撤回 enrollment 与 gfx946 guard并重生成 | 三端均确认 exact target 不存在（`TARGET_PRESENT=0`） | 不保留未通过 gfx946 correctness 的声明 |

FP16 MMAC、普通及 alt matrix-read 在本计划其他 gfx946 example 中各有通过证据，因此不能把 NaN 单点归因到某条 opcode。当前仅把 `NT v1 + wavelet + column-major A` 的整体 PMD 执行链标为缺口，按任务规则不改写真机已正确的 kernel。最终将撤回全部候选；gfx936/gfx938 的 binary identity 已直接排除性能变化，无需真实性能 A/B；gfx946 全程 timing off。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN。三端撤回确认完成；本步单独提交本文。至此当前 `05_gemm` 的 6 个注释 target 全部形成独立证据，下一优先级转入 commented legacy grouped-conv forward。

### Step 5.7：commented grouped-conv forward NCHW FP32 — SKIP-ARCH-INSTRUCTION（2026-09-02）

- 临时启用 `example_grouped_conv_fwd_mmac_nchw_fp32`。为避免默认大卷积拖慢 CPU 模拟，正式正确性 shape 使用 `G=1,N=1,K=128,C=4,Y=X=1,H=8,W=16`，对应单个 `128×128×4` GEMM tile；三端均为 `verification=1,init=1,time=0`。
- gfx936/gfx938 健康控制均完整通过。gfx946 guard 前输出全 0；放开共享 device wrapper 后仍全 0，说明还有更内层的能力边界。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.7-E0 | gfx936/gfx938 健康控制 | 两架构 reference PASS；binary SHA256=`09dea954a35f88b18dcd8f45177f677071bd7a6f07255a070861411c3721b072`/`3d5f3bf25b250acd738df07ce5ac84a66aec9da653c30f1aa03652e2d72c2515`，日志 SHA256 均为 `37bdf7bd46f15224e225e339011e19c657bf995752dfb15be586509aac63d2a3` | G0 healthy-control PASS |
| P5.7-E1 | gfx946 原 guard 对照 | dispatch 后输出全 0，max error=68；binary/log SHA256=`362fa9737f3ac7b11c341d3c26c6a3d98546e35bd56c76cc6790cfbb3638a341`/`4cafc1b5ebf80d48aa337b7c3628c8f44c3ad7e7293d5d3b7e4915fac33691b1` | empty-wrapper control |
| P5.7-E2 | 只放开 `device_grouped_conv_fwd_mmac.hpp` gfx946 guard | kernel body进入但输出仍全 0；candidate binary/log SHA256=`cf70ab921fb356d476dd3d0594b7063857f294c50709f5daf021e88a7cefbf80`/`7fcf5926d35c40afafc25814fec72db3f0fbe7868f54421a916f3fe13cec5f8b` | 外壳已排除，继续追踪 MMAC selector |
| P5.7-E3 | selector/ISA 审计 | 实例 `K0PerBlock=4,K1=4`，实际 reduction tile=16，因此 float selector 选择 `v_mmac_16x16x8_f32`；gfx946 code object/ISA SHA256=`e9ad13694850fe560beac76f658a4031bc7061754cf46fce10df1c01d10be13d`/`03a46d87392bf3f0853197000d6513cb99d53a50c3429167e3898685abde0587`，MMAC count=0，最终 buffer store 写零 | 精确定位为缺失 FP32×8 MMAC 路径 |
| P5.7-E4 | FP32×4 builtin 单点探针 | 普通 builtin 与知识库可核对的 `_vstep` 形式均可完成前端/target build，但 binary 仍与 P5.7-E2 相同，因为 selector 实际不走 x4 | 否证错误的 x4 单点修复，已撤回 |
| P5.7-E5 | 把 reduction tile 临时改为 `K0PerBlock=1,K1=4` 强制 x4 | 三架构均编译失败；首错为 `SliceLengths ... % SrcScalarPerVector == 0` 静态断言（`1 % 4`），表明必须联动重调 transfer/thread cluster；PMD/gfx936/gfx938 build-log SHA256=`943244b42013ac7e9572eedbfdb9d0226b5f3cb07372a364e4f20c07b4fb66f4`/`a3484d65a3c2ebfea50146362070f48350b09d52e47931ca8ccbe5dde6e33c4c`/`63f1dca92e4cd537593a32d129c1bded63be471df8930476be67f55f44bc2ae3` | 非安全的一行 retile；停止扩大修改 |
| P5.7-E6 | 撤回 enrollment、wrapper、builtin 与 retile 探针并重生成 | 三端均确认 exact target 不存在（`TARGET_PRESENT=0`） | `SKIP-ARCH-INSTRUCTION` |

少伯 v1.0 指令清单明确列出 FP32 HI-MMAC `v_mmac_16x16x4_f32`，未列出当前实例选择的 `v_mmac_16x16x8_f32`；生成 ISA 的零 MMAC 结果与此一致。改用 x4 需要联动重新设计 block transfer/thread cluster，不属于可由 Perf Model 证明安全的局部适配。按任务规则记录并跳过，不为通过模拟器而改写真机 kernel。最终无代码候选，gfx936/gfx938 无性能变化面，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0-G2 PASS；G3 `SKIP-ARCH-INSTRUCTION`；G4 PASS（候选条件隔离期间真机仍正确）；G5/G6 NOT RUN。三端撤回确认完成；本步单独提交本文，继续 grouped-conv forward NHWC FP32。

### Step 5.8：commented grouped-conv forward NHWC FP32 CShuffle — SKIP-ARCH-INSTRUCTION（2026-09-02）

- 临时只启用 `example_grouped_conv_fwd_mmac_nhwc_fp32`，使用与 P5.7 相同的最小卷积 `G=1,N=1,K=128,C=4,Y=X=1,H=8,W=16`，`verification=1,init=1,time=0`。
- 该 runner 的实例 TypeString 为 `DeviceGroupedConvFwd_Mmac_CShuffle-256-128-128-16-...`，源码 `KPerBlock=16,AK1=BK1=4`；float `MmacSelector` 因 reduction tile 16 选择与 P5.7 完全相同的 `v_mmac_16x16x8_f32`。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.8-E0 | gfx936 正式 correctness case | PASS；binary/log SHA256=`8f5111fc6d76e05afa3c64df819b223dd8b40e77238f383a8cd19d4d398fa6a2`/`72d6ebbb56e1fc9b00b95c0b7c57c0e0519b6c848103257d4b87b379f4b5f455` | G0 healthy-control PASS |
| P5.8-E1 | gfx938 同一正式 case | PASS；binary/log SHA256=`a414ba826a44511201bee443340ca529763c9174987e42dd943466b78dff1e16`/`72d6ebbb56e1fc9b00b95c0b7c57c0e0519b6c848103257d4b87b379f4b5f455` | G0/G4 PASS |
| P5.8-E2 | gfx946 PMD | NOT RUN；静态 selector 与 P5.7 的零-MMAC code object 已形成精确同指令闭包 | `SKIP-ARCH-INSTRUCTION`，避免重复已知无效模拟 |
| P5.8-E3 | 撤回临时 enrollment并重生成 | gfx936/gfx938 均确认 exact target 不存在（`TARGET_PRESENT=0`）；PMD 从未同步 enrollment | 恢复原始注释状态 |

P5.7 已证明当前 CK/toolchain 对 gfx946 不生成该 FP32×8 MMAC，且改用 x4 需要跨层重调 transfer/thread-cluster。NHWC CShuffle 的输出搬运不同，但计算指令选择相同，无法绕过该架构缺口，因此按模块证据跳过。最终无源码候选，不运行 gfx936/gfx938 性能 A/B；PMD 未测试性能。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-ARCH-INSTRUCTION`；G5/G6 NOT RUN。双真机撤回确认完成；本步单独提交本文，继续 grouped-conv forward NHWC FP16。

### Step 5.9：commented grouped-conv forward NHWC FP16 CShuffle — PASS-PMD（2026-09-02）

- 恢复 enrollment：`example_grouped_conv_fwd_mmac_nhwc_fp16`，并将其重新加入 `example_grouped_conv_fwd_mmac` aggregate target。没有修改 runner、tile、device wrapper 或 builtin。
- 正式正确性用例为 `G=1,N=1,K=128,C=8,Y=X=1,H=8,W=16`，对应单个输出 tile；三端均使用 `verification=1,init=1,time=0`。该实例 `KPerBlock=32,AK1=BK1=8`，float accumulation 下选择已验证的 `v_mmac_f32_16x16x16_f16`。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.9-E0 | gfx946 focused build/run | PASS；真实 `DeviceGroupedConvFwd_Mmac_CShuffle` dispatch，完整 CPU reference 无错误；binary/log SHA256=`be20f5191782568c5613b830a7c03b8a1b0e3241b01216c4ed35e99130e38f3d`/`7f0dda81e6b8d8f4994e6f8036e7bdfb2be38ff0f2f06effcf8c4dbfbdce689a` | G1-G3 PASS |
| P5.9-E1 | gfx936 同一 correctness case | PASS；binary/log SHA256=`23eed2419261fe53ca8b7df97bd211a287f6c0b43d5e1ed8098e878748374297`/`6982aa07113ccd69ae41519b3ad7d7d7e12b7e3dcb5a41ca7d3330997e62bbe7` | G0/G4 PASS |
| P5.9-E2 | gfx938 同一 correctness case | PASS；binary/log SHA256=`73c85edd8039c72c739481e7f519c70da6db5d6a97bead5a3df23c11a4f3d25f`/`6982aa07113ccd69ae41519b3ad7d7d7e12b7e3dcb5a41ca7d3330997e62bbe7` | G0/G4 PASS |
| P5.9-E3 | 三架构 aggregate build | `example_grouped_conv_fwd_mmac` 在 gfx946/gfx936/gfx938 三棵既有 build tree 均 `AGGREGATE_BUILD=PASS` | enrollment closure PASS |

PMD 仅报告一次 pending-cacheline warning，真实 dispatch 与完整 reference 均通过。最终差异只新增一个 executable 和 aggregate dependency，不改变任何已有 gfx936/gfx938 device kernel 或运行时选择，性能影响可直接排除，无需性能 A/B；gfx946 全程 timing off。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（build-graph-only change）。本 example 验证后立即单独提交，下一切片继续 NCHWC32 FP16。

### Step 5.10：commented grouped-conv forward NCHWC32 FP16 CShuffle — PASS-PMD（2026-09-02）

- 恢复 `example_grouped_conv_fwd_mmac_nchwc32_fp16` enrollment 及 aggregate dependency；runner/device 源码不变。
- 正式正确性用例 `G=1,N=1,K=128,C=32,Y=X=1,H=8,W=16`，三端均为 `verification=1,init=1,time=0`；物理张量为 `NGCHWc32/GKCYXc32/NGKHWk32`，独立覆盖 32-channel blocking 与写回。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.10-E0 | gfx946 focused build/run | PASS；真实 CShuffle dispatch、完整 CPU reference 无错误；binary/log SHA256=`4cbb6140cb2f85639215c24645a476676367a5346a5bed32af43e78b37ad721a`/`a58b37d973f96213314854804a8199e88220d33b7ac5cb02da3269d034a772c8` | G1-G3 PASS |
| P5.10-E1 | gfx936 同一 correctness case | PASS；binary/log SHA256=`d797de7117237903a5d9e02ae44146eada2335604b034f31c02064eb422e1867`/`11f92065bb68963d9f4baa7738d57d5829f50445887deb25d0905c8170563834` | G0/G4 PASS |
| P5.10-E2 | gfx938 同一 correctness case | PASS；binary/log SHA256=`efb3b14b7f7df3e90c3c33ae527a1b9b5eb8ca2c3c4f46ff891a0b67ef7a3850`/`11f92065bb68963d9f4baa7738d57d5829f50445887deb25d0905c8170563834` | G0/G4 PASS |
| P5.10-E3 | 三架构 aggregate build | `example_grouped_conv_fwd_mmac` 在 gfx946/gfx936/gfx938 均 `AGGREGATE_BUILD=PASS` | enrollment closure PASS |

PMD 仅报告一次 pending-cacheline warning，数值完整通过。最终仅改变构建图，不修改既有 device kernel/dispatch，故 gfx936/gfx938 性能影响可直接排除，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（build-graph-only change）。本 example 单独提交后继续 NHWC INT8。

### Step 5.11：commented grouped-conv forward NHWC INT8 CShuffle — real-GPU PASS / SKIP-PMD（2026-09-02）

- 临时恢复 `example_grouped_conv_fwd_mmac_nhwc_int8` enrollment，正式正确性用例为 `G=1,N=1,K=128,C=64,Y=X=1,H=8,W=16`，`verification=1,init=1,time=0`。
- 源码固定 `int8_t × int8_t -> int32_t`、`MPerMmac=NPerMmac=16,KPerBlock=64`；`MmacTraits<int8_t,...>` 唯一选择 `mmac_i32_16x16x32i8`。legacy wrapper 在 gfx946 当前走 `swallow`，而开放该分支会调用 P4.17 已实测失败且 PMD 横幅明确 `dp4x:disable` 的 `__builtin_hcu_mmac_i32_16x16x32_i8`。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.11-E0 | gfx946 focused build + code-object/ISA | BUILD PASS；binary/code-object/ISA SHA256=`28c495c4faabbdf7f372f418db73b6e1ae7fa5325eb4b21feefe363dde1673bd`/`de56687a959bd9f3acafea161b05e0eedd93b444eeed7c1dc7f1612fdbf02122`/`69ae7c5e4146e3bf380cf64cfa192664369785075a304d105a73d15e752ffa5d`；INT8 MMAC count=0 | G1/G2 PASS；gfx946 guard 精确落入无计算分支 |
| P5.11-E1 | gfx936 同一 correctness case | exact PASS；binary/log SHA256=`c3809da8d73621e157dad1daef0d575a9720dd54f1130c034dfb19131e47b524`/`e7230e3d0eb6383c7ea34ff04226fb8b3922f549bedb9198af0b66b5efa4e931` | G0/G4 PASS |
| P5.11-E2 | gfx938 同一 correctness case | exact PASS；binary/log SHA256=`b6ef3fc64ed2ba5ac2f8bf06bcd7f5200c95ce4e47dce7da6444c0c64d691a5a`/`e7230e3d0eb6383c7ea34ff04226fb8b3922f549bedb9198af0b66b5efa4e931` | G0/G4 PASS |
| P5.11-E3 | 撤回临时 enrollment 并重生成 | gfx946/gfx936/gfx938 均为 `TARGET_ABSENT` | 恢复原始注释状态 |

PMD 不运行该 binary：当前 guard 会静默输出零结果，不构成有效 correctness；增加 gfx946 guard 又会进入当前模拟器明确关闭、P4.17 已直接验证失败的整数点积/MMAC 路径。故记录为 Perf Model 指令模拟缺口并跳过该模块，不以错误 fallback 改写真实 GPU kernel。最终无业务源码或构建图差异，gfx936/gfx938 不存在性能不确定性，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-PMD`；G5/G6 NOT RUN（evidence-only）。本步单独提交本文后继续 NHWC v2 FP16。

### Step 5.12：commented grouped-conv forward NHWC v2 FP16 — shared broken baseline（2026-09-02）

- 临时恢复 `example_grouped_conv_fwd_mmac_nhwc_v2_fp16` enrollment。原始实例参数为 `KPack=4`，但 A/B `BlockTransferSrcScalarPerVector=2`；三架构均在同一 `ThreadGroupTensorSliceTransfer_LdsSwizzle` 约束处编译失败：`2 % 4 == 0`。
- 受控候选仅把 A/B source vector length 从 2 调到 4，使三架构均越过编译；随后用 `G=1,N=1,K=128,C=8,Y=X=1,H=8,W=16`、`verification=1,init=1,time=0` 做完整 reference 验证。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.12-E0 | 原始参数三架构 focused build | gfx946/gfx936/gfx938 均 FAIL，首个终止点同为 `static assertion failed due to requirement '2 % 4 == 0'`；build-log SHA256=`6eca6a7f36ab2d2c4f44ccabba34c238d73fc0419a589affe02ba1d036bebc1b`/`7566acfba6e1e947d36a63a0dc6a306d485638b2e556752afcaa029e7f98346d`/`0c4a07562bc6ea54a994e9070cb88c9913cb448359f24220655f057057efb3ce` | 三架构共享 baseline 编译缺陷，不是 Shaobo 回归 |
| P5.12-E1 | vector-length=4 候选，gfx946 build/run | BUILD PASS、真实 dispatch；全零型错误，max error=120；binary/log SHA256=`c0b63c51ef92078f1b5f8e83ddf2c59fb6e1d3a30979cf90e56dfb563b7bef2f`/`a22be05cec8cd082f1dd125a9153b67238361dd0b4f2e312fc8b1ca0871486f9` | candidate correctness FAIL |
| P5.12-E2 | 同候选 gfx936/gfx938 correctness | 两端均 FAIL；gfx936 max error=113.7734，binary/log=`91f5674ff7cc757a91bd7e082adeeb52a1015d342b0c8e3a2495467e4a17a27c`/`8ba627a93895fdfc7dd803a7a3c096f8b8b83991a8374d7864ec61c0120597c4`；gfx938 max error=115.8281，binary/log=`15d7f26341245a59908baef9d9ac9fc129ebe71f75dc6564775cc2a6ae95d0a2`/`1dc89540eb236c27e16b5938284d7e4ca0a108cb161cf145061186a2bc1ea39f` | 双真机否决候选，不能作为 Shaobo 修复 |
| P5.12-E3 | 撤回候选源码与 enrollment | gfx946/gfx936/gfx938 均为 `TARGET_ABSENT` | 恢复原始注释状态 |

该 example 在 gfx936/gfx938 baseline 即无法编译，最小向量参数修正又被双真机完整数值验证否决；继续重写 v2 transfer/tile 管线已超出 Shaobo 单架构适配边界。本步按共享坏例跳过，不保留源码改动。最终 candidate 与 baseline 相同，故没有 gfx936/gfx938 性能不确定性，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0/G1 PASS；G2 baseline FAIL；G3/G4 candidate FAIL；G5/G6 NOT RUN。三端撤回确认完成，本步单独提交本文后继续 NHWC v2 FP16 wavelet。

### Step 5.13：commented grouped-conv forward NHWC v2 FP16 wavelet — shared broken baseline（2026-09-02）

- 临时恢复 `example_grouped_conv_fwd_mmac_nhwc_v2_fp16_wavelet` enrollment，使用与 P5.12 相同的最小完整校验用例：`G=1,N=1,K=128,C=8,Y=X=1,H=8,W=16`、`verification=1,init=1,time=0`。
- 与普通 v2 不同，该 wavelet target 以原始参数在三架构均可构建；因此独立执行三端 correctness，而未沿用 P5.12 的编译失败分类。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.13-E0 | gfx946 focused build/run | BUILD PASS、真实 dispatch；输出全零、max error=120；binary/log SHA256=`202d261157b939143e4e2eee06f9782a79e75bf034eb04e173facf52119336c9`/`01075479b2d986a3a657d6c77dfeb07ce1009e0bb4198767e9d1802240b08286` | G2 PASS、G3 FAIL；未测试性能 |
| P5.13-E1 | gfx936 同一 correctness case | FAIL，max error=169；binary/log SHA256=`43360b1b138838ecce4015b4e75a941fe5b1d2caa8057038fa52707564a7df5d`/`484fde6b11318b99b3edc141fb66849ba7a672d48dc54fb368ab5ed4a5eeb3a2` | G4 baseline FAIL |
| P5.13-E2 | gfx938 同一 correctness case | FAIL，max error=169；binary/log SHA256=`39ba8cc688fde8407241ead541ef4a20919ac91f9b19aebefa8ec88cdaa1c26d`/`484fde6b11318b99b3edc141fb66849ba7a672d48dc54fb368ab5ed4a5eeb3a2` | G4 baseline FAIL；双真机日志逐字一致 |
| P5.13-E3 | 撤回临时 enrollment 并重生成 | gfx946/gfx936/gfx938 均为 `TARGET_ABSENT` | 恢复原始注释状态 |

双真机在相同输出位置产生相同错误且日志哈希一致，足以证明 wavelet runner/pipeline 的 baseline correctness 已损坏；gfx946 的全零表现不能单独归因于 Shaobo。最终不保留源码或构建图差异，因此不存在 gfx936/gfx938 性能不确定性，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0-G2 PASS；G3/G4 FAIL（shared broken baseline）；G5/G6 NOT RUN。本步单独提交本文后继续 NCHWC FP16。

### Step 5.14：commented grouped-conv forward NCHWC FP16 CShuffle — PASS-PMD（2026-09-02）

- 恢复 `example_grouped_conv_fwd_mmac_nchwc_fp16` enrollment 及 aggregate dependency；runner/device 源码不变。
- 正式正确性用例 `G=1,N=1,K=128,C=32,Y=X=1,H=8,W=16`，三端均为 `verification=1,init=1,time=0`；物理布局为模板化 `NGCHWc<32>/GKCYXc<32>/NGKHWk<32>`，实例使用 `KPerBlock=16,AK1=BK1=4`，与 P5.10 的 alias/KPerBlock=32 路径独立。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.14-E0 | gfx946 focused build/run | PASS；真实 CShuffle dispatch、完整 CPU reference 无错误；binary/log SHA256=`8eaedb6410d7c31e2413f175eb1436b3838606fc232449a32d82258fbd192509`/`20a7b31643719e542ee6eecec711130acf38fe251f2249abdf2a5ea3ec41ede7` | G1-G3 PASS |
| P5.14-E1 | gfx936 同一 correctness case | PASS；binary/log SHA256=`9a69f3c7019642749dcb28855588b95c5e6f5a806c90e920a792c4439b5f32f2`/`60078db830fa97f5a32f7ab4a8cb7ac493356e4a16430d6f0683b1a4f9928388` | G0/G4 PASS |
| P5.14-E2 | gfx938 同一 correctness case | PASS；binary/log SHA256=`7b3c4a7013058c690c8126a73060b24611348daeff4555b841a20d517ca122b7`/`60078db830fa97f5a32f7ab4a8cb7ac493356e4a16430d6f0683b1a4f9928388` | G0/G4 PASS |
| P5.14-E3 | 三架构 aggregate build | `example_grouped_conv_fwd_mmac` 在 gfx946/gfx936/gfx938 均 `AGGREGATE_BUILD=PASS` | enrollment closure PASS |

PMD 报告少量 `v_mad_u64_u32` read-before-write 与 pending-cacheline warning，但真实 dispatch 和完整 reference 均通过，只记录为模拟器 warning。最终仅改变构建图，不修改既有 device kernel/dispatch，因此可直接排除 gfx936/gfx938 性能影响，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（build-graph-only change）。本 example 单独提交后继续下一处未入图能力。

### Step 5.15：commented grouped-conv backward-data NHWC FP16 CShuffle — guard fix / PASS-PMD（2026-09-02）

- 恢复 `example_grouped_conv_bwd_data_mmac_nhwc_fp16` enrollment 及 aggregate dependency；正式用例为 `G=1,N=1,K=128,C=128,Y=X=1,H=8,W=16`、stride1/dilation1/pad0，三端均 `verification=1,init=1,time=0`。
- 初始 gfx936/gfx938 完整 reference PASS，而 gfx946 真实 launch 后输出全零。定位到外层 `device_grouped_conv_bwd_data_mmac_cshuffle.hpp` device body guard 缺少 `__gfx946__`；底层 CShuffle gridwise 已包含 gfx946。最终改动只为该 guard 增加 gfx946。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.15-E0 | guard 修复前 gfx946 focused build/run | BUILD PASS、dispatch 后全零，max error=428；binary/log SHA256=`5e80495fd339fa1fff0691702ad7b662682e05c6499bd9574420098bc8103ea8`/`3c3ac166eba987c1aee09609a41cbb387d60e3f8c4ebe36b7dfcef273ce9f41e` | empty-wrapper control |
| P5.15-E1 | guard 修复后 gfx946 同一 correctness case | PASS；真实 device body + 完整 CPU reference；binary/log SHA256=`86637bc0041fb700c45de54815a7ec25f9e9758a1e4ef483c703cab8b3254bd2`/`8e5208fab4b52f7b6fc43483acbb450eab3c7b8198d97da30c4a0702b35c3a64` | G1-G3 PASS；未测试性能 |
| P5.15-E2 | gfx936 guard 前后 correctness/binary | 两次均 PASS；binary 始终 `38341849429cb4cb3b32bd30319dfa333cb3900b57db9dd68e33495ba829fdbc`，log 始终 `24ce7801974b314d2ea2c2e5fe414b9ec26211bc25826181b0cb8e70f1d4a61f` | G4 PASS；binary identity |
| P5.15-E3 | gfx938 guard 前后 correctness/binary | 两次均 PASS；binary 始终 `3b0a747b89add6ca5711db877504be0bd53d17cbcdde9936e6cdc00b581f7191`，log 始终 `24ce7801974b314d2ea2c2e5fe414b9ec26211bc25826181b0cb8e70f1d4a61f` | G4 PASS；binary identity |
| P5.15-E4 | 三架构 aggregate build | `example_grouped_conv_bwd_data_mmac` 在 gfx946/gfx936/gfx938 均 `AGGREGATE_BUILD=PASS` | enrollment closure PASS |

gfx946 PMD 仅报告少量 `v_mad_u64_u32` read-before-write 与 pending-cacheline warning，数值校验完整通过。gfx936/gfx938 修复前后 binary 逐字相同，已直接排除性能变化，无需性能 A/B；gfx946 未测试性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（true-GPU binary identity）。本 example 单独提交后继续 backward-data NCHWC32 FP16。

### Step 5.16：commented grouped-conv backward-data NCHWC32 FP16 CShuffle — PASS-PMD（2026-09-02）

- 恢复 `example_grouped_conv_bwd_data_mmac_nchwc32_fp16` enrollment 及 aggregate dependency；复用 P5.15 已提交的 gfx946 wrapper guard，runner/device 源码不再修改。
- 正式用例 `G=1,N=1,K=128,C=128,Y=X=1,H=8,W=16`、stride1/dilation1/pad0，三端均 `verification=1,init=1,time=0`；物理布局为 `NGKHWk32/GKCYXc32/NGCHWc32`。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.16-E0 | gfx946 focused build/run | PASS；真实 blocked-layout backward-data dispatch、完整 CPU reference 无错误；binary/log SHA256=`85e628067f459f829abb05f7050d4f8e76cbd9914ea6b3df049af773fa611d6e`/`10bdd7146c22bd325ad0220a670749638ba21d99afe5c13556d0edb2bd19ed56` | G1-G3 PASS；未测试性能 |
| P5.16-E1 | gfx936 同一 correctness case | PASS；binary/log SHA256=`8c01030ad260775efa5c98c8c0b43b9ee02f6919f5721525379101476c79b204`/`fe908ae34164dcede9cde0fdd60abb1ab0026f17114a2ead8809193897ed0e6c` | G0/G4 PASS |
| P5.16-E2 | gfx938 同一 correctness case | PASS；binary/log SHA256=`309c5abf1cb1c2453819b03ee9bb625bdf019a224dad42c4201bb77e92173810`/`fe908ae34164dcede9cde0fdd60abb1ab0026f17114a2ead8809193897ed0e6c` | G0/G4 PASS |
| P5.16-E3 | 三架构 aggregate build | `example_grouped_conv_bwd_data_mmac` 在 gfx946/gfx936/gfx938 均 `AGGREGATE_BUILD=PASS` | enrollment closure PASS |

最终差异只新增 executable 与 aggregate dependency；P5.15 的 device guard 已独立提交并验证。故本步不会改变任何既有 gfx936/gfx938 kernel 或 dispatch，性能影响可直接排除，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（build-graph-only change）。本 example 单独提交后继续 backward-data v2 FP16。

### Step 5.17：commented grouped-conv backward-data NHWC v2 FP16 — shared broken baseline（2026-09-02）

- 临时恢复 `example_grouped_conv_bwd_data_mmac_nhwc_v2_fp16` enrollment；正式用例沿用 runner 默认问题：`G=1,N=1,K=64,C=16,Y=X=3,H=W=4`、stride1/dilation1/pad1，三端均 `verification=1,init=1,time=0`。
- 三架构 focused build 均通过并真实 dispatch，但 gfx936/gfx938 从同一 flat index 64 开始给出逐项相同错误，日志 SHA256 完全一致。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.17-E0 | gfx946 focused build/run | FAIL；全零型错误、max error=601；binary/log SHA256=`cd64cbb9503fe7a779b179dc7f8b44ce416e0d1f5e98180ea044ac0548ca91aa`/`22dfba578eb2dbe5fb854e3fee54f0da134e80458d4e8dd6e95e07387bc5e054` | G2 PASS、G3 非归因 |
| P5.17-E1 | gfx936 同一 correctness case | FAIL；从 flat index 64 起错误，max error=772；binary/log SHA256=`848abdab1306a56324177c0d4e397b1d3be70c11fecd7dcb6d8d0abf6fcda1cc`/`c27dadfd41f637df0334e0b8ef782880eaa69e3d1ebb7960fa99f322afce857d` | G4 baseline FAIL |
| P5.17-E2 | gfx938 同一 correctness case | FAIL；错误位置、数值和 max error 与 gfx936 完全相同；binary/log SHA256=`0b1a2efdecd9e0a9d3b80085340337c639446b8560abf277a678d24241e3c4a1`/`c27dadfd41f637df0334e0b8ef782880eaa69e3d1ebb7960fa99f322afce857d` | G4 baseline FAIL；`SKIP-BASELINE` |
| P5.17-E3 | 撤回临时 enrollment 并重生成 | gfx946/gfx936/gfx938 均为 `TARGET_ABSENT` | 恢复原始注释状态 |

双真机相同错误证明该非 CShuffle v2 backward-data baseline 已损坏，不能把 gfx946 全零归因于 Shaobo。最终无源码/构建图候选，故没有 gfx936/gfx938 性能不确定性，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0 FAIL；G1/G2 PASS；G3 非归因；G4 FAIL；G5/G6 NOT RUN；overall `SKIP-BASELINE`。本步单独提交本文后继续 backward-data v2 FP16 wavelet。

### Step 5.18：commented grouped-conv backward-data NHWC v2 FP16 wavelet — shared broken baseline（2026-09-02）

- 临时恢复 `example_grouped_conv_bwd_data_mmac_nhwc_v2_fp16_wavelet` enrollment；使用与 P5.17 完全相同的默认问题合同：`G=1,N=1,K=64,C=16,Y=X=3,H=W=4`、stride1/dilation1/pad1，`verification=1,init=1,time=0`。
- 三架构 focused build/dispatch 均完成；gfx936/gfx938 错误位置、数值、max error 和日志哈希与 P5.17 普通 v2 完全一致。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.18-E0 | gfx946 focused build/run | FAIL；全零型错误、max error=601；binary/log SHA256=`113ff3422fce4595325a58a18a60547bcebaf5f101a3cf28477b80077a9dac2c`/`d27070153b264659491749b5fd99f7c0316da4d94c69fc0a99de95d94121e39c` | G2 PASS、G3 非归因 |
| P5.18-E1 | gfx936 同一 correctness case | FAIL；从 flat index 64 起错误，max error=772；binary/log SHA256=`38400b76c297b28653d2226c2c2a778d0c8cafbf050c9acbfed929a4ecae3e1a`/`c27dadfd41f637df0334e0b8ef782880eaa69e3d1ebb7960fa99f322afce857d` | G4 baseline FAIL |
| P5.18-E2 | gfx938 同一 correctness case | FAIL；错误与 gfx936/P5.17 完全相同；binary/log SHA256=`65e5e8cd6a7867d90a4970fc877ccc3e118eb3b29a6f623fa19ce705f201bfb0`/`c27dadfd41f637df0334e0b8ef782880eaa69e3d1ebb7960fa99f322afce857d` | G4 baseline FAIL；`SKIP-BASELINE` |
| P5.18-E3 | 撤回临时 enrollment 并重生成 | gfx946/gfx936/gfx938 均为 `TARGET_ABSENT` | 恢复原始注释状态 |

双真机以及普通/wavelet 两种 v2 runner 的错误日志完全相同，证据指向共享 backward-data v2 baseline 缺陷，而非 gfx946 或 wavelet 单点问题。本任务不扩展为跨架构算法重写；最终无源码/构建图候选，无需 gfx936/gfx938 性能 A/B；gfx946 未测试性能。

门禁结果：G0 FAIL；G1/G2 PASS；G3 非归因；G4 FAIL；G5/G6 NOT RUN；overall `SKIP-BASELINE`。本步单独提交本文后继续 backward-data FP32 架构边界。

### Step 5.19：commented grouped-conv backward-data NCHW FP32 — real-GPU PASS / SKIP-ARCH-INSTRUCTION（2026-09-02）

- 临时仅在 gfx936/gfx938 恢复 `example_grouped_conv_bwd_data_mmac_nchw_fp32` enrollment；正式健康控制为 `G=1,N=1,K=128,C=128,Y=X=1,H=8,W=16`、stride1/dilation1/pad0，`verification=1,init=1,time=0`。
- 实例精确为 `K0PerBlock=4,K1=4,MPerMmac=NPerMmac=16`，float reduction tile=16；`MmacSelector` 因此选择 `mmac_f32_16x16x8f32`。外层 backward-data wrapper 与该 intrinsic 都没有 gfx946 有效分支，和 P5.7 已提取的零-MMAC FP32×8 code object 构成完全相同的指令闭包。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.19-E0 | gfx936 同一 correctness case | PASS；binary/log SHA256=`4471a2c038917a7cdaa0660bd32fc30ac126066893c23968a91c430a4ba672d8`/`24ce7801974b314d2ea2c2e5fe414b9ec26211bc25826181b0cb8e70f1d4a61f` | G0/G4 healthy-control PASS |
| P5.19-E1 | gfx938 同一 correctness case | PASS；binary/log SHA256=`0cc5fbff8fb2e76b2d2c928f993ce0ffbde922772ae47233e60d8dcf4d85ea29`/`24ce7801974b314d2ea2c2e5fe414b9ec26211bc25826181b0cb8e70f1d4a61f` | G0/G4 healthy-control PASS |
| P5.19-E2 | gfx946 PMD | NOT RUN；P5.7 已证明相同 selector 在 gfx946 code object 中 MMAC count=0，强制 x4 又需要跨层 transfer retune | `SKIP-ARCH-INSTRUCTION`，避免重复无效模拟 |
| P5.19-E3 | 撤回临时 enrollment 并重生成 | gfx936/gfx938 均为 `TARGET_ABSENT`；PMD 从未同步 enrollment | 恢复原始注释状态 |

两台真机完整 reference PASS，证明算法合同健康；gfx946 缺失的是当前实例不可替代的 FP32×8 MMAC 路径，按任务规则记录并跳过，不改写真机 kernel。最终无源码候选，不存在 gfx936/gfx938 性能变化，无需性能 A/B；PMD 未运行、未测试性能。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-ARCH-INSTRUCTION`；G5/G6 NOT RUN。本步单独提交本文后继续 backward-data NHWC FP32。

### Step 5.20：commented grouped-conv backward-data NHWC FP32 — real-GPU PASS / SKIP-ARCH-INSTRUCTION（2026-09-02）

- 临时仅在 gfx936/gfx938 恢复 `example_grouped_conv_bwd_data_mmac_nhwc_fp32` enrollment；正式健康控制为 `G=1,N=1,K=128,C=128,Y=X=1,H=8,W=16`、stride1/dilation1/pad0，`verification=1,init=1,time=0`。
- 实例精确为 `KPerBlock=16,AK1=BK1=4,MPerMmac=NPerMmac=16`，float reduction tile=16；`MmacSelector` 选择 `mmac_f32_16x16x8f32`。P5.15 已修复外层 CShuffle wrapper 的 gfx946 guard，但 `hcu_mmac.hpp` 中该 FP32×8 intrinsic 仍无 gfx946 有效分支，P5.7 的 gfx946 code object 已证明同 selector 的 MMAC count=0。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.20-E0 | gfx936 同一 correctness case | PASS；binary/log SHA256=`1890919266f2bdc131c14c55a9a5e90b040d14043364a0768fe3b14e709342e8`/`24ce7801974b314d2ea2c2e5fe414b9ec26211bc25826181b0cb8e70f1d4a61f` | G0/G4 healthy-control PASS |
| P5.20-E1 | gfx938 同一 correctness case | PASS；binary/log SHA256=`a9e271f4884f4727c199e6b717a5d47223e5d820b2ca830d1176507a4f09637e`/`24ce7801974b314d2ea2c2e5fe414b9ec26211bc25826181b0cb8e70f1d4a61f` | G0/G4 healthy-control PASS |
| P5.20-E2 | gfx946 PMD | NOT RUN；当前 wrapper 会进入 device body，但唯一计算 intrinsic 在 gfx946 仍为 swallow；P5.7 已有相同 selector 的零-MMAC code-object 证据 | `SKIP-ARCH-INSTRUCTION`，避免把确定的静默零输出重复当作 correctness 运行 |
| P5.20-E3 | 撤回临时 enrollment 并重生成 | gfx936/gfx938 均为 `TARGET_ABSENT`；PMD 从未同步 enrollment | 恢复原始注释状态 |

两台真机完整 reference PASS，证明 NHWC CShuffle backward-data 算法合同健康；gfx946 的阻塞点仍是当前实例不可替代的 FP32×8 MMAC，而不是 P5.15 已修复的 wrapper guard。最终无源码/构建图差异，不存在 gfx936/gfx938 性能变化，无需性能 A/B；PMD 未运行、未测试性能。

门禁结果：G0-G2/G4 PASS；G3 `SKIP-ARCH-INSTRUCTION`；G5/G6 NOT RUN。本步单独提交本文后转入 grouped-conv forward bias/activation 未入图队列。

### Step 5.21：commented grouped-conv forward bias+ReLU NHWC FP16 — shared broken baseline（2026-09-02）

- 临时恢复 `example_grouped_conv_fwd_bias_relu_mmac_nhwc_fp16` enrollment，并为 `device_grouped_conv_fwd_bias_activation_mmac_v2.hpp` 外层 wrapper 临时加入 gfx946 guard；其底层 FP16 v2 gridwise 已存在 gfx946 分支。
- 先测 `G=1,N=1,K=C=128,Y=X=1,H=8,W=16`，再用 runner 默认 `G=1,N=1,K=C=16,Y=X=1,H=W=4` 复核；所有运行均 `verification=1,init=1,time=0`。两种问题在 gfx936/gfx938 上均逐项产生完全相同的错误，排除 Shaobo 单架构归因。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.21-E0 | gfx936/gfx938 自定义 1x1 case | 两端 FAIL，max error=404，日志 SHA256 均为 `2b12c429ce9e31b73fb3b337df066c73100188c03c498728a7287ba97dda97cf`；binary 分别为 `4777c9aad92de222833fa7ab6315eded568c1a3f338d25b17baa9b5a6fa32a64`/`b6f94e80363a5181814a88e76bd705976eabf47a5bcdabae8acfa9d0fb03182e` | G4 baseline FAIL |
| P5.21-E1 | gfx936/gfx938 runner 默认 case | 两端 FAIL，max error=92，日志 SHA256 均为 `d6e2965ef0c58408b14bc3061e9f9b4d2388bbdfb910e83bcc7518e6f983244d` | 第二个独立 shape 复现共享 baseline 缺陷 |
| P5.21-E2 | gfx946 PMD focused build/run | BUILD/dispatch PASS；自定义与默认 case 均为全零型 FAIL，max error=377/100；binary SHA256=`5f681a8b00a02297d04e55beb5342197702bc0422ed7cbbbade47842dce9f6c4`，日志 SHA256=`593a5c89ef8829a3371e56c5128398a9e2bec82174dbf48dbf0f1d5b61749d7c`/`45d325d643d099391a17a0a33d1c6e30e13b8fff28e9409b365d459e67dabe83` | G2 PASS；G3 非归因 |
| P5.21-E3 | 撤回临时 wrapper guard/enrollment 并重生成 | gfx946/gfx936/gfx938 均为 `TARGET_ABSENT` | 恢复原始注释状态；`SKIP-BASELINE` |

PMD 同时提示 `v_add_f16`/`v_max_f16` 尚未测试及 pending-cacheline warning，但双真机两种 shape 已先证明 non-CShuffle v2 bias+ReLU example 的 correctness baseline 损坏，故本任务不把 PMD 全零定性为独立模拟器缺陷，也不扩展为跨架构算法修复。最终没有源码/构建图候选，不存在 gfx936/gfx938 性能不确定性，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0 FAIL；G1/G2 PASS；G3 非归因；G4 FAIL；G5/G6 NOT RUN；overall `SKIP-BASELINE`。本步单独提交本文后继续 bias+ReLU wavelet。

### Step 5.22：commented grouped-conv forward bias+ReLU NHWC FP16 wavelet — shared broken baseline（2026-09-02）

- 临时仅在 gfx936/gfx938 恢复 `example_grouped_conv_fwd_bias_relu_mmac_nhwc_fp16_wavelet` enrollment；使用 runner 默认 blocked-layout 问题 `G=1,N=1,K=C=32,Y=X=1,H=W=4`，`verification=1,init=1,time=0`。
- 该路径若继续适配 gfx946，需要同时开放 device wrapper 与 `gridwise_gemm_mmac_tn_mk_nk_mn_v2r1_wavelet.hpp` 两层 guard；在做这些改动前先执行真机健康控制，避免掩盖既有 baseline 缺陷。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.22-E0 | gfx936 默认 correctness case | FAIL；max error=158；binary/log SHA256=`e6b8cf5691f5472b56333deab0ea478ae5675cd18df63b45fe090a363b29bb83`/`a1cb856261ff10c83861e2163434d20af90be9547c05205717b7b5f8999dcef7` | G4 baseline FAIL |
| P5.22-E1 | gfx938 同一 case | FAIL；错误位置、数值和 max error 与 gfx936 完全相同；binary/log SHA256=`6731d614e97704c86081cbaa7ca3396f8a81a91c5bcf249952f9f2566c107a42`/`a1cb856261ff10c83861e2163434d20af90be9547c05205717b7b5f8999dcef7` | G4 baseline FAIL；`SKIP-BASELINE` |
| P5.22-E2 | gfx946 PMD | NOT RUN；真机 baseline 已失败，且 enrollment/两层 gfx946 guard 从未同步 | 避免无归因价值的模拟 |
| P5.22-E3 | 撤回临时 enrollment 并重生成 | gfx946/gfx936/gfx938 均为 `TARGET_ABSENT` | 恢复原始注释状态 |

双真机完全一致的 reference 错误证明 wavelet example 自身 baseline 损坏，不能把后续潜在的 gfx946 结果解释为 Shaobo 适配结论。本任务不扩展为跨架构 wavelet 算法重写；最终无源码/构建图候选，无需 gfx936/gfx938 性能 A/B；gfx946 未运行、未测试性能。

门禁结果：G0 FAIL；G1/G2 PASS；G3 NOT RUN；G4 FAIL；G5/G6 NOT RUN；overall `SKIP-BASELINE`。本步单独提交本文后继续 bias+ReLU CShuffle mixed。

### Step 5.23：commented grouped-conv forward bias+ReLU CShuffle mixed — shared compile/correctness failure（2026-09-02）

- 临时恢复 `example_grouped_conv_fwd_bias_relu_mmac_nhwc_fp16_cshuffle_mixed` enrollment；默认问题为 `G=1,N=1,K=C=16,Y=X=3,H=W=8`、stride1/dilation1/pad1，`verification=1,init=1,time=0`。
- 原实例的 A/B 源向量长度均为 2，而 LDS swizzle 目标向量长度为 4；gfx936/gfx938 编译均在 `thread_group_tensor_slice_transfer_lds_swizzle.hpp:83` 触发同一 `2 % 4 == 0` static assertion。随后仅把两处源向量长度 2→4 作为机械候选复核。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.23-E0 | gfx936/gfx938 原实例 focused build | 两端均 FAIL；首个终止错误相同：`static assertion failed due to requirement '2 % 4 == 0'` | G2 shared baseline compile FAIL |
| P5.23-E1 | 源向量长度 2→4 候选 + gfx936 | BUILD PASS，reference FAIL，max error=305.5312；binary/log SHA256=`0ac79eada9e9fd4caa0aa7c4a9ce34f9ff0a228ce73b72d7506baa0fec902d9a`/`c5366ecb243a582f6b12ba2bfa2275dd4a3134c7fb9fa77aa24620b2693f8ef6` | 候选不可用 |
| P5.23-E2 | 同一候选 + gfx938 | BUILD PASS，reference FAIL，max error=`inf`；binary/log SHA256=`3cda9843663bf7668cf8a8b47238d196e34f1be5dea4b5f6b754388970bb1ce9`/`1f4d74846cf43cb00e0fc72e4b6ee9ce040b33897b86a002909859eae9bbe48a` | G4 FAIL；`SKIP-BASELINE` |
| P5.23-E3 | gfx946 PMD | NOT RUN；原实例共享编译失败，候选又在两台真机 correctness 失败；enrollment/source/gfx946 两层 guard 从未同步 | 无可归因候选 |
| P5.23-E4 | 撤回 source/enrollment 并重生成 | gfx946/gfx936/gfx938 均为 `TARGET_ABSENT` | 恢复原始注释状态 |

P5.12 已观察到同类 2→4 候选会破坏普通 grouped-conv v2 correctness，本步又在 mixed bias+ReLU 路径独立复现。继续开放 gfx946 wrapper/gridwise guard无法修复共享 baseline，因此停止本模块而非引入错误 kernel。最终无源码/构建图候选，无需 gfx936/gfx938 性能 A/B；gfx946 未运行、未测试性能。

门禁结果：G0 FAIL；G1 PASS；G2 FAIL；G3 NOT RUN；G4 FAIL；G5/G6 NOT RUN；overall `SKIP-BASELINE`。本步单独提交本文后继续 bias+ReLU NCHW CShuffle。

### Step 5.24：commented grouped-conv forward bias+ReLU NCHW FP16 CShuffle — 4B LDS direct-load fix / PASS-PMD（2026-09-02）

- 恢复 `example_grouped_conv_fwd_bias_relu_mmac_nchw_fp16_cshuffle` enrollment；正式用例为 runner 默认 `G=1,N=1,K=C=16,Y=X=1,H=W=4`、stride1/dilation1/pad0，`verification=1,init=1,time=0`。
- gfx936/gfx938 原始实例均 PASS。gfx946 首先需要补 device wrapper 与 `gridwise_gemm_mmac_nn_m0k0k1m1_nk_nm_v1r1_cshuffle.hpp` 两层 guard；随后根据 `ThreadGroupTensorSliceTransfer_DirectLoad` 的一 DWORD 合同，把 gfx946 A-load 从 8×FP16/线程改成 2×FP16/线程，并把线程簇从 `S<1,4,1,16,4>` 调整为 `S<1,1,1,16,16>`，保持 256 线程且单 wave 连续写 64 DWORD。
- 最后定位到 `hcu_direct_load_global_to_lds()` 能力表遗漏 gfx946，导致 4B direct-load 仍被编译为 `swallow`；将 gfx946 纳入仅支持 `ReadSize==4` 的分支后 correctness 通过。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.24-E0 | gfx946 两层 guard + 原实例 | BUILD FAIL；`read_size==4` static assertion 实际为 16B | 收窄 A-load 到一 DWORD |
| P5.24-E1 | gfx946 A 向量 2、原线程簇 | BUILD FAIL；`AreThreadClusterLengthsValid()` static assertion | 按类内公式调整线程簇 |
| P5.24-E2 | gfx946 合法线程簇、direct-load 能力表修复前 | BUILD/dispatch PASS 但全零，max error=100；binary/log SHA256=`ee8f183ea402790ee9c856a5856522ed3f74817631f16647f55abf396fd7556a`/`54732bacf5881207e785ad496f60ff502d518b97e0a22cee8c33b193d4251db0` | code object 有 48 条 FP16 MMAC、无 `buffer_load_* ... lds`，定位 A-load 被吞 |
| P5.24-E3 | gfx946 4B direct-load 能力表修复后 | PASS；binary/log SHA256=`594d108b13969ef14ae55388aec7a511c0754a96d32914f74414c1e287871e9c`/`49a1abe415b94ff07c47a5e4de972f4f04f975b4f347c52b9977018e9ee4c387` | G1-G3 PASS；未测试性能 |
| P5.24-E4 | gfx946 code-object 复核 | code object/ISA SHA256=`343483bb5dedffa489661b402e396bbbfc87be67159fa94be90e4745048fc6e9`/`96e1fe394a5054c85da6d80089c7fd7f9f3222498010c9623cd6a78e65ff0a01`；48 条 `v_mmac_f32_16x16x16_f16`，12 条 `buffer_load_dword ... lds` | 计算与 4B LDS direct-load 均真实生成 |
| P5.24-E5 | gfx936 修复前后 correctness/binary | 两次均 PASS；binary 始终 `67710c19f123726692a1edd47ea77d6d880becbecb99c854bb4d77dcad95966e`，log 始终 `c7af2578a9093a93caa525fe7f1a1c62a8f9f3ec9ad5f06a18469e02b8c58cf9` | G4 PASS；binary identity |
| P5.24-E6 | gfx938 修复前后 correctness/binary | 两次均 PASS；binary 始终 `609e30f29c3fdb200fd2d7b1e7bd8ab85f2b8576852751aac6f595aa8b62a059`，log 始终 `c7af2578a9093a93caa525fe7f1a1c62a8f9f3ec9ad5f06a18469e02b8c58cf9` | G4 PASS；binary identity |
| P5.24-E7 | 三架构 aggregate build | `example_grouped_conv_fwd_bias_activation_mmac` 在 gfx946/gfx936/gfx938 均 `AGGREGATE_BUILD=PASS` | enrollment closure PASS |

PMD 仅记录 pending-cacheline warning，真实 dispatch 与完整 CPU reference 均通过。gfx946 参数由 `CK_ENABLE_HCU_GFX946_INSTANCES` 独立选择，common helper 只新增 gfx946 分支；gfx936/gfx938 修复前后 binary 逐字相同，已直接排除性能变化，不执行性能 A/B。gfx946 未测试性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（true-GPU binary identity）。本 example 单独提交后进入下一批未入图能力。

### Step 5.25：BatchNorm forward-inference NHWC FP16 unique target — PASS-PMD（2026-09-02）

- 修正 P4.20 盘点结论：三个 NHWC runner 不在 example_hcu/08_batchnorm 同目录，而是已经存在于同仓 example/34_batchnorm；本步没有跨树移植或复制源码。
- 以唯一 target example_batchnorm_forward_inferring_nhwc 复用现有 inference runner，避免与 NCHW HCU target 以及 regular example target 重名；forward-training/backward 仍保持注释，继续遵守“一 example 一 commit”。
- 正式 correctness case：FP16/NHWC、N=2,H=4,W=4,C=64、verify=1,init=2,time=0。三端远端 runner/CMake SHA256 分别一致为 f023c335f155eca0bbaf394fdd942cb9ddbf56fdfd56348d0541d34a9ab66fd5/62f6ab1898831877f6e9c64ecca4f461308db43586363943f9d202cf0fe68591。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.25-E0 | 同仓 runner 与 AMD 参考核对 | 算法主体一致；目标 checkout 只有版权行和 namespace 写法差异 | G0/G1 PASS；复用目标仓现有来源 |
| P5.25-E1 | gfx946 focused build/run | PASS；binary/log SHA256=39ba7f63c5f17dda629707ff23289f97e8bf4fd1cf2e286071f688b568a5a0a7/d601b58dad888083b46b2c46da8358f5c37b33288ac55144dcab3b9544ad061e，RUN_RC=0 | G2/G3 PASS；实际 packet 执行，未测试性能 |
| P5.25-E2 | gfx936 同一 correctness case | PASS；binary/log SHA256=40574e5c825d56efd938d9d9d0e04ed7e7b7c0d750ec465a93d38fcb1b0503fb/空日志哈希 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855，RUN_RC=0 | G4 PASS |
| P5.25-E3 | gfx938 同一 correctness case | PASS；binary/log SHA256=8002613273779e0314f34b3feb398f25a9f958eb22f3892f9fd7835515260c0e/空日志哈希 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855，RUN_RC=0 | G4 PASS |

PMD 报告 Inst_VOP3__V_DIV_FIXUP_F32 not test、VOP3P__V_MAD_MIXLO_F16 not test 和 pending-cacheline warning，但 reference correctness 与进程退出码均通过，故这些告警在本 runner 中为非阻塞记录。候选只新增 CMake target，不改变 runner、device/kernel 源码或已有 target 的编译参数，因此 gfx936/gfx938 不存在 kernel 性能变化，不执行性能 A/B；gfx946 未测试性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（build-graph-only enrollment）。本 example 单独提交后继续 BatchNorm forward-training NHWC。

### Step 5.26：BatchNorm forward-training NHWC FP16 unique target — PASS-PMD（2026-09-02）

- 以唯一 target example_batchnorm_forward_training_nhwc 复用同仓 example/34_batchnorm runner；不修改算法、device 实现或既有 NCHW target。
- 正式 case：FP16/NHWC、N=2,H=4,W=4,C=64、verify=1、update-moving-average=1、save-mean/inv-variance=1、init=2、time=0、single-block Welford。该合同同时比较 normalized output、running mean/variance 与 saved mean/inv-variance。
- 三端远端 runner/CMake SHA256 分别一致为 d3f56595268c2dcab2cbfb14603d610ac7866da732c52a9114b17c2ee60a85d8/2d22ae582b1bd8e5fb9e6d6dcc2f1372577c9218812c188fd2e7d0eba64064a3。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.26-E0 | 同仓 runner 与 AMD 参考核对 | 算法主体一致；差异为版权/namespace 及参考错误消息增强 | G0/G1 PASS；复用目标仓现有来源 |
| P5.26-E1 | gfx946 focused build/run | PASS；一个 grid=(1024,1,1)、wg=(256,1,1) kernel dispatch；binary/log SHA256=c96a7d5eba0534f5cbd853c51482771681659f96dd632006f24f7fd38b8c5867/3b658e17b40be09d28c0bf0cdfbe5d67cd0de4cdb19e8372240943905fe8da04，RUN_RC=0 | G2/G3 PASS；未测试性能 |
| P5.26-E2 | gfx936 同一 correctness case | PASS；binary/log SHA256=fa0e98d5571684124643bbbd99f7ccd4d0025937f001f49eb7ab2d659ed27aff/空日志哈希 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855，RUN_RC=0 | G4 PASS |
| P5.26-E3 | gfx938 同一 correctness case | PASS；binary/log SHA256=26cc34e2f9f686ec040cbaeb0aea6f1e7e1ac0e949f4251f637b4f0c01fb8aca/空日志哈希 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855，RUN_RC=0 | G4 PASS |

PMD 再次报告 V_DIV_FIXUP_F32/V_MAD_MIXLO_F16 未测试及 pending-cacheline warning，但完整 reference contract 和退出码通过，继续作为非阻塞模拟器告警记录。候选仍只有 build-graph enrollment，未改变任何真机 kernel code object 的来源或编译参数，故 gfx936/gfx938 无性能 A/B 必要；gfx946 未测试性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（build-graph-only enrollment）。本 example 单独提交后继续 BatchNorm backward NHWC。

### Step 5.27：BatchNorm backward NHWC FP16 unique target — PASS-PMD（2026-09-02）

- 以唯一 target example_batchnorm_backward_nhwc 复用同仓 example/34_batchnorm runner；至此原 08_batchnorm 的三个 NHWC 注释声明均已消除重名并逐 example 接入。
- 正式 case：FP16/NHWC、N=2,H=4,W=4,C=64、verify=1、使用 saved mean/inv-variance、init=3、time=0、single-block Welford；reference 覆盖 dx、dscale、dbias。
- 三端远端 runner/CMake SHA256 分别一致为 8d386338d9ecc4fd5f414837188735962c35df8b3486d7d9f6babc5111fcf690/82947951a4ddade75342221c29a90c0dae82908b41aead0004b6841cd1bffd3c。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.27-E0 | 同仓 runner 与 AMD 参考核对 | 算法主体一致；差异仅版权/namespace 与 epsilon 作用域 | G0/G1 PASS；复用目标仓现有来源 |
| P5.27-E1 | gfx946 focused build/run | PASS；一个 grid=(1024,1,1)、wg=(256,1,1) kernel dispatch；binary/log SHA256=72cbd6b71c3c1b387994f635b6b29a9b3e5bfa4ea726d428b56c3b3d47c917/e69169cfbf00a665759b100b044f5144d07fd6044e0b03fee0095075a9c9e94c，RUN_RC=0 | G2/G3 PASS；未测试性能 |
| P5.27-E2 | gfx936 同一 correctness case | PASS；binary/log SHA256=13a4627a9f2049b10c93399318f89a4b28a41f711733775f2562644f2c8d8d5b/空日志哈希 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855，RUN_RC=0 | G4 PASS |
| P5.27-E3 | gfx938 同一 correctness case | PASS；binary/log SHA256=a649b2c56433cb31162c5031bf0d20531d407e510edb33c7dab084e37d554737/空日志哈希 e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855，RUN_RC=0 | G4 PASS |

PMD 的 V_DIV_FIXUP_F32 未测试和 pending-cacheline warning 未影响完整 backward reference contract，作为非阻塞模拟器告警记录。候选仍只有 build-graph enrollment，不改变 gfx936/gfx938 的已有 kernel 源或编译参数，故无需性能 A/B；gfx946 未测试性能。

门禁结果：G0-G4 PASS；G5 NOT RUN；G6 NOT RUN（build-graph-only enrollment）。本 example 单独提交后进入下一批未入图能力。

### Step 5.28：commented CK Tile fused-conv base Wasp V2R1 — real-GPU PASS / SKIP-PMD（2026-09-02）

- 临时以唯一 target tile_example_fused_conv_base 恢复 fused_conv.cpp enrollment。原 runner 强制计时一次、丢弃 run_fused_conv 返回值，导致 reference 即使通过仍退出 1；诊断候选新增默认保持开启的 time 参数，time=0 只发射一次 correctness kernel，并正确传播 parse/unsupported/reference 状态。
- 第一候选 N=1,K=128,C=32,H=8,W=16 在 gfx936/gfx938 均被 IsSupportedArgument 一致拒绝（日志哈希均为 1de414594b5576e083015130e74e0b3fd5a685b40bc860051be03d7d5a0d5ba2），随后采用 P4.2 健康合同 G=1,N=64,K=C=256,H=W=2,R=S=1、stride1/pad0。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.28-E0 | gfx936 原 runner 健康合同 | reference 无错误输出；runner 固有退出码错误使 RUN_RC=1；`.hip_fatbin` SHA256=bcd9ece93be10960dcb10ac6be64dc773ae26ba9fc954699ee19f3ce8f84f832，日志=4e71f53eedcc6c85c9be703ef19e46124d2fca90a35fcf2852befa8cd8402b62 | G0 healthy-control PASS |
| P5.28-E1 | gfx938 原 runner 健康合同 | reference 无错误输出；同样 RUN_RC=1；`.hip_fatbin` SHA256=e9ee8e55d9bd4c2a7c412ec2816765b0d1f927f76e3ca546e9223b1718d93a73，日志=770f49bc6eb5f23da2f5b1ab6f8dee0d06a10f7e7229dd01fd51b29f936cbe6e | G0 healthy-control PASS |
| P5.28-E2 | runner 候选 + gfx936/gfx938 | 两端 time=0/time=1 均 reference PASS、RUN_RC=0；time=0 日志哈希均为 f0b37ffa370870a836aa9c05775e3439a85c973825289264465dadf26f79924f | G4 PASS；默认行为 smoke PASS |
| P5.28-E3 | 真机 device payload 对照 | gfx936 baseline/candidate `.hip_fatbin` 均为 bcd9ece9...f832；gfx938 均为 e9ee8e55...3a73；两端 cmp PASS | G6 PASS；排除 kernel/性能变化 |
| P5.28-E4 | gfx946 time=0 focused build/run | 真实 grid=(1024,1,1)、wg=(512,1,1) dispatch，但 37670/65536（57.47986%）错误，max err=0.02471924；binary/log SHA256=585830961ad487ef33197d4a4f870037441cf91209a509a00681d8289e37c0eb/955a7c1319d9cc850c376f5bb85f090d35c9f7b78352174d3ce263b8f133f185，RUN_RC=1 | G3 SKIP-PMD |
| P5.28-E5 | PMD 终止证据 | dispatch 中明确报告 src1_neg should not set 与 src1_abs should not set，随后产生部分错误结果 | Wasp V2R1 modifier 解码/执行链缺口 |
| P5.28-E6 | 撤回 runner/enrollment 并重生成 | gfx946/gfx936/gfx938 均 TARGET_ABSENT | 未保留未通过 PMD correctness 的 target |

两台真实 GPU 已证明健康合同和 Wasp V2R1 kernel 正确，且 runner 修复前后 device payload 逐字一致；gfx946 PMD 的 modifier 告警与部分错误结果构成明确的模拟器执行缺口。本任务不改写真机已正确的 Wasp pipeline，按规则记录并跳过该模块。最终源码/build graph 全部撤回，真机无性能变化；无需额外性能 A/B，gfx946 全程 time=0。

门禁结果：G0-G2/G4/G6 PASS；G3 SKIP-PMD；G5 NOT RUN。下一切片继续 fused_conv_tls.cpp，并仍先修正 correctness-only runner 后判断 PMD。

### Step 5.29：commented CK Tile fused-conv TLS V1 — SKIP-BASELINE（2026-09-02）

- 临时以唯一 target tile_example_fused_conv_tls 恢复 fused_conv_tls.cpp enrollment；先在两台真实 GPU 使用 runner 默认 F3x3_S1_D1_P1 合同 G=1,N=32,K=C=256,H=W=4。
- 该 runner 与 P5.28 一样强制计时且反转成功退出码，但这里 reference 本身已在两端失败，因此不再制作 correctness-only runner 候选，也不启动 gfx946 PMD。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.29-E0 | gfx936 默认健康控制 | FAIL；129470/131072（98.77777%）错误、输出近全零，max err=0.08221436；binary/`.hip_fatbin`/log SHA256=5cb8cc1d1fb604115e49a253265b86b32a1fb555268b760c80d417c4e0320d03/1d3a26a2fc1d746fad3f7d968f17192ab5d1e550521b87a5f944260f46188a47/46314c74c800c0d77fb9888651e8217d6c7df5ef509ecbed3d4bd802ba02d725 | G0 baseline FAIL |
| P5.29-E1 | gfx938 同一健康控制 | FAIL；131035/131072（99.97177%）错误、输出 NaN，max err=65504.03；binary/`.hip_fatbin`/log SHA256=eedf0e716ba0fd640164d28d7557eff005c20ca4c9cdbb327b99fd76e9c2e937/74258c5cccc60170eaab0aab409181d0dc4255ecfa8c6dfeee3431724ecff665/f1e7a15d8407b31687c2095f71ce5ba4ba4d8d11b5de70ae6dbd651cbf2cb66a | G0/G4 baseline FAIL |
| P5.29-E2 | gfx946 PMD | NOT RUN；两个真实架构已证明默认 TLS V1 baseline 损坏 | 不作 Shaobo/PMD 归因 |
| P5.29-E3 | 撤回 enrollment 并重生成 | gfx946/gfx936/gfx938 均 TARGET_ABSENT | 恢复原始注释状态 |

两端失败表现不完全相同，但完整 reference contract 在 gfx936 与 gfx938 上都未通过，足以否决该长期注释 example 的健康基线。本适配不扩展为跨架构 TLS V1 算法修复；最终没有源码或 build-graph candidate，故不存在性能变化面，也无需性能 A/B。Perf Model 未运行。

门禁结果：G0 FAIL；G1/G2 PASS；G3 NOT RUN；G4 FAIL；G5/G6 NOT RUN；overall SKIP-BASELINE。下一切片继续 fused_conv_mls.cpp。

### Step 5.30：CK Tile fused-conv MLS V1 unique target — PASS-PMD（2026-09-02）

- 以唯一 target tile_example_fused_conv_mls 恢复 fused_conv_mls.cpp enrollment。原 runner 强制计时且丢弃 run_fused_conv 返回值；最小修复新增默认保持开启的 time 参数，time=0 单次 correctness launch，并正确传播 parse/spatial-dim/reference 状态。
- 正式合同为 runner 默认 F1x1_S1_D1_P0：G=1,N=32,K=C=256,H=W=4。gfx936 对 MLS 路径为预期 unsupported，gfx938 与 gfx946 为 correctness 目标。
- 三端 candidate 源/CMake SHA256 均为 85c9b79de199630bbc48e718b16c4b209782e8f7ee529a54f3df57aae1bc75ac/5c5a6fd6345374639bc356c33ec36ef778e29c92d23ea9a52dad78ec1f9f922a。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.30-E0 | gfx936 baseline/candidate | 两版均构建成功并在 IsSupportedArgument 拒绝，日志 SHA256=672fad2ece233c7314b2277217039996e0331c377ea7f87b40c82bb6ad81fb7d；`.hip_fatbin` 均为 4d80286b4618520bc5486e0b3e8411a9498ef3c150c3c79312ea67d5ceca61a6、cmp PASS | G4 architecture-bound PASS |
| P5.30-E1 | gfx938 原 runner 健康控制 | reference 无错误；原退出码反转为 RUN_RC=1；binary/`.hip_fatbin`/log SHA256=ea0f6b750692182da362e9c77f1f20c744efec00be5df0ed20663c56ef0214c1/55842d586ba50f3a951d60930e691ea6ce928a31a8bf929a5733ff9494492d69/ff0145c806095ccbfd736e614bafeeb7a429d7c5b714242654cbeaa5a68e6678 | G0 healthy-control PASS |
| P5.30-E2 | gfx938 candidate time=0 | reference PASS、RUN_RC=0；log SHA256=407b6bcc5dcd168788fb17c2a04e403165fcaf2010cc8aa642f1f7a0aaa6062c；candidate `.hip_fatbin` 仍为 55842d58...2d69、cmp PASS | G4/G6 PASS |
| P5.30-E3 | gfx946 candidate time=0 | PASS；grid=(2048,1,1)、wg=(256,1,1) 实际 dispatch；binary/log SHA256=7469bd8751423be56b1881458d44176c866e9ff379d755eac76016672c774d65/33c9cc04579737710a0a4ad666aafa75ba2a4aecd49452ec84ceee5246367237，RUN_RC=0 | G2/G3 PASS；未测试性能 |
| P5.30-E4 | PMD matrix-load 观察 | 报告 matrix_load_32x32_b16 与 ds_read_matrix_trans 处于 testing，但完整 reference 与退出码通过 | 非阻塞模拟器告警 |

该非-Wasp MLS V1 路径在 gfx946 PMD 正确，不沿用 P4.2 MLS-Wasp 的 SKIP-PMD 结论。runner 改动只影响 host launch/状态传播，gfx936/gfx938 baseline/candidate device payload 均逐字一致，已直接排除真机 kernel 性能变化，因此不执行性能 A/B；gfx946 全程 time=0。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 unique target 与 correctness-only runner，并在本 example 单独提交后继续 fused_conv_cshuffle.cpp。

### Step 5.31：commented CK Tile fused-conv CShuffle Wasp V2R1 — real-GPU PASS / SKIP-PMD（2026-09-02）

- 临时以唯一 target tile_example_fused_conv_cshuffle 恢复 fused_conv_cshuffle.cpp；正式健康合同为 G=1,N=64,K=C=256,H=W=2,R=S=1、stride1/pad0。
- 原 runner 同样强制计时并丢弃 correctness 返回值；诊断候选新增默认保持开启的 time，time=0 单次发射且传播 reference 状态。该变体使用 PassThrough epilogue 与 MWGs=2/NWGs=1，独立于 P5.28 AddRelu/base 配置。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.31-E0 | gfx936/gfx938 原 runner 健康控制 | 两端 reference 无错误，但原退出码反转为 RUN_RC=1；binary 分别 fe246c70257423021ec0bc86875a61e3a1b8003e983dbe9537d107c92c6c0ac1/86d652e050ec2704b4837fef1190805546ee7e499766c5c3c3f06508fc18a81a | G0 healthy-control PASS |
| P5.31-E1 | gfx936/gfx938 candidate time=0 | 两端 reference PASS、RUN_RC=0，日志 SHA256 均为 f0b37ffa370870a836aa9c05775e3439a85c973825289264465dadf26f79924f | G4 PASS |
| P5.31-E2 | 真机 device payload 对照 | gfx936 baseline/candidate `.hip_fatbin` 均为 756b080949422027ffa1ccf1d249838da40168dd012a2609090546aadb96b29b；gfx938 均为 999ea082040cee6c1f9a898db7d147cea6f5f869ba1144f14adc5172ba0187be；两端 cmp PASS | G6 PASS；排除 kernel/性能变化 |
| P5.31-E3 | gfx946 candidate time=0 | 真实 grid=(1024,1,1)、wg=(512,1,1) dispatch，但 53823/65536（82.12738%）错误，max err=0.03041077；binary/log SHA256=671e4499c52ab295f3d2b55d249023e58eebc9e66bd1d40aa1a6cde642fe63d4/0eed179b5c9de866300bf8316b77e0b8ca724fbf39767958ebf8b5053541b85c，RUN_RC=1 | G3 SKIP-PMD |
| P5.31-E4 | PMD 终止证据 | 与 P5.28 独立复现 src1_neg/src1_abs should not set，随后产生部分错误输出 | Wasp V2R1 modifier 解码/执行链缺口 |
| P5.31-E5 | 撤回 runner/enrollment 并重生成 | gfx946/gfx936/gfx938 均 TARGET_ABSENT | 未保留未通过 PMD correctness 的 target |

PassThrough/CShuffle 配置也在相同 Wasp V2R1 pipeline 上复现 PMD modifier 告警和错误，进一步把问题限定到该 pipeline 执行链，而非 AddRelu 单点。按规则不改写真机正确 kernel，最终源码/build graph 全撤回；device payload identity 已排除真机性能变化，无需额外性能 A/B。gfx946 全程 time=0。

门禁结果：G0-G2/G4/G6 PASS；G3 SKIP-PMD；G5 NOT RUN。下一切片继续 fused_conv_tls_wasp.cpp。

### Step 5.32：commented CK Tile fused-conv TLS-Wasp V1 — SKIP-BASELINE（2026-09-02）

- 临时以唯一 target tile_example_fused_conv_tls_wasp 恢复 fused_conv_tls_wasp.cpp；该 runner 配置 F3x3_S1_D1_P1、MPerWG=NPerWG=128，并使用 TLS-Wasp V1 pipeline。
- runner 默认 G=1,N=64,K=64,C=128,H=W=2 因 K 小于固定 N tile，在 gfx936/gfx938 均被 IsSupportedArgument 拒绝，日志 SHA256 均为 5a46e8f6bdf89da6363ea0288bc54bd9bdaec50af1c487942270642e3c45553a。
- 仅把 K 提升为 128 做最小合法化 A/B；两端均能发射 kernel，但 reference 都失败。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.32-E0 | gfx936 K=128 健康控制 | FAIL；17360/32768（52.97852%）错误、max err=0.03250122；binary/`.hip_fatbin`/log SHA256=a0dc5ba6d430a4bd02265f4d37a65abf9817f61e5ebf0d95ca2ea7522b4abaa5/836080471ea5dd64aa53194748f4a88a669f32ac6d889391df7163f44c9bab9c/400e108a40cf2a041941f2cbe7a30dceddd71262d2d8230152e9ffaeccc12db0 | G0 baseline FAIL |
| P5.32-E1 | gfx938 同一 K=128 case | FAIL；21266/32768（64.89868%）错误、max err=65504；binary/`.hip_fatbin`/log SHA256=a850569fef50ceb78e046cd1eeca8e46c5a723a350a664f0b6ecdc973715fc39/4356ac625298cd189b9bb7a31668b92467172534df857c3c57ed071fbf7ba104/7c8ada1a5f41b26088d481796641015005417cc7e204746745d72dde287c2b2b | G0/G4 baseline FAIL |
| P5.32-E2 | gfx946 PMD | NOT RUN；默认 case 不支持，最小合法化 case 又在两台真机 reference 失败 | 不作 Shaobo/PMD 归因 |
| P5.32-E3 | 撤回 enrollment 并重生成 | gfx946/gfx936/gfx938 均 TARGET_ABSENT | 恢复原始注释状态 |

默认参数与固定 tile 不匹配，修正唯一明显约束后又在两个真实架构上同时错误，证明该长期注释 TLS-Wasp V1 runner 没有健康验收基线。本适配不扩展为跨架构 pipeline 修复；最终无源码/build-graph candidate，真机无性能变化面，无需性能 A/B。Perf Model 未运行。

门禁结果：G0 FAIL；G1/G2 PASS；G3 NOT RUN；G4 FAIL；G5/G6 NOT RUN；overall SKIP-BASELINE。下一切片继续 fused_conv_tls_wasp_persistant.cpp。

### Step 5.33：CK Tile fused-conv TLS-Wasp persistent V1 gfx946-only target — PASS-PMD（2026-09-02）

- 原 runner 默认 `K=64` 小于固定 `NPerWG=128`，因此先把默认值修正为可执行的 `K=128`；同时新增默认保持开启的 `time` 参数，`time=0` 只发射一次，并把 parse、spatial-dim 与 reference 结果正确传播到进程退出码。
- 初始跨架构 enrollment 暴露了清晰边界：gfx936 在 `IsSupportedArgument` 拒绝；gfx938 编译期报 `warp_dsreadm_format_attribute_impl.hpp:100: error: unsupport ds read matrix trans format on this GPU`。最终 target 仅在 `GPU_TARGETS MATCHES "gfx946"` 时注册，不把 gfx946 专属 persistent TLS-Wasp 路径强加给两台既有架构。
- 最终三端源码/CMake SHA256 均为 d4221f6a78205a00d0b55bb3fdaf1f256a5289a6c9a90f59a1a84df802474b45/4adb8f3e220b263593bd3886ca68b4ae024678bb2ba0d1a072b08a03ec6dcf39。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.33-E0 | gfx936 初始跨架构 candidate | 构建成功；G=1,N=64,K=C=128,H=W=2,R=S=3 时由 `IsSupportedArgument` 拒绝，日志 SHA256=4d29640b3a42064733545ceb53333af4328a096449d0360dde75a0730df57908 | 架构能力边界，不作为 Shaobo 错误 |
| P5.33-E1 | gfx938 初始跨架构 candidate | 编译期被 `unsupport ds read matrix trans format on this GPU` 拒绝 | 证明 target 不能进入 gfx938 build graph |
| P5.33-E2 | gfx946 最终 target、time=0 | reference PASS、RUN_RC=0；实际 grid=(36864,1,1)、wg=(768,1,1)；binary/log SHA256=eeb57dde0a3d9841e5a538e399febccc1319fd4c50d4d481786acdb48a51a47e/6df4a1ff18a18f18aff428e0927a26a995f28669d9b821705bd65a9c9650ef13 | G2/G3 PASS；未测试性能 |
| P5.33-E3 | PMD 指令观察 | 报告 ds_read_matrix_trans testing、src1_sext/neg/abs 与 TLS testing/pending cacheline 告警，但完整 reference 通过 | 非阻塞模拟器告警，保留原文便于后续 PMD 修复 |
| P5.33-E4 | gfx936/gfx938 最终 build graph 回归 | 两端 `tile_example_fused_conv_tls_wasp_persistant` 均 TARGET_ABSENT；既有 `tile_example_fused_conv` 与 P5.30 `tile_example_fused_conv_mls` 均构建通过 | G4/G6 PASS |

最终变更对 gfx936/gfx938 是严格的 build-graph no-op，既有 target 和设备 kernel 均未改动，所以不存在待测的真机性能影响，不执行性能 A/B；若后续放宽该架构 guard，则必须重新执行 correctness 与性能门禁。gfx946 仅做 time=0 correctness。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 gfx946-only unique target 与 correctness-only runner，并在独立提交后继续 `fused_conv_mls_wasp_persistant.cpp`。

### Step 5.34：commented CK Tile fused-conv MLS-Wasp persistent V1 — real-GPU PASS / SKIP-PMD（2026-09-02）

- 临时以唯一 target `tile_example_fused_conv_mls_wasp_persistant` 恢复 enrollment；先在 gfx938 使用原 runner 建立健康控制，再以与 P4.2 相同的架构选择方式把硬编码 gfx938 改为构建期 `HcuArch`，并加入 `time=0` 单次 correctness 发射和正确退出码传播。
- 正式合同为 G=1,N=64,K=C=256,H=W=2,R=S=1、stride1/pad0；它使用 persistent partitioner 与 AddRelu epilogue，不能由 P4.2 的非 persistent/PassThrough probe 替代。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P5.34-E0 | gfx938 原 runner 健康控制 | reference 无错误，旧 runner 因丢弃结果而 RUN_RC=1；binary/`.hip_fatbin`/log SHA256=8953d4c225213a55e5b61e8aecb7dff0469742ee3325ea0e9d8fac7fc2248a82/8a05b1a884e6db14727ad8a9f7a72dd989cfcb29ac42cfaad8ae0f922e723c40/74b683dee28115e4dc760e20be9e681a0a0ca660394d1c47d54dc29d46a09bae | G0 healthy-control PASS |
| P5.34-E1 | gfx938 candidate time=0 | reference PASS、RUN_RC=0；binary/log SHA256=a6c173d71cf0e7c26f9eb62599e339dc4884f15bd34dc487e48968bf4271bddd/f0b37ffa370870a836aa9c05775e3439a85c973825289264465dadf26f79924f | G4 PASS |
| P5.34-E2 | gfx938 device payload 对照 | baseline/candidate `.hip_fatbin` 均为 8a05b1a884e6db14727ad8a9f7a72dd989cfcb29ac42cfaad8ae0f922e723c40，cmp PASS | G6 PASS；排除真机 kernel/性能变化 |
| P5.34-E3 | gfx946 candidate time=0 | 实际 persistent dispatch grid=(73728,1,1)、wg=(768,1,1)，但 34885/65536（53.23029%）错误、max err=0.03067017；binary/log SHA256=30b97c6851f73ac01697e3ab3048a9294117fd615527fe8654f7ec85557be97f/e4b161810ccad67501b240e2fc54e1f741e453d4244aefd39a93df524392b687，RUN_RC=1 | G3 SKIP-PMD；未测试性能 |
| P5.34-E4 | PMD 终止证据 | dispatch 明确报告 `ds_read_matrix_trans : testing`，随后部分输出为零 | 与 P4.2 MLS-Wasp transpose-format DSReadM 缺口一致 |
| P5.34-E5 | 撤回 enrollment/runner 并重生成 | gfx946/gfx936/gfx938 均 TARGET_ABSENT；三端既有 fused-conv target focused build PASS | 恢复原始注释状态 |

同一健康合同在真实 gfx938 正确，且架构选择/host runner 修改没有改变 gfx938 device payload；gfx946 PMD 的 DSReadM 告警与部分零输出构成明确的模拟器执行缺口。按规则不为模拟器改写真机正确的 MLS-Wasp pipeline，最终代码和 build graph 全部撤回。gfx936/gfx938 没有最终性能变化面，且候选 payload identity 已直接消除不确定性，所以不执行真实性能 A/B；gfx946 全程 time=0。

门禁结果：G0-G2/G4/G6 PASS；G3 SKIP-PMD；G5 NOT RUN。至此当前 checkout 内 36 条 commented executable declaration 均已逐项归类；下一阶段按既定优先级从 AMD 参考树选择性迁移 `04_img2col`，先冻结最小依赖闭包而非复制整个目录。

### Step 6.1：AMD-to-HCU CK Tile img2col minimal closure — PASS-PMD（2026-09-03）

- 选择性迁移来源为 AMD `example/ck_tile/04_img2col`：原始功能提交 `de3e3b642402eac5b4a466f6a2fa5e9f022ba680`，并带入 `78082855d8195467c36eabb5cd2428b94777231d` 对 `time=0` 的除零/伪性能输出修复；没有复制 AMD 当前整个 example 树。
- HCU checkout 已有 `include/ck_tile/ops/image_to_column{,/kernel,/pipeline}` 与 `host/reference/reference_im2col.hpp`，且 `host.hpp` 已公开 reference；本步最小闭包仅新增 `example_hcu/ck_tile/04_img2col/{CMakeLists.txt,image_to_column.hpp,image_to_column.cpp}`，并在父 CMake 恢复子目录。target 使用 `EXCLUDE_FROM_ALL`，不进入 all/install/check。
- 保留与本树同代的 `make_kernel<blocks.x,kBlockPerCu>` API；审计时另发现上游 runner 把 `argv[4]` spatial-dim 重复作为 G 解析，本迁移把 `parse_conv_param` 起始位置修正为 `threshold_to_catch_partial_args + 1`。首轮偏移用例不计入正式验收。
- 正式 correctness 合同为 FP16 NHWGC，G=1,N=1,K=8,C=8，F=3x3、H=W=4、stride/dilation=1、左右 pad=1，verification=1、integer init、time=0；覆盖 K=72 的 tile 尾部和 padding 取值。
- 三端父 CMake/子 CMake/header/source SHA256 均为 a6eba15aaa763ecc2a6f063dd5cf09c950fc6d87ec4e8469868029eaf4507e9f/6cd6e216699cc5a2afece98082a271d12078f1c2dd605080943d8720686eb8b6/6c5bab2b7734781445d5461272be64ccaed60887a696f14d703c2f6a3e04eaf9/235e0e94bd17f5038d8d2d7c40ef3b4d6e6e3a6c4f293c78d93bff07b768b476。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.1-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=73a725312642f84da79bec46d68693f9dea076b4a165eeb8d5e69880755c2f71/bfd7842ebf57e71cb98c36670e50ee95a6b0698c1cabc086c14e7bb4643823c4/3a1ddad95d2bf18733e8991c6cf37a44e16947c0690b80ffc8afc96aff8aa983 | G1/G2/G4 PASS |
| P6.1-E1 | gfx946 PMD、正式 time=0 case | 单次 dispatch，grid=(256,1,1)、wg=(256,1,1)，`valid:y`、RUN_RC=0；日志 SHA256=51ca2eb94ac7f713f30fd5d7853b91f48f7c01a87ced52cffd1a1c36b78c4f3c | G3 PASS；未测试性能 |
| P6.1-E2 | gfx936 同一 correctness case | `timing:off`、`valid:y`、RUN_RC=0；日志 SHA256=bc2c473c49decc0d642b0aa4558cbbcba892a5ee94dee8f8567f4a0869c722c0 | G4 PASS |
| P6.1-E3 | gfx938 同一 correctness case | `timing:off`、`valid:y`、RUN_RC=0；日志 SHA256=bc2c473c49decc0d642b0aa4558cbbcba892a5ee94dee8f8567f4a0869c722c0 | G4 PASS |

本步没有修改任何既有 device/header 实现；新增 target 又是 `EXCLUDE_FROM_ALL`，不会重编译、替换或改变 gfx936/gfx938 的已有 kernel，因此不存在不确定的真机性能影响，不执行性能 A/B。gfx946 只运行 correctness 单发射。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 img2col 最小闭包并独立提交；下一切片 P6.2 审计 `05_reduce` 的 example 与 HCU 已有 reduce kernel/header 闭包。

### Step 6.2：AMD-to-HCU basic CK Tile reduce minimal closure — PASS-PMD（2026-09-03）

- AMD 当前 `05_reduce` 已在 `ea10a782036688cdc2a91266f675125bf1c5c59d` 重构为 generic kernel/pipeline，并新增 threadwise/multiblock runners；HCU 树没有该整层 API。为避免不受控跨版本移植，本步以与 HCU 现存 block-reduce API 同代的 `0b8f117f1ae765f0e490368de7f0c5d7591b17b6` basic FP16 Add reduction 为最小闭包。
- 新增 `example_hcu/ck_tile/05_reduce/{CMakeLists.txt,reduce.hpp,reduce.cpp}` 并恢复父子目录；只复用 HCU 已有 `BlockReduce2dDefaultPolicy`、block reduce 与 host reference，不修改公共 reduce header。target 为 `EXCLUDE_FROM_ALL`，不带入 AMD CMake 的 global `RULE_MESSAGES` 副作用。
- runner 新增默认保持上游行为的 `time=1`，正式 PMD 验收使用 `time=0`，将 warmup/repeat 强制收口为 0/1 并输出 `timing:off`。正式 shape 为 M=N=128、FP16、Add、CPU validation=1，恰好覆盖一个 256-thread block 与一次 N tile reduction。
- 三端父 CMake/子 CMake/header/source SHA256 均为 afba38521c3251a7839f7da921e016ce7e758ab5baeb60b91b436a46fc131e1e/efcbbcacfb0d0e4ca7b9276f2bd62858461654ecee6c5c0fcfa626cf8854598a/bdbe7532f0721e5d82ddb18a5745b9508ff2a18412981bc7fbe2361c14508b01/734046cb5840265c244a3164270e4d65ce74dc94b51eecc3175aa8be4dcb91c7。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.2-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=3f9a9a0c3bdfc7ff1e9949eee3b7fb29f9e5c1f3542cc45b609f4f718ca6a9f6/3a9dfd03905749c461c621d8fb7b039f801896239a49cab61fe36b53d9e8d4f8/4cb004c466a5dc9c7cffa99cc87c98f650cabd8314873b0c6e30249e54b3adfd | G1/G2/G4 PASS |
| P6.2-E1 | gfx946 PMD time=0 | 单次 grid=(256,1,1)、wg=(256,1,1) dispatch，`valid:y`、RUN_RC=0；日志 SHA256=4ea1a1e830e09a7656e6965c15dcf509dc38bbd0784b36d32840b6415cda89e9 | G3 PASS；未测试性能 |
| P6.2-E2 | PMD 指令观察 | 报告 src1_sext/neg/abs modifier 与 pending-cacheline warning，但完整 reduction/reference 正确 | 非阻塞模拟器告警 |
| P6.2-E3 | gfx936/gfx938 同一 correctness case | 两端均 `timing:off`、`valid:y`、RUN_RC=0；日志 SHA256 均为 783b21d9dca546935e245a2ce40c7791f321303db47337eccd9bdb37a90aa8a8 | G4 PASS |

公共 device/header 完全未改，新 target 又不进入默认构建图，因此不会改变 gfx936/gfx938 既有 binary 或 dispatch，真机性能影响可直接排除，不执行性能 A/B；gfx946 只做 correctness。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 basic reduce 最小闭包并独立提交；AMD 新版 multi-reduce API 作为后续独立能力，不与本 example 混合。下一切片 P6.3 审计 `06_permute`。

### Step 6.3：AMD-to-HCU generic CK Tile permute minimal closure — PASS-PMD（2026-09-03）

- 迁移来源为 generic permute 初始提交 `9fbd72e97e34f530ae370527755b655bf390d9ee`，与 HCU checkout 已有 `GenericPermuteProblem/GenericPermute/reference_permute` API 同代；最小闭包只新增 `example_hcu/ck_tile/06_permute/{CMakeLists.txt,permute.hpp,permute.cpp}` 并恢复父子目录。
- 不迁移同一 AMD example 中可选的 `alternative_impl/matrix_core_swizzle`：它实例化 GEMM/MMAC 专用 swizzle，是独立优化路径而非 generic permute 的依赖。保留 FP8/FP16/FP32 generic dispatch，当前切片仅验收 FP16。
- runner 新增默认保持上游行为的 `time=1`；`time=0` 强制 warmup/repeat=0/1 并输出 `timing:off`。正式 case 为 shape=[2,3,4,5]、perm=[2,0,3,1]、FP16、固定 seed、bit-exact CPU validation；120 个元素还覆盖 256-thread block 的线程尾部 guard。
- 三端父 CMake/子 CMake/header/source SHA256 均为 a2e3a5f24f443efa2d5bfbe6eb44c56884b8ad37432e02cc14315f3b6c1ebe50/f2a6802ef6c3b0051a2951f5eba650a8558f49c1dd2920cab239d7b7f0b26378/9b481edd596c8fb07ca56a78be05cf340f09353235887218fb51dbf636783052/6887e3a112fe8e5eb4cef5141241e68b3fc46df7158415211421b9d68617ef00。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.3-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=59731872018964173d958b881e58f53b5c394fb764d314b0b0866a7bcc079d50/18c54920ef3f8cfebcdf5d11491c401aa97931146e598ec5e0a65da52c5b5487/5687fd854af14a52284a6f7273b3a967ed98083962910dfe2b520c1da55e6b3b | G1/G2/G4 PASS |
| P6.3-E1 | gfx946 PMD time=0 | 单次 grid=(256,1,1)、wg=(256,1,1) dispatch，bit-exact `valid:y`、RUN_RC=0；日志 SHA256=594bd2ac3a46c3a9ee106e1989490fb26f66acc5c7469639c7709c1e443e52c7 | G3 PASS；未测试性能 |
| P6.3-E2 | PMD 指令观察 | 报告 `ignore GPUVM APE` 与 pending-cacheline warning，但完整排列/reference 正确 | 非阻塞模拟器告警 |
| P6.3-E3 | gfx936/gfx938 同一 correctness case | 两端均 `timing:off`、bit-exact `valid:y`、RUN_RC=0；日志 SHA256 均为 b5e9bce687d64678c2ede9a547e3f82130d7577b071da853c18cd17b0f9b40aa | G4 PASS |

公共 permute kernel/header 未修改，新增 target 为 `EXCLUDE_FROM_ALL`，既有 gfx936/gfx938 target、binary 和 dispatch 均不受影响；因此性能影响不是不确定项，不执行真实性能 A/B。gfx946 只做 correctness。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 generic permute 最小闭包并独立提交；下一切片 P6.4 审计 `12_smoothquant`。

### Step 6.4：AMD-to-HCU basic CK Tile smoothquant minimal closure — PASS-PMD（2026-09-03）

- HCU 已包含 smoothquant kernel、one/two-pass pipeline、default policy 和所需 host reference；本步以初始功能提交 `fbd654545a2644f99c3e7a493ebcc2169938583b` 的 standalone `example_smoothquant.cpp` 为同代 API 来源。
- 最小闭包只新增 `example_hcu/ck_tile/12_smoothquant/{CMakeLists.txt,example_smoothquant.cpp}` 并恢复父子目录。未迁移当前 AMD `tile_smoothquant` 的 20+ shape-specific instance/API 文件，也未引入 MoE smoothquant；这些是独立生产调度闭包。
- runner 新增默认保持上游行为的 `time=1`，`time=0` 强制 warmup/repeat=0/1 并输出 `timing:off`。正式 case 为 M=2,N=stride=128、FP16 输入、FP32 column scale/row scale、INT8 quant 输出、two-pass pipeline，CPU 同时校验 yscale 与 qy。
- 三端父 CMake/子 CMake/source SHA256 均为 4add24b44d677f5b4732c83f782f56d89056ffc0086ad74edcdaf45231f4a896/9737c3e909276617bc075e5f555b364c413f763aa5efe62f9cc69b5191a47a6b/b249ef5b105234ab6a0e84f42f5987aaecd866a8980d989df663d7ac741b30e5。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.4-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=1148cde5e63810951e2156f4f2f38e994d7111cd05c385440227117f35137e6b/1b918fce81d32946cfa184e730c83d50553c38d859ba53897b600813327ecb1a/736859f07afaf21f9fd7ba691431a56f20d8deaddca4b0663484f094f4193d32 | G1/G2/G4 PASS |
| P6.4-E1 | gfx946 PMD time=0 | 单次 grid=(256,1,1)、wg=(256,1,1) dispatch，yscale/qy reference `valid:y`、RUN_RC=0；日志 SHA256=5668ae80dc5648740d996c7463656f8bf7a87b5af6baeaa54b8975a218fa6426 | G3 PASS；未测试性能 |
| P6.4-E2 | PMD 指令观察 | 报告 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，但两份输出完整正确 | 非阻塞模拟器告警；该 kernel 不执行 dp4x |
| P6.4-E3 | gfx936/gfx938 同一 correctness case | 两端均 `timing:off`、`valid:y`、RUN_RC=0；日志 SHA256 均为 4934d1b3f5f3e963e170dae9c1760733bc9e4aa08f2390835c4ab1d72e4fd950 | G4 PASS |

公共 smoothquant/device header 未修改，新 target 是 `EXCLUDE_FROM_ALL`；gfx936/gfx938 既有 binary 与 dispatch 没有变化面，故不执行真实性能 A/B。gfx946 仅做 correctness，不使用模拟器 timing。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 basic smoothquant 最小闭包并独立提交；下一切片 P6.5 审计 `09_topk_softmax`。

### Step 6.5：AMD-to-HCU CK Tile topk-softmax minimal closure — PASS-PMD（2026-09-03）

- HCU checkout 已有 `TopkSoftmaxKernel`、warp-per-row pipeline、block top-k/softmax 与 CPU reference；本步以初始功能提交 `b098b71b05e4c06310f2e74056282a796f3cfd13` 的 example 为同代 API 来源，不迁移其提交中已在 HCU 树存在的公共基础设施。
- 最小闭包只新增 `example_hcu/ck_tile/09_topk_softmax/{CMakeLists.txt,topk_softmax_api.hpp,topk_softmax_api.cpp,topk_softmax.cpp}` 并恢复父子目录。target 为 `EXCLUDE_FROM_ALL`，只复用 `utility` 与既有公共 header。
- runner 新增默认保持上游行为的 `time=1`；`time=0` 强制 warmup/repeat=0/1 并输出 `timing:off`。正式 case 为 tokens=4、experts=8、topk=2、packed stride、FP16 input/FP32 weight、固定 seed=11939，逐 token 同时校验 softmax top-k value 与 index。
- 首轮三端均被同一个 host-side `stream_config.log_level` narrowing 诊断拦截；把本树 `int` 字段改为传入 `int` 后完成构建。该机械修复不涉及任何 device/架构分支。
- 三端父 CMake/子 CMake/API header/API source/runner SHA256 均为 cdea980b80b1de772199d55122987a713a8f64b735fb395361369b907d4d094b/329e0df919db9730ff33d41b07c0dbaf53dc618ee38dd18b6515c5325ea6851f/b132c1bd1fb89d2b136b43fe893f3dc1193c8366d4636a03f9dbfde6e1c08365/36523fbaf84dd0ab1ad8341fa1baaa4a9c746c11186c4521258038e5c338bf60/57f716c01a7c1dda73f3c0b1a6ed367bd99c4c0b768dc36408903204f36a9c8e。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.5-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=217bbc18e784757f354fdf7d710af11b27618dad4e4d7de2b9dd943325dcaf63/893dffdaa72ff451f7599ca00d6926db3aa05bf0ea54ddf53940a01f9d84bd76/1b10d1f983bc94e49d1fbc386c6e0c821c75585226ffc290153e836820f4c0d1 | G1/G2/G4 PASS |
| P6.5-E1 | gfx946 PMD time=0 | 单次 grid=(256,1,1)、wg=(256,1,1) dispatch，value/index reference `valid:y`、RUN_RC=0；日志 SHA256=f7e61e6d833db9fc56d4f0e91655728981bc7aba0b0ed861fcd3837b84155168 | G3 PASS；未测试性能 |
| P6.5-E2 | PMD 指令观察 | 报告 src1_sext/neg/abs modifier、`V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，但两类输出均正确 | 非阻塞模拟器告警 |
| P6.5-E3 | gfx936/gfx938 同一 correctness case | 两端均 `timing:off`、value/index `valid:y`、RUN_RC=0；日志 SHA256 均为 0c71fd15710e4426e92a4cf9778236cc62b54a444c1af24929a27cc166197a96 | G4 PASS |

公共 top-k/softmax/device header 未修改，新增 target 又是 `EXCLUDE_FROM_ALL`，不会改变 gfx936/gfx938 既有 binary 或 dispatch；因此不存在不确定的真机性能影响，不执行性能 A/B。gfx946 仅做 correctness 单发射。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 topk-softmax 最小闭包并独立提交；下一切片 P6.6 重新枚举当前 enabled target 与 AMD 参考树差异，按“已有公共依赖、可形成独立 correctness 闭包、无已知硬指令缺口”重排后续迁移优先级。

### Step 6.6：新增 target 与未入图能力覆盖复核 — COMPLETE（2026-09-03）

- P4.20 冻结的五个同代缺口 `04_img2col/05_reduce/06_permute/12_smoothquant/09_topk_softmax` 已在 P6.1-P6.5 全部形成独立 target，并完成 gfx946 PMD correctness 与 gfx936/gfx938 回归；当前 HCU 父 CMake 不再有注释子目录。
- AMD 当前目录与 HCU checkout 已明显跨代，不能再按目录编号机械复制。`16_batched_gemm/17_grouped_gemm/19_gemm_multi_d/20_grouped_convolution/22_gemm_multi_abd` 与 HCU 已有 legacy/CK Tile 验收面存在功能重叠，应先按新增 contract 而非目录存在性立项；`37_transpose` 当前只有 README，不是 executable target。
- `14_moe_smoothquant/18_flatmm/35_batched_transpose/36_pooling/40_streamk_gemm/41_batched_contraction/42_mx_gemm/49_sageattention/50_sparse_attn/51_tile_distr_enc_reg_map/52_cshuffle_lds` 均依赖 HCU 树尚无的独立公共算子或更新代 pipeline，不纳入“仅补 example”队列。
- 下一优先级选择 AMD `21_elementwise` 的 basic FP16 Add：HCU 已有 `tuple::apply`、binary reference 与 Add functor，最小新增公共闭包只需 generic `ElementWiseKernel`、shape/problem/default-policy，加一个独立 example。其 unary/transpose/4D variants 后续分别验收，不与首个 target 混合。

本步是只读覆盖审计与文档更新，没有 device 源码候选，不运行回归或性能 A/B。下一切片 P6.7 以 `606b0cc947cbab4c3313e3ca8c41c7b7647b6d25` 为来源实现 basic FP16 Add，并为 Perf Model 增加 `time=0`。

### Step 6.7：AMD-to-HCU basic CK Tile elementwise Add — PASS-PMD（2026-09-03）

- 迁移来源为 elementwise 初始提交 `606b0cc947cbab4c3313e3ca8c41c7b7647b6d25`。HCU 已有 `tuple::apply`、binary CPU reference 与 Add functor；最小公共闭包新增 `ElementWiseKernel`、`ElementWiseShape`、problem/default-policy 并从 `ops/elementwise.hpp` 公开。
- 首个 example 只注册 `EXCLUDE_FROM_ALL` 的 `tile_example_elementwise_add`，不同时迁移 unary/transpose/4D runners。为复用本树已有 Add 特化，FP16 slice 以 FP16 作为 compute type；正式 case 为 M=32,N=stride=64、固定 seeds=11939/11940，恰好一个 2048-element block、512 threads。
- runner 新增默认保持 benchmark 行为的 `time=1`；Perf Model 与两台真机均使用 `time=0` 单发射并输出 `timing:off`。
- 三端父 CMake/子 CMake/example/umbrella/kernel/shape/problem/policy SHA256 均为 9b2567c4e8dcd431f8344fb45b92fd51a26e06e98f5747f0eeb9fe2a9a7da635/623d85102ffc1438cdbe4564ed5ada319741956d89470f0827370a41dfd221b1/8e6d82cc0a0951283be04fe435e2473421e1f423d7605a1b776a9dba62b0525d/045787efa4a23ec2073b7b49f11b4257394f3ed7c5fbcbc2fb4175eee1f15da4/c7a4f511a132abf67a94968af19acf3caea9b7824cfb500d73030edc085b1f09/9c22192c30d347bf8664142c37775fb6905bf18c8459a29686af7ac44e39d980/e5fda4cb4d7ab8b2cf46dc0d5265d4f3ff09b8f2550b51ffc99ea66d2c5bf156/209ee363a2b09e3c7cd83f86c43db80dd0562ea796976ad8d2e095c18d0210cd。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.7-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=2ca6efb30876bdef3d0294da85f8230b621555facc95693d787a7a528f3cd2cc/de6cd9edcc842006ac36a2caa8f6a4f98280d0491bdb07c5ddf8c21b58bc6655/80438d2eff1acf2601df3a974a98a4a189a445cf1e1102e010c74d8fb70a60d7 | G1/G2/G4 PASS |
| P6.7-E1 | gfx946 PMD time=0 | 单次 grid=(512,1,1)、wg=(512,1,1) dispatch，CPU reference `valid:y`、RUN_RC=0；日志 SHA256=ff1993bb2d06bc416fdd4bcb263820c21e80750409356a6dc1409502ea2fd67a | G3 PASS；未测试性能 |
| P6.7-E2 | PMD 指令观察 | 报告 `V_ADD_F16 not test`、src1 modifier 与 pending-cacheline warning，但全部输出正确 | 非阻塞模拟器告警 |
| P6.7-E3 | gfx936/gfx938 同一 correctness case | 两端均 `grid:1, block:512, timing:off, valid:y`、RUN_RC=0；日志 SHA256 均为 c431fde5ac91dbc7ff69dc8c2ea3e2ecad7290b47d44bc2820513d6f72af6846 | G4 PASS |
| P6.7-E4 | 公共 umbrella 既有 consumer payload A/B | 以 P6.4 smoothquant binary 为修改前基线，重编译后 gfx936 `.hip_fatbin` SHA256 均为 f1db6286efc35ab98146e1d21e9599858b54a0c7766c15546ef25e01c9cb72bd，gfx938 均为 15237af2870f022c09c7c59ebef9831343402fa7de50524c7e7a19bea8d6c405；`cmp` 与 correctness 均 PASS | G6 PASS |

新增公共类型没有改变既有 consumer 的 device payload，已直接排除 gfx936/gfx938 性能变化，故不执行运行时性能 A/B；gfx946 仅做 correctness。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 generic elementwise kernel 与 basic Add example并独立提交；下一切片 P6.8 只新增 unary square target，复用本步公共闭包并单独验收。

### Step 6.8：CK Tile elementwise unary square — PASS-PMD（2026-09-03）

- 复用 P6.7 的 generic elementwise kernel 和 HCU 已有 `UnarySquare` functor；本步只在同目录新增 `tile_example_elementwise_unary_square` 与独立 runner，不改任何公共 header/device 实现。
- 正式合同仍为 FP16、M=32,N=stride=64、seed=11939、一个 2048-element block/512 threads；CPU reference 对每个输入执行 square。runner 支持 `time=0`，三端均为单发射 correctness。
- 三端 CMake/source SHA256 均为 ceb7eb6fc6f8d80bce9ea31771db7df49e56592e0670a46ef794d1e7cd4bb7bb/f6ae9f7b02cb253bb0a78dd7e3f9e08f32edc328b9ec89a081ced3a6c56ce075。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.8-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=ed07f79ff60adee4371844e3ed1134eebd6bc6e610d48154c03a759611d52c5b/04092372a003c7a95775c9bc743f9e14e1d2e9ec31e1ff36fbf4776090cdb981/3534cff7c6aefd6edf306fdcf8c4a70bac7cab3bdbc1787ea66f9cb640a1a088 | G1/G2/G4 PASS |
| P6.8-E1 | gfx946 PMD time=0 | 单次 grid=(512,1,1)、wg=(512,1,1) dispatch，CPU reference `valid:y`、RUN_RC=0；日志 SHA256=6c465432ac853c360417d64d4f4ec41968504dcf439902658e1b1a7ee8de7717 | G3 PASS；未测试性能 |
| P6.8-E2 | PMD 指令观察 | 无未测试算术指令告警，仅报告 pending-cacheline warning，完整输出正确 | 非阻塞模拟器告警 |
| P6.8-E3 | gfx936/gfx938 同一 correctness case | 两端均 `grid:1, block:512, timing:off, valid:y`、RUN_RC=0；日志 SHA256 均为 c431fde5ac91dbc7ff69dc8c2ea3e2ecad7290b47d44bc2820513d6f72af6846 | G4 PASS |

本步仅新增 `EXCLUDE_FROM_ALL` target/source，既有 gfx936/gfx938 代码对象与 dispatch 没有变化面，性能影响可直接排除，不运行性能 A/B；gfx946 只做 correctness。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 unary square target 并独立提交；下一切片 P6.9 以同一 generic kernel 验证 output-stride 实现的 FP16 2D transpose，不把它与 generic permute 的多维合同混同。

### Step 6.9：CK Tile elementwise 2D transpose — PASS-PMD（2026-09-03）

- 复用 P6.7 的 generic elementwise kernel 与 HCU 已有 `PassThrough` functor；本步只新增 `tile_example_elementwise_transpose` target/source。kernel 逻辑长度为 M×N，input strides=`{stride_in,1}`，output strides=`{1,M}`，从而把原坐标 `(m,n)` 写到连续输出 `(n,m)`。
- 正式合同为 FP16、M=32,N=stride_in=64、seed=11939，一个 2048-element block/512 threads；runner 内逐元素构造 transpose reference，并以零容差做 bit-exact 校验。三端均使用 `time=0` 单发射。
- 三端 CMake/source SHA256 均为 03305be5450c916ea99d7510692791b3fcd48db94bd3f9bd3639ea838d53d25f/199c18c61a3acce4f424cf50d10c65b6944f88f8042a515ff0de04e516297278。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.9-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=d35434a39cbecbb97f7f13da1b5118ef7283a3211b93a3aeef7b07f1b15d2326/e811096e09ac5c93768f60e0a5489b147dcd60cc1685639e939d264b345cbf6e/e63ade10bb64ddb85e2d3f4239f31bc6b9e631e37fad9366b1f377a741571472 | G1/G2/G4 PASS |
| P6.9-E1 | gfx946 PMD time=0 | 单次 grid=(512,1,1)、wg=(512,1,1) dispatch，bit-exact `valid:y`、RUN_RC=0；日志 SHA256=a2412946d03b110856e68d075b7f7c370badfc8e456ddb3b36b7b943849d0bf2 | G3 PASS；未测试性能 |
| P6.9-E2 | PMD 指令观察 | 无未测试算术指令告警，仅报告 pending-cacheline warning，完整转置正确 | 非阻塞模拟器告警 |
| P6.9-E3 | gfx936/gfx938 同一 correctness case | 两端均 `grid:1, block:512, timing:off, valid:y`、RUN_RC=0；日志 SHA256 均为 c431fde5ac91dbc7ff69dc8c2ea3e2ecad7290b47d44bc2820513d6f72af6846 | G4 PASS |

本步仅新增 `EXCLUDE_FROM_ALL` target/source，不改变既有 gfx936/gfx938 device code 或 dispatch，性能影响可直接排除，不执行性能 A/B；gfx946 只做 correctness。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 elementwise transpose 并独立提交；下一切片 P6.10 验证四维长度/stride 经 merge transform 后的 binary Add 路径，完成初始 `21_elementwise` 四个 target 的独立闭环。

### Step 6.10：CK Tile 4D elementwise Add — PASS-PMD（2026-09-03）

- 复用 P6.7 generic elementwise kernel/FP16 Add functor；本步只新增 `tile_example_elementwise_add_4d` target/source。逻辑 shape 固定为 `[2,4,8,32]`，显式 dense strides=`[1024,256,32,1]`，验证四维 `make_merge_transform`、双输入 load 与输出 store 的组合。
- 正式合同共 2048 元素、一个 block/512 threads、输入 seeds=11939/11940；CPU binary reference 覆盖完整 4D tensor。三端均以 `time=0` 单发射运行。
- 三端 CMake/source SHA256 均为 a9339274613ae8fe96e010b9ee277c025c4826714c2bb0f35946259ad3cb8d98/289c1db219b447a293041387256be88ed26be988cf9db9543e05c40eb4e927e6。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.10-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=677dc5240c6c99fe46fa02ccd8e0dc318d21f8de0cbbbb443406e278830506fe/304ffd2b0e901573aace0d8948b296b4b562e0430ad87e7808d930ebd631a537/0b36f66ce97583746b621b9a48a862ad7636bde951752cfa478e0e72334d0c6e | G1/G2/G4 PASS |
| P6.10-E1 | gfx946 PMD time=0 | 单次 grid=(512,1,1)、wg=(512,1,1) dispatch，完整 4D reference `valid:y`、RUN_RC=0；日志 SHA256=a131ef2b50a431ec3be21cfde5aa41aac7e5ae69239ee1048c022807860935aa | G3 PASS；未测试性能 |
| P6.10-E2 | PMD 指令观察 | 报告 `V_ADD_F16 not test`、src1 modifier 与 pending-cacheline warning，但全部输出正确 | 非阻塞模拟器告警 |
| P6.10-E3 | gfx936/gfx938 同一 correctness case | 两端均 `grid:1, block:512, timing:off, valid:y`、RUN_RC=0；日志 SHA256 均为 c431fde5ac91dbc7ff69dc8c2ea3e2ecad7290b47d44bc2820513d6f72af6846 | G4 PASS |

本步只新增 `EXCLUDE_FROM_ALL` target/source，不改变既有 gfx936/gfx938 device code 或 dispatch；性能影响可直接排除，不执行性能 A/B。gfx946 只做 correctness。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 4D Add target 并独立提交；至此 AMD 初始 `21_elementwise` 四个执行 target 均完成。下一切片 P6.11 重新按公共实现闭包大小排序 batched-transpose、pooling、multi-reduce 等候选，先选能独立 correctness 的最小项。

### Step 6.11：batched-transpose / pooling / multi-reduce 闭包优先级审计 — COMPLETE（2026-09-03）

- batched-transpose 初始提交 `c5fff071e5c60af87ed7e3a9d130d8151b353384` 共约 760 行，核心闭包是 umbrella、kernel 与三个 problem/pipeline/policy header；初代路径不使用 MMAC/DSReadM，CPU reference 可收口在 runner 内。
- multi-reduce 的 general 2D kernel 从 2025-08 的 `4750b293fe0abfa44a32181742a48b1dfec468f7` 系列引入，当前需要 multi-block/threadwise/partitioner 及通用 reduce problem/policy，公共代码量约 40K，明显大于单一 example 入口。
- pooling 初版直到 2025-10 的 `7b6451b68efada41ede3e1effb2c0aa9cf1ca314` 才引入，kernel/policy/problem/shape 与 runner 约 45K；它与当前 HCU fork 的 API 距离更远。
- 因此下一优先级为初代 basic batched-transpose；LDS/vectorized 后续提交 `9d1e44e56ab15e65b86b0bb8fda82f9caed3af92` 作为独立优化路径，不与首个 correctness 闭包混合。

本步仅作只读来源/闭包审计并更新文档，没有代码候选，不运行回归或性能 A/B。下一切片 P6.12 迁移 basic batched-transpose，先验证现存 core tensor API 的同代兼容性。

### Step 6.12：AMD-to-HCU basic CK Tile batched-transpose — PASS-PMD（2026-09-03）

- 来源从初始功能提交 `c5fff071e5c60af87ed7e3a9d130d8151b353384` 收敛，但没有照搬其错误的 `dim_block_h/w` 与二次 block-offset 计算；采用后续已修复的 grid/offset 语义以及 HCU 已具备的 `transpose_tile2d`/静态分布原语，只迁移非 LDS pipeline。
- 最小公共闭包新增 `ops/batched_transpose.hpp`、kernel、problem/pipeline/common-policy/default-policy；CPU reference 直接在 runner 内构造，不修改 `host.hpp`。新增 target 为 `EXCLUDE_FROM_ALL`，LDS/vectorized 路径继续独立。
- 正式合同为 FP16、batch=2、每批 32×32、packed batch stride；grid=(1,1,2)、block=64，同时覆盖 tile 内寄存器 transpose 与 batch offset，输出以零容差 bit-exact 校验。三端均使用 `time=0`。
- 三端父 CMake/子 CMake/example/umbrella/kernel/common-policy/pipeline/policy/problem SHA256 均为 7ad82213047351dda86ae8433866aba5f62ac2cdb12ef2e088a45d22afff2003/46aabca6c2dd9514957f89e2fd99b09d3227fe38b456906ff0e6aa3ea900e9f6/4273ab969c8cedab1537bd1d8012312b5f66e8fbf16d6dfed46e22568470859f/a959c53e7a8fafd62b0323170c26780a7b99dbc9b9f6e8baa1098ae04a8ce4c9/2ab72821da58462f31597cab12a1065feeb070fcac2d5f270ab78068cdb1f962/a2dc8dd85037d4b07b5b1d0ad11df8c6eeb18b7794e782b7bed620fa38bc0169/b4b72a0fe65dfd334ae1a8abafb768606dbb522bc4e0184af403c9bd8d9ef5d1/fe2c0f20883339985d4900af2c232779f95c9f9b03507449ec68fab8d522dc6f/acdfd8ba8ebdb81c0b305665f70368bc02c31d340556aa22ae11319cf023a383。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.12-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=7745a3ec047bbf4e528449fd04005fbf52ad2c5bf5d7cdf2633b7c5a308c5150/8c8babc5ae815f29930f337bf5d3fb40f7a60099ead1c5205b9ddeab2ed231fc/ca4b50ee9b2b2f0ff3551ab153ddec3b323ba948c66bdedfa84ee067b1b8deaa | G1/G2/G4 PASS |
| P6.12-E1 | gfx946 PMD time=0 | 单次 dispatch，实际 grid=(64,1,2)、wg=(64,1,1)，两个 batch bit-exact `valid:y`、RUN_RC=0；日志 SHA256=352f80fdc0d5a4a0cbb931871e0eb7b1fb6cc9cc02e3769ce425ffd8ac80d38e | G3 PASS；未测试性能 |
| P6.12-E2 | PMD 指令观察 | 报告 `v_mad_u64_u32 ... read vgpr before writing` 与 pending-cacheline warning，但两批完整输出正确 | 非阻塞模拟器告警，保留原文 |
| P6.12-E3 | gfx936/gfx938 同一 correctness case | 两端均 `grid:(1,1,2), block:64, timing:off, valid:y`、RUN_RC=0；日志 SHA256 均为 e6ced0e0028f543593383c22f3bd57f0dc822ffcad39f8f6087da955d3e0342b | G4 PASS |

新增公共 umbrella 没有挂入既有全局 header，唯一新 target 又是 `EXCLUDE_FROM_ALL`；gfx936/gfx938 既有 binary/dispatch 没有变化面，性能影响可直接排除，不执行性能 A/B。gfx946 只做 correctness。

门禁结果：G0-G4/G6 PASS；G5 NOT RUN。保留 basic batched-transpose 最小闭包并独立提交；下一切片 P6.13 增加 33×35 padded/tail dispatch，单独验证边界 tile，仍不混入 LDS 优化。

### Step 6.13：non-LDS batched-transpose padded/tail 合同 — SKIP-CORE-API（2026-09-03）

- 候选只把 P6.12 runner 的分布改为 `PadM=true, PadN=true`，正式 probe 为 FP16、batch=2、height=33、width=35，目标是覆盖两个维度的尾 tile；没有修改公共 kernel/core。
- gfx946/gfx936/gfx938 三端编译器在同一公共 HCU core 边界失败：实例化 `transpose_vectors<_Float16, 16, 1>` 时触发 `NX % 2 == 0 && NY % 2 == 0`，继而 `static_for<0, 1, 2>` 触发 `(NEnd - NBegin) % Increment == 0`。这是 scalar padded thread distribution 与当前 `transpose_tile2d` 的共享编译约束，不是 gfx946 PMD 指令执行问题。
- 因三端均未形成 binary，gfx946 PMD 明确 `NOT RUN`；不扩张本步范围去修改共享 core transpose primitive。失败候选已从本地及三台远端完全撤回。
- 撤回后复跑 P6.12 对齐合同，gfx946/gfx936/gfx938 均恢复 `valid:y`、RUN_RC=0，binary SHA256 仍为 7745a3ec047bbf4e528449fd04005fbf52ad2c5bf5d7cdf2633b7c5a308c5150/8c8babc5ae815f29930f337bf5d3fb40f7a60099ead1c5205b9ddeab2ed231fc/ca4b50ee9b2b2f0ff3551ab153ddec3b323ba948c66bdedfa84ee067b1b8deaa；本次 PMD 恢复日志 SHA256=4161ebe707b7a9c0a9e5045f62411e9d0ac3b848ab9a9ec771d2f022a5109bab，双真机日志 SHA256 均为 e6ced0e0028f543593383c22f3bd57f0dc822ffcad39f8f6087da955d3e0342b。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.13-E0 | 三架构 padded/tail focused build | FAIL；三端同一 `transpose_vectors<half,16,1>`/`static_for<0,1,2>` static assertion | G1/G2/G4 FAIL；共享 core API 边界 |
| P6.13-E1 | gfx946 PMD correctness | NOT RUN；无 binary | G3 NOT RUN；非 PMD 指令问题 |
| P6.13-E2 | 候选撤回与三端基线复验 | PASS；三端 P6.12 binary hash 不变且 `valid:y` | 恢复面 G4 PASS |

本步没有保留任何代码候选，不改变 gfx936/gfx938 device code、binary 或 dispatch，性能影响可直接排除，不执行性能 A/B。门禁结论为 `SKIP-CORE-API`；下一切片 P6.14 独立审计 LDS batched-transpose 闭包，优先判断 LDS 路径能否绕开当前寄存器 `transpose_tile2d` 的 scalar padded 限制。

### Step 6.14：LDS batched-transpose HCU 闭包审计 — SKIP-CORE-API（2026-09-03）

- 上游 LDS 路径来自 `821cd26c13323672b50c4cd2b35510d94f2199b8`，最小 operator 增量表面上是 `batched_transpose_lds_problem/policy/pipeline.hpp` 三个 header；实际读取闭包后确认它依赖 AMD core 的 `LaneGroupTransposeTraits`、`make_transposed_distr_encode` 以及 LDS tensor view 的 transpose-load 入口。
- 当前 HCU core 没有 `LaneGroupTransposeTraits`/`make_transposed_distr_encode`；其 `load_tile_transpose.hpp` 使用另一套 `DefaultTranspose`/`InputTileDistributionEncoding`。更关键的是 `tile_window_linear.hpp::load_transpose_linear` 调用了 `tensor_view::get_transpose_vectorized_elements`，但当前 HCU `tensor_view.hpp`/`buffer_view.hpp` 没有定义该入口。因此直接迁移三个 operator header 不能形成可实例化闭包。
- 少伯 ISA 资料确认 gfx946 提供 HCU `ds_read_matrix_trans_format` 与对应 matrix-format builtin；它的 fragment、element/row/col/alt 参数 ABI 不等于上游 AMD `ds_read_tr*`。正确适配需要在 HCU core 内新增指令绑定、tile distribution encoding 和 tensor-view 访问闭包，并分别定义 gfx936/gfx938 fallback，已超出单一 example 的最小独立修改边界。
- 本步是源码/ISA 只读审计，没有制造不能工作的 operator 候选，也没有运行 PMD。保留 P6.12 non-LDS PASS 路径；LDS 路径标记 `SKIP-CORE-API`，待独立 core 项目处理，不能把“少伯有同名 transpose 能力”误写为 CK LDS pipeline 已支持。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.14-E0 | 上游 LDS operator 最小闭包审计 | FAIL；三个 header 仍依赖未落地的 AMD transpose encoding/tensor-view API | G0 FAIL；HCU core ABI 缺口 |
| P6.14-E1 | HCU core/ISA 对照 | gfx946 有独立 matrix-transpose DS 指令族，但 CK core 无完整绑定 | 非 PMD 支持结论；不得直接照搬 |
| P6.14-E2 | build/PMD/真机回归 | NOT RUN；无可实例化候选 | G1-G5 NOT RUN |

本步只更新文档，没有代码、build graph、device binary 或 dispatch 变化，故不存在 gfx936/gfx938 性能影响，不执行性能 A/B。下一切片 P6.15 转向复用已三架构通过的 Smoothquant pipeline，只补 Moe routing kernel 与最小 FP16→INT8 CPU-reference runner。

### Step 6.15：AMD-to-HCU Moe Smoothquant FP16→INT8 minimal closure — PASS-PMD（2026-09-03）

- 复用 P6.4 已通过的 HCU `SmoothquantPipelineTwoPass`，只新增 `MoeSmoothquant` routing kernel、`14_moe_smoothquant` 单 target/runner，并在现有 smoothquant umbrella 暴露新 kernel；未迁移上游二十余个 shape-specific instance/API 文件。
- 正式合同为 FP16 input、FP32 expert smooth scale/row scale、INT8 output，tokens=2、hidden=128、experts=4、topk=2；路由固定为 `[0,1]`/`[2,3]`，每个 `(topk,token)` row 使用对应 expert scale。CPU reference 分别校验四个 yscale 与 512 个量化值，输入 seeds=11939/11940，三端均 `time=0`。
- 首轮三端编译均暴露同一个同代命名差异：上游 kernel 取 `Problem::SmoothScaleDataType`，当前 HCU problem 名为 `XScaleDataType`；只修正这一个 alias 后三端构建通过，没有架构分支。
- 三端父 CMake/子 CMake/runner/umbrella/kernel SHA256 均为 315ad4d0c5dc6796ca178977b8744a6929a0fead99cebaf264eeb5b91916d57e/94a6dbd8511cc4cd4df4e680151f699c964978ddac0cb65417fdd3478fcb56c2/9e1a01ea367c59bc5b83ab783ff2cf757104c38b6171719e3dced4167bccdfe7/ec21392830bc4c307dee32eb2899414df2e8157bbb3fada7cd031e7429a1f37e/14920369dc9b3898ae529e14b1275eb2e993f61a9843a54ea3da178e510f82ff。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.15-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=609de9cab3302ada1b45cb8d91949ccca57b555448dfea54b063203bdb7deca4/b5bd5ec0fc701ee3efa4e23851397f84e5bf74510fe066342a567b8d9f7be12a/610b10d1c4283c7fa1eb4abd6ce59e1bf9b347e32ad6ba934e5b65933a26493a | G1/G2/G4 PASS |
| P6.15-E1 | gfx946 PMD time=0 | host grid=(2,1,1)、block=256，实际 grid=(512,1,1)、wg=(256,1,1)；四条 routing row 全部 `valid:y`、RUN_RC=0，日志 SHA256=cf0f5ee4b5e18b7e09b2df19ae25334246b58fae883a5f3e54c7595ad926db90 | G3 PASS；未测试性能 |
| P6.15-E2 | PMD 指令观察 | 报告 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，yscale/qy 全量 reference 正确 | 与 P6.4 相同的非阻塞模拟器告警 |
| P6.15-E3 | gfx936/gfx938 同一 correctness case | 两端均 `grid:(2,1,1), block:256, timing:off, valid:y`、RUN_RC=0；日志 SHA256 均为 e369882070676d83a96e4d515d72293b1072766b1c69ed84a9226c86709b0e2b | G4 PASS |
| P6.15-E4 | 既有 P6.4 smoothquant umbrella 影响复核 | gfx936/gfx938 重建 binary SHA256 仍为 1b918fce81d32946cfa184e730c83d50553c38d859ba53897b600813327ecb1a/736859f07afaf21f9fd7ba691431a56f20d8deaddca4b0663484f094f4193d32，correctness 日志 hash 也不变 | device code identity PASS |

新 target 为 `EXCLUDE_FROM_ALL`；虽然修改了公共 umbrella，但两个真机上既有 Smoothquant device binary 与 P6.4 完全相同，因此可直接排除已有路径性能变化，不执行性能 A/B。gfx946 仅做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.16 单独增加 hidden 非 128 对齐的 padded-N 变体，验证 MoE routing 与尾列 mask 组合。

### Step 6.16：Moe Smoothquant padded-N hidden-tail — PASS-PMD（2026-09-03）

- 新增独立 `tile_example_moe_smoothquant_tail` target/source，复用 P6.15 公共 routing kernel；正式合同保持 FP16→INT8、tokens=2、experts=4、topk=2 与相同路由/seed，只把 hidden 改为 96，并实例化 Block_N=128、`kPadN=true`，CPU reference 仅覆盖 96 个有效列。
- 为隔离验收与性能影响，最终实现没有修改 P6.15 的 `moe_smoothquant.cpp`；tail 使用独立 source。子 CMake/tail source SHA256 在三端均为 ac947346764150f3423d2a245b52ce21c841154502463e7ffe706359da871e83/ba0119e60892f805680ff70a78ce4cd383609722d3c03c4a280ff69b15f17eda。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.16-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=6569f66b46ca50e12bdb307551ad18a5fec597ce6ddf1c822ed8a2a5ed6d2403/5f2587015d985a9429972f9d2c3f4c7ebd08636081e6369be1259360f9782075/ec824d257d5834bd517919469c06f65b1554b15274ae5e2ebc91319325cfe2aa | G1/G2/G4 PASS |
| P6.16-E1 | gfx946 PMD time=0 | 实际 grid=(512,1,1)、wg=(256,1,1)，四条 routing row 的 96-column tail reference 全部 `valid:y`、RUN_RC=0；日志 SHA256=919841b4a18ea33c466b12b52d646872fe413bd0c8454c61f854f137d1a0c2f6 | G3 PASS；未测试性能 |
| P6.16-E2 | PMD 指令观察 | 仍只有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，masked 输出正确 | 非阻塞模拟器告警 |
| P6.16-E3 | gfx936/gfx938 同一 correctness case | 两端均 `grid:(2,1,1), block:256, timing:off, valid:y`、RUN_RC=0；日志 SHA256 均为 e369882070676d83a96e4d515d72293b1072766b1c69ed84a9226c86709b0e2b | G4 PASS |
| P6.16-E4 | P6.15 aligned target 隔离复验 | gfx936/gfx938 binary SHA256 恢复并保持 b5bd5ec0fc701ee3efa4e23851397f84e5bf74510fe066342a567b8d9f7be12a/610b10d1c4283c7fa1eb4abd6ce59e1bf9b347e32ad6ba934e5b65933a26493a，correctness PASS | 既有 source/device identity PASS |

本步只新增 `EXCLUDE_FROM_ALL` target/source，P6.15 既有 source 和两个真机 binary 均未变化，因此性能影响可直接排除，不执行性能 A/B。gfx946 仅做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.17 审计 `16_batched_gemm` 对当前 HCU Universal GEMM kernel/host-args 的最小依赖差，并判断是否可复用已通过的 FP16 GEMM pipeline。

### Step 6.17：CK Tile batched-GEMM 最小闭包审计 — COMPLETE（2026-09-03）

- 上游 `example/ck_tile/16_batched_gemm` 直接依赖较新的 `BatchedGemmKernel`、`UniversalGemmHostArgs`、spatially-local partitioner 与通用 CShuffle epilogue；当前 HCU tree 已有后续合入的 Universal kernel 片段，但 `example_hcu/ck_tile/03_gemm` 的已验证基础路径仍使用较早的 `GemmKernel`、`GemmTilePartitioner`、`MmacGemmPipelineAGmemBGmemCRegV1` 和 `Default2DEpilogue`。
- 若直接复制上游 runner/kernel，会把 host-args、partitioner、epilogue 与 pipeline 四类迁移混成一个提交，无法区分 Shaobo MMAC 问题和代际 API 问题。最小闭包改为新增与 HCU `GemmKernel` 同代的 batched sibling：保留 x/y tile 映射，让 grid.z 表示 batch，给 A/B/C 指针分别增加 batch stride；首个合同固定 split-K=1。
- P6.18 将复用 P6.1 之前已经三架构验证的 FP16 MMAC 基础 shape，使用 batch=2、M=N=128、K=32、packed row/column/row layout，CPU 对每批独立 GEMM 做 reference。该切片不引入生产 selector、CShuffle、多 D tensor 或 split-K。

本步仅作只读闭包审计并更新文档，没有代码候选，不运行 PMD/真机或性能门禁。下一切片 P6.18 实施上述最小 batched-GEMM kernel/runner；只有三端 focused build/correctness 全部通过后才保留代码。

### Step 6.18：同代 CK Tile FP16 batched-GEMM minimal closure — PASS-PMD（2026-09-03）

- 新增与当前 HCU `GemmKernel` 同代的 `BatchedGemmKernel`：复用 x/y tile partition、MMAC pipeline 与 Default2DEpilogue，仅由 grid.z 选择 batch，并分别应用 A/B/C batch stride；没有引入较新 Universal host-args、spatially-local partitioner、CShuffle、多 D tensor 或 split-K。
- 正式合同固定为 FP16 input/output、FP32 accumulate，batch=2、M=N=128、K=32，A/C row-major、B column-major，packed batch stride。CPU 以 FP32 逐 batch 独立累加后转 FP16 reference；三端均使用 `time=0`，运行时 grid=(1,1,2)、block=256。
- 公共 kernel 由本 example 直接 include，没有挂入既有 `ops/gemm.hpp` umbrella；新增 target 为 `EXCLUDE_FROM_ALL`。父 CMake/子 CMake/runner/kernel SHA256 分别为 add579b43cc19a9b8d9852f5b67a2eeafdd7553975080f712a5f7921745efcc5/5228d3b934119cfa9b261b6b291bda94eba5a6fc3f220932924f3deae491854e/f3e9c75c965658bfd76c60bcc27effbd1024e4e967463d137482a319139c2c2b/121cbad87a7fcc769b2beea487c2b506078b0a176909565d57440805f955ce3f。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.18-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=bf1e4f29221849310c364ef73a8a7cb1908ab1427c84960fc2d5666e30b0885f/5a1561d93cf0bb51f12a140cda66bae99ca4388f6e2d2a8e80f92114a65c1a54/2e6cbf7c0ca97d10b29b22fd714e160da236accb1b3755e33bdf656a14d0e7fd | G1/G2/G4 PASS |
| P6.18-E1 | gfx946 PMD time=0 | 单次 batched dispatch，实际 grid=(256,1,2)、wg=(256,1,1)，两个 batch reference `valid:y`、RUN_RC=0；日志 SHA256=9efa95a32945009a86064238286d5fbec6ab794b39c67d8801074ec5f37082c5 | G3 PASS；未测试性能 |
| P6.18-E2 | PMD 指令观察 | 报告 `v_mad_u64_u32 ... read vgpr before writing` 与 pending-cacheline warning，但两个 batch 的完整 GEMM 输出正确 | 非阻塞模拟器告警，保留原文 |
| P6.18-E3 | gfx936/gfx938 同一 correctness case | 两端均 `grid:(1,1,2), block:256, timing:off, valid:y`、RUN_RC=0；日志 SHA256 均为 1c6dccf60cd33956ac8884ccea3fa6d81b2a2749631422a1ed74b33c19992683 | G4 PASS |

本步没有修改既有 GEMM umbrella/pipeline/device 实现，唯一 build-graph 变化是新增 `EXCLUDE_FROM_ALL` target；gfx936/gfx938 既有 binary 与 dispatch 没有变化面，性能影响可直接排除，不执行性能 A/B。gfx946 仅做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.19 审计 `19_gemm_multi_d`，先区分可复用的同代 GEMM pipeline 与较新多 D/CShuffle epilogue ABI。

### Step 6.19：同代 CK Tile FP16 GEMM Multi-D minimal closure — PASS-PMD（2026-09-03）

- 上游当前 `GemmKernelMultiD` 依赖较新的 Universal kernel/host-args、spatially-local partitioner 与通用 CShuffle epilogue。为避免把四类代际迁移混入同一提交，本步新增与 HCU 基础 MMAC GEMM 同代的 `GemmMultiDKernel` 和无 LDS 的 `Default2DMultiDEpilogue`，直接复用 accumulator tile distribution 读取两个 D tensor。
- 正式合同为 FP16 input/D/output、FP32 accumulate，M=N=128、K=32，A/E/D0/D1 row-major、B column-major，语义为 `E=(A×B)×D0×D1`；seeds=11939/11940/11941/11942。CPU 逐元素以 FP32 完成 GEMM 与两个乘法后只在输出处转 FP16，覆盖完整 128×128 输出。
- kernel/epilogue 都由本 example 直接 include，不挂入既有公共 umbrella；target 为 `EXCLUDE_FROM_ALL`。父 CMake/子 CMake/runner/epilogue/kernel SHA256 分别为 129ba54ffef1cd88814a7c37f8b23c96210425c0fff46753a55a6d60f14ffa21/aa1af6e93e77e854ca8fbbffdbf59d1fd715077857533a85f624121d98da4004/44fe01aaca7922b6450c2b33a3a15b0e433b8b7d8e600e4fb1218b770e30039e/a89080e50a1de7db807a2fc9cefe30225d803293b43934ec3221f84f782ae95e/7a18cda18fcc198ed0c457a1dd3243af8b511ac602f6bfe6f87edbb3d6149e9c。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.19-E0 | 三架构 focused configure/build | PASS；gfx946/gfx936/gfx938 binary SHA256=654cf09791e43d45382cd4850c7b6aa683a22bfadf63b662ee9cdbfec7cbd56c/8a308462f40e1ad8ab9e9714f7d290154761172378ee0c47490560488238d629/812e5da770281131d3eeb1b875e7318c8c2f1ac6d05c42b928e53e82f0efe849 | G1/G2/G4 PASS |
| P6.19-E1 | gfx946 PMD time=0 | 单次 grid=(256,1,1)、wg=(256,1,1) dispatch，完整 Multi-D reference `valid:y`、RUN_RC=0；日志 SHA256=a73799970137af359406790a063f0849e8757f154db455e53fb8e48a1d1ca906 | G3 PASS；未测试性能 |
| P6.19-E2 | PMD 指令观察 | 报告 `v_mad_u64_u32 ... read vgpr before writing`、src1_sext/neg/abs modifier 与 pending-cacheline warning，完整输出正确 | 非阻塞模拟器告警，保留原文 |
| P6.19-E3 | gfx936/gfx938 同一 correctness case | 两端均 `grid:(1,1,1), block:256, timing:off, valid:y`、RUN_RC=0；日志 SHA256 均为 84d20727df64e1a8f0fb98f76c488ba082805b67490f7b804ae816493898c32c | G4 PASS |

本步没有修改任何既有 GEMM/epilogue umbrella、pipeline 或 device 实现，唯一 build-graph 变化是新增 `EXCLUDE_FROM_ALL` target；gfx936/gfx938 既有 binary 与 dispatch 没有变化面，性能影响可直接排除，不执行性能 A/B。gfx946 只做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.20 审计 `22_gemm_multi_abd`，重点判断多 A/B 的 tuple load 是否可在当前 MMAC pipeline 外做无歧义的同代组合。

### Step 6.20：CK Tile GEMM Multi-ABD HCU 闭包审计 — SKIP-CORE-API（2026-09-03）

- 上游 `GemmKernelMultiABD` 要求 A/B/D 的 data type/layout 都是 tuple，并以 `AddScale(A0,A1)`、`AddScale(B0,B1)` 形成实际 GEMM 输入，再以 Multi-D op 处理输出；它依赖较新 Universal pipeline 对多 A/B tile 的联合读取和 elementwise 融合。
- 当前 HCU `MmacUniversalGemmPipelineProblem` 的 A/B 类型与 layout 是标量；`MmacGemmPipelineAGmemBGmemCRegV1` 只接收一份 A window、一份 B window，其可选 element function 也只作用于单个已加载 tile。HCU 树同时没有上游 `AddScale` functor。
- 当前树的 `UniversalGemmKernel` 虽能创建 tuple tensor views/windows，但 `RunGemm` 最终明确只把 `as_block_window[I0]` 与 `bs_block_window[I0]` 传给现有 pipeline。若只迁移上游 host wrapper/kernel façade，A1/B1 会被静默忽略，可能在弱 reference 下产生假通过，不能作为适配结果。
- 正确闭包需要新增 HCU MMAC multi-input pipeline：每个 K loop 同分布加载 A0/A1、B0/B1，按目标 functor 融合后写 LDS，再执行 MMAC；然后才能复用 P6.19 Multi-D epilogue。该工作跨越 pipeline problem/policy/load loop，不属于单一 example 的最小独立闭包。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.20-E0 | 上游 Multi-ABD contract 审计 | 需要 2A+2B+2D tuple、A/B AddScale 与 Multi-D output op | G0 来源确认 |
| P6.20-E1 | HCU Universal/MMAC 数据流对照 | tuple view 存在，但当前执行路径只消费 A0/B0；MMAC problem/pipeline 仍为 scalar A/B | G0 FAIL；共享 pipeline API 缺口 |
| P6.20-E2 | build/PMD/真机回归 | NOT RUN；没有制造会忽略 A1/B1 的错误候选 | G1-G5 NOT RUN；非 PMD 指令问题 |

本步只更新审计文档，没有代码、build graph、device binary 或 dispatch 变化，gfx936/gfx938 性能影响可直接排除，不执行性能 A/B。结论为 `SKIP-CORE-API`；后续若立项，应以 HCU MMAC multi-input pipeline 为独立项目。下一切片 P6.21 审计 `20_grouped_convolution` 与现有 `14_conv`/legacy grouped-conv 的重复面，优先选择无需新 pipeline 的独立 correctness contract。

### Step 6.21：CK Tile grouped-convolution G=2 launch closure — PASS-PMD（2026-09-03）

- 上游 `20_grouped_convolution` 的 forward 合同与现有 `example_hcu/ck_tile/14_conv/conv_fwd.cpp` 重叠；现有 runner 已使用 NHWGC/GKYXC/NHWGK grouped descriptor，但 P3.1 只验证过 G=1。本步不复制新一代 Universal/CShuffle runner，而是补测现有同代实现的真实 group offset。
- 初始 `G=2,N=4,C=K=64,H=W=4,Y=X=3,pad=1` 在 gfx946/gfx936/gfx938 三端均失败：runner 只发射 `tile_partitioner.CalculateGridSize()`，没有乘 group 数，实际仅一个 256-thread workgroup；device kernel 又以 `get_grid_size()/num_group` 推导每组 block 数。三端输出几乎全错，证明是共享 host launch baseline，不是 Shaobo 指令问题。初始 binary SHA256=ddf6cfa0cbdab042cbc88a64a3f3621458cd074cebd5abf81e96857c51b38b82/2c4ed0c28a8a5e8aac996323cbf46f7d488193cc900fd1d7ccc795b62d6cb077/a8c735aa72f36dcb6a05ab6d0f651855f6aa8dcdd4ea061cb17dbe4c4298979f，失败日志 SHA256=58102511012dd5abd474496fc83c11e4e45526adfc8d4045e893bb916d4f0fd0/b3430672b3af9eef3fa3758d83a0eccd6a678b3543f1dc5e3a952a8fdcee39ed/d327db3f95f7990bc07bbfb4dddcfd72eb111b44889a73d27c38606543e08f89。
- 最小修复只把 launch grid 改为每组 grid×`conv_param.G_`；kernel、descriptor、tile、MMAC、LDS CShuffle、参数 ABI 均未修改。source SHA256=489ccf0624854bd6fb0ded921822cceb87e39f3195659490be1f0c4064a8de87。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.21-E0 | 三架构 G=2 baseline | FAIL；均只发射一个 workgroup，约 99.8% 输出错误 | G0 shared baseline failure |
| P6.21-E1 | 修复后 focused build + G=2 correctness | PASS；gfx946/gfx936/gfx938 binary SHA256=8c6e57cf0613d46200cd54f183e6fdf79b5ee7b445eabbd1f965de96673b9092/b3e5dc505658e0be25a9f03341c7f9661218485892ef66cf20dfa285c1f1fe79/4ee35faf997d218cb87922a34ce3585997a4f52599812cdee4dc79904d7ef795；日志 SHA256=c92530447a28985db29ec5eb21b7472b93facefdd079ecc80bddaa99a428ed59/40fcf47ed40ef4987f463b21ea0ee3f08c276654a5febbb9c3923ff37b68573d/40fcf47ed40ef4987f463b21ea0ee3f08c276654a5febbb9c3923ff37b68573d | G1-G4 PASS |
| P6.21-E2 | gfx946 PMD 指令观察 | 修复后实际 grid=(512,1,1)、wg=(256,1,1)，完整 reference 返回 0；仅 pending-cacheline warning | G3 PASS；未测试性能 |
| P6.21-E3 | G=1 既有合同回归 | 三端均返回 0；gfx946/gfx936/gfx938 日志 SHA256=024f5130ec342757d474ada746b1ee36a8277b07c383e08ef2336936b51b35fa/4cc5310c5b8470bf99826b19570fdc58db8bea39294b211842aa9f4ee8392afc/4cc5310c5b8470bf99826b19570fdc58db8bea39294b211842aa9f4ee8392afc | P3.1 regression PASS |
| P6.21-E4 | gfx936/gfx938 device payload A/B | `.hip_fatbin` 修改前后 SHA256 分别恒为 872e4a8f262605114646cef6c29957c9ca0660b23d928f00e7d68d20f747e37a/1ad31db4b9aef34bbe22f5724beb98d156d77869dd77283cc0b8b8cf8fedc8b1，`cmp` PASS | G6 PASS |

gfx936/gfx938 device code 完全相同；G=1 launch grid 也完全不变，G=2 修改前不存在正确性能基线，因此性能影响不是不确定项，不执行性能 A/B。gfx946 只做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.22 审计 grouped-convolution backward-data 的现有 CK Tile 闭包；若只有新 Universal kernel 路径，则转向尚未覆盖且闭包更小的 operator。

### Step 6.22：CK Tile grouped-convolution backward-data/weight 闭包审计 — SKIP-CORE-API（2026-09-03）

- 上游 `20_grouped_convolution` 的 backward-data/weight 不是 forward kernel 的参数变体：分别依赖 `GroupedConvolutionBackwardDataKernel`/`GroupedConvolutionBackwardWeightKernel`、`GroupedConvTraits`、方向专用 implicit-GEMM transformer、Universal pipeline/partitioner 和对应 epilogue；backward-weight 另含 split-K/two-stage/StreamK 分支。
- 当前 HCU `include/ck_tile/ops` 没有 `grouped_convolution` operator 目录，也没有上述 kernel、traits、host args 或方向专用 transformer。现有 `ops/conv` 只有 P3/P6.21 已验收的 `ConvIgemmFwdKernel` 及其 forward transform，不能通过交换指针或 layout 正确表达 backward-data/weight。
- legacy `example_hcu/02_grouped_conv_bwd_data_mmac` 和 `03_grouped_conv_bwd_weight_mmac` 已在 P2/P5 分 target 验收，但它们不能替代新 CK Tile API 的迁移闭包。若继续此项，需要先移植整个 grouped-convolution core，并为 MMAC/epilogue 做 HCU 分支，明显超出单一 example 的独立最小改动。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.22-E0 | 上游 backward-data/weight source closure | kernel/traits/transformer/Universal pipeline/epilogue 多层依赖 | G0 来源确认 |
| P6.22-E1 | HCU CK Tile 能力对照 | 仅有 forward `ops/conv`，方向专用公共类型全部缺失 | G0 FAIL；共享 core API 缺口 |
| P6.22-E2 | build/PMD/真机回归 | NOT RUN；无可实例化最小候选 | G1-G5 NOT RUN；非 PMD 指令问题 |

本步只更新文档，不改变代码、build graph 或任何 device payload，故不执行性能 A/B。结论为 `SKIP-CORE-API`，legacy 已验收路径继续保留。下一切片 P6.23 扩展 P6.18 batched-GEMM 到每 batch 多个 output tile，验证 grid.x/y 与 grid.z 组合映射。

### Step 6.23：CK Tile batched-GEMM multi-tile contract — PASS-PMD（2026-09-03）

- P6.18 只覆盖每 batch 一个 128×128 output tile。本步保持同一 FP16/FP32 MMAC kernel 与 batch=2、K=32，只把 runner 的 M/N 限制放宽为正的 128 倍数，并由 K/N 动态生成 packed A/B/C stride；没有修改 kernel、pipeline、tile 或 epilogue。
- 正式 case 为 batch=2、M=256、N=128、K=32，host grid=(2,1,2)，同时覆盖 blockIdx.x 的第二个 M tile 与 blockIdx.z 的第二个 batch；CPU reference 覆盖 65536 个输出。source SHA256=73a0d821ac029b1f85dc8939b7969670e82a0e2ed7a47b2ec37b85c5afc493aa。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.23-E0 | 三架构 focused build + multi-tile correctness | PASS；正式运行 binary SHA256=c364fa2be58f627b5b56f62261a394948e05148613967b5697cb726132ff0a9a/b087486de615d2e2e1478dd3a52ba081c68305bcedb1b96b1ab3a1c8b0acdf7d/8c1ba0daa15be13842b4146fe5281f8cf2c6ee022ea314b0b0cf741e635df8a4，均 `grid:(2,1,2), block:256, timing:off, valid:y`；日志 SHA256=dcb6d976756a6f0b67f478746222f684302d9da972e6457d187f5897c885dda5/32bdc5b56a71eba1de3a3ace2e80bd362e477e3f9a82e7543053cf70327dd2ff/32bdc5b56a71eba1de3a3ace2e80bd362e477e3f9a82e7543053cf70327dd2ff | G1-G4 PASS |
| P6.23-E1 | gfx946 PMD 指令观察 | 实际 grid=(512,1,2)、wg=(256,1,1)，完整 reference `valid:y`；仍仅有 `v_mad_u64_u32 ... read vgpr before writing` 与 pending-cacheline warning | G3 PASS；未测试性能 |
| P6.23-E2 | P6.18 128×128 合同回归 | 三端均 `grid:(1,1,2), valid:y`；PMD/gfx936/gfx938 日志 SHA256=f6269c498a4047728d5177e47ebeb777c5b307537a016fbc302d32ed5eb08098/1c6dccf60cd33956ac8884ccea3fa6d81b2a2749631422a1ed74b33c19992683/1c6dccf60cd33956ac8884ccea3fa6d81b2a2749631422a1ed74b33c19992683 | 既有 contract PASS |
| P6.23-E3 | gfx936/gfx938 device payload A/B | `.hip_fatbin` 修改前后 SHA256 分别恒为 cd6b11863466163d8c8bb592f2b56183b00a63161a359a137a29c3864b0db983/7045cdd85225abcc872a81a15bd9d4aa32e057f7156146890030261ed2bf5696，`cmp` PASS | G6 PASS |

本步只改变 host 参数校验和 stride 构造，两个真机的 device payload 完全相同，已直接排除性能变化，不执行性能 A/B；gfx946 只做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.24 审计同 kernel 的 N 方向多 tile 与 M/N 二维组合，避免只由 x 维覆盖掩盖 y 维映射问题。

### Step 6.24：CK Tile batched-GEMM x/y/z grid mapping — PASS-PMD（2026-09-03）

- 复用 P6.23 已提交的 batched-GEMM binary，不修改任何源码；正式 case 为 batch=2、M=N=256、K=32，对应 host grid=(2,2,2)，把 M-tile、N-tile 与 batch 三个维度组合在一次 launch 内。
- gfx946 PMD 实际 dispatch grid=(512,2,2)、wg=(256,1,1)，完整 131072-element CPU reference `valid:y`，binary/log SHA256=c364fa2be58f627b5b56f62261a394948e05148613967b5697cb726132ff0a9a/6f7baaaf0902cf0d0418ea400d17796e2cd0079a110a2b978e7f67340dfa0d49；仅报告既有 `v_mad_u64_u32 ... read vgpr before writing` 与 pending-cacheline warning，未测试性能。
- gfx936/gfx938 使用同一合同均 `grid:(2,2,2), block:256, timing:off, valid:y`；binary SHA256=06ef82b023ec3a17008a81e143a6ce1230bb826902838f866ea0b10dc9ec5370/7a5f5cb4e871aee3a93b0eca752256749f29e918bc475f2548cf972851ac90c8，日志 SHA256 均为 62caf563eab640bacc9ad8aa5b04eb10f60e15ee024005c02e7386546031b1e5。

本步是对已提交、device payload 已在 P6.23 证明不变的 binary 增加 correctness case，没有候选代码或性能变化面，不执行性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.25 重新盘点未覆盖目录，优先排除只含 README、缺失整套 core API 或依赖不支持指令的路径，再选择下一个可形成最小闭包的 target。

### Step 6.25：剩余 CK Tile example 依赖闭包再排序 — COMPLETE（2026-09-03）

本步逐项对照上游剩余目录与当前 HCU `include/ck_tile`，只做静态能力盘点，不把缺少公共 operator 的 runner 误当成可直接适配的 example。

| 上游目录 | 上游规模/用途 | 当前 HCU 缺口 | 结论 |
| --- | --- | --- | --- |
| `36_pooling` | 3 files / 14878 bytes，Pool3D correctness | `ops/pooling`、problem/shape/kernel/reference 全部缺失 | `SKIP-CORE-API` |
| `37_transpose` | 仅 1 个 README，无可执行 target | 没有验收程序 | `NO-TARGET` |
| `40_streamk_gemm` | 5 files / 34492 bytes，StreamK GEMM | 只有零散 `streamk_common.hpp`，缺 kernel/partitioner/host contract；PMD 也不能验证调度性能收益 | `SKIP-CORE-API` |
| `41_batched_contraction` | 4 files / 44568 bytes | `ops/batched_contraction` 整体缺失 | `SKIP-CORE-API` |
| `42_mx_gemm` | 5 files / 35834 bytes | MX GEMM problem/kernel/pipeline 与数据格式闭包缺失 | `SKIP-CORE-API` |
| `49_sageattention` / `50_sparse_attn` | example 分别为 16 files / 161007 bytes、13 files / 148172 bytes | 不是单一 example 最小移植，需单独拆 operator、mask/layout、pipeline 与 ISA 门禁 | `DEFER-LARGE-CLOSURE` |
| `51_tile_distr_enc_reg_map` | 3 files / 9556 bytes，AMD target 的 encoding/register mapping 工具 | 缺 `core/arch/mma` 下 calculator/mapper；示例目标为 gfx908/gfx950/RDNA，不表达 HCU MMAC mapping | `SKIP-CORE-API` |
| `52_cshuffle_lds` | 4 files / 16772 bytes，纯 LDS 带宽/bank-conflict microbenchmark | 缺 `tile_load_store_microkernels.hpp`；虽有旧版 HCU CShuffle 部分 API，但该目录只输出计时结果、没有 correctness reference | `SKIP-CORE-API`，且禁止在 PMD 做性能验收 |

这些结论都不是“Perf Model 不支持某条 Shaobo 指令”：当前阻塞发生在 source/core 闭包之前，因此不制造编译失败候选，也不运行 PMD。`49/50` 保留为后续独立大项目，不能在本阶段通过复制 runner 混入 attention 全栈。

本步只更新文档，没有代码、build graph、device binary 或 dispatch 变化，gfx936/gfx938 性能影响可直接排除，不执行性能 A/B。下一切片 P6.26 扩展已经三架构通过的 P6.19 GEMM Multi-D 到 M/N 二维多 tile，覆盖 tile partition 与逐 tile D0/D1 epilogue 地址映射。

### Step 6.26：CK Tile GEMM Multi-D x/y multi-tile contract — PASS-PMD（2026-09-03）

- 保持 P6.19 的 FP16 input/output、FP32 accumulate、K=32、`E=(A*B)*D0*D1` kernel/pipeline/epilogue 不变，只把 runner 的 M/N 限制放宽为正的 128 倍数，并由运行时 K/N 构造 packed A/B/D/E stride。
- 正式 case 为 M=N=256、K=32，host grid=(2,2,1)，同时覆盖第二个 M tile、第二个 N tile，以及四个 output tile 各自的 D0/D1 load 和 E store 地址；CPU reference 覆盖 65536 个输出。source SHA256=ae679d8af105c9036695847ed5952029572e46f994622fd45b0556ad2a350d7c。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.26-E0 | 三架构 focused build + 256×256 correctness | PASS；gfx946/gfx936/gfx938 binary SHA256=c36f3346726add21cae1e3a0940e2657f410fd7b0017d38dc4abfb2b9fe833e6/51ea77220f47c9d5ecfa164f1d48213704a6324abdd0a1d909a2f3145e85a316/13a118a036ae596b01f51e3ac44eacc891b64a3444cbc1bdad4f4c4c9ed11093；均 `grid:(2,2,1), block:256, timing:off, valid:y` | G1-G4 PASS |
| P6.26-E1 | gfx946 PMD 指令观察 | 实际 grid=(512,2,1)、wg=(256,1,1)，完整 reference `valid:y`、RUN_RC=0；日志 SHA256=3b9fee8a566f5e0c21571e46bab0809567e77ded6132cb111408601e3bdf1051 | G3 PASS；未测试性能 |
| P6.26-E2 | gfx936/gfx938 正式 case | 两端均 RUN_RC=0，日志 SHA256 均为 ecf263f3f15c8600d300ad86a30a0af14d6dd693d21f65867663f4f5c821a393 | G4 PASS |
| P6.26-E3 | P6.19 128×128 合同回归 | 三端均 `grid:(1,1,1), valid:y`；PMD/gfx936/gfx938 日志 SHA256=2add281c861df7d846214985288630344a65328b46c341bcc61e2fcc7b424496/84d20727df64e1a8f0fb98f76c488ba082805b67490f7b804ae816493898c32c/84d20727df64e1a8f0fb98f76c488ba082805b67490f7b804ae816493898c32c | 既有 contract PASS |
| P6.26-E4 | gfx936/gfx938 device payload A/B | `.hip_fatbin` 修改前后 SHA256 分别恒为 89bf3c7f89eece97ab0efdcaf9e7329afbacda1f2465b225c2ba92f3d27be2a6/c66d7476878558519e44f703364346a74db6c1ab346f15f16a8c813245b999b5，`cmp` PASS | G6 PASS |

PMD 仍只报告 P6.19 已记录的 `v_mad_u64_u32 ... read vgpr before writing`、src1 modifier 与 pending-cacheline warning，四个 tile 输出全部正确，属于非阻塞模拟器告警。由于本步只改变 host 参数校验/stride，且两个真机 device payload 完全相同，性能变化已直接排除，不执行性能 A/B；gfx946 只做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.27 为 P6.19 同一 kernel 增加非方形 multi-tile case，交叉验证 stride_N 与单轴 tile mapping，之后转入 attention 大闭包的可拆分入口审计。

### Step 6.27：CK Tile GEMM Multi-D rectangular single-axis mapping — PASS-PMD（2026-09-03）

- 不修改源码，复用 P6.26 已提交且两个真机 device payload identity PASS 的 binary；分别运行 M=256,N=128,K=32 与 M=128,N=256,K=32，隔离验证 `blockIdx.x`/M offset 和 `blockIdx.y`/N offset。
- gfx946 PMD 的两个实际 dispatch 分别为 grid=(512,1,1) 与 (256,2,1)，wg=(256,1,1)，均 `valid:y`、RUN_RC=0；日志 SHA256=ad9cb79fd9a690add0b4f588ff85a798712215158e1bcd48c5607fb4a150eb67/3f14ccdc39b862b089aa0fb5cc86543a13868f6dbb0b6d2cf5f9ad2fa8ecf782。仍只有既有 `v_mad_u64_u32`、src1 modifier 与 pending-cacheline 非阻塞告警，未测试性能。
- gfx936/gfx938 两端均得到 host grid=(2,1,1)/(1,2,1)、`timing:off, valid:y`、RUN_RC=0；M-axis 日志 SHA256 均为 e369882070676d83a96e4d515d72293b1072766b1c69ed84a9226c86709b0e2b，N-axis 日志 SHA256 均为 c7da768bd33bc79647aab9bf83281c2e7552f41fe4e80dc777e553a52bbba88a。

本步没有代码、build graph 或 device payload 变化；性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.28 对 `49_sageattention` 与 `50_sparse_attn` 的入口、公共 attention 复用面和 ISA 特化点分层审计，只选择可独立 correctness 的最小切片。

### Step 6.28：SageAttention / SparseAttention 分层闭包审计 — DEFER-LARGE-CLOSURE（2026-09-03）

- 更正 P6.25 的目录名与规模：上游是 `50_sparse_attn`，其 example 有 13 个文件、148172 bytes；不是空目录。`49_sageattention` example 为 16 个文件、161007 bytes。
- SageAttention 不只是 FMHA runner 变体：还需 `ops/sageattention` 的 11 个 header、189493 bytes，包含独立 quant-scale enum、kernel、shape/traits、同步/异步 pipeline 与 custom policy；主要合同是 FP8、INT8/INT4 Q + FP8 K/V、BF16 output 和多种量化 scale。
- SparseAttention 也需 `ops/sparse_attn` 的 4 个 header、97054 bytes，并分成 Jenga one-hot block map 与 VSA delta-LUT 两套 kernel/pipeline；example 另含两套约 33 KB 的 codegen 和当前 HCU 缺失的 `reference_blocked_attention.hpp`。
- 当前 HCU 虽已在 P4.1/P4.7 验证 basic FMHA forward/backward，但它不是上游当前 attention core 的同代快照：对两侧 `ops/fmha` 做只读 diff，涉及 65 个文件，上游相对当前 HCU 约 29837 insertions/3804 deletions。新 Sage/Sparse pipeline 直接引用这一代 async default policy、problem/traits、mask/dropout 与 kernel Kargs，不能只复制 4/11 个 operator header。
- Sparse Jenga 的 block-map LDS staging 还调用 `amd_direct_load_global_to_lds<bool,4>`；HCU 有自己的同名实现，说明它不是“公共 builtin 完全缺失”，但其正确性必须与 HCU async FMHA pipeline 一起验证，不能据同名函数推断 ABI/时序兼容。Sage 的 gfx9 codegen 同样只是 AMD arch family 过滤，不等于 gfx946 MMAC/FP8 合同已适配。

结论：这两项均保留为独立的大闭包项目，后续顺序应为 Sparse FP16 Jenga（one-hot）→ Sparse FP16 VSA（LUT）→ Sage BF16 no-quant pipeline probe → Sage quantized dtype；每项先完成 FMHA core 代际对齐，再做 PMD 指令门禁。当前不制造会在 host/core 层失败的候选，也不把 P4.1 basic FMHA 结果外推为支持。

本步只更新文档，没有代码、build graph、device payload 或 dispatch 变化，故不运行三架构和性能 A/B。下一切片 P6.29 回到当前树已经存在但 P4.1 明确未覆盖的 `01_fmha` group/varlen mode，先用 FP16 两 batch 不同 Sq/Sk 做独立 correctness 验收，为未来 Sparse attention 的 batch/head/addressing 提供本代 HCU 控制基线。

### Step 6.29：CK Tile FMHA group/varlen 最小闭包探测 — SKIP-CORE-API（2026-09-03）

- 正式 probe 为 FP16、group mode、batch=2、head=1、D=Dv=64、Sq=[32,48]、Sk=[48,64]、无 bias/mask/dropout/LSE、row-major V，三端均 `time=0`。原 build graph 的 `fwd_blob_list.txt` 在 gfx946/gfx936/gfx938 上均为 `GROUP_BLOBS=0`，运行时三端一致在 host selector 返回 `not supported yet`、RC=254，没有 kernel dispatch；PMD/真机日志 SHA256=b27da485f882abe61496794409c1b4de52eb98ecdee6557782fe7dfa78afbee1/9f5cd76a3cdbb0f24070cfe447696c6780e07c5762951d632d1c69bd74f4929d/9f5cd76a3cdbb0f24070cfe447696c6780e07c5762951d632d1c69bd74f4929d。
- 根因第一层是 HCU codegen factory 只有 `F_spad=t,F_skpad=f` 的 async pipeline，而公共 compatibility rule 对 group mode 强制 `F_spad=t,F_skpad=t`。最小候选增加 fully-padded pipeline，并用 mode rule 保证旧 pipeline 只生成 batch、新 pipeline 只生成 group；本地 codegen 自检通过，为四种 dtype/head-dim 组合生成 4 个 group blob且不改变原 4 个 batch blob。
- 三架构 focused build 随即在完全相同的共享 core API 处失败：`FmhaFwdKernel` 没有 group runner 调用的 `MakeKargsImpl`，其 `GridSize` 只接受 `(batch,nhead,seqlen_q,hdim_v)` 四参，而 group runner 需要额外的 `seqlen_k_ptr != nullptr` 五参路径。失败发生在 device binary 生成之前，不是 Shaobo/PMD 指令支持问题；gfx946/gfx936/gfx938 构建日志 SHA256=937cf69bdd0067783d29f9c8d39015d19312d2853307382d342d1ad21918f6e5/ea4b59902bc2d6c93a4da46074740fb2e4f2963986c1457136d0175c51162040/2da406ec508c96f0b128f2e12bd480fedf8a4f641f5395fbaaa6fe70c65db15b。
- 候选已从本地和三台远端完整撤回；三端重新 configure/build 后均恢复 `GROUP_BLOBS=0`。P4.1 FP16 batch `b=h=1,Sq=Sk=D=64` 再验收均 `timing:off, valid:y`：gfx946/gfx936/gfx938 binary SHA256=595cb7954ca537c0c50fedb53fc557d6f8140b2c09bd85aff1de3860acaf9c54/ab345c932cf3022f8eb36d30d9432d5d8d4e940211d7e682978ca0596c2d0f46/5d7e31a103aca9cf26adb762c228759537390e7b8257dc280ee632f57e1aaf2a，日志 SHA256=291597073a565eb53c3bde98486ff01d0342f49696d50b7162f1fc99f857909c/0bb95e0f73a188c7c720e39498dea8e5354ba6f305483da858409cf80fa16eac/0bb95e0f73a188c7c720e39498dea8e5354ba6f305483da858409cf80fa16eac。

结论为 `SKIP-CORE-API`：正确适配需要先为 HCU `FmhaFwdKernel` 补 group Kargs、per-batch sequence-start/length 与 grid partitioner 闭包，不能只扩大 codegen。最终没有代码或 binary 候选，gfx936/gfx938 性能 A/B 不适用；gfx946 没有运行 group kernel，更未测试性能。下一切片 P6.30 审计当前 `01_fmha` split-KV 生成/API 闭包，优先判断它是否已由独立 kernel 实现而不依赖缺失的 group Kargs。

### Step 6.30：CK Tile FMHA split-KV 最小闭包探测 — SKIP-CORE-API（2026-09-03）

- split-KV 确有独立 `FmhaFwdSplitKVKernel`/combine kernel、Kargs 与 grid helper，不直接依赖 P6.29 的普通 FWD group Kargs；但默认 `FMHA_FWD_ENABLE_APIS=fwd` 不生成它，extended-FWD headers 也受宏门禁。原 codegen 还把 gfx936/gfx938/gfx946 都归入通用 `gfx9` factory，生成的 device body 只检查 `__gfx9__` 并使用 `gfx9_t` launch tag，与 basic FWD 已有的精确 HCU factory 不一致。
- 最小候选为三架构各自增加 exact factory（`__gfx936__`/`__gfx938__`/`__gfx946__`、`void` launch tag），只在显式启用 split-KV 时打开 extended-FWD headers，并增加默认空的 codegen filter。FP16 batch D=64 的最小 filter 在每个架构只生成 5 个文件：1 个 basic FWD kernel/API、1 个 split kernel、1 个 combine kernel和 split API；预定 correctness case 为 `b=h=h_k=1,Sq=64,Sk=128,D=Dv=64,num_splits=2,time=0`。
- 三端 focused build 均在 device binary 生成前暴露相同的 codegen/core 代际 ABI 漂移：当前 `TileFmhaFwdSplitKVTraits` 的参数合同使生成代码把 `bool` 传给 `BlockAttentionBiasEnum`；生成 API 引用当前 core 不存在的 `BlockFmhaSplitKVCombinePipelineTileSizes`。其余 `fmha_trait`/pipeline/kernel 未声明错误都是这两个首错的级联。gfx946/gfx936/gfx938 build log SHA256=4fa7360587f102e03ef4f53fcf9265ee58afde66d607b35c78907110a11c6cd1/efc228c8954f63796b6fc4ff74e1fadb1a9a03f95eaa4f37e17b579ea30ea4a8/f9615a6978415f9e922a51de52c1511cea32730e4a94da3e367b6d287a73073c。
- 候选已从本地和三台远端完整撤回，并显式清除 build cache 中临时 API/filter 配置。三端均恢复默认 5 个 FWD blob，P4.1 FP16 batch 64 再验收 `timing:off, valid:y`；binary SHA256 恢复为 595cb7954ca537c0c50fedb53fc557d6f8140b2c09bd85aff1de3860acaf9c54/ab345c932cf3022f8eb36d30d9432d5d8d4e940211d7e682978ca0596c2d0f46/5d7e31a103aca9cf26adb762c228759537390e7b8257dc280ee632f57e1aaf2a。gfx936/gfx938 与候选前保存的 binary 逐字节相同；恢复日志 SHA256=ba2469073d78de6935247e960222157c71942574bf56f3c62d949c425c4edf7d/568e17186b386b09488cc8f68703ade4aa0837b17da7868c586ecab267e52db4/6f21d4dcd3404450f5011cb7dec680a7e59e3af5660eebab3756ddbbe247a225，运行日志 SHA256=6c0cacbe02ef1d04a0d71023ae9361e6b4140adc7487bd3862969eab2c151377/0bb95e0f73a188c7c720e39498dea8e5354ba6f305483da858409cf80fa16eac/0bb95e0f73a188c7c720e39498dea8e5354ba6f305483da858409cf80fa16eac。

结论为 `SKIP-CORE-API`：split-KV 需要先将 extended-FWD codegen、traits、combine tile-size、pipeline 与 kernel core 对齐为同一代，再进行 HCU MMAC/PMD 指令验收；本次失败不是 Shaobo 指令模拟问题。最终没有候选 binary，且两个真机恢复后与 baseline byte-identical，性能变化已直接排除，不执行性能 A/B；gfx946 全程未测试性能。下一切片 P6.31 审计 append-KV/paged-KV 是否共享同一 extended-FWD ABI 缺口，并把 attention 扩展路径的阻塞边界一次收口，随后转向已通过 example 的更深 tail/stride correctness 合同。

### Step 6.31：CK Tile FMHA append-KV / paged-KV 扩展路径收口 — SKIP-CORE-API（2026-09-03）

- append-KV 不是可绕过 P6.30 的独立入口：runner 在 `CK_TILE_FMHA_FWD_APPENDKV_API && !CK_TILE_FMHA_FWD_SPLITKV_API` 时直接 `#error`，并明确依赖 split-KV fallback；因此 P6.30 已确认的 traits/combine/core 代际漂移是其前置阻塞。append codegen 本身也仅有通用 `ArchTrait("gfx9")`，尚无 basic FWD 已具备的 gfx936/gfx938/gfx946 精确 factory。
- paged-KV prefill 同样只把所有 HCU 归入通用 `gfx9` factory；更早的闭包缺口是当前 `include/ck_tile/ops/fmha` 完全没有 paged-KV kernel、tile partitioner 或 pipeline header，而 codegen 直接引用 `TileFmhaFwdPagedKVTraits`、`BlockFmhaFwdPagedKVPipelineProblem`、`BlockFmhaFwdPagedKVPipelineQRKSVS` 与 `FmhaFwdPagedKVKernel`。现有 extended-FWD umbrella 门禁只包含 append/split headers，不能通过打开一个宏补齐 paged-KV。
- 两条路径均在 device 编译和指令执行之前有确定的公共 core 阻塞；再次在三端制造与 P6.30 同源的失败不能增加证据，因此本步不生成候选、不运行 PMD/真机。它们不是 Perf Model 的 Shaobo 指令支持问题，后续必须作为同代 extended-FWD core 迁移整体处理，顺序为 split-KV core/API → append-KV → paged-KV core/codegen。

本步只更新文档，没有代码、build graph、device payload 或 dispatch 变化，gfx936/gfx938 性能影响可直接排除，不执行性能 A/B；gfx946 不测试性能。attention 扩展路径至此完成阻塞边界收口。下一切片 P6.32 回到已通过的 `16_batched_gemm`，用 batch=3、M=N=256 的三维 grid 做更深 correctness 验收，覆盖 z 维第三批次与每批四个 output tile 的组合地址映射。

### Step 6.32：CK Tile Batched GEMM batch=3 × x/y multi-tile — PASS-PMD（2026-09-03）

- runner 原先把 `batch` 人为固定为 2，而 kernel 的 `BatchedGemmHostArgs`、batch stride 和 `GridSize(M,N,batch)` 均已是运行时合同。本步只把 host 校验放宽为正 batch，保留 M/N 为正 128 倍数、K=32 的既有最小 kernel 边界；source SHA256=ee967d0829e1060557c0cf434e7eedbca526484d9cac8bca70e95920bde6c8fa。
- 正式 case 为 FP16 input/output、FP32 accumulate、batch=3、M=N=256、K=32，host grid=(2,2,3)，同时覆盖第三批次的 A/B/C batch stride 与每批 2×2 output tile。gfx946 PMD 实际 dispatch 为 grid=(512,2,3)、wg=(256,1,1)，reference `valid:y`，日志 SHA256=38b274ea806f2cd31f3ecc587b913d0adaf8803aa59f803c75cc46c39a25e57b；只出现 P6.18/P6.23 已记录的 `v_mad_u64_u32` read-before-write 与 pending-cacheline 模拟器告警，未测试性能。
- gfx936/gfx938 正式 case 均 `grid:(2,2,3), timing:off, valid:y`，日志 SHA256 均为 ac652b7e64e0465459f5a08e6b0389bbe42eeb8799f1e04f356ccc5e68cd28d4；修改后 binary SHA256=74e05cc5c5ca90c40da314fdca972672f4236171065b91c81dce6ef1c5adfa60/e174e96d90109a11abffed6a80826b19c1f36d85c4d737b106ec6353a8321deb。gfx946 binary SHA256=91c06e02adcb60cca1d1b6657691714be14476783ad26b543095677f3d6b4570。
- P6.18 的 batch=2、M=N=128、K=32 合同在三端再次 `grid:(1,1,2), valid:y`；PMD日志 SHA256=de4e5abd5fdba764ee6cb5ba1a7461d6d9468156105c723ef001d8d8b8716c19，gfx936/gfx938 均为 1c6dccf60cd33956ac8884ccea3fa6d81b2a2749631422a1ed74b33c19992683。

本步只修改 host 参数校验；gfx936/gfx938 修改前后 `.hip_fatbin` 分别恒为 cd6b11863466163d8c8bb592f2b56183b00a63161a359a137a29c3864b0db983/7045cdd85225abcc872a81a15bd9d4aa32e057f7156146890030261ed2bf5696，均 `cmp` PASS，device payload 与 dispatch 实现不变，性能影响已直接排除，不执行性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.33 运行 batch=1、M=128,N=256 的单批次/单轴边界，验证放宽后的最小 z 维和 N 方向第二 tile 组合。

### Step 6.33：CK Tile Batched GEMM batch=1 × N multi-tile boundary — PASS-PMD（2026-09-03）

- 不修改源码，复用 P6.32 已提交并证明 device payload identity 的 binary；正式 case 为 FP16 input/output、FP32 accumulate、batch=1、M=128、N=256、K=32，隔离覆盖最小 z 维和 N 方向第二 output tile。
- gfx946 PMD 实际 dispatch 为 grid=(256,2,1)、wg=(256,1,1)，完整 CPU reference `valid:y`，日志 SHA256=3f1ec2fa69aaf684995ff7ad4edd9a369a5037ab3422363aa5f6ccaa239f5141；仍只有既有 `v_mad_u64_u32` read-before-write 和 pending-cacheline 模拟器告警，未测试性能。
- gfx936/gfx938 均为 `grid:(1,2,1), block:256, timing:off, valid:y`，日志 SHA256 均为 c7da768bd33bc79647aab9bf83281c2e7552f41fe4e80dc777e553a52bbba88a。

本步没有代码、build graph 或 device payload 变化；P6.32 已以修改前后 fatbin 逐字节相同直接排除性能影响，故不重复运行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.34 盘点已通过的基础数据搬运 example，选择一个尚未覆盖的非对齐/高维 stride correctness 合同，避免继续堆叠同一 GEMM 的等价尺寸。

### Step 6.34：CK Tile Generic Permute maximum-rank stride contract — PASS-PMD（2026-09-03）

- 当前 `GenericPermuteHostArgs::kMaxRanks=8`，而 P6.3 只覆盖 rank=4。本步不修改源码，复用已提交 binary，正式 case 为 FP16、shape=[2,1,3,2,2,1,2,3]、perm=[7,0,5,2,6,1,4,3]、固定 seed、`time=0`；输出 shape=[3,2,1,3,2,1,2,2]。144 个元素覆盖全部 8 个 stride/perm slot，并保留 256-thread block 的线程尾部 guard。
- gfx946 PMD 单次实际 dispatch 为 grid=(256,1,1)、wg=(256,1,1)，bit-exact `valid:y`，binary/log SHA256=59731872018964173d958b881e58f53b5c394fb764d314b0b0866a7bcc079d50/04cf2626262cc0852c7211e584f171ca0c81a45ecd8511198507a76dd32b439e；只报告 P6.3 已记录的 `ignore GPUVM APE` 与 pending-cacheline warning，未测试性能。
- gfx936/gfx938 均得到相同输出 shape、`timing:off, valid:y`；binary SHA256=18c54920ef3f8cfebcdf5d11491c401aa97931146e598ec5e0a65da52c5b5487/5687fd854af14a52284a6f7273b3a967ed98083962910dfe2b520c1da55e6b3b，日志 SHA256 均为 0eb3116810bf886da3ae92cf900fb57c546869cfae40b997e29f55a772f8e170。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.35 验收同一 generic permute 的 FP8/FP32 dtype dispatch，补齐 P6.3 保留但尚未运行的两种数据搬运合同。

### Step 6.35：CK Tile Generic Permute FP8 / FP32 dtype dispatch — PASS-PMD（2026-09-03）

- 不修改源码，复用 P6.3/P6.34 binary；两个正式 case 均使用 shape=[2,3,4,5]、perm=[2,0,3,1]、固定 seed、`time=0`，分别实例化 1-byte FP8 和 4-byte FP32 data-movement dispatch，并以对应位宽整数做 bit-exact CPU comparison。
- gfx946 PMD 两次实际 dispatch 均为 grid=(256,1,1)、wg=(256,1,1)，FP8/FP32 均 `valid:y`；binary SHA256=59731872018964173d958b881e58f53b5c394fb764d314b0b0866a7bcc079d50，日志 SHA256=8a1f3866d50448e080ccc522a6b3f325cb55439ecc1a69d80ea591499e4a1bc7/4056a4250ba15778eb8f58585861565ad35ab0b880dfea8848390754cd92b8b3。PMD 仍只报告 `ignore GPUVM APE` 与 pending-cacheline warning，未测试性能。
- gfx936/gfx938 两端 FP8/FP32 均输出 shape=[4,2,5,3]、`timing:off, valid:y`；FP8 日志 SHA256 均为 a4cf2e2166d8f2f17330bfc546e7ed9933af57f44ce9103c6386b65f2b8d4bf0，FP32 均为 a9065758010955495b00c31b94a268b41792bb92afb2338eff181a8b4c598557。binary 保持 P6.3 的 18c54920ef3f8cfebcdf5d11491c401aa97931146e598ec5e0a65da52c5b5487/5687fd854af14a52284a6f7273b3a967ed98083962910dfe2b520c1da55e6b3b。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。至此 generic permute 的全部公开 dtype dispatch 和最大 rank 均完成三架构基本验收。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.36 审计 `04_img2col` 当前 runner 保留但未验收的 dtype/layout/stride 合同，选择不需要新增 core API 的最小扩展。

### Step 6.36：CK Tile Img2Col group / stride / dilation / asymmetric-padding contract — PASS-PMD（2026-09-03）

- 当前 `04_img2col` 的公开合同固定为 FP16/NHWGC；P6.1 只覆盖 `G=N=1,C=8`、3×3 filter、4×4 input、单位 stride/dilation 和对称 pad=1。本步不修改源码，正式 case 为 `G=N=2,C=K=8`、filter=3×2、input=5×7、stride=(2,1)、dilation=(1,2)、left-pad=(1,2)、right-pad=(0,1)、`time=0`，对应 output spatial=(2,8)、`NHoWo=32`、`CYX=48`，同时覆盖 group offset、非单位 H-stride、非单位 W-dilation 与非对称边界填零。
- gfx946 PMD 实际 dispatch 为 grid=(256,1,2)、wg=(256,1,1)，第二个 group 由 grid.z 独立覆盖，完整 CPU reference `image_to_column: pass`、`valid:y`；binary/log SHA256=73a725312642f84da79bec46d68693f9dea076b4a165eeb8d5e69880755c2f71/0746ecc883d766a6e04a65a88b7680ddbf2e0fe6a3605550b9dacd7a5d00394f。PMD 仅报告 pending-cacheline warning，未测试性能。
- gfx936/gfx938 使用同一合同均 `image_to_column: pass, timing:off`、`valid:y`；binary SHA256=bfd7842ebf57e71cb98c36670e50ee95a6b0698c1cabc086c14e7bb4643823c4/3a1ddad95d2bf18733e8991c6cf37a44e16947c0690b80ffc8afc96aff8aa983，日志 SHA256 均为 bc2c473c49decc0d642b0aa4558cbbcba892a5ee94dee8f8567f4a0869c722c0。

本步只增加对已提交 binary 的 correctness case，没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.37 审计已通过的数据规约 example，优先选择能覆盖非整 tile 尾部、多 row 或负/非单位 stride 的最小 correctness 合同。

### Step 6.37：CK Tile Reduce M-tail launch 与 N-vector contract — PASS-PMD（2026-09-03）

- `05_reduce` 的 kernel 使用 M/N 运行时 tensor view，但 runner 原 grid 为 `m/128` 向下取整。三架构 baseline `M=129,N=256` 均只发射一个 block，前 128 行正确而 `out[128]=0`，`valid:n`、RC=254；gfx946/gfx936/gfx938 baseline 日志 SHA256=cae20ca615e0ab89e29cbc0d410e0e0d0960072e5d54a4b15ca179ecbc00df75/9883b0cf11b1c12e8d107b506c445c564800db1aeb2dbf588c91546f86982bcc/9883b0cf11b1c12e8d107b506c445c564800db1aeb2dbf588c91546f86982bcc。这是共享 host 欠发射，不是 Shaobo 指令问题。
- 独立控制实验 `M=129,N=257` 在两个真机上 129 行全错，说明本代 kernel 的 8-element vector load 还要求 N/row-stride 为 8 的倍数。最小修复把 M-grid 改为 `integer_divide_ceil(m,128)`，并在分配/launch 前拒绝非正 M/N 或 `N%8!=0`，避免 silent wrong；不修改 reduce kernel、block policy、tile、向量宽度或公共 header。source SHA256=f78617db0ff19864079aaebc6fe48010cff943ab02209bbc5cd8590a11813937。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.37-E0 | 三架构 focused build + `M=129,N=256` | PASS；均输出 `grid size 2, timing:off, valid:y`；gfx946/gfx936/gfx938 binary SHA256=c5e485120ee24843dc3c42202e2ff7aede38386af8fed07cc1add1abbf522bf1/13ea157fa274307acd109a908fb99cc00c79e6e35099f029f0a060f08731ffe5/19b10a8460f0322a829f11fd6126d2037d681577c94224e9bfc7b1e2bf807952，日志 SHA256=ff1ebbf17b788097dc9681418eb0192f68553eccc4a495a52ae890a376f826b3/6d2a4ac4b91acb373506c43e11484b3e0f002f0571642ab07698660f3dc8658d/6d2a4ac4b91acb373506c43e11484b3e0f002f0571642ab07698660f3dc8658d | G1-G4 PASS |
| P6.37-E1 | gfx946 PMD 指令观察 | 实际 grid=(512,1,1)、wg=(256,1,1)，完整 129-row reference `valid:y`；仅有 P6.2 已记录的 src1 modifier 与 pending-cacheline warning | G3 PASS；未测试性能 |
| P6.37-E2 | `N=257` 参数边界 | 三端均在 device allocation/dispatch 前输出 `n must be a multiple of 8`、RC=254；日志 SHA256 均为 9bf22b234af633669b3b4c2a3930de40a19df6d559f623ca53019fdc9282d14e | silent-wrong 已收口 |
| P6.37-E3 | P6.2 `128×128` 回归 | 三端均 `grid size 1, timing:off, valid:y`；PMD/gfx936/gfx938 日志 SHA256=ea3f65fef0322265bb7a7caa7dd053e66d76c260fe71a9a1364f4042d80468d8/783b21d9dca546935e245a2ce40c7791f321303db47337eccd9bdb37a90aa8a8/783b21d9dca546935e245a2ce40c7791f321303db47337eccd9bdb37a90aa8a8 | 既有 contract PASS |
| P6.37-E4 | gfx936/gfx938 device payload A/B | `.hip_fatbin` 修改前后 SHA256 分别恒为 03fe25a4c4570c3500c9f33df618148f1403eb378f74ba2c2c1e64018e04e225/30397d5b2d07c9ac4886e7a4e661617e9b2ec815e9d572dd901a0e18de1f2a2c | G6 PASS |

两个真机的 device payload 完全相同，既有合法 shape 的 grid 也不变；性能变化已直接排除，不执行性能 A/B。gfx946 只做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.38 验证同一 reduce kernel 的多次 N-tile reduction（N>128）与多 block M 组合，确保 M-tail 修复没有掩盖 N-window 迭代问题。

### Step 6.38：CK Tile Reduce multi-M-block × multi-N-tile contract — PASS-PMD（2026-09-03）

- 不修改源码，复用 P6.37 已提交且两个真机 device payload identity PASS 的 binary；正式 case 为 FP16 Add、`M=257,N=384,time=0`。M 维产生 3 个 block（末 block 仅 1 个有效 row），N 维让每个 block 的 tile window 连续移动并累加 3 次，组合覆盖 M-tail store mask 与多轮 N reduction。
- gfx946 PMD 实际 dispatch 为 grid=(768,1,1)、wg=(256,1,1)，完整 257-row CPU reference `valid:y`、RUN_RC=0；binary/log SHA256=c5e485120ee24843dc3c42202e2ff7aede38386af8fed07cc1add1abbf522bf1/7114389e87aa91117d065f4e88e35862a726c538266128232816609ccf4a02bd。仍只有 P6.2/P6.37 已记录的 src1 modifier 与 pending-cacheline 模拟器告警，未测试性能。
- gfx936/gfx938 均输出 `grid size 3, timing:off, valid:y`、RUN_RC=0；binary SHA256=13ea157fa274307acd109a908fb99cc00c79e6e35099f029f0a060f08731ffe5/19b10a8460f0322a829f11fd6126d2037d681577c94224e9bfc7b1e2bf807952，日志 SHA256 均为 54e797642b677e275f313277fef358fcf5c1738e226baef7e5c04f67c65368c7。

本步没有代码、build graph、binary 或 dispatch 实现变化，且 P6.37 已证明真机 device payload 不变，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.39 审计 `09_topk_softmax` 的非整 block item 数与多 row 合同，优先发现/收口 host grid 或尾部 mask 边界。

### Step 6.39：CK Tile TopK-Softmax row-tail / expert-padding / strided-IO contract — PASS-PMD（2026-09-03）

- 不修改源码，复用 P6.5 已提交 binary；正式 case 为 FP16 input/FP32 weight、tokens=33、experts=13、topk=3、input/output row stride=17/5、固定 seed=11939、`time=0`。selector 将 13 experts 映射到 16-expert 编译实例；该实例每 block 处理 32 rows，因此一次覆盖第二个 block 的单 row 尾部、experts padding 以及输入/输出非 packed 地址。
- gfx946 PMD 实际 dispatch 为 grid=(512,1,1)、wg=(256,1,1)，33 行的 top-k value/index reference 全部 `valid:y`、RUN_RC=0；binary/log SHA256=217bbc18e784757f354fdf7d710af11b27618dad4e4d7de2b9dd943325dcaf63/d6f98ac0f06250a13298870ffa3cb93369e3dc2f8a18c6040c4f19caa7418a68。仍只有 P6.5 已记录的 src1 modifier、`V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936/gfx938 同一合同均输出 `[fp16|fp32]tokens:33, experts:13, topk:3, st_i:17, st_o:5, timing:off, valid:y`、RUN_RC=0；binary SHA256=893dffdaa72ff451f7599ca00d6926db3aa05bf0ea54ddf53940a01f9d84bd76/1b10d1f983bc94e49d1fbc386c6e0c821c75585226ffc290153e836820f4c0d1，日志 SHA256 均为 bbded23540d121f9ce3f1b8e1f2aecb4b7acf501040add02d50298c3a72420dc。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.40 使用同一 row-tail/stride 合同验收 BF16 input dispatch，补齐 runner 已公开但 P6.5 尚未运行的 dtype。

### Step 6.40：CK Tile TopK-Softmax BF16 dispatch — PASS-PMD（2026-09-03）

- 不修改源码，保持 P6.39 的 tokens=33、experts=13、topk=3、input/output row stride=17/5、seed=11939、`time=0`，只把 input dtype 切换为 runner 已公开但 P6.5 未验收的 BF16；FP32 weight 与 index reference 合同不变。
- gfx946 PMD 实际 dispatch 为 grid=(512,1,1)、wg=(256,1,1)，33 行 value/index 全部 `valid:y`、RUN_RC=0；binary/log SHA256=217bbc18e784757f354fdf7d710af11b27618dad4e4d7de2b9dd943325dcaf63/4ef02490fa08e53e2ad4776f596a690b28401e675d09ef8bff857c27585959fc。没有 BF16 执行终止，仅有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936/gfx938 同一合同均输出 `[bf16|fp32]tokens:33, experts:13, topk:3, st_i:17, st_o:5, timing:off, valid:y`、RUN_RC=0；binary SHA256=893dffdaa72ff451f7599ca00d6926db3aa05bf0ea54ddf53940a01f9d84bd76/1b10d1f983bc94e49d1fbc386c6e0c821c75585226ffc290153e836820f4c0d1，日志 SHA256 均为 e02da9e6b2a335df9d8c5ff30d283dfedfc153d78123e309da3f26cf56d5f4f5。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。至此 topk-softmax 两种公开 input dtype 均完成 padded-expert/row-tail/strided-IO 三架构验收。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.41 审计 `12_smoothquant` 的非整 block row、N 向量尾部与 stride 参数域，优先收口 silent-wrong 或欠发射边界。

### Step 6.41：CK Tile SmoothQuant M-tail / N-tail / strided-row contract — PASS-PMD（2026-09-03）

- 当前 standalone SmoothQuant 为 `Block_M=2,Block_N=128,Vector_N=1,kPadN=true` 的 FP16→INT8 two-pass pipeline，grid 已对 M 做 ceil-div，pipeline 也按 `ceil(N/128)` 迭代。P6.4 只覆盖 2×128 packed；本步不修改源码，正式 case 为 `M=3,N=193,stride=211,time=0`，组合覆盖第二个 M-block 的单 row 尾部、第二个 N-window 的 65-column 尾部、reverse second-pass store 与非 packed row stride。
- gfx946 PMD 实际 dispatch 为 grid=(512,1,1)、wg=(256,1,1)，yscale 和全部 3×193 有效 qy 输出均通过 CPU reference，`valid:y`、RUN_RC=0；binary/log SHA256=1148cde5e63810951e2156f4f2f38e994d7111cd05c385440227117f35137e6b/4fef4fe45acec46b1756f4bbc05ee9062dfc7145f1b2cc85132f3b50c36f589a。仍只有 P6.4 已记录的 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936/gfx938 同一合同均输出 `timing:off, [fp16] m:3, n:193, stride:211, valid:y`、RUN_RC=0；binary SHA256=1b918fce81d32946cfa184e730c83d50553c38d859ba53897b600813327ecb1a/736859f07afaf21f9fd7ba691431a56f20d8deaddca4b0663484f094f4193d32，日志 SHA256 均为 4e04aa651c1274e0467119b408193629f36137e3ce5299e641803d22849783dc。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.42 审计 standalone SmoothQuant BF16 input 是否能在现有同代 kernel 中形成独立最小闭包；若只需实例化 runner，则构建并按三架构 correctness/性能影响门禁验收。

### Step 6.42：CK Tile standalone SmoothQuant BF16 input — PASS-PMD（2026-09-03）

- 静态闭包确认 standalone runner 已由 `run<DataType>` 模板表达，现有 `SmoothquantPipelineProblem`、two-pass pipeline、host reference 和 BF16 type conversion 均无需公共改动；缺口仅是 main 没有实例化/分发 BF16。本步把 `prec` help 扩为 fp16/bf16，并增加 `run<ck_tile::bf16_t>` 分支；source SHA256=3fc965f4273ea5889edc7cf568c01870311e07eeda1d1d080b49bf35d2476f54。
- 正式 BF16 case 复用 P6.41 的 `M=3,N=193,stride=211,time=0`，同时覆盖 BF16 input→FP32 scale/compute→INT8 quant、M/N tail 与非 packed stride。三端均 focused build PASS；gfx946/gfx936/gfx938 binary SHA256=9a1059c7099d2f6f866e0bc41589dbbae22ce1f95f9d6bbf9d94e4b1f366c650/645132fc4f999a428b33a6734e7398d63f707601646283b5051f92db35ce56bc/f8712bbeccad0cd72d9a828fa998d0f7e1876054bd8c8320b02b4a869e61e008。
- gfx946 PMD 实际 dispatch 为 grid=(512,1,1)、wg=(256,1,1)，yscale/qy reference `valid:y`、BF16_RC=0；日志 SHA256=0b12635cf98e0459c608c9a46bf6c2ec5824fe6c3df82f3893808863ff4f93ae。没有 BF16 终止指令，仅有既有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning；未测试性能。
- gfx936/gfx938 BF16 均 `valid:y`、BF16_RC=0，日志 SHA256 均为 fb5cf55c1e3e4b8e92ba1c3614179789a203a37367abd193b47aadcab560104f；P6.4 FP16 `M=2,N=128` 在三端也均 `valid:y`，PMD/gfx936/gfx938 日志 SHA256=ad67181627a2b1bbcdcf8a6969ef04c9ec70ea600838abdc8d208d6f7626032a/4934d1b3f5f3e963e170dae9c1760733bc9e4aa08f2390835c4ab1d72e4fd950/4934d1b3f5f3e963e170dae9c1760733bc9e4aa08f2390835c4ab1d72e4fd950。

新增 BF16 实例使 gfx936/gfx938 `.hip_fatbin` SHA256 分别由 31f4efbf29b93487d89022e967ee42a73a89d47a1fe7f981e1b82b49a072a516/ab10947362e089d637f9d47b9d1ef67e2e413708069c9c5862767bb8163e208a 变为 c678e8ab63764c009746da0e7b2c5bc176644b349b0d18f5d5bed0f4c9b634c2/db03e699966e9cb1a706b4831042db7896fd3a6891c0d6e8829006e08505fb97，不能用 payload identity 排除 FP16 性能影响，因此已在两端空闲（HCU use/memory use 均为 0）时执行真实性能 A/B。固定 FP16 `M=3328,N=stride=4096`、warmup=20、repeat=100、独立运行 5 次：gfx936 baseline=[0.147508,0.147481,0.147515,0.147492,0.147482] ms，candidate=[0.147532,0.147524,0.147475,0.147507,0.147555] ms，中位数 0.147492→0.147524 ms（+0.0217%）；gfx938 baseline=[0.181326,0.181150,0.181325,0.181258,0.181056] ms，candidate=[0.181226,0.181073,0.181159,0.181097,0.181250] ms，中位数 0.181258→0.181159 ms（-0.0546%）。两端变化方向相反且绝对值小于 0.06%，判定为测量噪声，无性能回退；baseline/candidate 日志 SHA256=9290128b2b8ec440ad84ac2b979217257ae96f9c8bb9e82ffa02ccda0b588449/66da4d031d7ea6ef739b72da9f40ba5fb9fd23dc4d75715d24f5e81ea57377be（gfx936）、b0153db189bb5ef791ae918f868e90a3962a3e58453e0e033be781b492d76699/e417a60d2a14803ecad50ac5a024338846789d1132a24250461c5d1c0eb9ad27（gfx938）。

门禁结果：G0-G6 PASS。下一切片 P6.43 审计 `21_elementwise` 已提交 kernel 的非整 block element tail 与非 packed stride，优先选择可证明 padding/guard 的现有 target；若 runner 限制掩盖合法 kernel 参数，则做最小 host-contract 修复。

### Step 6.43：CK Tile Elementwise Add block-tail 与 packed-stride contract — PASS-PMD（2026-09-03）

- `ElementWiseKernel` 先对二维 lens 做 `make_merge_transform`，再按合并后的一维逻辑索引访问；runner 原来却允许 `stride>=n`。正式反例 `M=257,N=8,stride=13` 在 gfx946/gfx936/gfx938 均发射两个 block 后 `valid:n`、RC=254，错误从第二行物理 gap 开始，共 1280 个错误、38.3693%；日志 SHA256=b02ea088c4cde6e961752ff79342bb202e464fa23607feb0220363937b820b68/d822fc7ec1557323e9abcec8fab4b96219ce46a1f1178c738db30b59ffb760aa/d822fc7ec1557323e9abcec8fab4b96219ce46a1f1178c738db30b59ffb760aa。这是共享 host/API contract 缺陷，不是 Shaobo 指令问题。
- 保持总逻辑元素 2056、改为 packed `stride=8` 后三端均 `grid:2,block:512,timing:off,valid:y`；第二个 2048-element block 仅含一个 8-element FP16 vector，证明 `kPad=true` 的 block tail 健康。gfx946/gfx936/gfx938 健康控制日志 SHA256=64708d45d32a7d0999c940c36c2ea750204147b2cb4b54f41b2970a359f6dcbc/ada33fb5a37977f46b5ab3bbf0831c2dc507c81b0c8d0ab8d61d51b597474a05/ada33fb5a37977f46b5ab3bbf0831c2dc507c81b0c8d0ab8d61d51b597474a05。
- 最小修复只把 runner 参数合同从 `stride>=n` 收紧为 `stride==n`，非法 stride 在 device allocation/dispatch 前返回 RC=252 和明确错误；kernel、merge transform、tile/vector、Add functor 与公共 header 均未修改。source SHA256=7580c8b5a323ed888e922cbaf6b8a33e1cbeb0658cf4b61c96e8b86f8b1dc5f4。

| ID | 单一动作 | 结果 | 分类/下一步 |
| --- | --- | --- | --- |
| P6.43-E0 | 三架构 focused build + packed block-tail | PASS；gfx946/gfx936/gfx938 binary SHA256=4d6e43aafe3fe8f0183ac636bfb548c91d18bfc70723209dccca06798013348d/64ff5272b82746a00115827443878a7e842729a5b2c3e4e769d7704a15f118cb/160c526ac61b87f5611ab05578b860b1cf337030c53b2cec7f0e9d78c7bd5923，均 `valid:y` | G1-G4 PASS |
| P6.43-E1 | gfx946 PMD 指令观察 | 实际 grid=(1024,1,1)、wg=(512,1,1)，2056-element reference 正确；仅有 P6.7 已记录的 `V_ADD_F16 not test`、src1 modifier 与 pending-cacheline warning | G3 PASS；未测试性能 |
| P6.43-E2 | 非 packed stride 拒绝 | 三端均在 dispatch 前输出 `this merged kernel requires stride == n`、RC=252；日志 SHA256 均为 8cd586ddf30c484e1a6ebfd2fa2aea91228c997c3caf77afdaf29f7e50c5f428 | silent-wrong 已收口 |
| P6.43-E3 | P6.7 `M=32,N=64` 回归 | 三端均 `grid:1,block:512,timing:off,valid:y`；PMD/gfx936/gfx938 日志 SHA256=a2e33cdfcf75e34014ddaa2c4e0b3e48178d340ad632c7fab6c52f50fc45e747/c431fde5ac91dbc7ff69dc8c2ea3e2ecad7290b47d44bc2820513d6f72af6846/c431fde5ac91dbc7ff69dc8c2ea3e2ecad7290b47d44bc2820513d6f72af6846 | 既有 contract PASS |
| P6.43-E4 | 三架构 device payload A/B | gfx946/gfx936/gfx938 `.hip_fatbin` 修改前后 SHA256 分别恒为 c075a7f6d18f45b454012ca40748f00d039f485f18658a1ff31e11072064c244/b9d73edde9b9d4ed0d7d1ba82fe4232069c7e89fbadb0f2fc34c297d0f72310a/5bd0dbe3099c0306bb3734195952e42d0fb97ae8271f4752c2b170332c0912a1 | G6 PASS |

device payload 完全相同，既有 packed shape 的 grid/执行实现也不变，性能影响已直接排除，不执行 gfx936/gfx938 性能 A/B；gfx946 只做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.44 审计 unary-square/transpose/4D elementwise runner 是否也把非 packed stride 暴露给同一 merge kernel；共享问题只修各自实际暴露的 host 参数，不扩大公共 kernel。

### Step 6.44：CK Tile Elementwise Unary-Square packed-stride contract — PASS-PMD（2026-09-03）

- 静态审计确认 4D Add runner 始终内部构造 dense stride，没有暴露 P6.43 问题；Unary-Square 和 Transpose 则分别暴露 `stride`/`stride_in>=n`。本步按“一 example 一 commit”只处理 Unary-Square，Transpose 留给 P6.45。
- Unary-Square 复用与 Add 完全相同的二维 merge kernel。反例 `M=257,N=8,stride=13` 在 gfx946/gfx936/gfx938 均 `grid:2` 后 `valid:n`、RC=254，共 1266 个错误、37.94964%；日志 SHA256=d6969c42cf8afd8bf2d4df63ee65180789842be42bece46c0ca67749c584bf96/08ce8d0f93a723a7c336bce6c407585afe76a64e8931b179c54f12e87e2b4a89/08ce8d0f93a723a7c336bce6c407585afe76a64e8931b179c54f12e87e2b4a89，确认是同一共享 host contract，不是 Shaobo 指令问题。
- 最小修复只把 Unary-Square runner 收紧为 `stride==n`，source SHA256=55ba6111366e5a729a6ef3ea34ee350cfb24ab31ca161b49f200f45472f3c0e4；不改 kernel、tile/vector、UnarySquare functor或公共 header。
- 修改后三端 `M=257,N=8,stride=8` 均 `grid:2,block:512,timing:off,valid:y`，gfx946/gfx936/gfx938 日志 SHA256=8c58747362a3dfdf0be3e57a070777de1f262b0a87305078606a9920e36a9036/ada33fb5a37977f46b5ab3bbf0831c2dc507c81b0c8d0ab8d61d51b597474a05/ada33fb5a37977f46b5ab3bbf0831c2dc507c81b0c8d0ab8d61d51b597474a05；P6.8 `M=32,N=64` 回归也均 `valid:y`，日志 SHA256=c313218d35277169eeee8e47a15f902ab79e29def2010740fd67af8e5ac90fa9/c431fde5ac91dbc7ff69dc8c2ea3e2ecad7290b47d44bc2820513d6f72af6846/c431fde5ac91dbc7ff69dc8c2ea3e2ecad7290b47d44bc2820513d6f72af6846。非法 stride 三端均在 dispatch 前 RC=252，日志 SHA256=8cd586ddf30c484e1a6ebfd2fa2aea91228c997c3caf77afdaf29f7e50c5f428。
- gfx946 PMD packed-tail 实际 grid=(1024,1,1)、wg=(512,1,1)，只出现 pending-cacheline warning；未测试性能。gfx946/gfx936/gfx938 `.hip_fatbin` 修改前后 SHA256 分别恒为 1b16482b06e6c64131e22ee9ac556fb33fc074afc1e452e76955db192305b6f5/1a1a699ee8344b194855a6eaf1cda23ad4f667a3810c0a9c34135ae8bdd88c2d/cae21f6bd54e92c8d9601fbe2afc4824ffbe1a8594407c04f8ac88d3f943b3c5。

device payload 完全相同，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。下一切片 P6.45 对 Transpose 的 `stride_in` 做独立反例、packed 控制与三架构收口；不得与本提交混合。

### Step 6.45：CK Tile Elementwise Transpose block-tail 与 strided-input contract — PASS-PMD（2026-09-03）

- 静态复核纠正了 P6.44 的初步风险判断：Transpose 虽也把二维 lens 合并为一维 tile，但同时把 `input_strides=(stride_in,1)` 和转置后的 `output_strides=(1,m)` 传入 kernel descriptor；它不是 Add/Unary-Square 那种只按 packed 一维地址解释输入的路径。因此保留公开合同 `stride_in>=n`，不做错误的 packed-only 收紧。source SHA256=199c18c61a3acce4f424cf50d10c65b6944f88f8042a515ff0de04e516297278。
- 正式 case 为 FP16 `M=257,N=8,stride_in=13,time=0`，总逻辑元素 2056，组合覆盖第二个 2048-element block 的单个 8-element vector 尾部和每行 5-element 物理 gap。gfx946 PMD 实际 dispatch 为 grid=(1024,1,1)、wg=(512,1,1)，输出 `grid:2,block:512,timing:off,valid:y`、RUN_RC=0；binary/log SHA256=d35434a39cbecbb97f7f13da1b5118ef7283a3211b93a3aeef7b07f1b15d2326/bdba9b70fd2245b16e2f0fe453a691c6e651f2c8d17a64f8637c703e22c29fe7，仅有 pending-cacheline warning，未测试性能。
- gfx936/gfx938 同一 strided-input 合同均 `grid:2,block:512,timing:off,valid:y`、RUN_RC=0；binary SHA256=e811096e09ac5c93768f60e0a5489b147dcd60cc1685639e939d264b345cbf6e/e63ade10bb64ddb85e2d3f4239f31bc6b9e631e37fad9366b1f377a741571472，日志 SHA256 均为 ada33fb5a37977f46b5ab3bbf0831c2dc507c81b0c8d0ab8d61d51b597474a05。
- packed 健康控制 `M=257,N=8,stride_in=8` 在三端也均 `valid:y`、RUN_RC=0；gfx946/gfx936/gfx938 日志 SHA256=5d732734a16e3e5514a4122c9f8d50649652543b26fe7ce4ae1eaa6c3ba1db69/ada33fb5a37977f46b5ab3bbf0831c2dc507c81b0c8d0ab8d61d51b597474a05/ada33fb5a37977f46b5ab3bbf0831c2dc507c81b0c8d0ab8d61d51b597474a05，排除仅靠物理 gap 偶然通过的解释。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。至此 `21_elementwise` 三个公开二维 runner 的 stride 合同已分别用反例/对照闭合；下一切片 P6.46 独立验收 `elementwise_add_4d` 的 4D dense-stride、多维边界与 block tail。

### Step 6.46：CK Tile Elementwise Add-4D multi-dimension / block-tail contract — PASS-PMD（2026-09-03）

- P6.10 的固定 `[2,4,8,32]` 恰为 2048 elements，只覆盖单个完整 block。本步不修改源码，将正式 shape 改为 `[3,5,7,24]`：前三维都是非 2 次幂，dense strides 为 `[840,168,24,1]`，总元素 2520；最后一维保持 FP16 8-element vector 对齐，同时让第二个 2048-element block 只处理 472 elements，组合覆盖四维 merge 映射和 block padding/guard。source SHA256=289c1db219b447a293041387256be88ed26be988cf9db9543e05c40eb4e927e6。
- gfx946 PMD 实际 dispatch 为 grid=(1024,1,1)、wg=(512,1,1)，输出 `grid:2,block:512,timing:off,valid:y`、RUN_RC=0；binary/log SHA256=677dc5240c6c99fe46fa02ccd8e0dc318d21f8de0cbbbb443406e278830506fe/4c1afd178585a24d38870af8d832968c8f0d398ac88ab2489a7453935008ad26。只出现既有 `V_ADD_F16 not test`、src1 modifier 与 pending-cacheline warning，没有终止或数值错误；未测试性能。
- gfx936/gfx938 同一 shape 均输出 `grid:2,block:512,timing:off,valid:y`、RUN_RC=0；binary SHA256=304ffd2b0e901573aace0d8948b296b4b562e0430ad87e7808d930ebd631a537/0b36f66ce97583746b621b9a48a862ad7636bde951752cfa478e0e72334d0c6e，日志 SHA256 均为 ada33fb5a37977f46b5ab3bbf0831c2dc507c81b0c8d0ab8d61d51b597474a05。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。至此 `21_elementwise` 四个 executable 的基础、stride 和 tail 合同均闭合；下一切片 P6.47 回到已启用 target 清单，按“公开 dtype/参数分支尚未覆盖”重新筛选最小独立验收项。

### Step 6.47：CK Tile basic Batched-Transpose rectangular multi-grid contract — PASS-PMD（2026-09-03）

- 公开 dtype 分支复盘后，最小未覆盖参数路径是 P6.12 basic batched-transpose 只验过 `batch=2,height=32,width=32` 的 grid=(1,1,2)。本步不修改源码，正式 case 改为 FP16 `batch=3,height=64,width=96,time=0`，仍遵守 basic pipeline 的 32-aligned 合同，不与 P6.13 已记录的 padded core-API 缺口混淆；预期并实际得到 grid=(2,3,3)，独立覆盖 height、width、batch 三个 grid 维度都大于 1 的 offset 映射。source SHA256=4273ab969c8cedab1537bd1d8012312b5f66e8fbf16d6dfed46e22568470859f。
- gfx946 PMD 实际 dispatch 为 grid=(128,3,3)、wg=(64,1,1)，3×64×96 输出以零容差通过，runner 输出 `grid:(2,3,3),block:64,timing:off,valid:y`、RUN_RC=0；binary/log SHA256=7745a3ec047bbf4e528449fd04005fbf52ad2c5bf5d7cdf2633b7c5a308c5150/833fe897a3d4b9c7510a543c7779f4aaf087c415844c1ce24de9f8984331e10c。PMD 报告 18 条 P6.12 同类 `v_mad_u64_u32 ... read vgpr before writing` 和一次 pending-cacheline warning，但完整输出正确；未测试性能。
- gfx936/gfx938 同一 multi-grid case 均 `grid:(2,3,3),block:64,timing:off,valid:y`、RUN_RC=0；binary SHA256=8c8babc5ae815f29930f337bf5d3fb40f7a60099ead1c5205b9ddeab2ed231fc/ca4b50ee9b2b2f0ff3551ab153ddec3b323ba948c66bdedfa84ee067b1b8deaa，日志 SHA256 均为 d1eb643f51f47e0c1a4d13669d85bd6773ab00459b8e6516e4107578aa544982。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。P6.12 的 basic aligned pipeline 现已覆盖单 tile 与三维 multi-grid；P6.13/P6.14 的 padded/LDS core-API 边界保持不变。下一切片 P6.48 继续筛选已启用 runner 的公开 mode/shape 分支，优先选择可在现存 binary 上独立收口的边界。

### Step 6.48：Add-RMSNorm-RDQuant BF16→INT8 scalar-store contract — PASS-PMD（2026-09-03）

- P1.5b 的 PMD 单变量扫描只验证 FP16→INT8；本步补齐 instance runner 已公开但未独立运行的 BF16 input。正式 case 为 `M=4,N=stride=64,save_x=1,quant=int8,time=0`，命中 `add_rmsnorm2d_rdquant_fwd_bf16_4x64_4x1_1x64_1x1_wpr_op_pn_x`，将 `Vector_N=1` 固定为已知健康标量 store，从而只判断 BF16 add/RMSNorm/scale/quant 执行，不混入 P1.5b 的 vectorized INT8 packing PMD 缺口。runner/API/BF16 n64-n128 instance SHA256=77c480127d148be5bba9774914c38fbc94ff4eff758aa99a459396dcdc30a6c0/0f7e335f12dc2dcb99e4c2e3a82c7e458558bcfa52ebf4661f8c3e5305459b45/891a229d3eda9c8a0e1e87af19fb2d0d8d3a03458fb91381e7b1676ae6d8c0cc。
- gfx946 PMD 实际 dispatch 为 grid=(256,1,1)、wg=(256,1,1)，保存的 BF16 X、FP32 yscale 与 INT8 qy 全部通过 reference，`timing:off,valid:y`、RUN_RC=0；binary/log SHA256=d816c64fe9fcc3a807dc613782ebf864cc259a163993b4002a08e550bfdec309/5e4b570432e7d5639fd1a7bff9be5699bb6542207f6c4132a6fb317a9bd4e124。仅有既有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936/gfx938 同一 BF16→INT8 case 均命中同名 scalar instance 并 `timing:off,valid:y`、RUN_RC=0；binary SHA256=0c80ce0ec1639b3e04b32ff3114ec53ecefb76098be9ca648e0bcbe2228dd369/63db89b80525ff566139fe26d138b02c1e458b5283d795f40085adb49d60d6d2，日志 SHA256 均为 6c46990e0a371924ee52fecdcd21adb57a67861ecba2a91244505c8a2319a171。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。结论限定为 BF16→INT8 `Vector_N=1`，不改变 P1.5b 对 `Vector_N>1` 的 `SKIP-PMD`。下一切片 P6.49 使用同一 scalar instance 边界独立验收 FP8 quant output，判断 gfx946 的 conversion/store 是否可运行。

### Step 6.49：Add-RMSNorm-RDQuant BF16→FP8 scalar-store contract — PASS-PMD（2026-09-03）

- 复用 P6.48 的 BF16 `M=4,N=stride=64,save_x=1,time=0`，唯一变量是 quant output 从 INT8 改为 FP8；selector 仍命中 `...bf16_4x64_4x1_1x64_1x1_wpr_op_pn_x`，即 `Vector_N=1`。该 A/B 将 BF16 input 和 add/RMSNorm/reduction 保持不变，独立验证 gfx946 的 FP8 conversion/scalar store 与 FP8 host reference。
- gfx946 PMD 实际 dispatch 为 grid=(256,1,1)、wg=(256,1,1)，保存的 BF16 X、FP32 yscale 与 FP8 qy 全部通过 reference，`timing:off,valid:y`、RUN_RC=0；binary/log SHA256=d816c64fe9fcc3a807dc613782ebf864cc259a163993b4002a08e550bfdec309/c3e337626b5e62497f03abf4478fc687d99a69afcd6ff55aadc6e219f3153a7f。仅有既有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936/gfx938 同一 BF16→FP8 case 均命中同名 scalar instance 并 `timing:off,valid:y`、RUN_RC=0；binary SHA256=0c80ce0ec1639b3e04b32ff3114ec53ecefb76098be9ca648e0bcbe2228dd369/63db89b80525ff566139fe26d138b02c1e458b5283d795f40085adb49d60d6d2，日志 SHA256 均为 f1c5ec2f659cd81d8193b3411c5fb0abcb5184bc0636f443e56c4c5dced752d6。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。结论限定为 FP8 `Vector_N=1`；下一切片 P6.50 把 N 提升到 128 命中 `Vector_N=2`，判断 P1.5b 的 PMD vectorized packing/store 缺口是否也影响 FP8 output。

### Step 6.50：Add-RMSNorm-RDQuant BF16→FP8 Vector_N=2 contract — PASS-PMD（2026-09-03）

- 复用 P6.49，只把 `N=stride` 从 64 提升到 128；selector 实际从 `...1x1...` 切换到 `add_rmsnorm2d_rdquant_fwd_bf16_4x128_4x1_1x128_1x2_wpr_op_pn_x`，即 `Vector_N=2`。这与 P1.5b 的 BF16→INT8/FP16→INT8 无关，只用于判断 FP8 两元素向量 conversion/packing/store。
- gfx946 PMD 实际 dispatch 为 grid=(256,1,1)、wg=(256,1,1)，保存的 BF16 X、FP32 yscale 与 FP8 qy 全部通过 reference，`timing:off,valid:y`、RUN_RC=0；binary/log SHA256=d816c64fe9fcc3a807dc613782ebf864cc259a163993b4002a08e550bfdec309/839d14aff284a02daba02b0660b640254ed785469c716dca13b7fdaa284498c4。除了既有 `V_DIV_FIXUP_F32 not test`/pending-cacheline，还出现 src1 modifier warning，但没有终止或数值错误；未测试性能。
- gfx936/gfx938 同一 `Vector_N=2` case 均命中同名 instance 并 `timing:off,valid:y`、RUN_RC=0；binary SHA256=0c80ce0ec1639b3e04b32ff3114ec53ecefb76098be9ca648e0bcbe2228dd369/63db89b80525ff566139fe26d138b02c1e458b5283d795f40085adb49d60d6d2，日志 SHA256 均为 aa491d0553baa715ee729176db091bb43f82d438004879c6d4783dc661cc2d27。

本步没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。FP8 `Vector_N=2` 正确而 P1.5b INT8 `Vector_N=2` 稳定奇数槽清零，进一步把既有 PMD 缺口限定到 INT8 SDWA byte packing/store，不能外推为一般向量低精度 store 不支持。下一切片 P6.51 继续验证 FP8 `Vector_N=4/8`，闭合现有 instance 的主要向量宽度。

### Step 6.51：Add-RMSNorm-RDQuant BF16→FP8 Vector_N=4/8 — PARTIAL / SKIP-PMD-VEC4（2026-09-03）

- 同一未修改 binary 上运行 BF16→FP8 `M=4,save_x=1,time=0`：`N=256` 命中 `...bf16_4x256_4x1_1x256_1x4...`，gfx946 PMD 两次均稳定 `valid:n`、RC=254；首次/复验日志 SHA256=6fa785cec78e4fdc0ac76534e723d09d8d2de680c11506737c6169974567b074/9364884812da202a2b5e5e8b1e2664d35457e6da390e1080012d8c5cf2785e97。两次签名相同：保存的 X 有 1014/1024（99.02344%）错误，FP8 qy 有 1023/1024（99.90234%）错误，输出近全零。
- 相邻健康控制均通过：保持 BF16→FP8、改 `N=512` 命中 `Vector_N=8`，PMD `valid:y`、RC=0，日志 SHA256=ac9c53548891e3619e8c121628f4059937d19ad9f1b53bf19e6322c2ee9d64c9；保持 `N=256,Vector_N=4`、只把 input 改为 FP16，PMD 也 `valid:y`、RC=0，日志 SHA256=044aba7811f4fd3245682edcba5f5233e6d775f3eced5eee2c7cbf38ff850527。结合 P6.50 的 BF16→FP8 `Vector_N=2` PASS，失败被限定为 BF16×FP8×Vec4 实例组合，而不是 BF16、FP8 或向量 store 的一般不支持。
- gfx936/gfx938 的 BF16→FP8 Vec4/Vec8 四次都命中预期实例并 `valid:y`、RC=0；两端 binary SHA256=0c80ce0ec1639b3e04b32ff3114ec53ecefb76098be9ca648e0bcbe2228dd369/63db89b80525ff566139fe26d138b02c1e458b5283d795f40085adb49d60d6d2，Vec4/Vec8 日志 SHA256 均分别为 0182640819b7f9edcf588345a30889ca4722ff8569a72eba115fb7734ff4a853/2e1baaf30ee962e33eb86b0f8ed44f01345f10a457ec025699f6cf546d723eaf。
- gfx946 code object SHA256=228b5a9e5a3c2c4d9190a7c5b683c53bf70eb043418d68d01f0f7bb834af0fda；BF16→FP8 Vec8/Vec4/Vec2 的独立 ISA 区间 SHA256=fb51c0ec4386f90c3943df6afafb1c2b8758ecb5c1e3a86d03e1bf6996d87595/b8d15347f700b3d1340943d1aec2e6e3ae53fdae7ec54060fc5352ed539bbfbe/5cf2ffe0c3a170f6046686cc0451cef8480eb26d32f12a9f3cd71c2f01e602c9。Vec4 区间包含 `buffer_store_dwordx2 ... offen offset:1024` 等特有的打包/地址组合，但 PMD 没有给出显式 unsupported-instruction 终止，且 FP16 Vec4 正确，故不把任何单条指令写成已证实根因。

按“PMD 支持问题无法在 CK 内可靠解决则记录并跳过”处理：BF16→FP8 Vec4 标记 `SKIP-PMD-VEC4`，保留 Vec2/Vec8 PASS；不为了 CPU 模拟器绕过而修改两台真卡均正确的 production selector。没有代码、build graph、binary 或 dispatch 实现变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B；gfx946 未测试性能。门禁结果：G0-G2/G4/G6 PASS；G3 PARTIAL；G5 NOT RUN。下一切片 P6.52 审计 `save_x=0` 的公开 host dispatch 是否真实选择 no-save 实例，先判定合同问题再决定最小修复边界。

### Step 6.52：Add-RMSNorm-RDQuant BF16 no-save host dispatch — PASS-PMD（2026-09-03）

- 静态闭包确认 `N∈(4096,8192]` 的 API/instance 已同时提供 `kSaveX=true/false`，但 main 的 BF16→INT8 和 BF16→FP8 两个 `save_x=0` 分支都误调用 `run<...,true>`；FP16 分支则正确传 `false`。修改前在三架构运行 BF16→INT8 `M=1,N=stride=4097,save_x=0,time=0`，虽均 `valid:y`，实际 kernel 名都以 `_x` 结尾，直接证明公开 no-save 参数没有生效。gfx946/gfx936/gfx938 baseline 日志 SHA256=dbbdef1e04663c73512561a8452b3bf98c3aef2a990f653a9bb8afdafd757939/56ae48eb1ac90cf1adad92bebbf7b799d5898c9c7ee92f755bc0144e859c174a/56ae48eb1ac90cf1adad92bebbf7b799d5898c9c7ee92f755bc0144e859c174a。
- 最小修复仅把这两个 BF16 no-save host 模板参数改为 `false`，不改 API、instance、pipeline 或 device kernel；source SHA256=d4bdaa2679106652dbcb482f6dfa43ca3070b7acbcef297584e3c12d5129429c。三端 focused build PASS；gfx946/gfx936/gfx938 candidate binary SHA256=271735b1e629bacefd20e11b62e3017e81fbf5171a280d1bbc3ebedc5cc995a6/150e0f45184796477fcef616942b1f8bc8ee2b78560445269905646a530e9e1e/ef56531bcee61af89885a9bdeb6e200e74de652c8a34345564e0554a0ab9f6f9，build-log SHA256=60c236bf557a984a13961d684b0553d4cf9029ca0823f0b1b592297045780b40/e64318ccea2f121b7aec18326417ac584575ae76f40f7754a8959989b47452cd/3edd2384b24f84ec318e5ad4f15239a62e431d45fff4bcc5bc95942b2f6a5523。
- 修改后三端 BF16→INT8/FP8 `save_x=0` 均命中不带 `_x` 的 no-save instance 并 `valid:y`、RC=0；gfx946 INT8/FP8 日志 SHA256=e361a9c2bd451aae2d1406e1303af7bd975cdb8167e55464a5cc2141bc80c58e/8a19aab14a8ad0d33ad2e96959faf5556ceb621836078d77c8e7c53711b67ac5，gfx936/gfx938 对应日志均为 2448f7ef53e98b2400ffc797d5efbc7fe18aff0091733e4830dfbcd06437fd04/309787d9726507c874810dd7392079cd8b36f3f0ccde2aefaa2af6c8a176b7e3。BF16→INT8 `save_x=1` 三端仍命中 `_x` 且 `valid:y`；PMD/双真机日志 SHA256=0448144d38937ef055106648f6d33f7b1c63b57f36a73fd231d0135661703b5b/56ae48eb1ac90cf1adad92bebbf7b799d5898c9c7ee92f755bc0144e859c174a。
- 重链接后 gfx946/gfx936/gfx938 `.hip_fatbin` 文本哈希分别为 7d41cb4b1f7693499055b540b962f74fd610229220455f1d6b9bf1214734f44e→c5ad7bae67de5bf481dbb046b0d5c50d7f99e30644fc7c7d29c5be147b105c90、590ab1235e7930ec7d79b70f654a29f4b65001825e130dec72d46a03e6fc3ee7→0268c45761541795ba4129bea70a9eda5017208567b79be3d6816d9367ab3161、42d02c477401536b1f9bf376d765d96169e2d1b1e7ebe5fa1ff1df64eadf7f67→7be311b933e9efc15a52389d0b195e7cc9c9ac2b7a0183957928116476914ce4；且 no-save 实际 dispatch 已改变，故不能用 payload identity 排除性能影响。
- gfx936/gfx938 测试前 7 张 HCU 的 use/memory-use 均为 0。固定 BF16→INT8 `M=3328,N=stride=8192,save_x=0,warmup=20,repeat=100,v=0`，交替运行 baseline/candidate 五对：gfx936 baseline=[186.720,189.374,189.358,189.003,186.719] us，candidate=[141.131,145.533,145.527,144.460,142.246] us，中位数 189.003→144.460 us（-23.57%）；gfx938 baseline=[201.372,201.165,201.154,201.206,201.143] us，candidate=[158.751,158.816,158.815,158.808,158.797] us，中位数 201.165→158.808 us（-21.06%）。性能日志 SHA256=aeb0f045605e187a79c95b27ba522f71a197688b04d1a0d275ffa930856f7ac0/08ac24ef2e0e19420a19ea0a271f8f1fcb5a3df86cef1c262fe15ac4f70063b0；GB/s 口径会随 `SaveX` 改变计数字节数，跨版本判定只使用同 workload 的 kernel latency。

门禁结果：G0-G6 PASS。修复恢复了 BF16 no-save 公开合同，并在 gfx936/gfx938 分别带来 23.57%/21.06% 延迟下降，无性能回退；gfx946 PMD 全程只做 correctness。下一切片 P6.53 独立检查 `N<=4096` 时 API 无条件选择 save-X 的现有参数语义，区分设计边界与 runner 误导，避免把大 N 修复错误外推。

### Step 6.53：Add-RMSNorm-RDQuant no-save shape-domain audit — DEFER-SHARED-INSTANCE-CLOSURE（2026-09-03）

- 静态审计确认 no-save 并非所有 N 的完整公共能力：API 在 `N<=4096` 的所有 selector 分支无条件实例化 `kSaveX=true`；仅 `4096<N<=8192` 同时按 runtime `save_x` 选择 true/false；`N>8192` three-pass 又无条件为 true。显式 `kSaveX=false` 的模板实例也只集中在 n8192 instance TU。因此 P6.52 只修复“已有 no-save device 实例却被 BF16 main 布尔参数挡住”的 host 缺陷，不能外推到全部 shape。
- 修改后的同一 binary 上运行 BF16→INT8 `M=1,N=stride=64,save_x=0,time=0`，gfx946/gfx936/gfx938 均 `valid:y`、RC=0，但 kernel 名仍为 `...bf16_4x64_4x1_1x64_1x1_wpr_op_pn_x`，证实小 N 会额外写 X。PMD/gfx936/gfx938 binary SHA256=271735b1e629bacefd20e11b62e3017e81fbf5171a280d1bbc3ebedc5cc995a6/150e0f45184796477fcef616942b1f8bc8ee2b78560445269905646a530e9e1e/ef56531bcee61af89885a9bdeb6e200e74de652c8a34345564e0554a0ab9f6f9，日志 SHA256=3132fdf27ab5ab7fa25995856dfdc5ac28edb155dd709df464a26666789eaf9c/b968ccfcddd355870263b3ea35a0f69c7ae7811057376a8acd9597557ab10345/b968ccfcddd355870263b3ea35a0f69c7ae7811057376a8acd9597557ab10345。
- 这不是 gfx946 指令缺口，三架构行为完全一致。若要把 no-save 扩到小 N/three-pass，需要新增并注册完整 FP16/BF16×INT8/FP8×shape 的 device 实例矩阵，再对 code size、selector 与两架构性能单独治理；只在 example 层拒绝参数不能保护直接 API consumer。本步不扩张为该共享能力项目，记录 `DEFER-SHARED-INSTANCE-CLOSURE` 并保留失败域证据。

本步没有代码、build graph、binary 或 dispatch 变化，不运行 gfx936/gfx938 性能 A/B；gfx946 未测试性能。门禁结果：G0-G4/G6 PASS（行为观察），G5 NOT RUN；能力结论为只有 `4096<N<=8192` 支持真实 no-save。下一切片 P6.54 转向 P1.6 已指出但未校验的 Moe-Sorting `tokens_positions_per_expert` 输出，先审计 reference 与 buffer contract 的最小补强闭包。

### Step 6.54：Moe-Sorting expert token-count / padded-position output — PASS-PMD（2026-09-03）

- kernel 合同确认 `tokens_positions_per_expert[0:E]` 为每个 expert 的原始 token 命中数，`[E:2E]` 为按 `unit_size` 向上 padding 后的起始位置。runner 原来分配并传入该输出，却既不回读也不校验。本步在无 local-expert mask 的范围内补回读与 CPU reference；P6.57 的 sparse case 进一步确认 count 对全部 expert 有定义，而 start 只对非零 expert 有定义，最终校验按该边界执行。初版 source SHA256=deaec0e8be3017f25f44c6067e7dedd42268c6b19832f24baeef61440a3653a6。
- 正式覆盖两条既有执行路径：oneshot `tokens=32,experts=8,topk=2,unit=32,seed=17`，以及 multi-phase `tokens=1024` 同其余参数。gfx946 PMD 两档分别完成 1/3 次 dispatch，既有 sorting outputs、total padded tokens 以及新增 count/start reference 全部通过，均 `valid:y`、RC=0；binary SHA256=8bc2fd931f61ed26fc66bcb1da0a21be4f7aa08cd66d4ed3303ec6f28df3c368，oneshot/multi-phase/build 日志 SHA256=eb60129e8d812020d5b7fdf20bfec81a3ef4de9c0db6ebb4ef7616f5099a627d/97429bd8bc5ef5d1a4709684541df93366fda5f6ea8e9e220d8440d2226b66b1/845735efe39c107cf277b1f8e10aff84a76fe5791806712b4d0ee594452b5f04。PMD 仍只有 P1.6 已记录的 SGPR-before-write/src1 modifier/pending-cacheline 等 warning；未测试性能。
- gfx936/gfx938 两档也全部 `valid:y`、RC=0；candidate binary SHA256=251393f5066125515f16edfcc357c994ed25789255ae9246012efbc032d0c95c/dc73c3b50b3a57a4e92af57a375d64466d867148d164c31cc5dc0d8e545c0974，oneshot/multi-phase 日志 SHA256 均为 c54994576ca06329f49225b9b0d3e214d6e8bc57e9de0be93ab205de889c462a/98755fa70406e2673136afe8d1e1cedd7b85ce3587f7e036b1acfe599bed5c3c。
- gfx936/gfx938 baseline/candidate `.hip_fatbin` 文本 SHA256 分别恒为 0d7fb981cdeebc2ae2cdd1223531923d2a670674dc434141ca6abe012d3ca8e6/096dd0e0633336507dc8f55a1cceafc0cfe66dedbdffea9eec31c6f3e3012807，逐端完全相同；新增逻辑只发生在 timing 后的 host readback/reference。

device payload 与 kernel 计时区间未变，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B；gfx946 只做 correctness。门禁结果：G0-G4/G6 PASS；G5 NOT RUN。P1.6 留下的未校验 API 输出已在无 mask 的 oneshot/multi-phase 两条路径闭合；下一切片 P6.55 独立推导并验证 local-expert masking 下 count/start 的口径，不与本提交混合。

### Step 6.55：Moe-Sorting local-expert count / padded-position contract — PASS-PMD（2026-09-03）

- 源码审计确认：local-expert masking 下，跨 oneshot/multi-phase 都有定义的合同是“启用 expert 的原始 token 命中数”和“仅由此前启用 expert 的 padded size 累加出的起点”。被屏蔽 expert 的两个槽位不应由 runner 固化：oneshot 会计算其 count、multi-phase 会因 mask 提前退出而保持初始化值。因此本步只逐 expert 精确校验 mask=1 的 count/start，不读取或约束 mask=0 槽位；source SHA256=f4acf5ddd57a390625c1329d14f403ec64e71bf01fe25878377ef388fca211db。
- 正式覆盖 local-mask oneshot `tokens=128,experts=8,topk=4,unit=32,local_eid=0,2,4,6,seed=17` 和 multi-phase `tokens=1024,experts=8,topk=2` 同 mask。gfx946 PMD 两档分别得到 total padded tokens 288/1120，新增 enabled-expert count/start 及既有全部输出均 `valid:y`、RC=0；binary SHA256=59a17ac25ae3ffa9e844971e42b9ef04aafb6b8649869fef8b5c092261b5461c，oneshot/multi-phase/build 日志 SHA256=7844843ccc7232bc56e9d735dc670bf4f61854461ef98585bbadca87d8cd9107/c4278a163368b3a52ea24c4e5afe5e40d0dcb44363edfb932e9210908ea4fcd1/5617193a64ceac3ce2bdaa032f395621df8f93169efe5b55b498cbde3fd98cb4。PMD 仍只出现 P1.6 已记录的 pending-cacheline/src1 modifier warning；未测试性能。
- gfx936/gfx938 两档 correctness 同样 total=288/1120、`valid:y`、RC=0；candidate binary SHA256=67d02574692a5df2e883a2bcd494d60228210a64b11bdf18dc2c1cf59e39ea44/e3f1773f0548c68ae6d2e714f118082111747c8bb40eca575bac0740962865b3，oneshot/multi-phase/build 日志 SHA256=a4a932401bf3c3f1ffc7b32ef66ab11ff316c71ee2ff2b6ccfd6844291de2116/2e0eb109d14af0619acbb0bca8f406f42a797e2a285407d9de13b41fd89e7f11/239cd20fd9ad263cb87447abf1b114ece5456e5436a4a707e4220855dd1bbf1f（gfx936），以及相同 correctness 哈希/aaeb081399d4dd3ff060d44850e4f541434519f1a95a4d466f5e72bfc6666242（gfx938 build）。
- 因 host 与 device 共处同一 translation unit，gfx936/gfx938 `.hip_fatbin` 文本 SHA256 分别由 0d7fb981cdeebc2ae2cdd1223531923d2a670674dc434141ca6abe012d3ca8e6/096dd0e0633336507dc8f55a1cceafc0cfe66dedbdffea9eec31c6f3e3012807 变为 304c2cfa3f950423aa0b8a02b405e2157e1a687e1028704ac3395cdad434858a/cae246fb4f313ca18766a4ceb1330a7a2a99a70893be4b8373bc69276426a810，不能仅凭逻辑位置排除性能影响。两机 GPU 均为 0% use/0% memory 后，先做 200-repeat×5 对 A/B，再做 warmup=50、repeat=1000、10 对且交替起始顺序的平衡复测；正式平衡结果 gfx936 baseline/candidate 中位数 17.1945/17.2090 us（+0.08%），gfx938 18.4090/18.3995 us（-0.05%），无可测回归。平衡日志 SHA256=17ed21cabe48dc39e0b36cfea38a739d6a3f96b9e8ab30d5832e8fe37134ee6d/c250874cc4800b94d7f306602163ea344e1b3e060697256b589e47083e9bf910；首轮日志 SHA256=05916937b12452fae27298c58fe428a130a3aa411a86df3bbe73c45152e18db3/cef88021130cff6f750f4f9a8ebfe7785d07561643abd67bbe2243ee99fc1104。

门禁结果：G0-G5 PASS，G6 PASS（真机平衡 A/B）。local-mask 的启用 expert positions 合同已闭合；下一切片 P6.56 使用极小 token 数覆盖 zero-token expert/重复 padded-position 边界，继续复用本 runner 的完整 `2E` 校验。

### Step 6.56：Moe-Sorting zero-token expert / repeated start boundary — PASS-PMD（2026-09-03）

- 固定 `tokens=1,experts=8,topk=1,unit=32,seed=17,time=0`，强制只命中一个 expert，其余 7 个 expert 的 count 为 0；这直接覆盖 `SkipExpertsWithZeroTokens` 及 P6.54 新增 reference 的零计数边界。P6.57 随后依据 multi-phase 源码把 zero-token expert 的 start 明确为未定义槽位，最终 runner 不再约束这些 start。
- gfx946 PMD/gfx936/gfx938 均得到 `total_tokens_post_pad:32(32), valid:y`、RC=0。使用 P6.55 已构建 binary，SHA256 分别为 59a17ac25ae3ffa9e844971e42b9ef04aafb6b8649869fef8b5c092261b5461c/67d02574692a5df2e883a2bcd494d60228210a64b11bdf18dc2c1cf59e39ea44/e3f1773f0548c68ae6d2e714f118082111747c8bb40eca575bac0740962865b3；PMD 日志 SHA256=d06ef7873cd29319059fcae8431a8359a79ceb0b5dcbaa2ad35f27edb7ad4cdc，gfx936/gfx938 日志 SHA256 均为 1230124e76b557a665440ad80ac44cf5e61365a785fdab4720f201b4a0b6fbd9。PMD 只出现既有 pending-cacheline warning，未测试性能。

本步无代码、build graph、binary 或 dispatch 变化，不运行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS（复用已验 binary），G5 NOT RUN；下一切片 P6.57 将 zero-token expert 扩展到 multi-phase sparse-expert 路径。

### Step 6.57：Moe-Sorting multi-phase sparse expert contract correction — PASS-PMD（2026-09-03）

- 首轮在 gfx936/gfx938 运行 `tokens=1024,experts=256,topk=1,unit=32,seed=17,time=0`，主 sorting 输出及 `total_tokens_post_pad:8032(8032)` 一致，但新增 reference 对 5 个 zero-token expert 的 start 期望为累计位置，device 槽位为初始化值 0，双方同样 RC=255；失败日志 SHA256 均为 c63489b116bfbd6604fc41c183b4af10535eb681fe062782dbf817c893227814。源码确认 multi-phase P23 在 `e_start==e_end` 时早于 start 写回直接返回；oneshot 则会写 start，因此 zero-token start 不是跨路径合同，不能作为 kernel 错误。
- runner 修正为：无 mask 时全部 expert 的 count 必须精确匹配、仅 count>0 时校验 start；有 mask 时仅 mask=1 的 count 有定义、且同样仅 count>0 时校验 start。最终 source SHA256=7b61f3b8c3174e18d61b52d74f5413228bdd1a6939a58ccf276cea40333d3b3e。
- 修正后在三架构重跑六档矩阵：P6.54 无 mask oneshot/multi-phase、P6.55 local-mask oneshot/multi-phase、P6.56 单 token sparse oneshot、本步 256-expert sparse multi-phase，全部 `valid:y`、MATRIX_RC=0。gfx946 binary/matrix/build SHA256=8a00344b5a52c403df95084a41b145ac36b18b63a4862c79282bbdd79660afc2/6dcfdb3d98e8db837921578f01965c35bed98a53f6adc3d3f67a713902a02ddd/ee587ebafb13cce9c4c6e22e2cb0696a7377ad480bbf8b154c43f5afcd1d2adb；256-expert case 会大量报告既有 SGPR-before-write 及 src1 modifier PMD warning，但结果正确，记录为模拟器 warning 而非阻塞，未在 PMD 测性能。
- gfx936 candidate/matrix/build SHA256=a71edd8a77d09a89801b92c81cf6c2ba6da4041360111dd9536e29cab54939bf/a2ad9a5a1ccef5a78410cf4b79c75a4a5f738f6c30f3c7d5ba6d22ad7e868b11/208e46ff166fb7ae9d62f38af36673866ec782af9360b9613fe7de40d7de99a4；gfx938 为 a50b2aa4d7121ff5146aabbfdf9bcb6cccc12820e771bcfe4949fa5a9b24af61/相同 matrix hash/6f8a30c1db9899e1fb1b424d0cb899f7481a0004e2c254fc6b85fb51c6a14e89。两端 fatbin 文本 SHA256 从 P6.55 的 304c2cfa3f950423aa0b8a02b405e2157e1a687e1028704ac3395cdad434858a/cae246fb4f313ca18766a4ceb1330a7a2a99a70893be4b8373bc69276426a810 变为 0d7fb981cdeebc2ae2cdd1223531923d2a670674dc434141ca6abe012d3ca8e6/096dd0e0633336507dc8f55a1cceafc0cfe66dedbdffea9eec31c6f3e3012807，因此执行性能门禁。
- 两机均确认 0% GPU use/0% memory，使用 sparse multi-phase shape、warmup=50、repeat=1000、10 对且交替起始顺序。gfx936 baseline/candidate 中位数 16.4585/16.4540 us（-0.03%），gfx938 17.921/17.884 us（-0.21%），无可测性能回归；性能日志 SHA256=3774d958f902acdc856ad7b11bcd1382dc42b811c1447399fb7da2858b888e1d/35f0084d570f7f2a0eff2733e0283c6845d919c2b23dd82581b7f5072d90e9ec。

门禁结果：G0-G5 PASS，G6 PASS（真机平衡 A/B）。`tokens_positions_per_expert` 的可移植合同现已覆盖 mask/no-mask、oneshot/multi-phase、dense/sparse expert；下一切片 P6.58 返回全局 example 清单，选择尚未覆盖的最小独立 dtype/shape 分支。

### Step 6.58：Moe-Quant upper odd-hidden scalar dispatch — PASS-PMD（2026-09-03）

- 选择 `ck_tile/18_moe_quant` 已公开但此前未覆盖的 `hidden_size=8191`：它进入 `<=8192` 最后一档 dispatch，奇数 N 强制 `Vector_N=1`，既覆盖 scalar-output 参数上界，也与 P3.6 已标记 `SKIP-PMD` 的 Vector_N=2/4/8 明确隔离。固定 `tokens=1,prec_o=int8,time=0`，分别运行 FP16/BF16 input。
- gfx946 两档均选择 `moe_quant_{fp16,bf16}_i8_1x8192_1x16_1x64_1x1_bpr_op_pn`，grid/workgroup=`1024/1024`，row scale 与 8191 个 INT8 输出完整 `valid:y`、MATRIX_RC=0；binary/log SHA256=5e87f1435318425a6a01c30560e2e4a19bd0edd8b1a6503f14ac7aa24d0736ba/7af1f841dcffb92344343dfbe6e1c293e7ec5ead3b650e9af008ce905b1a2c3a。PMD 仅报告 P3.6 已有的 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936/gfx938 同两档均 `valid:y`、MATRIX_RC=0；binary SHA256=b6658c58ea33157affadff37285ecf093b6783fee113a7f98b0808b46add5695/20d41af288b41756832c57781485028a97b4da8e9e0f85d62a5ad257df7ea49e，日志 SHA256 均为 1207486c6d19ea07dbabe28dcb2b92645694fa5518384884ade62952cf7072c5。

本步复用已验 binary，没有代码、build graph、device payload 或 dispatch 变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。门禁结果：G0-G4/G6 PASS，G5 NOT RUN；下一切片 P6.59 审计 runner 参数说明中的 FP8 output 与实际 main/API/instance 闭包是否一致，只在完整实例闭包存在时实施。

### Step 6.59：Moe-Quant FP8 scalar-output 最小实例闭包 — PASS-PMD（2026-09-03）

- 静态审计确认 runner 的 `prec_o` 帮助文本原先宣称支持 `int8/fp8`，但 main、API 与显式实例只形成 FP16/BF16→INT8 闭包。为避免把另一算子 `moe_smoothquant` 的 expert/topk 合同错误搬入，本步仅补齐当前 Moe-Quant pipeline 可独立表达的 FP16/BF16→FP8 scalar-output 最小闭包：`hidden_size<=64`、`Vector_N=1`、`Block_N=64`；更大 FP8 shape 明确报不支持，不伪装成已有能力。main/API/new instance source SHA256=eda6560e129ff8a912934052ccefd615125fa46a13c7de839450abf43e100858/ef5d7452a27a06bcd33c689f71ac8608e932137cdc0043c6018d15cbe9621b60/04a977e59cbe8b3450456a0cdec6fd09082fa9918f895e7d18787c7e83411517。
- 首轮 gfx936/gfx938 链接缺少新 instance symbol，原因是现有构建目录的 `file(GLOB instances/*.cpp)` 不会在新增文件后自动重扫；gfx946 PMD 还额外缺少该新文件的精确同步。补齐单文件并对三端执行 `cmake -S . -B <existing-build-dir>` 后，link line 均包含 `moe_quant_fp8_n64_instance.cpp.o`，未修改公共 CMake 或清理构建目录。
- 正式 correctness 矩阵为 tokens=3、FP16/BF16→FP8、hidden_size=63/64，并保留 FP16→INT8 H=64 控制项；三端五档均命中预期 `...fp8_4x64_4x1_1x64_1x1_wpr_op_pn` 或原 INT8 instance，`valid:y`、MATRIX_RC=0。gfx946 PMD binary/matrix/build SHA256=bb54e7e26af9ddceae22493e3b71e14a9d4cfa46406b7740e1f1d57784abba24/3888cc18d46f6667b0c738b5df8ab8bbf80f0014a150ef116458087c598040b4/cad92254fa259504ed896f9499985ed751e929cfd83580e5b5a8cd772dd3c937；只出现既有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，无 FP8 指令终止，且未测试性能。
- gfx936 candidate/fatbin/matrix/build SHA256=71fe0d3ab74166082205f5e858c9a710b2f9e296abb5c9dda623dc9ffe83c83c/a4f278b60715582032e17a0e4d3e119df1b9c3c688212cf00b8e97360b053de3/12438c1b765718922f387850ed2d56d60774ad28ccabcdc875a5f94a4977bcf8/54af6d003459aebdcb5e07336e1ea2b29637a46f0a0ad81989fb8543d20b2435；gfx938 为 dd3c41f3b7ff6a7fc2d5c1017ac59afc962aa73fe027cc3dd964d7ee2faf5ef7/a2567439e9d954a579a106f34b8feb5460c936945c78af4afb07884f34bce087/相同 matrix hash/0ca71d98fd8eb2a6f65bdfe74e280f281f96f7c6d06d86e3e3e426f919d4d74a。
- 新增 FP8 code object 使 gfx936/gfx938 fatbin 分别由 54578edb49af26d081e549e91dfc242c45102f70fec6adb74da70767ea4a3d81/9f78b394932d1e67dd8107c7e4af7fa2671b5b07e541fe712baea85bde48782e 变为上述 candidate hash，因此按要求对原 FP16→INT8 `tokens=128,hidden=8192` 做真机性能 A/B。两机 7 张 HCU 均为 0% use/0% memory；首轮 warmup=50、repeat=1000、10 对平衡中位数为 gfx936 6.865975→6.897030 us（+0.45%）、gfx938 7.056860→7.060275 us（+0.05%）。为收敛 gfx936 边界，继续执行 warmup=50、repeat=2000、20 对平衡复测，gfx936 6.823225→6.825095 us（+0.027%），gfx938 7.000570→7.001060 us（+0.007%）；样本高度重叠，无可测性能回退。首轮/复测日志 SHA256=14e7b3d2b0d5e13b5e997aea34826d9b1d14ae209e9a8cc1f8e76ee84ed37c4c/982bc1d4532fb72c2ddac9d7d2eb025c3407db6f68f9e3860d5773284e7c073d（gfx936）、95eb636e4f7b26d8c420a4302a73a56bf1cfac70e3c77be559b87fd859d211c6/aa6576822225bba82e47d1707d96e15ab9dfc5c52baefd82ce1b610934e13f50（gfx938）。

门禁结果：G0-G6 PASS。Moe-Quant 的公开 FP8 output 不再是虚假帮助项，当前可移植边界明确为 FP16/BF16 input、`hidden_size<=64` 的 scalar-output 实例；下一切片 P6.60 独立审计 FP8 H>64 的实例扩展代价与最小验证边界，不把缺失实例误记成 gfx946 指令问题。

### Step 6.60：Moe-Quant FP8 H≤128 scalar-output 扩展 — PASS-PMD（2026-09-03）

- P3.6 已证明该 pipeline 的 INT8 `Vector_N=2/4/8` 在 PMD 上出现规律性零值，而 `Vector_N=1` 在三架构正确。因而本步不机械复制 INT8 H≤128 的偶数 H vector2 selector，而新增 FP16/BF16→FP8 `Repeat_N=2,ThreadPerBlock_N=64,Vector_N=1` 实例：H=65..128 都用 128-column block 的 scalar store，扩大可验证能力且不重入已知模拟器缺口。main/API/new n128 instance source SHA256=dfbecf91b1230e14bff8e72015b64d842dc364561528e1bed5859e59681ac44f/4c0120837e040f537fb7a879269141d9678043f37f9d8a42195cbf42924948fa/8b2486975eb0877f91544f2276078ca619c41c3c54c0aebfe0d090198f58b201。
- 正式矩阵为 tokens=3、FP16→FP8 H=64/65/128、BF16→FP8 H=65/128，以及 FP16→INT8 H=127 scalar 控制。三端六档均 `valid:y`、MATRIX_RC=0；H=65/128 均命中 `moe_quant_{fp16,bf16}_fp8_4x128_4x1_1x64_1x1_wpr_op_pn`。gfx946 PMD binary/matrix/build SHA256=6a0031bb2175fce2bc3b9a28d3c5732adfc84c737a29baefa3df297404ea2b84/71010cf42ae3032f8ab6fd2f2ba3918b1146e0226246aa9390acf734b3c4ccf4/c735c495fae08c34fff0fa040f75a191b187e9a6db1b7406ec25dc8c5e4e7057；只出现既有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- 首轮特意加入的 INT8 H=128 vector2 对照在 gfx936/gfx938 正确，但 PMD 复现 189/384（49.2188%）输出为零、RC=254，并仅在此失败档出现 `src1_sext/src1_neg/src1_abs should not set`；探索日志 SHA256=506d8abb3dc777a438447f9f905d86e0959eb12df0606f868f77fd7adb9225a4。这与 P3.6 的数量、位置和 warning 完全一致，反向证明新增 FP8 H=128 scalar 路径避开了该 PMD vectorized-output 缺口；不修改真实卡正确的 INT8 selector。
- gfx936 baseline/candidate/fatbin-after/matrix/build SHA256=71fe0d3ab74166082205f5e858c9a710b2f9e296abb5c9dda623dc9ffe83c83c/62c949ca8445bfeb4d6dcfe7dc9cabcd5f998dd3640f0b6935c4e57ed9fae079/b4cdf00707f0a4ddb7ac1dea49dab696dbf4de5385329fb9fb1f9f32224b6308/09a48d67b249fa89f5ed013d77824d195ad26522634e0310e30baf53bd0223a3/913f390de92fae9f9e7992549b1ede31c53e6e2ffd9ec4aa0d06df9a86c20c09；gfx938 为 dd3c41f3b7ff6a7fc2d5c1017ac59afc962aa73fe027cc3dd964d7ee2faf5ef7/987aeb2c419f5650a6dbbe3501066fc97ed43c0328410fee7a05807ed3c515d5/5d6dd8eb1c450bc9a579af812d0d29a563ba8343955e226a4e9777d8403b5806/相同 matrix hash/c7a1e81aebed97d63f523d9851805812c39a88022e02fce921866a05ab279480。
- fatbin 新增 n128 code object，不能直接排除既有路径性能影响。两机 7 张 HCU 均为 0% use/0% memory，固定原 FP16→INT8 `tokens=128,hidden=8192`、warmup=50、repeat=2000、20 对平衡 A/B。首轮 gfx936 6.836580→6.831420 us（-0.08%）、gfx938 6.997045→7.016690 us（+0.28%）；因 gfx938 有单个 7.27166 us 离群点又完整复测一轮，得到 gfx936 6.836775→6.834390 us（-0.03%）、gfx938 7.005790→6.998015 us（-0.11%）。变化方向跨轮反转、样本主体重叠，判定无可测性能回退；首轮/复测日志 SHA256=bf70c4720cbc1dd99a383f7c2346f2a97ce42e03b2e64da07d3a16b4ac35bd4b/6684b4e5c6b86c5b2b62613300f4de69e0f0826069270d3eadfc7312476bd035（gfx936）、35ce6529deeeeba73cd537cf94dee741655c41728ced5773aa161d3b5e3aec43/7fec58be72ea943cf90d2e935becaa807ec7ca51327aae13dc9302148157d24c（gfx938）。

门禁结果：G0-G6 PASS。Moe-Quant 的 FP8 scalar-output 可移植边界扩大到 FP16/BF16 input、`hidden_size<=128`；P3.6 的 vectorized-output `SKIP-PMD` 结论保持不变。下一切片 P6.61 审计 H≤256 的 scalar fallback 是否能以独立实例继续扩展，并控制 code-size/真机性能门禁。

### Step 6.61：Moe-Quant FP8 H≤256 scalar-output 扩展 — PASS-PMD（2026-09-03）

- 沿用 P6.60 的 portable scalar 策略，新增 FP16/BF16→FP8 `Repeat_N=4,ThreadPerBlock_N=64,Vector_N=1` 显式实例，并把 FP8 dispatch/help 上界从 128 扩到 256；不注册已知在 PMD 失败的 vector2/vector4 路径。main/API/new n256 instance source SHA256=52fc85744c3006212272ed46b6ee2bf136c80c055eac0c852554d83ee01ea40a/60df97a8ff8d10109dd293d356fd20d651f653357ca53f18f2ad05f21dd46794/a3f80e4f8230cb6d5d7814a21b4269904bd8413f3aa05682b2e3c4f3213483bb。
- 正式矩阵为 tokens=3、FP16→FP8 H=128/129/255/256、BF16→FP8 H=129/256，以及 FP16→INT8 H=255 scalar 控制。三端七档均命中预期实例并 `valid:y`、MATRIX_RC=0；H=129..256 的 kernel 名为 `moe_quant_{fp16,bf16}_fp8_4x256_4x1_1x64_1x1_wpr_op_pn`。gfx946 PMD binary/matrix/build SHA256=7bcb2c303461977a078e888392640546e75d0a553087c17f609ae1eb5a5bd29f/35d48cab08a4e83bbcf9243aa05eb6f757fee2cd81fe9772270aa5d8a0af1751/89c372042e9fd56cdeec6c25b1048175498e14e31b1327763b6fe4d35eb162e8；仍只有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936 baseline/candidate/fatbin-after/matrix/build SHA256=62c949ca8445bfeb4d6dcfe7dc9cabcd5f998dd3640f0b6935c4e57ed9fae079/e1d73abae35a4343540a84e161dae6e15334e0eca1f150f5c82b48ff5d9abf1e/b3a3ffc57859ec5c8303441830b4fd909b791dff5cce3504c8d6f5cccbad848c/dbab70d5bbdbbd01dd6b77721e0c91339a0e057c9a630044a3edb971d2e5f0f8/c7671b4d35acbcef77e753841b99f1e4ba4e227854d139292266fc2bde3637c4；gfx938 为 987aeb2c419f5650a6dbbe3501066fc97ed43c0328410fee7a05807ed3c515d5/d0c109b729942146a9c2955f6bcf63b590dd3548f1aede7c8d3c6fa943b54a5c/6a7167ed0194de97c79aae758b8b5e8584699ba77734dde747dfb5e8ef5b8a14/相同 matrix hash/b2fec2227036b032315f0ef9efa44013c0bbdd3e77f01cdeb734ce3f7d3c4f34。
- fatbin 变化后，两机 7 张 HCU 均确认 0% use/0% memory；固定原 FP16→INT8 `tokens=128,hidden=8192`、warmup=50、repeat=2000、20 对平衡 A/B。gfx936 中位数 6.835285→6.832625 us（-0.04%），日志 SHA256=ca86c8806982775ba39beb7e8c4e3179b447f6d8073395f50d67f3cb8bd93771。gfx938 首轮 6.995970→7.015240 us（+0.28%），确认轮 7.009810→7.010015 us（+0.003%）；首轮偏移未复现，日志 SHA256=a6f0a739d1e8b4d5e9e9c7abbee4d8d53eb00bd9d80672392dc8d80ba0bcfada/7ac7f1a8e7c2a6ea0db433ad9ca7f45914b71b67316fe19f5847bcef2ee79a0c，无可测性能回退。

门禁结果：G0-G6 PASS。Moe-Quant FP8 scalar-output 可移植边界扩大到 `hidden_size<=256`；下一切片 P6.62 评估 H≤512 的 `Repeat_N=8,Vector_N=1` 实例，继续将 PMD correctness 与真机性能影响分开验收。

### Step 6.62：Moe-Quant FP8 H≤512 scalar-output 扩展 — PASS-PMD（2026-09-03）

- 新增 FP16/BF16→FP8 `Repeat_N=8,ThreadPerBlock_N=64,Vector_N=1` 显式实例，并把 FP8 dispatch/help 上界从 256 扩到 512；继续只注册 PMD 可验证的 scalar store。main/API/new n512 instance source SHA256=e2356a03c72dbef64b337aacb092f74ca7b4dc41a9b8b4011120b1ca8bcc896a/de8e55fa58648be92ac6e3a11be5a441336eef85aa77c2c18e7f92412f82fd67/01c159f01cf30cf791134cf7ef04808e001d92dc78962a0eaae7518b162ff9d1。
- 正式矩阵为 tokens=3、FP16→FP8 H=256/257/511/512、BF16→FP8 H=257/512，以及 FP16→INT8 H=511 scalar 控制。三端七档均命中预期实例并 `valid:y`、MATRIX_RC=0；H=257..512 的 kernel 名为 `moe_quant_{fp16,bf16}_fp8_4x512_4x1_1x64_1x1_wpr_op_pn`。gfx946 PMD binary/matrix/build SHA256=c4a54eef8918eaa748e0c1fbd35a405dbdcd14d04e9fb58434bfb38be7e8bc33/b65065ad5e42fc19cdcc720137eb6244a4dbbd805192b6608d0eade1995a9468/69bca9b64e1b0cd8936e1ae95eb6f774e805b7834ea71e92410e8de2c0ce22af；仍只有既有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936 baseline/candidate/fatbin-after/matrix/build SHA256=e1d73abae35a4343540a84e161dae6e15334e0eca1f150f5c82b48ff5d9abf1e/941f2c3ad216ea8dd85d1b4181728555151c1d1a8981283931028e8d747c5bd1/0a4080cf98149997d481d83f9570639cf6beb284989cb286727113259b9e6823/7de809e6ce3e46c35bb095d27b3b441b1d9ed3c795bf9895101a81377cd8140b/105a57dcc34f3a622da86093d6de06947231ac825db688052d41bc5675a59398；gfx938 为 d0c109b729942146a9c2955f6bcf63b590dd3548f1aede7c8d3c6fa943b54a5c/304d7369475ea2dfbe8f3f159349366183334800231036c0efa87bd8d7f103cf/a2301f72366935afb664878a385ca7c31ca70f524c6a52cbdce41b1fb07866bd/相同 matrix hash/39f95aab4a478ceab056ebbed4671966a8180037dd7756a3f690161f3b399116。
- fatbin 变化后，两机 7 张 HCU 均为 0% use/0% memory；固定原 FP16→INT8 `tokens=128,hidden=8192`、warmup=50、repeat=2000、20 对平衡 A/B。gfx936 中位数 6.838950→6.840805 us（+0.027%），gfx938 7.007845→7.002625 us（-0.074%），无可测性能回退；日志 SHA256=4bd8a6149bb3de60d962a9b0a18eeeca763a189ed042769dba5cca980bfbe60a/2fd26de7110b947f8aff4459ec26cd35a189b8d31c61ac0a9076dc85174101d3。

门禁结果：G0-G6 PASS。Moe-Quant FP8 scalar-output 可移植边界扩大到 `hidden_size<=512`；下一切片 P6.63 审计 512..1024 的线程拓扑切换，先以现有 INT8 odd-H scalar trait 推导最小 FP8 实例，再做三架构门禁。

### Step 6.63：Moe-Quant FP8 H≤1024 scalar fallback — PASS-PMD（2026-09-03）

- 对 512<H≤1024 复用现有 INT8 odd-H 健康拓扑 `Repeat_N=4,ThreadPerBlock_N=256,Vector_N=1`，以单个 FP16/BF16→FP8 实例覆盖整个区间。相较按 768 再分档，这一 correctness fallback 对 513..768 有额外 padding，但避免新增 vectorized PMD 风险，也为后续真实性能 selector 保留独立优化空间。main/API/new n1024 instance source SHA256=96e40dd018646cd884d1a86f69aa5e0586839a741c0192a34d5eb34e55343a7b/4a1e0d67138e04aecc5dd34d1ddc3fb536607f2bc0693ae46c32cbedb97db180/6155cda5018558df05f66a1a9682713c457a0dbdd421e82282e385c49b18c600。
- 正式矩阵为 tokens=3、FP16→FP8 H=512/513/768/769/1024、BF16→FP8 H=513/1024，以及 FP16→INT8 H=1023 scalar 控制。三端八档均命中预期实例并 `valid:y`、MATRIX_RC=0；新路径 kernel 名为 `moe_quant_{fp16,bf16}_fp8_1x1024_1x4_1x64_1x1_bpr_op_pn`。gfx946 PMD 的 H=1024、tokens=3 实际 grid=(256,3,1)、wg=(256,1,1)；binary/matrix/build SHA256=49ba01bbb16a3ece2efb6697fd879b383457b71f1c442d887fbfe088af667d23/a82e33b5d6dd86e0ad5d029f08e2bac30bbf8bfb23e0cea172b745dfc88d1bf5/e1e1dcc3602f373f11dff5d302702ea9dd6cdaa577e27d7663ccc023a38e7f9a。仍只有既有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936 baseline/candidate/fatbin-after/matrix/build SHA256=941f2c3ad216ea8dd85d1b4181728555151c1d1a8981283931028e8d747c5bd1/d1239629ee4ad452dd1debd41a49ecce78da2c8080dd9c8f7efc570dc1499e80/964533eb39c935aae335aa4acb71d835cce5135ba55cbde7e8e0e4d546413b6c/0cb429bff8fe4721cf3dd4bca9d343379b063fca305ee908fe2aab252d263526/e472bad03d397b4e04a132a4496cfcc2bba9d93ec1a592af62d9608daa5ac7f1；gfx938 为 304d7369475ea2dfbe8f3f159349366183334800231036c0efa87bd8d7f103cf/62f031156b2688e3c80b26be0271a319ae675c8bf594624d807188c90dbf64ea/f85699022059885e9d6b9b55e0f3c7c54b7876b8cc6afb1a4042b64640dc1e75/相同 matrix hash/3d0d14fc766119cc00bf869278ed63f2988af084235bb02a7c123283e1ee253a。
- fatbin 变化后，两机 7 张 HCU 均为 0% use/0% memory；固定原 FP16→INT8 `tokens=128,hidden=8192`、warmup=50、repeat=2000、20 对平衡 A/B。gfx936 中位数 6.882720→6.879905 us（-0.04%），gfx938 7.022380→7.004205 us（-0.26%），无性能回退；日志 SHA256=aa6f22c73bcd59358b34862a86f10650b83ef2bbbf87259d367e3fa9bda85580/c4dc3b7e32ca56fbf74109071ca8a03a1b2c089c30c5a75497dd194bf312f8f7。

门禁结果：G0-G6 PASS。Moe-Quant FP8 scalar-output 可移植边界扩大到 `hidden_size<=1024`；下一切片 P6.64 评估 1024..2048 的 `Repeat_N=8,ThreadPerBlock_N=256,Vector_N=1` scalar fallback，并继续限制单步实例数量。

### Step 6.64：Moe-Quant FP8 H≤8192 scalar fallback 完整闭包 — PASS-PMD（2026-09-03）

- 剩余 shape 仍可由已在 INT8 odd-H 路径验证过的 scalar 拓扑表达，因此本步一次补齐三个 FP16/BF16→FP8 实例族：H≤2048 使用 `1x8x1x256,Vector_N=1`，H≤4096 使用 `1x4x1x1024,Vector_N=1`，H≤8192 使用 `1x8x1x1024,Vector_N=1`；FP8 help/API 上界与既有输入上界统一为 8192。main/API/n2048/n4096/n8192 source SHA256=72f2685a656b4591a991c459bba319c494c798ad915afe3e9684c4d3a9ece2bf/63adc1efb47fb56bdc608bc8b0381821a15dfe82068e915b51154b4d58f54909/cb68ac2e2475b0bb0624fed8104175f661ec5f039c3e54768e6c4129d8617a69/3dd49bbffa1b8b0358cecc507bfa707d39eafe0d6d78e51c435f43a7e3fda1df/699aa91186343900dadee0af6f951a7abc455177c06b9b5f5e3cf50cfdaa6258。
- 正式矩阵固定 tokens=1，覆盖 FP16→FP8 H=1024/1025/2048/2049/4096/4097/8191/8192，BF16→FP8 H=1025/2048/2049/4096/4097/8192，以及 FP16→INT8 H=8191 scalar 控制，共 15 档。三架构全部命中预期 scalar instance 并 `valid:y`、MATRIX_RC=0；H=8192 的 PMD 实际 grid=(1024,1,1)、wg=(1024,1,1)，完整 FP8 输出正确。gfx946 binary/matrix/build SHA256=2828881026b7204657c0eb9372323e4e1f03e1204c48983c9d54cf65fe733b9a/f9632d7acb3108457dc853ce6631896acee8414912800e51d6a74d09a742aab4/ebedb2dca9d122f60625a5786a55d2f20a06f872d01df30f0048df28b7dae23f；仍只有既有 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。
- gfx936 baseline/candidate/fatbin-after/matrix/build SHA256=d1239629ee4ad452dd1debd41a49ecce78da2c8080dd9c8f7efc570dc1499e80/88dff58ed90808941e13d56c3b94a48598edb6d189c0ccdab94ce1b8200b365f/3e512fcdd5274f7b0ce5261be90b0af0ba0d29d29296965af176c6ca60f302d4/e8614ac447029109854e1ca93348a96e60323f66104be09d622245d2fe715f34/4e18dfd4af995d1060bfb431a614e9852d18781d51d786d7ae864458de280ce4；gfx938 为 62f031156b2688e3c80b26be0271a319ae675c8bf594624d807188c90dbf64ea/9346b54cac2307a5de6b990bcff3d37c38395bcfb1b3308a96a8b48f7cd2c75c/d11b2a4698c5bb2feedb6be7f5f2925daa6ec109aaed07380b6ab0cd5ef8f356/相同 matrix hash/7b6b4ddbb55f34c6da3bcaa334163366b93034407be9b33697208d3945c15333。
- fatbin 变化后，两机 7 张 HCU 均为 0% use/0% memory；固定原 FP16→INT8 `tokens=128,hidden=8192`、warmup=50、repeat=2000、20 对平衡 A/B。gfx936 中位数 6.837360→6.838100 us（+0.011%），gfx938 6.998475→7.006140 us（+0.110%），样本主体重叠、无可测性能回退；日志 SHA256=97f824522fb8624c8d9217ac44bf68c62c6e0f94c46765303aa27b43fe96f7ac/7f38c65400bba29cac8bf30f7609076e405a99fd5d90afb10ead6ac6548a45a7。

门禁结果：G0-G6 PASS。`18_moe_quant` 现已为 FP16/BF16 input × INT8/FP8 output 提供共同的 `hidden_size<=8192` 功能边界；gfx946 使用 scalar FP8 fallback，P3.6 的 vectorized-output `SKIP-PMD` 仅限制性能优化路径，不再阻塞 FP8 功能。下一切片 P6.65 验证 H>8192 的 host rejection 是否在分配/dispatch 前给出稳定错误，然后返回 enabled-example 清单选择下一独立模块。

### Step 6.65：Moe-Quant H>8192 前置拒绝与模块收口 — PASS-PMD（2026-09-03）

- 源码审计确认原流程会先构造 4 个 HostTensor、3 个 DeviceMem 并上传 input，之后才由 API dispatch 抛出 H>8192 异常；main 也没有 catch，因此拒绝会以 abort 而非稳定接口返回码呈现。本步在参数 parse 后、dtype dispatch 前增加统一上界检查，错误固定为 `[ERROR] hidden_size exceeds 8192`、进程返回 `-4`（shell RC=252），main source SHA256=f8eec2bfacbd6303e6e5d0372801146780e60682e5a9db48ee2aea002b8049b1。
- 正式矩阵覆盖 FP16/BF16 × INT8/FP8 的 H=8193 拒绝，以及 FP16→INT8 H=8191、FP16→FP8 H=8192 正确性控制。三架构均为 6/6 预期返回，H=8193 的 PMD 日志在错误行后直接结束，未出现 PMD 初始化或 kernel dispatch，证明拒绝发生在 host/device 分配与执行前；PMD/gfx936/gfx938 matrix SHA256=a6bd925f0863241fa3c33a8078bf4f4341b7588c535ab7e79d78e1d98eabf824/2915f71f1dee7fd423bb693610c4201c2d8f140f2a7a6569c74f0e47af09d0ab/2915f71f1dee7fd423bb693610c4201c2d8f140f2a7a6569c74f0e47af09d0ab，build log SHA256=343003a1c2665e400a13cef5a835577b0a783f72183fd46a45bfaeb62cc7683a/f40833aafe8ad4db8b35ed2075f3bba77eddc79d319f50ce17f8df25199f50cf/e4f2cf0f3b988fa088cb7d6d888769c87aec3dcf0827751041515233976319b6。
- 三架构 `.hip_fatbin` 编译前后完全不变：gfx946=e609535d56925227e570890356aa34f870b894f6a3f2478c71687c7b3922371e，gfx936=3e512fcdd5274f7b0ce5261be90b0af0ba0d29d29296965af176c6ca60f302d4，gfx938=d11b2a4698c5bb2feedb6be7f5f2925daa6ec109aaed07380b6ab0cd5ef8f356；候选 binary SHA256=a9de7a0de99058fd89650700149711370ec82ae93e802d17df72553f6e3200cb/8d097ee73a31e41a012b20807878717d31c5e519a7de8287189502a552a82bec/ba05f5c53de0dc939f6b7a237250cc1c59bc6a1d14d9d87570b63031526e1283。设备 payload 未变，可确定 gfx936/gfx938 kernel 性能不受影响，故本步不另做性能测试。
- 首轮 PMD 控制曾使用 FP16→INT8 H=8192，复现 P3.6 已归档的 `Vector_N=8` 模拟器写出缺陷（24.6704% wrong values，失败日志 SHA256=705a1783111e20b5c8aca724e8b3d48b433dbe8bb8198cc4bba38500bccf6e93）；同一二进制在 gfx936/gfx938 均 `valid:y`。该失败证据保留但不重复归因或改写 kernel，正式 PMD 控制改用 H=8191 scalar 路径并通过。

门禁结果：G0-G6 PASS。`18_moe_quant` 的可执行输入边界、FP8 gfx946 scalar fallback、三架构正确性与实机性能风险均已闭环，本 example 完成；下一切片返回 enabled-example 清单，按“最小独立编译目标、PMD 可观测正确性、低共享代码风险”选择下一模块。

### Step 6.66：LayerNorm2D BF16 no-save 与 `save_mv` 合同审计 — PASS-PMD / DEFER-SAVE-MV（2026-09-03）

- P1.3 只运行 FP16 no-save；本步选择相同 M=2,N=128、packed stride、无 bias/add/quant 的 BF16 分支，并以 FP16 为控制。gfx946/gfx936/gfx938 两种 dtype 均命中对应 `...4x128_4x1_1x128_1x8_wpr_pn`、`timing:off,valid:y`、RC=0；PMD baseline/run/build SHA256=89656a4aa85821811e7c2fc3fa557db560db94f2a9c8c110956724690dbdd639/57b0f29e05be3ed97ec9bf688f3aaa935c17f78eda05fdf780fb958fa8ea149b/f2d1da25a91a0ca77602a1ca6f5a7fce2cb860775a061ee2f26a0d817afa3d08，gfx936/gfx938 binary SHA256=18f3b54e059cc758f13486251c529c1f63c89df18624731fdba431d6ce63bfd2/00b374d155932b55209bd50ef341e72bf5edc60500ec837cc9166b426ab1774d，no-save 日志 SHA256 均为 f97af68237db7ca1536f52c6cfae6c6083ec4ef462880859ec65914843e53ccf。
- 静态追踪发现 main 的 BF16 `save_mv=0` 分支写成 `run<...,true>`，但它在当前闭包中是无行为影响的死参数差异：生成器所有 instance 均固定 `kSaveMeanInvStd=false`，生成 API 的 selector 条件也完全不比较 `t.save_mean_var`。generate/runner/header SHA256=27a8830dafd5dd8a944927e903086c1b71a84f410146098b0a2440f28c8ab887/d05b37ff9f95fcdc30c09f9e9fb3cd51d8ea50faa431e2f4d232ad5f735bf43b/3d3900defd1c5fe6f9c883c9ee711ba689043796b512e61038c5dac602707cd9。
- 反向用 `save_mv=1` 运行 FP16/BF16 后，三架构仍选择名称中无 `_mv` 的同一 no-save kernel 并仅校验 Y 为 `valid:y`；PMD/gfx936/gfx938 日志 SHA256=78ff753d0d376551f0e2c7811c6882d49854fd255e4c1bbeed1ce250effcb676/a6504d3fc0eefff83e373f9f9044aca8b49be2ebcb3245ed536e1c56162e8542/a6504d3fc0eefff83e373f9f9044aca8b49be2ebcb3245ed536e1c56162e8542。runner 又把 `p_mean/p_invStd` 固定传 `nullptr`，且不读取/校验 device mean/invStd，因此当前 `save_mv=1` 只是虚假训练合同。
- 单独把 BF16 main 的模板布尔值改为 false 不会改变 selector、kernel、输出或性能，却会掩盖真正缺失的 generator/API/device-buffer/reference-validation 闭包，故不保留机械修改。实现 save-MV 需要作为共享 LayerNorm generator/API 项目单独立项，本阶段标记 `DEFER-SAVE-MV`；本步没有代码/device payload/dispatch 变化，不执行 gfx936/gfx938 性能 A/B，gfx946 未测试性能。

门禁结果：BF16/FP16 no-save G0-G4/G6 PASS，G5 NOT RUN；`save_mv=1` 为 DEFER-SHARED-GENERATOR-CLOSURE。下一切片 P6.67 验收 `10_rmsnorm2d` 完整 instance runner 的 BF16 no-save 分支，区别于 P1.4 只覆盖的简单 FP16 example。

### Step 6.67：RMSNorm2D full-instance BF16 no-save correctness-only 闭包 — PASS-PMD（2026-09-03）

- P1.4 的 `tile_example_rmsnorm2d_fwd` 只含简单 FP16 kernel；完整生成式 target `tile_rmsnorm2d_fwd` 原先强制 `stream_config.time_kernel=true`，无法在 PMD 上执行单发射 correctness。本步为 runner 增加默认值保持为 1 的 `-time`，`time=0` 时固定 warmup/repeat=0/1、跳过吞吐除法并输出 `timing:off`；source SHA256=9e5a930c70008f86a6a263bdebb4e0f2cbee28715f3945aa9f2ca7ee99598986。
- 正式矩阵为 BF16 no-save M=2,N=128、M=4,N=1024，并以 FP16 M=2,N=128 控制，全部 packed stride、无 fused-add/quant。三架构均 3/3 `valid:y`、MATRIX_RC=0；PMD 分别命中 BF16 WPR 64-thread、BF16 WPR 128-thread 与 FP16 WPR 64-thread 单次 dispatch，仅有既存 `V_DIV_FIXUP_F32 not test`、src1 modifier 和 pending-cacheline warning，未测试性能。PMD matrix/build SHA256=0d977e68fe3ecd8eef58f3ba7bb9cfefbac6293fd5975ae86d96c1644f0f5a51/480eee1d69a9c25a0af561f4850487c830f832bd4e5872d5545f176d3aaeea53，gfx936/gfx938 matrix SHA256 均为 f53dfa3be4772064447fd3c7491c6bde8c3060858d83380fb8e03255494b32f8，build SHA256=196ea11477c276f2b183dbff2d2af32744853b063b5d013e1834312938a5c2fb/708246b430f218f760e6de43d730957cd0e7d3b824c4c356e72b2749a65fa5a7。
- baseline/candidate binary SHA256：gfx946=7cdfcd57df193155f407f86aa61a1de5bfe73ab132b5ac09c44db20611fd9f08/7c0a599b5f94b5f1d2fac36b2b76a6b52fd9556a1f9afa150da1283417f65f05，gfx936=cd280556ab0278328007c6741d34b369a43923365c373421a69b92324f51e690/641ae6e55d16a0d04f9f1c1cbe2e864d628758ee938a2393ff23bb275c45ada7，gfx938=09654930d1183ac84f04ed1ba5405ae03f00eb15814d82f28bedebbad7853f8b/9a5c30d0e5f97cb1c33fcd66362fc7886e6adccd9b2e59e39802015de78653b9。对应 `.hip_fatbin` 也变化为 2178e8732c057459dcf9de872e86f393153e982c8f9f31ff58798ec0382940a1→781e16400090687526dd04bf97f3277de08d286a526afb2472b9ffde540d8c4c、7927aefd5a0e4b7d236bbfa13abe773b60e81a0af2cbbd0d6747830c9653e4b4→8f59dc0364bd78dcaa23410cb82aac025bcc1d1a23ab31166011d5abb4089eda、4c81a109e841a6125e87b0b35d5c1a27f04e9e454ef00f8d1f40f8e6536f239f→c39affdd9ac1009e02160080d71a79ff8093e6f69afe5273fbaf00a3d4eaf994，因此没有仅凭 host diff 排除性能风险。
- 两台真机测试前 7 张 HCU 均为 0% use/0% memory；固定 BF16 no-save M=N=4096、warmup=50、repeat=2000、20 对交替顺序 A/B。gfx936 中位数 65.615850→65.571650 us（-0.067%），gfx938 68.305150→68.304500 us（-0.001%），无性能回退；日志 SHA256=62c62873903d6d2379c512882dd3d69aa021c93d96a025c1835b78a880600be3/5f25a62242bc1c983a0567309a7f9779192023c7bfb0310f6cf27fe9729b7c05。

门禁结果：G0-G6 PASS。完整 RMSNorm2D runner 现在可在 gfx946 PMD 做 correctness-only 单发射，BF16 no-save 的小/大 N WPR 分支均完成三架构验收；下一切片 P6.68 独立审计该 runner 的 `save_rms=1` 是否与 P6.66 LayerNorm 相同地暴露虚假训练输出合同。

### Step 6.68：RMSNorm2D `save_rms=1` 训练输出合同审计 — DEFER-SAVE-RMS（2026-09-03）

- generator/runner/header SHA256=99c75acccc1734755785d2554fa226dc07ad811341e6c8a8f08b0066b6465714/9e5a930c70008f86a6a263bdebb4e0f2cbee28715f3945aa9f2ca7ee99598986/bef78e3e2e38a0116d64c27d815766adfe8191a843a7c0fbeb853d2251015152。静态追踪确认所有生成 instance 固定 `kSaveInvRms=false`，API selector 不比较 `t.save_rms`，runner 又把 `p_invRms` 固定传 `nullptr` 且没有 device invRms 的回读/校验，因此公开的 `save_rms=1` 不可能产生训练所需 invRms 输出。
- 以 FP16/BF16 M=2,N=128、`save_rms=1,time=0` 在三架构探测，均选择名称中没有 `_rms` 的同一 no-save kernel，Y 输出 `valid:y`、RC=0；这只能证明 RMSNorm Y 正确，不能把静默忽略 invRms 误报为训练合同通过。PMD/gfx936/gfx938 日志 SHA256=2f83a6b1614c6c129b58fe37f5b45eabde12be15bc0ef8786ddd94ebce47895b/e517f40b2364ecd5afb24568a4e45d005a143f3253b05982669c5a5c1f1bdb1b/e517f40b2364ecd5afb24568a4e45d005a143f3253b05982669c5a5c1f1bdb1b。
- 该缺口发生在共享 generator/API/buffer/reference-validation 闭包，早于 gfx946 指令执行；不通过只改 main 布尔值制造虚假修复，标记 `DEFER-SAVE-RMS`。本步复用 P6.67 已验 binary，没有代码、build graph、device payload 或 dispatch 变化，不执行性能 A/B，PMD 未测试性能。

门禁结果：no-save 维持 PASS-PMD；`save_rms=1` 为 DEFER-SHARED-GENERATOR-CLOSURE。下一切片 P6.69 使用 N=63 的 scalar-output instance 验收完整 runner 的 BF16/FP16→INT8 dynamic-quant，隔离已知 PMD vectorized INT8 store 缺口。

### Step 6.69：RMSNorm2D FP16/BF16→INT8 scalar dynamic-quant — PASS-PMD（2026-09-03）

- 复用 P6.67 已提交 binary，固定 M=2,N=stride=63、`fquant=2,save_rms=0,save_unquant=0,time=0`；N=63 使 selector 跳过 Vector_N=8/4/2，分别命中 FP16/BF16 `...4x64_4x1_1x64_1x1_wpr_op_dqt_pn`，从而隔离验证 RMSNorm、row amax/FP32 scale、INT8 conversion 与 scalar store。
- gfx946 PMD 两种 input dtype 均单次 grid=(256,1,1)、wg=(256,1,1)，Y/scale reference `valid:y`、MATRIX_RC=0；只有既存 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。binary/log SHA256=7c0a599b5f94b5f1d2fac36b2b76a6b52fd9556a1f9afa150da1283417f65f05/828c947d59972264da3910a3aa7d0cab9dfb2c62feaa0a5db5699a53b4cb5eaf。
- gfx936/gfx938 同矩阵均 2/2 `valid:y`、MATRIX_RC=0；binary SHA256=641ae6e55d16a0d04f9f1c1cbe2e864d628758ee938a2393ff23bb275c45ada7/9a5c30d0e5f97cb1c33fcd66362fc7886e6adccd9b2e59e39802015de78653b9，日志 SHA256 均为 e012d67cdad78d4347ac9ddfa711e25bab6c3f196a50d9c46294c59c28f7f7b1。
- 本步没有代码、build graph、binary 或 dispatch 变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B；结论只覆盖 scalar INT8 store，不外推到已知 PMD vectorized INT8 路径。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。下一切片 P6.70 在相同 N=63 scalar 边界将 `fquant=2` 切换为 `fquant=1`，独立验收 smooth-dynamic-quant 的逐列 smooth scale 路径。

### Step 6.70：RMSNorm2D FP16/BF16→INT8 scalar smooth-dynamic-quant — PASS-PMD（2026-09-03）

- 完全复用 P6.69 的 M=2,N=stride=63、no-save、time=0 合同，唯一变量为 `fquant=2→1`；selector 分别命中 FP16/BF16 `...sxfp32_syfp32_4x64_4x1_1x64_1x1_wpr_op_smdqt_pn`，新增覆盖逐列 FP32 smooth-scale load/multiply，再做 row amax/scale 与 scalar INT8 store。
- gfx946 PMD 两种 input dtype 均单次 grid=(256,1,1)、wg=(256,1,1)，Y/scale reference `valid:y`、MATRIX_RC=0；只有既存 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。binary/log SHA256=7c0a599b5f94b5f1d2fac36b2b76a6b52fd9556a1f9afa150da1283417f65f05/d883de4b41cbbe96d09f60e93e75f0f6aa4b20c9a302ae3b1eb5158b1d99943d。
- gfx936/gfx938 同矩阵均 2/2 `valid:y`、MATRIX_RC=0；binary SHA256=641ae6e55d16a0d04f9f1c1cbe2e864d628758ee938a2393ff23bb275c45ada7/9a5c30d0e5f97cb1c33fcd66362fc7886e6adccd9b2e59e39802015de78653b9，日志 SHA256 均为 7ba9de2a5a9246938d42a5c9594aeda923d690f8a0e969eb9adf4a10c598f7b5。
- 本步没有代码、build graph、binary 或 dispatch 变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B；结论仍限定于 Vector_N=1 scalar INT8 store。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。下一切片 P6.71 在 dynamic/smooth-dynamic 两种模式下开启 `save_unquant=1`，验证量化 Y、scale 与量化前 RMSNorm 输出的三输出一致性。

### Step 6.71：RMSNorm2D INT8 quant `save_unquant=1` 三输出合同 — PASS-PMD（2026-09-03）

- 固定 M=2,N=stride=63、`save_rms=0,time=0`，覆盖 FP16/BF16 × dynamic(`fquant=2`)/smooth-dynamic(`fquant=1`) 四种组合，只把 P6.69/P6.70 的 `save_unquant` 从 0 改为 1。runner 的 `if constexpr(SaveUnquant)` 会同时回读并校验量化前 Y，另保留 INT8 Y 与 FP32 row-scale 校验。
- gfx946 PMD 四档均单次 grid=(256,1,1)、wg=(256,1,1)，三类输出全部 `valid:y`、MATRIX_RC=0；除既存 `V_DIV_FIXUP_F32 not test`/pending-cacheline 外，FP16 save-unquant 路径还报告 `V_MAD_MIXLO_F16 not test`，但 FP16/BF16 的 unquant Y 均与 reference 一致，按非阻塞模拟器告警记录，未测试性能。binary/log SHA256=7c0a599b5f94b5f1d2fac36b2b76a6b52fd9556a1f9afa150da1283417f65f05/126d5608bd1167cf60a40316de352b966a6359e6ce295b30cbc2621171d35c24。
- gfx936/gfx938 同矩阵均 4/4 `valid:y`、MATRIX_RC=0；binary SHA256=641ae6e55d16a0d04f9f1c1cbe2e864d628758ee938a2393ff23bb275c45ada7/9a5c30d0e5f97cb1c33fcd66362fc7886e6adccd9b2e59e39802015de78653b9，日志 SHA256 均为 16c20dc707155662887137d12c489a4acf31bbcf3b11c0f92919f5b80e03380c。
- 本步没有代码、build graph、binary 或 dispatch 变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B；PMD 的未实现指令告警已有正确输出证据，不构成跳过条件。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。下一切片 P6.72 验收 BF16 non-quant 的 `fadd=1/2` 两种 fused residual 模式，分别覆盖 pre-add-store 与 pre-add-only。

### Step 6.72：RMSNorm2D fused residual 合同与 unsupported-path 修正 — PASS-PMD / SKIP-FADD2（2026-09-03）

- 固定 M=2,N=stride=128、no-save/no-quant、`time=0`，覆盖 FP16/BF16 × `fadd=1/2`。首轮三架构完全一致：`fadd=1` 命中 `..._pras_pn`（pre-add + residual store）且 `valid:y`；`fadd=2` 没有 kernel 名和 dispatch，runner 却继续把未写出的全零 Y 与 reference 比较，235/256（91.79688%）错误并返回 RC=254。首轮 PMD 日志 SHA256=946f2d81131f278bf9275bc040924345955c70d9133867b2d2c22d12f2089575，gfx936/gfx938 日志 SHA256 均为 f37ce8050ae7587119be8c0beb4e69f55d5e07fc86ff2712a0e028939e1d9467；相同失败证明它不是 gfx946/PMD 指令问题。
- 静态链路确认 generator 明确使用 `fused_add_list=[0,1]`（注释中才保留 `[0,1,2]` 草案），生成 API 未命中时返回 `-1`；runner 的帮助文本却宣称支持 `fadd=2`，且遗漏 `ave_time<0` 检查。本步把公开合同收紧为 0/1，在 host tensor/device buffer 创建前拒绝其他值，并为所有 API 未命中补上统一 `not supported` 返回检查；runner/generator SHA256=ce0a01fd8e211b468dfa51913411ddf4762c314d4bb204010601d7a8cb11ba51/99c75acccc1734755785d2554fa226dc07ad811341e6c8a8f08b0066b6465714。没有盲目开启 `fadd=2` 的共享 generator/pipeline 闭包，标记 `SKIP-FADD2-SHARED-GENERATOR-CLOSURE`。
- 修正后正式矩阵在三架构均 4/4 符合预期：FP16/BF16 `fadd=1` 单 dispatch 且 `valid:y`、RC=0；两档 `fadd=2` 只输出稳定错误 `fadd only supports 0 (no fused add) or 1 (preadd+store).`、RC=254，PMD 不初始化 simulator、不产生 dispatch。PMD/gfx936/gfx938 matrix SHA256=28a836b056628a6ca35e741f9025909076ac7dc1b574b2a2b55c36ba83f3eb36/117df27fc53a033704553139c0782c47753bfa359e808377c8c07cbd9990194a/117df27fc53a033704553139c0782c47753bfa359e808377c8c07cbd9990194a；build SHA256=11957072da7a4ec60ab048366087076e7e4367e2734daca064d1ba8fcfbf69c0/0c1aff22cf5c5e8b54795c9fd432655fb466b63deb94c49d8604bec7c71e4cdc/100c272cdb06f08f9069a58a7772711ace74632b2c58c89fbe136192c80ca974，candidate binary SHA256=3698821b2db137ccd80ea959265d806b1273e460276f4faf139eb6260284ab81/876026d5a24503af7a5caf897fbc7e7a860ea464c7ba33f00203b96ef91fa84d/7fa796293328811411eb2084f13e6964a2bf1e037df6ba0a3e080c4ec5e31106。PMD 只运行 correctness，未测试性能。
- host runner TU 的 `.hip_fatbin` 相对 P6.67 前置 baseline 发生变化，因此按不能直接排除影响处理。两机测试前默认 HCU0 均 0% use/0% memory（gfx936 HCU6 有无关的 5% memory 占用，不进入本次设备），固定 BF16 no-add M=N=4096、warmup=50、repeat=2000、20 对交替顺序，以 P6.67 前置 binary 对当前 candidate：gfx936 65.637200→65.734900 us（+0.149%，日志 SHA256=be5384cd63107be0bda5c32888fccbc8c7424b76817155ab7a5daf2aabd2450f），gfx938 68.304500→68.302150 us（-0.003%，日志 SHA256=34d5477816618eb402884cc847ab3d27f43eb7b1a69fe6d78d11f116f2f51a93），样本区间重叠，无可测性能回退。gfx936 首次采样因重复任务并发写同一日志而废弃，未将其统计值作为证据；清洁重跑使用独立 41 行日志。

门禁结果：`fadd=1` G0-G6 PASS；`fadd=2` 在 device dispatch 前明确分类为共享生成器缺口并跳过，不再误报为 gfx946 correctness 失败。下一切片 P6.73 在 N=63 scalar-output 边界组合 `fadd=1` 与 dynamic/smooth-dynamic quant，验收 residual-store、量化 Y/scale 及可选 unquant Y 的组合输出合同。

### Step 6.73：RMSNorm2D fused residual + INT8 quant 组合闭包 — PASS-PMD（2026-09-03）

- 固定 M=2,N=stride=63、`fadd=1,save_rms=0,time=0`，交叉覆盖 FP16/BF16 input、smooth-dynamic/dynamic quant（`fquant=1/2`）以及 `save_unquant=0/1`，共 8 档。N=63 强制 Vector_N=1 scalar INT8 store；每档同时校验 pre-add 后 residual store、量化 Y 与 FP32 row scale，save-unquant 档再校验量化前 RMSNorm Y。
- gfx946 PMD 八档均命中对应 `..._pras_smdqt_pn`/`..._pras_dqt_pn` instance，单次 grid=(256,1,1)、wg=(256,1,1)，全部 `valid:y`、MATRIX_RC=0；仅有既存 `V_DIV_FIXUP_F32 not test`、pending-cacheline warning，FP16 save-unquant 仍出现已归档的 `V_MAD_MIXLO_F16 not test`，但所有输出正确。binary/log SHA256=3698821b2db137ccd80ea959265d806b1273e460276f4faf139eb6260284ab81/ca609311c0deb57d10bd8ff571de70f11d32d7b54a842d7d1e7f03dbd68af36b，未测试性能。
- gfx936/gfx938 同一 8 档矩阵均 `valid:y`、MATRIX_RC=0；binary SHA256=876026d5a24503af7a5caf897fbc7e7a860ea464c7ba33f00203b96ef91fa84d/7fa796293328811411eb2084f13e6964a2bf1e037df6ba0a3e080c4ec5e31106，日志 SHA256 均为 e069bbd55dc0d1a8ff36d7b08be2103769cdd358fdd512c750ca5c4f244d74d4。
- 本步复用 P6.72 已验 binary，没有代码、build graph、device payload 或 dispatch 变化，性能影响可直接排除，不重复执行 gfx936/gfx938 性能 A/B。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。RMSNorm2D 当前可生成的 no-save FP16/BF16→FP16/BF16/INT8、fadd=0/1、fquant=0/1/2 与 save-unquant 组合均已在 scalar 或标准 shape 上完成 gfx946 correctness；下一切片 P6.74 审计 N>8192 的 big-N two-pass instance 及 workspace/reference 合同，先以最小 M 验收 no-add BF16/FP16。

### Step 6.74：RMSNorm2D big-N two-pass no-add — PASS-PMD（2026-09-03）

- 静态审计确认 selector 在 N>8192 落入 `big` 档，使用 `kTwoPass=true` 的 four-trait 候选；pipeline 在单 kernel 内分两遍读取/归约，不需要 host workspace 或第二次 dispatch。big-N generator 明确排除 fused quant，但仍生成 no-add/fadd=1 的 FP16/BF16 instance。
- 固定最小 M=1、N=stride=8193、no-save/no-add/no-quant、`time=0`，分别运行 FP16/BF16。gfx946 PMD 两档均选择 `rmsnorm2d_fwd_{fp16,bf16}_1x4096_1x16_1x64_1x1_bpr_tp_pn_2p`，单次 grid=(1024,1,1)、wg=(1024,1,1)，`valid:y`、MATRIX_RC=0；只有既存 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。binary/log SHA256=3698821b2db137ccd80ea959265d806b1273e460276f4faf139eb6260284ab81/774749bc296039dfedd73841696fd08bc49df7bf91bd0399a05a30d14fd43c24。
- gfx936/gfx938 同两档均选择相同 two-pass trait、`valid:y`、MATRIX_RC=0；binary SHA256=876026d5a24503af7a5caf897fbc7e7a860ea464c7ba33f00203b96ef91fa84d/7fa796293328811411eb2084f13e6964a2bf1e037df6ba0a3e080c4ec5e31106，日志 SHA256 均为 27462977470e62bbcbb4b039e6b9ecf60e94b39f7c1a6fef2fcfff3a92386781。
- 本步复用 P6.72 已验 binary，没有代码、build graph、device payload 或 dispatch 变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。big-N two-pass no-add 已完成三架构正确性闭环；下一切片 P6.75 在相同 M=1,N=8193 上开启 `fadd=1`，验收 two-pass pre-add + residual-store 路径。

### Step 6.75：RMSNorm2D big-N two-pass fused residual — PASS-PMD（2026-09-03）

- 在 P6.74 相同 M=1,N=stride=8193、no-save/no-quant、`time=0` 合同上只开启 `fadd=1`，分别运行 FP16/BF16；这同时覆盖 two-pass 首遍 pre-add、第二遍 norm 写出以及 pre-add residual 独立 store。
- gfx946 PMD 两档均选择 `rmsnorm2d_fwd_{fp16,bf16}_1x4096_1x16_1x64_1x1_bpr_tp_pras_pn_2p`，单次 grid=(1024,1,1)、wg=(1024,1,1)，Y 与 residual 输出均 `valid:y`、MATRIX_RC=0；只有既存 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。binary/log SHA256=3698821b2db137ccd80ea959265d806b1273e460276f4faf139eb6260284ab81/220fcad26268654a7bfed0c69297b44a572985d3579448e882e476adc5b6c017。
- gfx936/gfx938 同两档均 `valid:y`、MATRIX_RC=0；binary SHA256=876026d5a24503af7a5caf897fbc7e7a860ea464c7ba33f00203b96ef91fa84d/7fa796293328811411eb2084f13e6964a2bf1e037df6ba0a3e080c4ec5e31106，日志 SHA256 均为 3c78a2777452d07c29ddcbe362eb3cc8c516718fc5c792732413027dfa574cb3。
- 本步复用 P6.72 已验 binary，没有代码、build graph、device payload 或 dispatch 变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。big-N fused quant 被 generator 明确排除，未把未生成能力误计为 gfx946 缺口。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。`10_rmsnorm2d` 当前真实生成闭包（no-save、FP16/BF16、one/two-pass、fadd=0/1、one-pass scalar INT8 quant/save-unquant）已完成三架构 correctness 与必要性能回归；下一切片返回 enabled-example 清单，选择下一个尚未覆盖的最小独立模块。

### Step 6.76：LayerNorm2D bias/residual + smooth-dynamic INT8 scalar 闭包 — PASS-PMD（2026-09-03）

- 固定 M=2,N=stride=63、`save_mv=0,fquant=1,time=0`，交叉覆盖 FP16/BF16 input、`xbias=0/1` 与 `fadd=0/1` 共 8 档。N=63 强制 Vector_N=1，分别命中普通、`_xbias`、`_pras`、`_xbias_pras` 的 smooth-dynamic quant instance；每档校验 LayerNorm、FP32 smooth scale、row scale、INT8 Y，并在 fadd=1 时校验 residual store。
- gfx946 PMD 八档均单次 grid=(256,1,1)、wg=(256,1,1)，全部 `valid:y`、MATRIX_RC=0；只有既存 `V_DIV_FIXUP_F32 not test` 与 pending-cacheline warning，未测试性能。binary/log SHA256=89656a4aa85821811e7c2fc3fa557db560db94f2a9c8c110956724690dbdd639/7d97c283d8338c3214b85c8873eb8e8710603404e48ed65e3754d0f4ec7aad62。
- gfx936/gfx938 同矩阵均 8/8 `valid:y`、MATRIX_RC=0；binary SHA256=18f3b54e059cc758f13486251c529c1f63c89df18624731fdba431d6ce63bfd2/00b374d155932b55209bd50ef341e72bf5edc60500ec837cc9166b426ab1774d，日志 SHA256 均为 2f632ea0466284c447c835482a2dfcdd8321f86b21f12976e957e9af843cf6c4。
- 本步复用 P6.66 已验 binary，没有代码、build graph、device payload 或 dispatch 变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。LayerNorm 的 scalar smooth-dynamic quant 与 bias/residual 组合已完成三架构闭环；下一切片 P6.77 修正 runner 虚假宣称的 `fadd=2/fquant=2`，在 host allocation 前稳定拒绝未生成组合。

### Step 6.77：LayerNorm2D CLI 生成合同收紧与前置拒绝 — PASS-PMD / SKIP-UNGENERATED（2026-09-03）

- 静态链路确认 generator 只生成 `xbias=0/1`、`fadd=0/1`、`fquant=0/1`，其中 1 仅代表 smooth-dynamic quant；runner 帮助文本却宣称 `fadd=2`（pre-add only）与 `fquant=2`（dynamic quant）。虽然 API 已能返回 `-1` 且 runner 会显示 `not supported`，但发生在 host/device tensor 创建之后，公开合同仍不准确。
- 本步把帮助文本收紧到真实生成集合，并在 tensor/device buffer 创建前统一校验三个枚举：非法值分别输出稳定的 `xbias/fadd/fquant only supports...` 并返回 false（shell RC=254）；同时把 quant dtype 错误文本从“1 or 2”修正为“1”。runner source SHA256=49fa2885ca3e9217ce83a26e50de333551876e71a4310048c2956b8eb01a7531。未盲目开启 generator 中注释掉的共享能力，`fadd=2/fquant=2` 记为 `SKIP-UNGENERATED-SHARED-CLOSURE`。
- 正式矩阵含 FP16/BF16 的 `xbias=1,fadd=1,fquant=1` 两档健康控制，以及 `xbias=2`、`fadd=2`、`fquant=2` 各两档拒绝。三架构均 8/8 符合预期：控制项 `valid:y`、RC=0；六个非法项只输出对应 host 错误、RC=254，无 kernel 名。PMD invalid cases 不初始化 simulator、无 dispatch；PMD/gfx936/gfx938 matrix SHA256=1faeea5eb2fac33ac304603f3d5516627e7f2a8a6e9db59b06cf82eb186acff9/fa16c88f9ea9dec54c7247b8620367fa5c2521db165fbe57940a89bfed2b2335/fa16c88f9ea9dec54c7247b8620367fa5c2521db165fbe57940a89bfed2b2335。
- 三端增量构建成功，build SHA256=99dd39c7a5311d3f533e6559d79e567b2c8224896e5a707b09e057e61c189df1/933d550d9ebf63e4d6d38e0350b68164071b0b095db2e046e2d04aba9292dff0/fd6ecbf39076f5a920e53df573f66da38100b2ef96999288a0877ea067651c33，candidate binary SHA256=607c4cfa558b81782e90460fb357a97cb9e1343d1f5083f8cda56d3f11f8ad05/ce9e996678c65ac2cdf6f6bfbca733d6aee6b161cd1eec9209aefa0003f05345/2a9a9558ced254682d3fa64e486ed9453309d9a6b626a2d553e9155dde8fe310。gfx936/gfx938 `.hip_fatbin` 修正前后完全不变，分别为 c9d1d009d77ef7cbd4eda4a63f1fadf8dbee2fa52c725f4f278641b4568adca1/01253a743b51e2d411a52a8a75f79f6db4ba9d9c3e807aa789b5eb10cdb79d76；设备 payload 未变，可直接排除性能影响，故不执行性能 A/B，PMD 未测试性能。

门禁结果：G0-G6 PASS；LayerNorm 的实际 inference 生成闭包和 unsupported-path 均已有三架构证据，save-MV 仍按 P6.66 保持共享生成器闭包延期。下一切片 P6.78 验收 LayerNorm N>8192 big-N two-pass 的 FP16/BF16 no-bias/no-add 路径。

### Step 6.78：LayerNorm2D big-N two-pass no-bias/no-add — PASS-PMD（2026-09-03）

- 静态审计确认 N>8192 进入 `big` 档，使用 `kTwoPass=true`、Welford-enabled 的 four-trait 候选；big-N 生成式明确排除 smooth-dynamic quant，但保留 bias/residual 组合。本步先固定最小 M=1、N=stride=8193、`xbias=0,fadd=0,fquant=0,save_mv=0,time=0`，分别运行 FP16/BF16。
- gfx946 PMD 两档均选择 `layernorm2d_fwd_{fp16,bf16}_1x4096_1x16_1x64_1x1_bpr_2p_pn`，单次 grid=(1024,1,1)、wg=(1024,1,1)，均值/方差归约与 Y reference `valid:y`、MATRIX_RC=0；仅有 pending-cacheline warning，未测试性能。binary/log SHA256=607c4cfa558b81782e90460fb357a97cb9e1343d1f5083f8cda56d3f11f8ad05/0757584ae44c776d47849e1beb7e49b129b7ea7d21968cc606b30d0b5a81020a。
- gfx936/gfx938 同两档均选择相同 two-pass trait、`valid:y`、MATRIX_RC=0；binary SHA256=ce9e996678c65ac2cdf6f6bfbca733d6aee6b161cd1eec9209aefa0003f05345/2a9a9558ced254682d3fa64e486ed9453309d9a6b626a2d553e9155dde8fe310，日志 SHA256 均为 5118429c09d51eea610b30e3c951e39b3e0549dda672467c1197bbd2858315bc。
- 本步复用 P6.77 已验 binary，没有代码、build graph、device payload 或 dispatch 变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。LayerNorm big-N two-pass 基础链路已闭环；下一切片 P6.79 在相同 N=8193 上交叉覆盖 `xbias=0/1` 与 `fadd=0/1`，完成 big-N inference 生成组合收口。

### Step 6.79：LayerNorm2D big-N two-pass bias/residual 组合 — PASS-PMD（2026-09-03）

- 复用 P6.78 的 M=1,N=stride=8193、`save_mv=0,fquant=0,time=0`，补测尚未覆盖的 bias-only（xbias=1,fadd=0）、residual-only（xbias=0,fadd=1）与 bias+residual（xbias=1,fadd=1），每种含 FP16/BF16，共 6 档；fadd=1 同时校验 pre-add residual store。
- gfx946 PMD 六档均命中相应 `_2p_xbias`、`_2p_pras` 或 `_2p_xbias_pras` instance，单次 grid=(1024,1,1)、wg=(1024,1,1)，所有 Y/残差输出 `valid:y`、MATRIX_RC=0；仅有 pending-cacheline warning，未测试性能。binary/log SHA256=607c4cfa558b81782e90460fb357a97cb9e1343d1f5083f8cda56d3f11f8ad05/88e9720d32b050a0c07e642ce1aa3848f3793fae82ce247dcd1ff33fad073d5b。
- gfx936/gfx938 同矩阵均 6/6 `valid:y`、MATRIX_RC=0；binary SHA256=ce9e996678c65ac2cdf6f6bfbca733d6aee6b161cd1eec9209aefa0003f05345/2a9a9558ced254682d3fa64e486ed9453309d9a6b626a2d553e9155dde8fe310，日志 SHA256 均为 4aecd3d9808b1d19951d2bf5f371f8c4f5299f79d3d9dcc5a1c7c1994a3cb4f2。
- 本步复用 P6.77 已验 binary，没有代码、build graph、device payload 或 dispatch 变化，性能影响可直接排除，不执行 gfx936/gfx938 性能 A/B。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。`02_layernorm2d` 当前真实 inference 生成闭包（FP16/BF16、one/two-pass、xbias/fadd=0/1、one-pass scalar smooth-quant）完成三架构 correctness，unsupported 输入已稳定拒绝；save-MV 仍按 P6.66 延期。下一切片 P6.80 返回全局 enabled target 与已验清单，优先选择尚未形成三架构 correctness 闭包的独立 example。

### Step 6.80：Fused-MoE `topk=2` 多路由 sorting/atomic 闭包 — PASS-PMD（2026-09-03）

- P4.19 已把当前 94 个 enabled target 全部映射到构建/运行结论，因此后续优先补真实 runner 内尚未覆盖的独立分支，而不是重复新增目录。P3.7 的 `17_fused_moe` 只验证了 `topk=1`；该路径会启用 deterministic temporary buffer，不能代表 `topk>1` 的多 route FP32 atomic 累加。本步固定 `t=2,e=4,k=2,h=1024,i=128,bm=16,act=SiLU,balance=1,fquant=0,time=0`，4 个 route 恰好均匀覆盖 4 个 expert，并分别运行 API 0（device sorting + MoE GEMM）与 API 1（host-reference sorting + MoE GEMM）。
- gfx946 PMD 两档均 `valid:y`、MATRIX_RC=0。API 0 实际发射 sorting `grid=(768,1,1),wg=(256,1,1)` 与 MoE GEMM `grid=(256,5,1),wg=(256,1,1)`；API 1 发射同 shape MoE GEMM，证明多 expert padding、sorted weights、两段 GEMM 与 topk-weight atomic accumulation 均可执行。只有 API 0 已归档的 pending-cacheline warning，无新指令终止；binary/log SHA256=69a3b4ab4f35bea889e32b6ee59046e90ad65aa59cd3584433bc6d8541739266/3e0619a3bf815f91dbf30ab989085835c913881fb8696fe8b65f9c61758661aa，未测试性能。
- gfx936/gfx938 同矩阵均 2/2 `valid:y`、MATRIX_RC=0；binary SHA256=24858396e44e00564308a380ddaf2672a2bf7246f9c184b8cb84fef08adc29ae/cca9a55f242ce671538ccc63134eeba99124f9ccbd055c733c8f52037c457218，日志 SHA256 均为 33f04e05f7fc3df43f3001183785481624fb86d1af34f5bfaf6af2dd0d7a3e95。
- 本步复用 P3.7 已提交、已做 device ELF identity 证明的 binary，没有代码、build graph、device payload 或 selector 变化，性能影响可直接排除，故不执行 gfx936/gfx938 性能 A/B。结果只扩展 correctness 覆盖，不把 PMD 用作性能证据。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。`17_fused_moe` 的非量化 FP16/FP32 输出现已覆盖 topk=1 deterministic 与 topk=2 atomic 两类执行链；下一切片 P6.81 审计 runner 暴露的 `fquant`、`gate_only`、`tp` 参数与实际生成实例，选择一个最小、可由 CPU reference 完整验证的新增分支。

### Step 6.81：Fused-MoE runner 真实参数合同与前置拒绝 — PASS-PMD / SKIP-UNEXPOSED-QUANT（2026-09-03）

- 静态闭包确认 target 虽然编入若干 quant kernel/API 模板，但当前 `main` 唯一可达的 C++ 类型实例是 FP16 input/weight、FP32 output；non-quant selector 还没有比较 `fused_quant`，reference 又完全忽略 scale tensor。改动前 gfx936 探测因此出现 `fquant=1/2` 均执行普通 FP16 kernel、`valid:y`、RC=0 的虚假量化通过；`gate_only=1` 则在 host/device tensor 创建及 API 0 sorting 准备后才返回 unsupported。探测日志 SHA256=d63d1b9eee6bc3bc855ffdc67da0f4d6420dd5f7830994f873998cf6ff4010da。
- 本步不把不可验证的量化类型强行接入 runner，而是把 help 收紧到真实可达合同，并在任何 tensor/device allocation 前校验正数 shape、`1<=topk<=experts`、stride、`block_m=16/32`、TP 正数整除、`fquant=0`、`gate_only=0`、API 0/1、balance 0/1 与 init 0/1/2；非法输入固定返回 `-4`（shell RC=252）。量化 kernel 源码没有删除，后续需以完整 dtype/scale/zero-point/reference 闭包独立启用，当前分类为 `SKIP-UNEXPOSED-RUNNER-CLOSURE`。main source SHA256=132cc2f1b03ef09f029926bd165f114f8491410dc63c0cb7ac3c5627edfc2e74。
- 正式矩阵用 `t=2,e=4,k=2,h=1024,i=256,tp=2,bm=16,balance=1` 的 API 0/1 作为健康控制，并覆盖 fquant 1/2、gate-only、API 2、topk 0/越界、短 stride、block_m 8、TP 0/非整除、balance 2、init 3 与 tokens 0 共 13 个非法项。三架构均 15/15 符合预期：两个控制 `valid:y`、RC=0；非法项仅输出稳定 host 错误、RC=252。PMD 非法项均未出现 `Run on Perf Model`，证明未初始化 simulator 或发射 device kernel；PMD/gfx936/gfx938 matrix SHA256=67ec55a00e4247c76fe675bda10630158e8b3172aa1c14a7ba8a753f8efc5ab5/27bce70521b16750b29220d3c8feab90846d2879e707563d1de3baa43a1cd055/27bce70521b16750b29220d3c8feab90846d2879e707563d1de3baa43a1cd055。PMD 未测试性能。
- baseline/candidate binary SHA256：gfx946=69a3b4ab4f35bea889e32b6ee59046e90ad65aa59cd3584433bc6d8541739266/b018131f39747b82ef7cfad40a24a0dcf312a4556080b27406ea4d705f207a73，gfx936=24858396e44e00564308a380ddaf2672a2bf7246f9c184b8cb84fef08adc29ae/41cd80d6412420e2d3a7089592727d998dab07e88869a8ddfac817eca5904786，gfx938=cca9a55f242ce671538ccc63134eeba99124f9ccbd055c733c8f52037c457218/4763c9f7cf7ae19ebff46da08311ba082ed3b57e2f00ae7911a555573074e81e。两台真机各提取 1 个 device ELF，gfx936 manifest 前后均为 f591862b373b3e428b6b67d044dd0f8d89cfbe663b241c51045b3e539dc7a605，gfx938 前后均为 464d6ab335289ff8bfec59c194c0af04c4d6f49d8e4c00ab2d2de4f678958373；device payload 逐字节不变，可直接排除 kernel 性能影响，故不执行性能 A/B。

门禁结果：G0-G6 PASS。runner 不再把 ignored quant 参数误报成量化支持，也不再为必然 unsupported/危险输入做大额分配；合法 TP 分片与 topk=2 路径保持三架构正确。下一切片 P6.82 使用当前独立 `block_m=32` instance 验收 sorting padding、deterministic topk1 与 topk2 atomic 两类分支。

### Step 6.82：Fused-MoE `block_m=32` 双真机基线分类 — SKIP-SHARED-BAD-BASELINE（2026-09-03）

- `block_m=32` 选择不同于已通过 bm16 的 256-wide intermediate tile instance。本步先按门禁在真机建立健康基线，固定 `t=2,h=1024,i=128,balance=1,time=0`，交叉覆盖 topk=1（e=2，deterministic temporary）/topk=2（e=4，atomic accumulation）与 API 0/1；首个 gfx936 topk1/API0 已有 99.12109% wrong values，因此没有直接送 PMD 归因 Shaobo。
- 完整诊断显示 gfx936 topk1 的 API 0/1 均为 2030/2048（99.12109%）错误、RC=254；topk2 的 API 0/1 都在 `grid=(256,5,1),wg=(256,1,1),group_segment_size=32768` 的 MoE GEMM dispatch 触发 invalid-address VMFault、RC=139。失败后 7 张 HCU 均为 0% use/0% memory，无残留占用；binary/log SHA256=41cd80d6412420e2d3a7089592727d998dab07e88869a8ddfac817eca5904786/6937a993ab59ae6e62fd4d0c957b44b12d302a41c48c2ad805c03a41342a2d46。
- gfx938 同四档均可返回但 correctness 全错：topk1 的 API 0/1 各 2037/2048（99.46289%）错误，topk2 的 API 0/1 各 2012/2048（98.24219%）错误，均 RC=254；binary/log SHA256=4763c9f7cf7ae19ebff46da08311ba082ed3b57e2f00ae7911a555573074e81e/a25f43685ffc0017af7e27d41f36a65a39d6acf4f0e108be313b7af061512dde。API 0 与 API 1 同错排除了 sorting 作为单一原因，双架构失败则排除了 gfx946/PMD 特有指令问题。
- 按“真机健康基线先于 PMD”门禁，本子路径标记 `SKIP-SHARED-BAD-BASELINE`，不运行 gfx946 simulator，不修改 device kernel，也不做无意义的性能测试。失败日志和 VMFault 证据保留，后续若单独修复 bm32 需作为跨架构 Fused-MoE correctness 项目处理。

门禁结果：G0-G2 PASS，G3 NOT RUN，G4 FAIL-SHARED，G5/G6 NOT RUN。下一切片 P6.83 回到已验证的 bm16 instance，使用 `stride>hidden_size` 覆盖 padded token row 的输入/输出寻址，并继续在 PMD 与双真机分离验收。

### Step 6.83：Fused-MoE padded token stride 与 topk 数值分层 — PASS-PMD（2026-09-03）

- 首轮固定 bm16、`t=2,e=4,k=2,h=1024,i=128,stride=1056,balance=1,time=0`。gfx946 PMD 与 gfx938 的 API 0/1 均 `valid:y`；PMD API 0 sorting/GEMM grid=(1024,1,1)/(256,5,1)，API 1 GEMM grid=(256,5,1)，binary/log SHA256=b018131f39747b82ef7cfad40a24a0dcf312a4556080b27406ea4d705f207a73/770bcf210cc70a2ded5e8de241556a876d75d0689b23da793f46e0d5b871ae66；gfx938 binary/log SHA256=4763c9f7cf7ae19ebff46da08311ba082ed3b57e2f00ae7911a555573074e81e/5dccfd7e7adf2daccadc1e598e6b6a67c5a84f391fe5bed9046476220afcc16c。
- 同一 topk2 case 在 gfx936 的 API 0 首次只有逻辑元素 out[1693] 超容差：-0.05759716 vs -0.04634571，1/2080（0.04807692%）错误。为避免把单点数值边界误判为 stride 寻址错误，在不降低容差的前提下对 API 0/1 各重复 10 次；20/20 均在完全相同元素、相同误差上 RC=254，证明它可重复且与 sorting 无关。gfx936 repeat log SHA256=5ddf955e99a27f3183a389d016723f80659428371263ab93af36d3c4e294bc3b；因同 device ELF 的 gfx938 与 PMD 均通过，这一项记录为 gfx936 既有 FP accumulation/reference tolerance 边界，不归因 gfx946。
- 为隔离 stride 的真实合同，正式三架构矩阵改用 topk=1 deterministic：`t=2,e=2,k=1,h=1024,i=128,stride=1056,bm=16,balance=1`，API 0/1 均 `valid:y`、MATRIX_RC=0。PMD API 0 sorting/GEMM grid=(1024,1,1)/(256,3,1)，API 1 GEMM grid=(256,3,1)，日志 SHA256=75d0c323962022e0ae4b0ea9587edda251432f0a1e9d76a583900e589eb35c52；gfx936/gfx938 日志 SHA256 均为 abb3a56146129bb05a32eafb5b84f602281b08bc650e08e89cea985201a3e917。
- 本步没有源码、build graph、binary、device payload 或 selector 变化；padded stride 只改变 runtime kargs，性能影响可直接排除，故不执行 gfx936/gfx938 性能 A/B，PMD 未测试性能。

门禁结果：topk1 padded stride G0-G4/G6 PASS、G5 NOT RUN；topk2 padded stride 在 gfx946/gfx938 PASS、gfx936 为稳定单点数值超差并保留失败证据。下一切片 P6.84 使用 bm16、topk1 与 `tokens>block_m`，让同一 expert 跨多个 sorted tile，验收 repeated expert-id tile 与 deterministic reduction。

### Step 6.84：Fused-MoE 单 expert 跨多个 sorted tile — PASS-PMD（2026-09-03）

- 固定 `t=17,e=1,k=1,h=1024,i=128,bm=16,act=SiLU,balance=1,time=0`，使唯一 expert 的 17 个 token 必须跨越两个 16-token sorted tile；分别运行 API 0（device sorting）与 API 1（host-reference sorting），验收 repeated expert-id tile、第二个 partial tile 以及 topk1 deterministic temporary/reduction 链路。
- gfx946 PMD 两档均 `valid:y`、MATRIX_RC=0。API 0 实际发射 sorting 相关 `grid=(24576,1,1)`、`grid=(256,1,1)`、`grid=(4608,1,1)`，随后 MoE GEMM `grid=(256,2,1),wg=(256,1,1)`；API 1 直接发射相同两-tile GEMM。只有 API 0 已归档的 pending-cacheline warning，无新指令终止；binary/log SHA256=b018131f39747b82ef7cfad40a24a0dcf312a4556080b27406ea4d705f207a73/454545b5a45b6b908bc1083ee4b608d448940592ea845fb57023365adb33922d，未测试性能。
- gfx936/gfx938 同矩阵均 2/2 `valid:y`、MATRIX_RC=0；binary SHA256=41cd80d6412420e2d3a7089592727d998dab07e88869a8ddfac817eca5904786/4763c9f7cf7ae19ebff46da08311ba082ed3b57e2f00ae7911a555573074e81e，两端日志 SHA256 均为 0fff6f80999330b41a3eed0c9554da73eeda7647337d4f77a4ddecea54034a7f。
- 本步复用 P6.81 已验 binary，没有源码、build graph、device payload、selector 或 launch 配置变化；仅扩展 runtime shape correctness 覆盖，性能影响可直接排除，因此不执行 gfx936/gfx938 性能 A/B，PMD 未测试性能。

门禁结果：G0-G4/G6 PASS，G5 NOT RUN。`17_fused_moe` 当前 runner 可达的 bm16 FP16-input/weight、FP32-output 闭包已覆盖 act=0/1、API 0/1、topk1 deterministic/topk2 atomic、TP 分片、padded stride 与单 expert 多 sorted tile；bm32 保持共享坏 baseline，quant/gate-only 保持未暴露合同。下一切片 P6.85 返回全局遗留队列，选择尚可独立闭环的最小共享生成器缺口。

### Step 6.85：RMSNorm2D N≤64 save-RMS 真实输出最小闭包 — PASS-PMD（2026-09-03）

- 全局遗留审计把尚未闭合的共享项限定为 Add-RMSNorm no-save 全 shape、LayerNorm save mean/invstd 与 RMSNorm save inv-RMS。RMSNorm core one/two-pass pipeline、kernel Kargs 和 CPU reference 已有 `kSaveInvRms` 完整实现，缺口仅在生成器从未生成 true trait、API dispatch 忽略 `save_rms`，以及 runner 传空指针且不回读。改动前 gfx936/gfx938 上 FP16/BF16 `m=17,n=63,save_rms=1` 因此都错误地选择 no-save kernel 并 `valid:y`、RC=0，属于可重复的虚假支持；baseline binary/log SHA256=876026d5a24503af7a5caf897fbc7e7a860ea464c7ba33f00203b96ef91fa84d/30d15b2f18a7b1a75ac3f34e66a7bd0ce033749bfc0d1f13ac97680b09a09323（gfx936）、7fa796293328811411eb2084f13e6964a2bf1e037df6ba0a3e080c4ec5e31106/8228c857592ec80183f0fdec13c3722fdbfb9ef42a5e0361301a55c7c996606e（gfx938）。
- 本步先建立最小独立能力：生成器只为 N≤64、FP16/BF16、non-quant、no-add 的三个向量候选各追加 save trait，并把 `t.save_rms` 纳入所有 API 匹配条件；runner 分配/传入 `[M]` inv-RMS buffer、回读并与 CPU reference 校验，同时修正性能字节口径从错误的 M×N 为 M。N>64 或 save+fused-add/quant 在任何 tensor/device allocation 前稳定拒绝，帮助文本同步真实边界。generate.py/runner SHA256=4d0e3623dfc1c4c1903c6c35ecae336110186bfbd3b1e54d971e43cd3f2ce371/bf124819ef9faefdfcc80e77cc8c668e5d49bfbac77480606813279342447550；三端生成 API/FP16-N64/BF16-N64 SHA256 均为 40def9783101d20ce9bb31c0b95b50e8bd6d3edd30ed3e6e9db8d20b02ec9e65/c5f905b764ad8ea5647192b772eb1d57583f476f29d955478e1ac54517dc5ccf/b01d9d66fd13db3aa078f268f2958f1628b4bec0b7ae6491104078e4d23ad800，三端完整 target 构建成功。
- 正式 correctness 矩阵交叉覆盖 FP16/BF16 × N=63 scalar-tail/N=64 vector，各 save case 均命中带 `_rms` 的真实 kernel，Y 与 inv-RMS 同时 `valid:y`；另含两种 dtype 的 N=63 no-save 控制，以及 N=65、N=64+fadd=1 共四项前置拒绝。三架构均 10/10 符合预期；PMD N=63/N=64 分别为 grid=(1280,1,1),wg=(256,1,1) 与 grid=(192,1,1),wg=(64,1,1)，只有既存 `V_DIV_FIXUP_F32`/FP16 mix/src1 modifier/pending-cacheline warning，无新终止。PMD binary/log SHA256=b42210f5a9741af7a7516d36436b33a0e5c316041bc4758dda3896eac70aafb6/6af6fc52f3a46e009cff91881237ae187b0e814dd570d68b89490d0f3e2e2cbe；gfx936 binary/log SHA256=3acc97f506793d477f83890f5ae60fb941df1869ff3f744ebc7f6ad78fa37f51/be324290efc2bfae96639402bcfc9a1ea57f4529dedb0049255b32e6d0240828；gfx938=db60aa692afe4ae48ee08e26dc85eda358bfb1bcd972c436c4b9035f35bbab65/e7d3765b24fbad66413077638df1acbe07d1a24cf51231282ec5fdca84701ef1。PMD 全程 `time=0`，未测试性能；四个非法项均无 `Run on Perf Model`。
- 生成 API 与 binary 都发生变化，不能用 payload identity 排除既有 no-save 性能影响，因此按要求在两台真机固定 `M=3328,N=64,save_rms=0,warmup=50,repeat=1000` 各做 FP16/BF16 五次 baseline/candidate。gfx936 中位数 FP16 3.46143→3.43983 us（-0.62%）、BF16 4.07486→4.03342 us（-1.02%）；gfx938 FP16 3.67165→3.63645 us（-0.96%）、BF16 4.33346→4.32493 us（-0.20%），无性能回退。

门禁结果：G0-G6 PASS。旧 runner 的 silent false-positive 已消除，N≤64 save-RMS 现在具有真实 device store 与 CPU-reference 证据；较大 N 仍明确保持未生成边界，不误报支持。下一切片 P6.86 沿同一独立闭包扩展 N≤128 one-pass shape class，仍只增加 FP16/BF16 non-quant no-add save trait，并复用相同三架构/性能影响门禁。

### Step 6.86 前置判定：既有能力边界与后续项目归档 — COMPLETE（2026-09-03）

- 判定基线为任务开始前 `main-local@07581109898bb779054f331ca4a0c13e3b1aad29`。RMSNorm 的 one/two-pass core、Kargs 与 reference 虽然已有 `kSaveInvRms` 参数，但原生成器从未产生 `kSaveInvRms=true` 实例，API 也不比较 `save_rms`，gfx936/gfx938 实测会静默选择 no-save kernel；因此 P6.85/P6.86 是本次已启动的深入适配，不把 N>128 扩展误写为真机原有能力。
- LayerNorm 同理：任务前 runner 暴露 `save_mv`、模板含 `kSaveMeanInvStd`，但生成器没有注册 true trait，runner 没有以真实 device mean/invstd 输出闭环；P6.66 已证明三架构均静默走 no-save。故 save-MV 全 shape 属于后续新增能力，不在 Step 1 继续实施。
- Add-RMSNorm-RDQuant 的真实 no-save 原有能力只覆盖 `4096<N<=8192`；P6.52 已修复 BF16 host dispatch 并完成三架构 correctness/性能闭环。`N<=4096` 原 API 无条件选择 save-X，`N>8192` three-pass 也只注册 save-X，因而小 N 与 big-N no-save 不是 gfx936/gfx938 原有能力。
- 按用户要求，以下能力统一记录为后续项目，不在 P6.86 后继续修改代码：RMSNorm save-RMS one-pass `128<N<=1024`、one-pass `1024<N<=8192`、big-N two-pass；LayerNorm save mean/invstd；Add-RMSNorm-RDQuant no-save `N<=4096` 与 `N>8192`。每项后续实施都必须补齐实例/selector/输出/reference，并重新执行 gfx946 PMD correctness 与 gfx936/gfx938 correctness/性能门禁。
- `52_cshuffle_lds` 与上述项目一并归档：当前缺 `tile_load_store_microkernels.hpp`，example 只报告 LDS timing、没有 correctness reference；后续必须先增加可确定输入模式、逐元素 CPU/reference 校验和非零错误返回，再谈 gfx946 指令验收。Perf Model 永远不用于该 microbenchmark 的性能结论。

本步只更新来源判定与工作边界，没有改变 device code、build graph 或 dispatch，故不运行三架构测试与性能 A/B。下一步完成已经在 P6.85 明确启动的 P6.86；P6.86 独立提交后结束本轮深入适配。

### Step 6.86：RMSNorm2D N≤128 save-RMS one-pass 闭包 — PASS-PMD（2026-09-03）

- 在 P6.85 的真实 inv-RMS 输出/reference 闭包上，把 FP16/BF16、non-quant、no-add save trait 从 N≤64 扩到第二个 one-pass shape class N≤128；runner 的前置合同同步扩到 N≤128，N>128 与 fused-add/quant 组合继续拒绝。`generate.py`/runner SHA256=569437bb75746ea17867fa0836047f4731756dd71366a9a5058278959101a8c6/91f2f259cc4366a6d3f0d63346bb35755034314c59b70e54493964aabed59449。三端重新生成的 API/FP16-N128/BF16-N128 SHA256 均为 a0a8e7f8172dec29c027940f8758aafcfd82ddd4cb50d8450ed9959f5f65d7cc/2342936e6f3186f55c43eee27dfe3aa5b1baa7d2d9d5965a7a20820dbb91972a/ca0b606d5f7081d6077dc4a26e50a2027319851e1f3d0258bb1d21f9dbdd02d1，三架构 focused target 均构建成功。
- 正式 correctness 矩阵覆盖 FP16/BF16 × N=127 scalar-tail/N=128 vector save case、N=127 no-save 健康控制，以及 N=129、N=128+fadd=1 的前置拒绝，共 10 档。gfx946 PMD 10/10 符合预期：N=127/N=128 分别命中带 `_rms` 的 1x64x1/1x128x8 kernel，Y 与 inv-RMS 同时 `valid:y`；四个非法项均 RC=254，未发射 simulator。只有 P6.85 已归档的 `V_DIV_FIXUP_F32`、FP16 modifier 与 pending-cacheline warning，无新指令终止；binary/log SHA256=c00b45a20cd462cd3b49be0e4263d726f6b0a0a2ed1afdc04bd68307520b64ad/951a3048eb9057d16a688454a621187b0d36c72f8559b50259e646dac63a7e3e。PMD 全程 `time=0`，未测试性能。
- gfx936/gfx938 同一 10 档矩阵均通过，save case 命中与 PMD 相同的 `_rms` instance，非法项均 RC=254；candidate binary/log SHA256=321da3bdb66682b915f7972824cf4561a05e240c18b4265c6744595e700f45c7/5593bb3b985d228dcdceaddb04c45ce0639b750f4ef0333ac013a55167a1b59e（gfx936）、c4e1748be189b4688ed5ac38e2b0299d68fd661a817e38e08b69ff8bc53c3f18/0301c2db09f1f10b0a8231e239334636689f82e3fa2a68707d8adff7c3cea5b4（gfx938）。
- 因生成 API/二进制改变，不能静态排除原 N=128 no-save 性能影响。两台真机均构造仅移除新增 save trait 的 registration baseline，并与 candidate 在同一 GPU 上按 ABBA/BAAB 顺序交错运行 7 轮、每轮各 2 次，固定 `M=3328,N=128,warmup=100,repeat=2000`。gfx936 14 次中位数：FP16 4.891000→4.909960 us（+0.388%），BF16 5.952455→5.960805 us（+0.140%）；gfx938：FP16 5.051585→5.055480 us（+0.077%），BF16 6.001290→5.988535 us（-0.213%）。四组均命中完全相同的 no-save kernel 名称，无性能回退；baseline/candidate binary SHA256=0099a426ec2f2abb9b70e0700e4e90a902903a49e86fa641c9752698eddcce38/321da3bdb66682b915f7972824cf4561a05e240c18b4265c6744595e700f45c7（gfx936）、8eaa913115664b91f0be6762ed4f7f1b4fe01cb2a7f43c3fac13f06dcb0d01dd/c4e1748be189b4688ed5ac38e2b0299d68fd661a817e38e08b69ff8bc53c3f18（gfx938），A/B 日志 SHA256=e80a39a2986bf3539842b51a5b8322231bbc4ba3e5fbf9fb79c02f77389b7480/6848df01a140c2d82d8748e5fa67028a66c4281f0dc506677a921581d2f6e9cc。

门禁结果：G0-G6 PASS。P6.86 完成本轮已规划深入适配；N>128 save-RMS、LayerNorm save-MV、Add-RMSNorm 缺失 no-save shape 与 CShuffle LDS correctness harness 均按上一节来源判定作为后续独立项目，不在 Step 1 继续实现。

## 8. 提交模板

每个 example 完成后执行：

```bash
git status --short
git add -- <该 example 的代码/测试路径> adapt_ck_to_shaobo_step1.md
git diff --cached --check
git diff --cached
git commit -m "feat(gfx946): adapt <example-name>"
```

不得使用 `git add -A`，不得把任务开始前的 `script/cmake-ck-dev.sh`、`.planning/` 或远端 `.hcu-remote/` 混入提交。

## 9. 继续执行提示

Step 1 已完成，没有待续的 `IN PROGRESS` 切片。后续若启动上一节归档的独立项目，先选择其中一个最小能力闭包，复核本地 dirty 与三台远端路径；完成一个 example 的 gfx946 PMD correctness 和 gfx936/gfx938 回归后，立即更新本文并仅提交该 example 的最小改动。若出现 Perf Model 指令阻塞，保留首个终止错误、最小复现和 A/B 分类，标记跳过后继续下一独立模块。gfx946 Perf Model 禁止性能测试；gfx936/gfx938 在不能直接排除性能影响时必须执行 baseline/candidate A/B。
