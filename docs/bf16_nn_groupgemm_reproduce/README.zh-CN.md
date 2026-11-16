# BF16 NN Grouped GEMM：交付分支与正式口径复测

本说明用于把本地 CK 工作区发布到新的远程仓库，让接收者 clone 后自行构建并生成新的正式口径测试结果。

交付分支：`bf16-nn-reproduce-handoff`，基于 `2ab0db23ca6e0dba6ee1e98e897e862c4cb5ce80`，包含原有9个已修改文件、10个未跟踪文件，以及本次复现入口与说明的更新。保留既有 kernel 内容和实验开关，不把未采用的原型当作生产优化。原 `v7-coverage` 分支保留。

**shape 和 BLASLt 均由使用者准备，本仓库不打包它们。** 通过 `--shapes` 和 `--blas-lib` 指定路径，配合 `--session-probe`，不需要任何旧实验目录中的探针或日志。本次整理只进行本地提交和 CPU 验证，没有上传、运行 GPU 或更新正式性能数据。

## 1. 原 git status 中的文件是什么

下面记录的是交付整理前的状态；这些内容会随交付分支提交。`modified` 是已跟踪文件相对提交的改动，`untracked` 是尚未进入 Git 的文件。内容签名保存在 [inventory.json](inventory.json) 的 `original_worktree_files`，共19个文件。

示例目录为 `example_hcu/ck_tile/19_grouped_gemm/`，GEMM 头文件目录为 `include/ck_tile/ops/gemm/`。

|原已修改文件|作用|
|---|---|
|示例目录 `CMakeLists.txt`|源文件清单变化时触发 CMake 重新配置。|
|示例目录 `grouped_gemm_decode_device_args_provider_sources.txt`|纳入 B-vector 新实例源文件。|
|示例目录 `grouped_gemm_device_args.cpp`|B-vector 开关解析、6001/6002 分派与候选信息处理。|
|示例目录 `grouped_gemm_device_args_query_test.cpp`|查询测试补入6001/6002，候选数由69增至71。|
|示例目录 `instances/grouped_gemm_device_args_provider.hpp`|补充新 decode 实例的分派处理。|
|GEMM目录 `grouped_gemm_device_args.h`|新增6001/6002实例枚举；没有新增公开结构字段。|
|GEMM目录 `grouped_gemm_selector.hpp`|B-vector 策略、实例注册和属性选择。|
|GEMM目录 `pipeline/gemm_pipeline_decode_nn_wave.hpp`|抽象 B 加载、重排和等待，默认仍使用标量对照策略。|
|GEMM目录 `pipeline/gemm_pipeline_v7_uniform.hpp`|保留原 `kPre4` 的 `__builtin_amdgcn_readfirstlane` 改动，参与V7构建。|

原两个未跟踪的 C++ 文件是：

- `example_hcu/ck_tile/19_grouped_gemm/instances/grouped_gemm_bf16_nn_decode_bvector_device_args.cpp`：B-vector 启动入口。
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_decode_nn_bvector.hpp`：B 合并加载、寄存器排列和等待实现。

它们是 A-LDS/B寄存器、K32、两级预取的实验，实例6001/6002，默认关闭，当前1193条正式结果未选用。保留其构建/分派接入时，这两个文件必须一并进入版本。

原 `tools/grouped_gemm_reproduce/` 的8个未跟踪文件是较早的复现工具：`reproduce.py`组织作业，`plan.py`解析配置，`selector_probe.cpp`做CPU选核核对，`runtime.py`和`health.py`管理门禁与恢复，`report.py`汇总，`fake_benchmark.py`与`test_reproduce.py`用于CPU模拟测试。

**当前正式口径使用 `tools/grouped_gemm_depth8_reproduce/`，不要混用两套同名模块。** 正式表六条采用的 A寄存器/B私有LDS八级流水是已入库的 `gemm_pipeline_decode_nn_depth8.hpp`，实例6003。

## 2. 发布者如何上传，接收者如何取得

交付分支已经包含源码和工具，不需要接收者再手工补齐未跟踪文件。发布者核对本地提交后，将该分支推送到新远程仓库。以下为之后人工发布的示例；本次整理不执行 push：

```bash
git remote add publish YOUR_NEW_REMOTE_URL
git push publish bf16-nn-reproduce-handoff:refs/heads/bf16-nn-reproduce-handoff
```

接收者：

```bash
git clone --branch bf16-nn-reproduce-handoff YOUR_NEW_REMOTE_URL composable_kernel
cd composable_kernel
git rev-parse HEAD
git status --short
```

核对发布者提供的提交号。下面所有命令均在仓库根目录执行。源码可位于任意目录，入口根据脚本位置定位仓库。

## 3. shape 和 BLASLt 怎么使用

使用者自行准备：

```bash
export CK_BF16_SHAPES=/your/path/shapes_1178.csv
export CK_BF16_BLAS=/your/path/hipblaslt-install/lib/libhipblaslt.so
```

### shape 文件

`--shapes` 指向1193条的完整 shape CSV。文件名可以不同，内容须与正式计划一致：ID1～1193，881 decode、312 prefill，至少提供 `ID,M,N,K,G`；M≤256 属于decode。原文件有UTF-8 BOM，读取时会处理。

为防误用其他测试集，当前入口固定核对正式 shape 文件 SHA256：

```text
5c3ae4d1d682b919ed90adb5cfed1cec755e3b519f9003afde8c46d282e77863
```

改写换行符或重新导出CSV也会改变SHA；应使用原始文件，不通过修改校验值代替确认测试集。仅给一个 `--shapes` 即可，不需要准备旧环境中的第二份镜像。

另一个 `--plan` 输入是仓库内的策略表：

```text
docs/data/v7_optimization_p6f/final_results/policy_overlay_incremental_1193.csv
```

shape 文件核对尺寸；策略表另有每条的 `enable_env`、`instance_id`、冻结 `frozen_BLAS_ms`。脚本逐条清理继承的实验开关、配置本条方案，把 M/N/K/G 转成 benchmark 的 `--m/--n/--k/--groups`，通过默认selector选核并核对实例，不强制指定ID。

`--depth8` 按属性把 M1～32、N256、K512、G64 的原六条覆盖为6003，其余1187条沿用策略表。仅启用一个V7开关不足以代表这套逐shape配置。历史 `results.csv` 列结构不同，不作为 `--plan`。

### BLASLt 库

`--blas-lib` 指向可用的 `libhipblaslt.so`，完整安装及其依赖由使用者管理。不要只提供断开的符号链接；需要能找到实际 `.so`、origami依赖、gfx936对应的kernel/library数据。

BLASLt 在本轮用于全输出精度参考。R仍按策略表的冻结BLAS耗时计算：`R_frozen=frozen_BLAS_ms/本轮CK_ms`，精度阶段的临时BLAS时间不会替换该参考，也不等于重新测过Native GroupGEMM基线。

原正式参考的主库SHA256为 `05524e5b1032409d3b4ba3926e26a4e989f2cccb4b867e0f1a2cfaa1e63ed5a5`。新会话会记录使用者实际库及动态依赖的SHA。若要比较历史性能，使用相同GPU、驱动/DTK和库版本；若库有变化，应如实标注。

`inventory.json` 中的绝对路径和旧探针项是原环境的溯源信息，不是要求接收者复制的附件清单。

## 4. 构建正式程序

现有环境使用 Linux x86_64、`/opt/dtk/bin/aicc`（clang18.0.0，HIP输出版本6.2.0-0）、CMake3.29.2，Release，`GPU_TARGETS=gfx936;gfx938`。还需要Python3、g++、ldd、hy-smi、可读的dmesg。复现入口固定物理GPU4/gfx936，门禁要求主机MemAvailable至少64GiB。

```bash
export PATH="/opt/dtk/bin:/opt/dtk/hip/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname "$CK_BF16_BLAS"):/opt/dtk/hip/lib:${LD_LIBRARY_PATH:-}"

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/aicc \
  '-DGPU_TARGETS=gfx936;gfx938' \
  '-DCMAKE_PREFIX_PATH=/opt/dtk;/opt/dtk/llvm;/opt/dtk/hip'

cmake --build build --target \
  tile_example_grouped_gemm_device_args_benchmark \
  tile_example_grouped_gemm_device_args_query_test \
  tile_example_grouped_gemm_device_args_smoke \
  tile_example_grouped_gemm_selector_test \
  tile_example_grouped_gemm_decode_depth8_selector_test \
  -j4

ldd build/bin/tile_example_grouped_gemm_device_args_benchmark
ldd "$CK_BF16_BLAS"
```

确认没有 `not found`。正式benchmark通过 `--blas-lib` 动态加载BLASLt，因此自身ldd不一定列出BLASLt。当前配置使用工具链头文件；如需指定BLASLt安装的头文件，可另设 `-DCK_GROUPED_GEMM_HIPBLASLT_ROOT=/your/path/hipblaslt-install`，并记录该配置差异。

先完成编译，再建立测量会话；测试中不要重新编译或改源码、库文件。新会话会冻结实际二进制、依赖和源码记录，文件变化会停止继续测量。

## 5. 测试步骤

### A. dry-run：CPU核对，不提交GPU

```bash
export CK_BF16_REPLAY_OUT="/workspace/v7_coverage/replay_after_clone_$(date -u +%Y%m%dT%H%M%SZ)_$$"

ulimit -c 0
python3 -B -u tools/grouped_gemm_depth8_reproduce/reproduce.py \
  --plan docs/data/v7_optimization_p6f/final_results/policy_overlay_incremental_1193.csv \
  --shapes "$CK_BF16_SHAPES" --blas-lib "$CK_BF16_BLAS" \
  --output "$CK_BF16_REPLAY_OUT" \
  --depth8 --session-probe --dry-run
```

预期 `PLAN_VERIFIED`、`DRY_RUN_PASS no GPU commands`。它会读取构建后的程序/库、运行ldd，并编译运行CPU选核probe；不会调用hy-smi或GPU，也不会运行恢复对照核。

第一次dry-run建立12小时会话：前10小时可提交GPU，后2小时收尾；resume不重置期限。任务授权若更早截止，应服从更早截止。准备后隔了很久再测，应使用新目录和相应授权，不能重置旧会话。换终端时重新设置同一个输出变量。

### B. 有限先导

以下真正使用GPU4，需要相应测试授权：

```bash
ulimit -c 0
python3 -B -u tools/grouped_gemm_depth8_reproduce/reproduce.py \
  --plan docs/data/v7_optimization_p6f/final_results/policy_overlay_incremental_1193.csv \
  --shapes "$CK_BF16_SHAPES" --blas-lib "$CK_BF16_BLAS" \
  --output "$CK_BF16_REPLAY_OUT" \
  --depth8 --session-probe --resume --pilot-only
```

先在健康门禁下验证本次构建中的V7实例5000：M1/N2048/K768/G4，三次全输出、真实guard/redzone；通过后，记录它作为本会话的恢复探针。无需旧 `bench_cache` 或旧 `jobs.jsonl`。初次对照尚未通过就发生精度或环境异常，会停止，不使用未验证的kernel做恢复。

随后执行selector/depth8-selector/query/smoke回归、M1～32全整数精度、10条边界回退及六组新旧A/B资格，最后测固定12条先导。预期结束标记为 `PILOT_STAGE_PASS`。额外资格/恢复对照不会计入正式1193条结果。

只做资格验证可把 `--pilot-only` 换成 `--qualification-only`。该步骤也会先建立本会话恢复证据。

### C. 之后需要全量时

本次整理不执行全量。将来获准复测全部1193条时，沿用本次目录：

```bash
ulimit -c 0
python3 -B -u tools/grouped_gemm_depth8_reproduce/reproduce.py \
  --plan docs/data/v7_optimization_p6f/final_results/policy_overlay_incremental_1193.csv \
  --shapes "$CK_BF16_SHAPES" --blas-lib "$CK_BF16_BLAS" \
  --output "$CK_BF16_REPLAY_OUT" \
  --depth8 --session-probe --resume
```

它继续未完成的条目；每条先全输出精度，再按warmup10/samples30/单launch/include_args_build0跑三轮，取三轮中位数的中位数。精度及计时必须来自同计划、二进制、配置、物理卡和epoch；不选最快attempt，不用历史值补缺项。

### GPU4与MKFD规则

- `HIP_VISIBLE_DEVICES`、`CUDA_VISIBLE_DEVICES`均固定4；进程内逻辑设备0仍是物理卡4。
- 使用全机共享锁 `/workspace/v7_coverage/p6b/run.lock`，自动创建父目录，需有写权限。所有GPU4作业沿用同锁，不并发起裸benchmark。
- 每条作业前后核对hy-smi与可读dmesg。卡忙不抢占，异常不换卡绕过。
- 恢复必须先有本会话真实精度证据，且核对原始日志、作业记录、二进制/依赖SHA；最多3次有间隔的健康观察，连续两次通过才在独立进程执行已验证对照。
- 恢复成功推进epoch，整条精度加三轮重做；恢复累计90分钟、探针失败、会话到期或永久stop则停止，不删除标记强行续跑。

不传新路径/`--session-probe`时仍保留旧调用行为，供历史记录兼容；新交付使用上面的完整命令。

## 6. 查看正式口径结果

输出位于 `$CK_BF16_REPLAY_OUT`，历史仓库结果不改写：

|文件|内容|
|---|---|
|`results.csv`、`results.json`|本轮每shape的CK ms、冻结BLAS ms、R、实际实例、配置、轮次/CV和日志来源。|
|`summary.json`|完成数、缺失、互斥分桶、高CV、恢复信息；全量成功应complete=true、accepted_shapes=1193、缺失为空。|
|`qualification_results.jsonl`|单独保存资格和六条A/B。|
|`probe_validation.json`|本会话恢复对照的真实验证证据，不是计时成绩。|
|`bin_crossings.json`、`all_regressions.json`|相对输入历史表的跨桶、耗时回退>2%条目；完整结果保留所有其他变化。|
|`all_failures.json`、`jobs.jsonl`、`jobs/`|失败和实际命令、环境、原始日志。|
|`frozen_plan.json`、`session.json`|输入路径、签名、计划与期限；独立恢复进程从冻结计划恢复使用者的库/shape路径。|

历史正式结果在 `docs/data/decode_depth8_replay_p6k/results.csv`，分桶如下：

|阶段|总数|R≥0.99|0.8≤R<0.99|0.5≤R<0.8|R<0.5|
|---|---:|---:|---:|---:|---:|
|prefill|312|292|20|0|0|
|decode|881|604|263|14|0|

R≥0.99是接近BLAS的分桶，严格达到BLAS为R≥1；TFLOPS=`2*M*N*K*G/(ms*1e9)`，包含G。冻结BLAS中ID1～1178来自物理卡0，1179～1193来自卡4，本轮CK均卡4。新测量应保留该来源说明。

同口径复测会生成新成绩，不保证每个毫秒值或临界桶逐项相同。高CV和约±2%的变化保留不确定标记。当前源码包含后续未采用实验，原正式二进制SHA为 `76642f45892426cb97e758306f780d22c95567b38f9da475105aa38a0456b5ae`；新构建不宣称与它位级一致。

只重建自己的新输出目录报告，不新增GPU测量：

```bash
python3 -B -u tools/grouped_gemm_depth8_reproduce/reproduce.py \
  --output "$CK_BF16_REPLAY_OUT" --report-only
```

这会写回该输出目录。不要对只有归档结果、没有现场jobs.jsonl的仓库历史文档目录执行report-only；历史成绩直接读取results.csv/summary.json。

## 7. 验证覆盖

本次对入口做CPU与模拟进程验证，覆盖输入路径、dry-run不发GPU、恢复证据缺失/篡改拒绝、恢复进程配置传递、完整精度与三轮、实例核对、截止和MKFD门禁。接收者构建后可运行相同CPU测试：

```bash
CK_BF16_TEST_SHAPES="$CK_BF16_SHAPES" CK_BF16_TEST_BLAS_LIB="$CK_BF16_BLAS" \
  python3 -B -m unittest discover -s tools/grouped_gemm_depth8_reproduce -v
```

fake进程仅为CPU测试夹具，不产生GPU成绩。31项CPU测试通过，并从干净本地clone核对了19个原文件的签名、1193条CPU选核和带显式输入路径的dry-run。克隆核对仅复用已有build/bin作文件签名和ldd检查，没有进行完整C++重建。

具体记录见 [validation.json](validation.json) 和 [干净克隆核对](clean_clone_validation.json)。本次没有执行新GPU先导或全量；接收者仍应记录实际构建、先导及需要时的全量结果。
