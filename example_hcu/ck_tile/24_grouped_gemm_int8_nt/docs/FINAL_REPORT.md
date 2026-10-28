# INT8 NT Grouped GEMM 正式验收记录

状态：TARGET_MET。1178条完整shape，数值通过1178条，R>=0.5为1178条。无显存跳过或缺失指标。

口径：INT8×INT8→INT32、无scale，NT，物理卡7；保持独立专家权重和完整M/N/K/E。公开默认select/query/run；BLAS使用指定安装包及已tuning的最快有效算法，同轮重新计时。

计时：正式一轮；每样本预热10次、计时50次，共3个HIP event样本取中位数。

| phase | shape数 | 数值通过 | R>=0.5 | R中位数 |
|---|---:|---:|---:|---:|
| prefill | 306 | 306 | 306 | 0.6115 |
| decode | 872 | 872 | 872 | 1.1447 |

最小R=0.508321，总体R中位数=1.0209。R=BLAS ms/CK ms，TOPS=2×E×M×N×K/(ms×10^9)。

正确性：每组全输出与独立精确可表达的FP32整数参考比较；输入范围[-8,8]，检查完整输出guard及输入抽样。完整范围[-128,127]、ragged/empty、stride/tail、非默认stream和错误参数由另存的FP64边界回归覆盖；这些验证结论不代表量化INT8输出或带scale路径已实现。

文件：results.csv、records.jsonl、manifest.json、audit.json；二进制和源码检查点在binaries/及source_checkpoint/。R未达标的shape会保留完整数值和性能记录，不从分母删除。

统一C++入口全1178条数值回归完成，1178 PASS、0失败；CPU精确抽样、全量CK/BLAS输出与guard检查通过。time=0证据仅作数值验证，不作为正式性能。两入口CK机器码bundle一致。
