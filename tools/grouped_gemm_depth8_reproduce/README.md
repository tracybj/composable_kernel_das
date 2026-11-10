# 按表复现 BF16 NN Grouped GEMM，并覆盖八级独立 Decode 候选

入口读取1193条历史策略表，清理继承的实验开关，逐条配置正式benchmark，核对默认selector的实际实例，先全输出精度验证，再测三轮单launch性能。结果中的CK耗时全部来自当前执行；BLAS耗时保持输入表的冻结Native GroupedGemm值及原来源。

`--depth8`按属性将gfx936、BF16 NN、common非空非ragged、M1～32/N256/K512/G64覆盖为实例6003。正式1193表仅命中原有六条，不增加26条合成shape。原表、历史CK耗时及其余1187条配置保持只读。实验开关默认关闭；此入口是按表配置的复测，不代表全局生产默认策略。

## 运行

在已获授权的任务中，先构建正式benchmark及selector、depth8 selector、query、smoke目标，再运行：

```bash
python3 -B -u tools/grouped_gemm_depth8_reproduce/reproduce.py \
  --plan docs/data/v7_optimization_p6f/final_results/policy_overlay_incremental_1193.csv \
  --output /workspace/v7_coverage/p6k_depth8_replay/NEW_AUTHORIZED_RUN \
  --depth8 --dry-run
```

对该输出目录依次使用同样的`--plan`和`--depth8`参数，添加`--resume --pilot-only`执行专项验收与先导样本；然后只保留`--resume`继续全部1193条。第一条dry-run即记录本次执行的12小时总期限，后续不会重置；若任务已预先写入session.json，则首次也需`--resume`。

一条命令也可以省略`--dry-run`，自动完成CPU核对、回归、专项验收、先导和全部1193条。全量之前先冻结计划和二进制；`--report-only --output DIR`只重新生成已有数据的报告。旧目录续跑与新的授权复测使用不同目录，不能覆盖历史结果或重置过期期限。

## 契约与验收

- 新实例源于已归档的A寄存器/B私有LDS八级实现，不使用当前6000的A-LDS/B寄存器实现；四wave各自完成完整K。
- 验证域为packed dense NN：lda=K、ldb=N、ldc=N，设备分配及向量访问满足对齐和范围要求。v1的host problem不包含设备kargs中的指针与stride；selector不回读它们。新开关的调用方须满足该既有存储契约；不宣称支持未验收的任意stride/地址。
- 全量前专项检查M=1～32全部整数的三次全输出、十条域外GPU回退、六条同二进制新旧配置三轮A/B。架构/精度/转置、空组/ragged/device length等拒绝条件另由CPU selector测试覆盖。
- 每条全量shape至少一次CK对逐组BLAS全输出及两侧真实guard；六条新实例重复三次。三轮计时统一warmup10/samples30/每样本单launch/include_args_build0，取三轮中位数的中位数。
- 正式入口固定`build/bin/tile_example_grouped_gemm_device_args_benchmark`，不强制指定实例。新开关与禁用开关由白名单和CPU probe共同解释；禁用优先。
- GPU仅物理卡4，统一锁串行；每个GPU作业前fresh hy-smi、空闲/内存/MKFD检查及可读内核日志。异常后独立进程有界恢复，整条跨epoch重做；卡忙不抢占，日志不可读停止提交，不换卡绕过故障。
- 第10小时停止新GPU提交，最后两小时审计与收尾；编译、等待和恢复计时，恢复相关等待累计90分钟停止GPU。超时不等于完成。
- `results.csv/json`只包含同计划、二进制、配置、物理卡和epoch内精度及三轮均完成的行；专项A/B不会混入全量结果。续跑选择最后一个完整有效attempt，禁止选最快attempt。
- `summary.json`列出完整数量、缺失、互斥分桶、高CV和恢复；`bin_crossings.json`保留所有跨桶，`all_regressions.json`保留回退，`all_failures.json`保留失败。
- 高CV、门槛附近和±2%变化标记不确定。预测611/256/14不是必须达到的成绩；未测项不会由历史补齐。

CPU故障注入与断点测试：`python3 -B -m unittest discover -s tools/grouped_gemm_depth8_reproduce -v`。测试通过不替代实际GPU精度与性能证据。
