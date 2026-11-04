# P6c 小 M 增量分桶评估

完整结论、互斥分桶和新增15条明细见 [report.md](report.md)。本阶段只测615个不同CK shape，不做1193全量；原578条沿用11-03。主方案模拟M≤64启用Tn128（5003），默认开关未提升。报告完整保留回退，未逐条挑选快者。

`measured.csv/JSON`保存618个原型对照（原600条+新增9条各两种tile），其中607条纳入主策略。`merged1193.csv`逐条记录baseline/after、R、实例和来源。新增6条prefill只有原路径基线，标为current_new_baseline；其余实测标为current_AB，历史沿用标为11-03_carried。`buckets.csv`保留所有对照表。

原1178的BLAS来自卡0冻结数据，新增15条是卡4Native GroupedGemm按原协议补测，因此扩展1193表含不同BLAS来源，属于参考拼接表。CK前后A/B均卡4；新增M128/256的原型对照不加入主策略。

`jobs.jsonl.gz`和`gate.jsonl.gz`为完整逐条命令、结果和门禁的无损压缩。原始logs、Native候选数据及冻结依赖留在 `/workspace/v7_coverage/p6c/` 和manifest列出的现场路径。脚本是现场环境的审计副本，重放需对应的旧/新benchmark与BLAS依赖。Native配置继承冻结基线协议模板，实际执行时间以jobs及每case的finished_at为准。

分析脚本严格要求3759项成功（accuracy18/newblas15/perf3726），核对分桶和、截图11-03原值及来源。`post_audit.json`确认数据/二进制哈希不变、readfirstlane差异仍保留。报告中只有本次新增decode的54次比较称为CK全输出验证；其他CK计时为CPU抽样检查。
