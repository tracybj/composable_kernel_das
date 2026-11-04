# P6c：增量评估小 M 原型与分桶

预先固定实验策略：gfx936 BF16 NN、common非ragged/empty组、M≤64、K64…4096且K%16=0、N%128=0、G≤1024时评估5003（64×128×16）；其余保留原路径。代码与默认开关不变，本阶段仅实验评估，未提升默认selector。该策略根据P6b的M128/256回退与N64回退事实固定，不按本次每条实测选快者，不做shape白名单。

原集合中600条符合条件；另纳入新增15条基线，其中7条M≤64进入主策略，M128/256两条额外评估5003/5004但不并入主策略，6条prefill仅建立旧路径基线。因此CK只测615个不同shape，原578条沿用11-03数据，未跑1193全量。

执行：卡4串行，每条GPU命令前hy-smi空闲门禁，MKFD/无卡行最多3次60秒重试，忙卡立即停。新decode9条×两个tile×3次全输出BLAS比对；600原shape及新增9decode三轮交替base→128，新增9再单列base→64，新增6prefill三轮仅base。正式CK计时由device_args_benchmark、默认selector、warmup10/samples30/单launch产生，每次CPU抽样及guard检查。此前P6b的162次全输出验证仍适用于相同二进制；本阶段没有将600条抽样检查说成全输出检查。

BLAS：原1178沿用冻结卡0 GroupedGemm算法及性能；仅新增15条用现有native GroupedGemm桥接库在卡4补基线。workspace32MiB、heuristic64、screen3×10×3、top3、refine/formal warmup5 repeat20 samples3，输入/输出/guard按现有调优工具验证，与历史协议相同、卡号不同。公开benchmark的BLAS路径是逐组matmul参考，仅用于正确性；不把该路径的计时当作native GroupedGemm基线。

统计：R=BLAS毫秒/CK毫秒，互斥区间[0.99,∞)、[0.8,0.99)、[0.5,0.8)、[0.3,0.5)、[0.2,0.3)、<0.2。复现截图两次原1178统计，再分别输出原1178与扩展1193的本次对照/优化增量拼接表；新增15也单列。相同范围本次base/after三轮中位数用于区分真实实验变化与历史计时漂移，未测范围不生成新数据。扩展1193包含旧卡0及新增卡4 BLAS基线，必须注明来源混合。
