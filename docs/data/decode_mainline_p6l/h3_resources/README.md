# H3两种资源因素：全部发现与SQTT证据

在同一正式二进制eb02cd5d…及epoch1，完成25项资格验证、46个预登记三轮配对和5组成对SQTT。所有精度和真实guard/redzone检查通过；全阶段345次全输出比对、失败0。原1193表和冻结BLAS未改。

四wave A2/B3在19条代表上GM candidate/base=1.018896，8条改善>2%、6条回退>2%、1条高CV；八wave D4在14条代表上GM=1.032618，6条改善>2%、8条回退>2%。两组集合不同，不能直接用这两个GM比较结构优劣；按相同14条比较的选择只用于profile。三轮发现不是七block稳定验收。全域采用不通过，Goal计数仍为0/0，未绘制新heldout或回填局部赢家。

对照证明缩小LDS可以恢复平均驻留wave，但不能独立保证耗时改善。M64四wave实际VGPR133（硬件分配显示136），仍明显高于旧小M对照80；待研究计算与读取的次序。PMC回放改变cache状态，HBM是全dispatch流量；wait和wave累计时钟不能直接相加当墙钟。

第8批perf下载包路径/大小/SHA见profile_delivery_batch8.json。10份有效SQTT全部保留，ID633在CU0/1两份稀疏尝试亦保留，只按原CU0/1/2规则成对回退，最终CU2有效。Windows XCompute未本地打开。此处只提交小型计数/ISA摘要，不提交perf、二进制或厂商ISA。

raw_logs.tar.gz包括405作业/1215原始文件，逐成员SHA验证；source_and_binary_index.json指向包含用户原有改动的精确会话源码/二进制快照，不代为提交用户源码。自有实现本地提交b78c07bf；本归档实验用，不合入，不push。下一轮代码和数据另存，原失败证据保持不改。
