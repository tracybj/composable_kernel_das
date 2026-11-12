# 协作A/B与三组操作数：代表集局部收益，未整域采用

85对发现全部完成，控制与候选均为同一build6正式benchmark/default selector、GPU4/epoch1。每臂全输出三次及真实guard，然后三轮AB/BA，warmup10/samples30/单launch/include_args_build0。全部回退和高CV保留，不混用不同构建的逐shape赢家，未做七block确认。

三个协作A/B配置K16D4、K16D8、K32D4的八代表GM candidate/base为1.115065、0.991590、0.942367。K32D4有6条改善>2%、2条回退>2%、1条高CV，5条点估计改善≥5%；没有证明完整属性域可采用，不计入50稳定/30可归因跨桶。短线三配置GM均>1。四组同GPU代码direct/legacy入口控制也未证明广泛host收益。详细正负结果见h3_discovery_analysis.csv/json。

H3独立实现使用完整CTA M所有权、协作A/B共享LDS、三组操作数读t+2/计算t，以及Global预取t+D+1。M≤32为1M×4N wave，M64为2M×2N；并非A寄存器/B私有LDS。共享LDS不是V7独有架构，独立代码是否有效须以当前最佳CK对照验证。旧5003实际64×128×16、5008实际64×128×32，不能把small_m128误解为M128，也不能把M64收益全归因去除M256填充。

18份有效SQTT组成9对，第5批包路径/大小/SHA见profile_delivery_batch5.json；包含固定收益、短线及回退、四组相同GPU代码的host路线控制。所有原名和sidecar保持。未在本机运行Windows XCompute，不能宣称GUI已解码。LDS冲突约.25～.28接近对照、HBM约一份输入，不支持B大规模重复的根因解释；两stage提前LDS读通过真实循环CFG及寄存器依赖静态核对，静态顺序不证明实际重叠。profiler时长和不同wave累计Latency不是正式成绩。

首次资格在任何H3 GPU启动前因sandbox无权读dmesg停止。原stop和manifest保留，具权限独立新进程依原门禁恢复：两次健康间隔60.196秒、固定SHA5000全输出探针PASS、epoch0→1、恢复445.379秒全部计入期限，无驱动重启/切卡/门禁绕过。build5另有query变量shadowing普通编译失败，保留结果；仅host变量改名后build6通过，不称MKFD。

短76原SQTT稀疏及后续两个采集位置稀疏均保留；只依登记的CU索引0/1/2顺序成对重采，不改shape/kernel/正式85对或4096token门槛。23个profile作业中18份成对有效、3份稀疏、2份原本完整但被成对重采替代；原数据仍在现场。全部作业ok表示进程/门禁有效，不表示每个SQTT都通过完整性门槛。

raw_jobs.tar.gz归档786物理作业、相关门禁/命令/脚本/manifest以及权限和稀疏事件，逐成员SHA核对；perf、二进制、厂商ISA不入Git。compiled_sources保留build6实际输入，与source_sha256逐字核对，包含已记录的初始用户overlay；源代码自有增量提交5e86b6e，初始用户修改未代为提交。原1193/冻结BLAS/旧Excel未改，无全量GPU复测。后续有限深度/host因素实验及其成绩不属于本归档。
