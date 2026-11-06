

### P6f decode流水架构图整理（2026-11-06；仅CPU文档）

用户明确要求按P6f最终per_shape_results.csv的优化状态绘制类似“主循环＋收尾”的架构图，并确认采用该图型。本次源码固定c39e6df，未纳入后续P6h实例6000。生成两张中文PNG、可编辑SVG、两页PDF和说明：docs/data/p6f_decode_pipeline_diagrams/。

- 图1以小M K16/K32 N128为例，区分A/B→CTA共享LDS→3组operand VGPR→MMAC、4槽LDS/12相位、两组wave指令次序、必要wait/barrier、旧尾段/drain/P6f steady与输出LDS排列。写回只有入口vmcnt0+CTA barrier，后续wave私有输出片仅LGKM等待；没有K分片partial C归约。
- 图2列出原V7、K16/K32、square、N尾块128/256的实际tile/wave/LDS/MMAC数量及P6f采用状态，区分K流水排空与N列尾块。图1wait数值不套到原V7或N64等其他几何。
- 重新核对最终CSV：288组/209shape，其中decode235组/191个不同shape；没有将其描述为881条decode全量实测。facts.json保留冻结源码与CSV SHA，图内保留精度通过不等于性能提升的采用边界。
- 两图已实际查看；SVG XML、PNG尺寸/文字边界/字号、PDF两页、相对链接、14个冻结源码SHA与CSV SHA均通过检查。只用CPU和既有记录，无新GPU/编译/测试、无选核或kernel改动。用户readfirstlane diff MD5仍4c7f49a5c7592767d53195a713db6b8a；暂停P6g tools保留。
