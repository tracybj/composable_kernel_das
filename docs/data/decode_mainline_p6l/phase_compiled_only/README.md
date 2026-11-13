# 提前计算MMAC调度原型：仅编译和静态验证

**未做GPU query/smoke、精度、性能或SQTT；未采用，不计入50/30。** 6026/6027是H3协作LDS的一个调度因素，不是第四种所有权结构。新开关默认关闭，已有实例语义保留。

同样的五个几何比较两种指令次序后，CPU预登记规则选择所有wave先计算当前MMAC，再执行原有等待/barrier、读取未来操作数并预取。双目标探针均无spill/scratch；gfx936 VGPR总和493→467，wide M32/M64为83→73、128→112，compact仍63/86/133。三bank CPU模型和实际CFG审计只证明所检查的静态依赖，不能代替GPU正确性或时间收益。

六目标正式CPU构建于2026-11-12 17:57:39 UTC结束。当前benchmark SHA见compile/build9/completed.json；1710个既有shader机器码保持，10个新shader无spill/scratch，5个gfx936循环依赖通过只读检查。纯提交树2663个新selector、1063个Depth8及原回归通过；89493个policy场景/83521比较、18432个公开禁用位场景通过。所有测试和编译都在原20h截止前完成；截止后只读审计。

原1193配置的CPU选核未变，不是1193 GPU复测。源码自有部分本地提交5f4da600822a38fc2f8db8f0c13d230c9005d408，保留初始用户overlay；精确22份实际编译输入只在outside_evidence.json指定现场保存，未再次代为提交用户源码。旧compact收尾注释before computing与compute-first分支不符：实际最后MMAC先于ready(0)，排空后Store C；源码保持构建时字节，详见实际CFG审计。

phase_discovery_plan.json及计划脚本仅为未执行证据：27项资格、46配对和profile9均未运行。脚本链没有GPU端到端通过记录，不能宣称一键验证已跑通。原截止和gpu_stop仍有效，后续实验需用户另行授权的新时间窗口、重新检查入口并冻结输入；不能运行旧session绕过门禁。
