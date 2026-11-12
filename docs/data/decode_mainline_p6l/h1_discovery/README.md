# H1有限发现：全部候选回退，未采用

当前最佳可启用CK作为同二进制对照。物理GPU4串行、统一锁/MKFD门禁，epoch0；非1193全量。正式benchmark/default selector、每臂三次全输出及真实guard、warmup10/samples30/单launch/include_args_build0，各三轮AB/BA交替，汇总轮中位数的中位数。

48组包括24中长K、18短K、6域外控制；所有24中长K和18短K候选均回退。几何平均candidate/base：K16 D4 operand2为1.283545、D8 operand2为1.213590、D4 serial为1.287495；短N64为1.984372、N32为2.997108。高CV、全部小回退及控制结果均保留。三轮发现不等于七block稳定/归因验收，不扩围该结构，不计入50/30目标。

H1为完整M/CTA、A寄存器/B波私有LDS，具有独立global环及操作数预读。资源和MMAC有效率改善未转化为耗时改善。ID621/805的候选VMEM/wave约为644/1284，对照270/526；等待计数占wavecycle约77%/82%，对照14%/17%。ID1072双方HBM均约一份输入，不能认定B大量重复是根因。PMC重放缓存和累计wave时钟局限保留；未在本机运行Windows XCompute。

四组配对SQTT下载包及SHA见profile_delivery_batch2.json，原名/依赖保留，Source主循环PC与Wavefronts说明位于现场delivery/batch2_h1_current_candidate/XCOMPUTE_README.md。perf/厂商ISA/二进制不提交Git，profile时长不作正式成绩。

raw_jobs.tar.gz含本阶段440个实际作业及门禁/服务日志，所有命令、环境、原始精度/计时和身份可审计；包含一个控制器结束前已完成但配对未完成的baseline精度作业。该组后来从精度起整组重跑，前一个作业保留但排除配对成绩。build1 query变量shadowing失败与修复记录保留在原现场，build2全目标成功；旧1574 shader不变，H1新增15个gfx936 shader无spill/scratch/barrier。

下一结构H2已在GPU实验前冻结h2_hypothesis.json，协作共享A并比较K16/K32及操作数预读。该文件只是设计，不代表GPU精度/性能已验证。原截止、旧队列暂停和用户修改保护不变。

本轮自有源码本地提交：b8f4e7f027a12cb9572fe9c4ce25df68845fefc6（实验用，不合入，未push）。提交包含H1与尚待GPU验证的H2。独立index剥离全部初始P6i/readfirstlane修改，纯提交树CPU647/Depth8 1063及原selector通过、C API host语法通过；提交后原9文件用户增删行精确保持，仅两处registry计数作显式归一。二进制来自含保留用户overlay的工作区，source_sha256.json记录实际编译字节，不能把纯Git树SHA冒充测量工作区。
