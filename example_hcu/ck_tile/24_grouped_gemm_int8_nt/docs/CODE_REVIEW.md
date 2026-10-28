# 实现约束复查

本次范围为gfx936、NT、有符号INT8输入和INT32输出、无scale。以下结论针对新增代码和已有公开分流的修改。

| 项目 | 实现与证据 |
|---|---|
| 数值语义 | 使用CK已有I8/I8/I32 MMAC attribute。无中间浮点转换、量化、split-K或原子累加。K上限131056保证完整INT8范围内数学部分和不溢出INT32。 |
| 计算所有权 | 每个CTA只写一组中的一个M/N输出tile；tile映射一一对应。分组调度末尾多出的CTA在访问前退出。 |
| 对齐与尾部 | A/B以16字节搬运，支持K16尾部；C以4个INT32向量写回。无mask实例必须由common尺寸、无ragged/empty标志及完整tile对齐共同保护。 |
| LDS生命周期 | 当前K tile完成读取后同步，再覆盖单缓冲；写入下一tile后同步。最后计算与epilogue复用LDS前另有同步。编译调度屏障不替代硬件同步。 |
| 输出重排 | 16行条带或合并wave条带；后者使用保持向量对齐的可逆列XOR。全输出、行padding及组前后guard回归覆盖。 |
| 大地址 | 全局行偏移使用64位乘法；统一C++数据初始化、比较和分配使用64位元素数，避免大输出计数溢出。 |
| 输入与stream | 输入只有读取路径；GMEM store目标为输出。设备参数仅在GPU读取。公开run沿用调用方stream，不分配、不回读、不做device同步。benchmark的setup/teardown及event同步位于计时契约之外。 |
| 错误与兼容 | 保留v1结构体及旧enum数值；新dtype、实例和reason兼容扩展。expected-instance错误在launch前拒绝。非零num_cu等尚未覆盖请求明确返回unsupported。 |
| 默认集成 | 实例由provider清单和CMake编译，公开selector按算子属性选择。未使用模型名、case ID或phase标签选核。 |
| 代码生成 | 独立构建与完整CK grouped-GEMM代码生成选项对齐。14个最终内核private segment均为0；这不等价于已测得occupancy。 |
| 统一入口 | 原C++ benchmark支持INT8，原CSV runner支持repeat、算法序号、TOPS和time=0验证。原615行FP16/BF16命令构造逐条相同；统一入口与正式库的CK GPU bundle SHA一致。 |

独立正确性与性能证据分别保存在全量记录、完整范围FP64边界测试、统一C++回归以及资源文件中。数值PASS、相对性能R达标和不同dtype/量化支持是不同结论。FP16/BF16的本次回归为原80条host selector检查，未声称做过其全量GPU数值/性能回归。

调度收益由受控计时和完整数值比较支持；缓存局部性解释属于结构假设，没有将未采集的硬件计数写作已测结论。正式达标情况以最终audit.json为准。
