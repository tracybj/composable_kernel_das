# 2026-11-04 shape 扩展

按用户要求追加 G=64、N=256、K=512，M 按 2 倍递增取 1…16384，包含 512/1024，共15条（ID1179…1193）。M≤256 的9条为decode，其余6条为prefill。EP沿用原K512/G64家族的4。

已同步更新：
- `/workspace/shapes_1178.csv`
- `/workspace/Grouped_Geem/blaslt_tuning_tools/shapes_1178.csv`

保留原文件名供现有调用方读取；当前实际1193条（881 decode、312 prefill）。原1178条和文件前36369字节完整保留，新ID连续无重复，两份CSV逐字节一致。只追加shape，未测性能；历史结果仍对应原1178条。

本目录CSV保存本次新增15条，validation.json记录主文件路径、取值、条数和前后SHA256。修改前原文件备份保存在 `/workspace/v7_coverage/shape_extensions/20261104_n256_k512_g64/shapes_1178_before.csv`。
