# HCU ck_tile 分组 GEMM

本目录是 HCU 侧 ck_tile 分组 GEMM 路径，和 legacy CK
`DeviceGroupedGemm_*` library/factory 路径不同，也和
`example_hcu/ck_tile/17_fused_moe` 下的 MoE 专用 group GEMM 不同。

## 关注的目标

当前 FP16/BF16 最小 smoke 使用已有 fast-build target：

```bash
cmake --build . --target tile_example_grouped_gemm_fp16_fast -j 16
cmake --build . --target tile_example_grouped_gemm_bf16_fast -j 16
```

显式 smoke target 会构建上述两个 target，并运行一个 2 组 grouped GEMM
correctness 检查：

```bash
cmake --build . --target tile_example_grouped_gemm_fp16_bf16_fast_smoke -j 16
```

等价的直接运行命令：

```bash
./bin/tile_example_grouped_gemm_fp16_fast -prec=fp16 -a_layout=R -b_layout=C -group_count=2 -Ms=128,256 -Ns=128,128 -Ks=128,128 -stride_As=128,128 -stride_Bs=128,128 -stride_Cs=128,128 -validate=1 -warmup=1 -repeat=1
./bin/tile_example_grouped_gemm_bf16_fast -prec=bf16 -a_layout=R -b_layout=C -group_count=2 -Ms=128,256 -Ns=128,128 -Ks=128,128 -stride_As=128,128 -stride_Bs=128,128 -stride_Cs=128,128 -validate=1 -warmup=1 -repeat=1
```

也可以通过 CTest 单独运行这两条 smoke：

```bash
ctest -R 'tile_example_grouped_gemm_.*_fast_smoke' --output-on-failure
```

## 当前边界

- 最小已验证 layout 是 A 行主序、B 列主序、C 行主序。
- smoke 有意使用从 M/N/K 128 起步的对齐 shape。
- 现有 FP16/BF16 fast target 使用 CompV4 persistent double-buffer 路径；
  当前要求 `num_loop = K / K_Tile >= 2`。
- FP16/BF16 的 `K_Tile=64`，因此当前 fast example 的 K 维 baseline
  应满足 `K>=128`。
- `num_loop < 2` 的小 K shape 已在 host-side launch/support check 中显式
  拒绝；若要真正支持，需要补充 small-K fallback。
- 本目标不注册 standard grouped GEMM factory/profiler instance，也不覆盖
  grouped GEMM MultiABD。

## `num_loop >= 2` 限制

`tile_example_grouped_gemm_fp16_fast` 对应的配置是
`GemmConfigComputeV4<fp16>`：`M_Tile=64`、`N_Tile=128`、`K_Tile=64`、
`DoubleSmemBuffer=true`、`Persistent=true`。BF16 fast target 同样使用
`K_Tile=64`。

CompV4 pipeline 的 `PrefetchStages=2`，tail 选择逻辑会把奇数 loop 映射到
`TailNumber::Three`。当前 fast example 以至少 2 个 K tile 作为支持前提：

```text
num_loop = K / K_Tile >= 2
```

对 FP16/BF16 fast target 来说，这等价于：

```text
K >= 2 * 64 = 128
```

因此 README 中的 correctness smoke 只选用 `K>=128` 的 shape。当前 fast path
会在 host 侧拒绝 `num_loop < 2` 的 group；后续如果要把 `K<128` 纳入支持范围，
应先增加专门的小 K 路径。
