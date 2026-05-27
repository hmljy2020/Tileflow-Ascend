# TileFlow/Timeloop 流水近似说明

考虑一个最小例子：

```text
for k_tile:
    C_tile = A_tile @ B_tile
    bypass C through L0/L1
    store C to GM
```

假设每个 tile 的代价是：

```text
计算 C: 100 cycles
写 C 到 GM: 180 cycles
```

TileFlow 会先根据 bypass 判断哪些层级需要统计 `C`：

```text
L0 bypass C   -> L0 不统计 C update
L1 bypass C   -> L1 不统计 C update
GM 不 bypass  -> GM 统计 C update/write
```

然后 Timeloop 风格的 buffer evaluation 会用平均带宽压力估计这一层时间：

```text
compute_cycles = 100 * tile_count
write_time     = C_total / GM_write_bandwidth
final_cycles   ~= max(compute_cycles, write_time)
```

如果有 10 个 tile：

```text
compute_cycles = 1000
write_time     = 1800
final_cycles   ~= 1800
```

这个结果表达的是：GM 写带宽成为整体吞吐瓶颈。

真实硬件的时序可能不同。如果矩阵乘单元只有一个输出寄存器或 accumulator
slot，那么当前 `C_tile` 没有 drain 出去之前，下一块 matmul 不能开始：

```text
real_time = 10 * (100 + 180) = 2800
```

如果存在输出 double buffer，当前 tile 的 drain 可以和下一块 tile 的 compute
重叠：

```text
real_time = 100 + 180 + 9 * max(100, 180) = 1900
```

总结：

```text
TileFlow/Timeloop 估计的是平均吞吐瓶颈：
    max(total_compute, total_write)

真实硬件还可能需要逐 tile 的时序：
    warmup + drain + (N - 1) * max(per_tile_compute, per_tile_write)

如果没有输出 buffering，更保守的模型可能是：
    N * (compute + write)
```

因此，TileFlow/Timeloop 可以反映 `C` bypass 到 GM 后 GM 写带宽成为瓶颈，
但不会精确建模每个 tile 的 accumulator 占用、输出 drain backpressure，
以及 warmup/drain 的启动和排空开销。
