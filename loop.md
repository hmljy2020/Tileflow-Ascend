# Loop 语义说明

这个目录里的 `map.yaml` 是面向后续 TileFlow-Ascend 后端的语义初稿。
它尽量靠近 TileFlow 原生 `Tile / Scope / Op / factors` 的写法，但仍然包含 `receive_tile`
这类当前 TileFlow parser 不支持的扩展字段。

## 当前计算

`prob.yaml` 描述的问题是：

```text
P[M, N] = Q[M, D] @ K[D, N]
S[M, N] = Exp(P[M, N])
```

当前全局问题规模是：

```text
M = 4096
N = 4096
D = 64
```

`map.yaml` 顶层 GM tile 是：

```text
M=2048 N=128 D=64
```

因此从全局问题看，外层至少会形成：

```text
M 方向: 4096 / 2048 = 2 个 GM tile
N 方向: 4096 / 128  = 32 个 GM tile
D 方向: 64 / 64     = 1 个 GM tile
```

也就是一共 `2 * 32 * 1 = 64` 个 GM 级 tile。每个 GM tile 内部再进入 `CoreGroup`，
然后按 `Pipeline` 顺序执行 Cube 分支和 Vec 分支。

## Tile 字段规则

当前 map 主要使用两个字段：

```text
factors
receive_tile
```

`factors` 在这里按“本层向下一层传递的 tile 大小”理解。比如：

```yaml
target: GM
factors: M=2048 N=128 D=64
```

表示 GM 每次向 CoreGroup 提供一个 `2048x128x64` 的计算块。

`receive_tile` 表示本层每次从父层接收并处理的 tile 大小。最新 map 里只有 L1 显式写了：

```yaml
target: L1
receive_tile: M=256 N=128 D=64
factors: M=256 N=128 D=64
```

这里 `receive_tile == factors`，所以 L1 本身不再额外生成“把大 tile 切成小 tile”的内部循环。
它每次就是接收一个 `256x128x64` 的 Cube 计算块，并把同样大小的块传给 L0。

## 最新 Map 的结构

当前 `map.yaml` 的核心结构可以抽象成：

```text
GM temporal tile: M=2048 N=128 D=64
  CoreGroup spatial tile: M=2048 N=128 D=64
    Scope Pipeline
      L1 temporal tile: M=256 N=128 D=64
        L0 temporal tile: M=256 N=128 D=64
          Op MatmulP

      VecLaneGroup spatial tile: M=1024 N=128
        UB temporal tile: M=1024 N=128
          Op ExpP
```

注意：`arch.yaml` 里硬件末端有 `Cube` 和 `Vec`，但当前 map 没有再显式写：

```text
L0 -> Cube -> Op
UB -> Vec  -> Op
```

而是直接写成：

```text
L0 -> Op MatmulP
UB -> Op ExpP
```

所以这版 map 的含义是：`MatmulP` 发生在 L0 下方对应的 Cube compute endpoint，
`ExpP` 发生在 UB 下方对应的 Vec compute endpoint。以后如果后端需要严格匹配 arch 树，
可以再把 `Cube` 和 `Vec` 作为显式 Tile 节点补回来。

## 生成的 Loop 草图

按照当前 `prob.yaml` 和 `map.yaml`，后端可以生成类似下面的逻辑：

```python
for m0 in range(0, 4096, 2048):
    for n0 in range(0, 4096, 128):
        for d0 in range(0, 64, 64):
            gm_tile = Tile(M=2048, N=128, D=64)

            # CoreGroup 是 spatial tile。
            # 当前 map 没有写 spatial_instances，因此具体映射到多少个 Core
            # 需要后端结合 arch.yaml 的 Core[0..23] 决定。
            for core_tile in spatial_dispatch(gm_tile, target="CoreGroup"):

                # Pipeline stage 1: Cube 分支计算 P。
                # L1/L0 每次处理 M=256, N=128, D=64。
                for m1 in range(0, 2048, 256):
                    l1_tile = Tile(M=256, N=128, D=64)

                    load_L1(Q[m0 + m1 : m0 + m1 + 256, d0 : d0 + 64])
                    load_L1(K[d0 : d0 + 64, n0 : n0 + 128])

                    load_L0(Q_tile(M=256, D=64))
                    load_L0(K_tile(D=64, N=128))

                    P_subtile = MatmulP(
                        Q_tile(M=256, D=64),
                        K_tile(D=64, N=128),
                    )

                    store_P(P_subtile)  # P tile shape = M=256, N=128

                # Pipeline stage 2: Vec 分支消费 P 并计算 S。
                # VecLaneGroup/UB 每次处理 M=1024, N=128。
                parallel for m2 in range(0, 2048, 1024):
                    load_UB(P[m0 + m2 : m0 + m2 + 1024, n0 : n0 + 128])
                    S_tile = ExpP(P_tile(M=1024, N=128))
                    store_output(S_tile)
```

## Cube 分支的循环

Cube 分支的输入 tile 来自 GM/CoreGroup：

```text
GM/CoreGroup tile = M=2048 N=128 D=64
```

L1 和 L0 的 tile 是：

```text
L1 factors = M=256 N=128 D=64
L0 factors = M=256 N=128 D=64
```

因此在一个 GM tile 内，Cube 分支主要沿 M 方向切 8 次：

```text
2048 / 256 = 8
```

每一轮 MatmulP 计算：

```text
Q tile: M=256 D=64
K tile: D=64  N=128
P tile: M=256 N=128
```

8 轮之后得到完整的：

```text
P group: M=2048 N=128
```

这里的 D 是完整 reduction 维度，因为 `D=64` 在 GM/L1/L0 都没有继续切小。

## Vec 分支的循环

Vec 分支只处理 `P` 和 `S`，因此没有 D 维：

```text
VecLaneGroup factors = M=1024 N=128
UB factors           = M=1024 N=128
```

在一个 GM tile 的 `P group = M=2048 N=128` 内，Vec 分支沿 M 方向切 2 次：

```text
2048 / 1024 = 2
```

每一轮 ExpP 计算：

```text
P tile: M=1024 N=128
S tile: M=1024 N=128
```

因此，最新 map 表达的是：

```text
Cube: 8 轮，每轮产生 256x128 的 P_subtile
Vec:  2 轮，每轮消费 1024x128 的 P_tile
```

也就是说 Vec 每轮消费的 P tile 等于 4 个 Cube 轮次的输出：

```text
1024 / 256 = 4
```

如果要求 Vec 必须等 Cube 完成整个 `2048x128` 的 P group 后再启动，
那么 Pipeline 语义应当理解为：

```text
先执行 8 轮 MatmulP，得到完整 P group
再执行 2 轮 ExpP，消费这个 P group
```

如果未来想表达更细粒度的 producer-consumer overlap，则还需要在 map 中额外描述
`P` 的同步粒度，比如 `producer_tile=M=256 N=128`、`consumer_tile=M=1024 N=128`，
或者让后端从 `P` 的 tile 粒度自动推导“4 个 Cube subtile 满足 1 个 Vec tile”。
