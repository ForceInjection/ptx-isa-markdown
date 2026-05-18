# NCU Profile 样例报告

针对 `examples/vectorAdd/solution.cu` 的向量加法 kernel，在 RTX 5090 (Blackwell, CC 12.0) 上使用 `--set launch` 模式生成。

## 文件说明

| 文件                    | 大小  | 说明                                      |
| ----------------------- | ----- | ----------------------------------------- |
| `sample-launch.ncu-rep` | 41 KB | 二进制 NCU 报告，使用 `ncu --import` 读取 |
| `sample-launch.txt`     | 5 KB  | 文本导出，含 50+ 设备属性                 |

## 生成命令

```bash
# 1. 构建自包含 profiling 可执行文件
python3 skills/kernel-benchmarker/scripts/ncu_profile.py examples/vectorAdd/solution.cu \
    --N=1000000 --build-only

# 2. 运行 NCU profiling
ncu --kernel-name solve --launch-skip 0 --launch-count 1 \
    --set launch -o report -f examples/vectorAdd/solution_bench \
    --N=1000000 --warmup=0 --repeat=1

# 3. 查看报告
ncu --import report.ncu-rep --print-summary per-kernel
ncu --import report.ncu-rep --print-summary per-kernel --page raw  # 详细属性列表
```

## 报告内容解读

### Kernel 启动配置

```text
solve (3907, 1, 1) x (256, 1, 1)
```

3907 blocks × 256 threads/block = 1,000,192 线程，处理 1,000,000 个元素。每 256 线程组成一个 block，block 数量根据元素数自动计算。

### 关键设备属性

| 属性                            | 值       | 含义                                  |
| ------------------------------- | -------- | ------------------------------------- |
| `compute_capability`            | 12.0     | Blackwell 架构 (RTX 5090)             |
| `clock_rate`                    | 2407 MHz | GPU 基础频率                          |
| `fb_bus_width`                  | 512-bit  | 显存总线位宽（带宽上限 ≈ 1.5 TB/s）   |
| `l2_cache_size`                 | 96 MB    | L2 缓存总量（大缓存利于数据复用）     |
| `limits_num_tpcs`               | 85       | SM 总数（计算单元数量）               |
| `limits_max_cta_per_sm`         | 24       | 每个 SM 最多同时驻留 24 个 block      |
| `max_block_dim_x`               | 1024     | 每个 block 最多 1024 个线程（x 方向） |
| `max_grid_dim_x`                | 2^31 - 1 | Grid 上限（足够覆盖极大问题规模）     |
| `cooperative_launch`            | 1        | 支持 Cooperative Groups 跨 block 同步 |
| `generic_compression_supported` | 1        | 硬件压缩支持（可压缩内存）            |

### `--set launch` vs `--set full`

| 模式           | 需要 PMU                        | 可回答的问题                                                                  | 不可回答的问题                     |
| -------------- | ------------------------------- | ----------------------------------------------------------------------------- | ---------------------------------- |
| `--set launch` | 不需要                          | 启动配置是否合理？block/SM 占比？寄存器压力上限？                             | SM 利用率？DRAM 带宽？瓶颈在哪里？ |
| `--set full`   | 需要（`perf_event_paranoid=0`） | 上述全部 + 动态指标：SM Throughput、DRAM/L1/L2 带宽、Occupancy、Warp 调度统计 | —                                  |

**受限环境（容器/云主机）使用 `--set launch`**，非受限宿主机可使用 `--set full` 获取完整性能指标。

> [!NOTE]
> 本样例在 AutoDL 容器环境（`perf_event_paranoid=4`，无法修改）中生成，使用 `--set launch` 模式。
