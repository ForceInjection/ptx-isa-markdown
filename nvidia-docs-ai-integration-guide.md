# 工欲善其事，必先利其器：面向 Vibe Coding 的 CUDA 代码技能库介绍

写给 vLLM 等大模型推理框架与 GPU 底层开发者！

> 项目地址：**<https://github.com/ForceInjection/cuda-code-skill>**
> 基于 [technillogue/ptx-isa-markdown](https://github.com/technillogue/ptx-isa-markdown) 的 fork 版本

---

## 目录

- [1. 你是否遇到过这些场景？](#1-你是否遇到过这些场景)
- [2. 这个项目是什么](#2-这个项目是什么)
- [3. 覆盖了哪些文档](#3-覆盖了哪些文档)
- [4. 六个真实场景](#4-六个真实场景)
- [5. Agent Skill 矩阵](#5-agent-skill-矩阵)
- [6. 怎么使用](#6-怎么使用)
- [7. 设计上的几个决策](#7-设计上的几个决策)
- [8. 结语](#8-结语)

---

## 1. 你是否遇到过这些场景？

你在优化 vLLM 的 attention kernel，用 `cuobjdump --dump-ptx` 反编译出了 PTX 汇编，但那条 `wgmma.mma_async` 指令的寄存器布局是什么？操作数约束是什么？

你打开浏览器，搜到 PTX ISA 官方文档——一个 **5 MB 的单页 HTML**，Ctrl+F 半天，翻到眼花。

或者这个场景：vLLM 接入 FP8 量化，`cublasGemmEx` 传入 `CUDA_R_8F_E4M3` 的时候，`computeType` 该用什么？scale 因子怎么传？你打开 cuBLAS 文档，发现光 API 页面就有 **300 多个子页面**，函数签名藏在第 4 层点击之后。

又或者：Tensor Parallel 跑起来，all-reduce 突然变慢，甚至 hang 住。想调一下 `NCCL_ALGO` 或 `NCCL_PROTO`，结果 NCCL 的环境变量文档散落在各个页面里，找全要花半小时。

**这些痛点，归根结底是一个问题：NVIDIA 的官方文档，不适合快速查阅，更不适合被 AI 工具消费。**

今天介绍一个我们在用的工具，专门解决这个问题。

---

## 2. 这个项目是什么

项目地址：[ForceInjection/cuda-code-skill](https://github.com/ForceInjection/cuda-code-skill)（fork 自 [technillogue/ptx-isa-markdown](https://github.com/technillogue/ptx-isa-markdown)）

原始项目只做了一件事：把 PTX ISA 这份 5 MB 的单页 HTML 拆成 405 个 Markdown 文件，让你可以用 `grep` 搜索。这个想法很好，但覆盖面太窄—— vLLM 开发日常还需要 cuBLAS 、 NCCL 、 CUDA Math API 等等。

**我们做的事情是把这个思路系统化。**

核心产出是一个统一的文档爬虫 `scrape_cuda_docs.py` ，加上一套可以直接装进 AI IDE 的多技能（ Skill ）目录。一句话概括：

> **把 NVIDIA 的 6 套官方文档，转换成 1000+ 个本地 Markdown 文件，结合 Agent 技能矩阵，让你和 AI 都能高效检索与自动优化 CUDA 代码。**

---

## 3. 覆盖了哪些文档

转换后的文档覆盖 vLLM 开发最常用的 6 套参考资料，总计约 8.7 MB、1032 个文件：

| 文档集                | 文件数 | 大小   | 主要内容                                            |
| --------------------- | ------ | ------ | --------------------------------------------------- |
| PTX ISA 9.1           | 405    | 2.3 MB | 完整指令集， `wgmma` 、 `cp.async` 、 `mbarrier` 等 |
| CUDA Runtime API 13.1 | 104    | 1.2 MB | 37 模块 + 66 数据结构                               |
| CUDA Driver API 13.1  | 129    | 1.2 MB | 49 模块（含虚拟内存、 context 管理）                |
| CUDA Math API 13.x    | 41     | 0.5 MB | `half` / `bfloat16` / FP8 内置函数，类型转换        |
| cuBLAS 13.2           | 319    | 2.9 MB | GEMM 、 `cuBLASLt` 、 FP8 GEMM 、 epilogue 融合     |
| NCCL                  | 34     | 0.5 MB | API + 使用指南 + 全量环境变量                       |

文档经过清洗：去掉重复 TOC 、导航栏、冗余 URL 、版权声明等噪音，**体积压缩 76-83%**，只保留你真正需要的内容。

---

## 4. 六个真实场景

下面用 6 个 vLLM 开发中的真实场景，演示这个工具怎么用。

### 4.1 场景一：读懂反编译出的 PTX 指令

你用 `cuobjdump --dump-ptx` 拿到 vLLM FlashAttention kernel 的 PTX，看到：

```bash
# 示例：TMA 异步拷贝指令
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [smem_desc], [gmem_desc], [mbar], {x, y};
```

这条 TMA 指令的操作数格式是什么？ `mbarrier::complete_tx::bytes` 是什么语义？

```bash
# 在 PTX 文档中搜索 TMA 指令
grep -r "cp.async.bulk.tensor" skills/cuda-knowledge/references/ptx-docs/9-instruction-set/
```

秒定位到 `9-instruction-set/` 下对应章节，操作数说明、约束条件、示例代码一目了然。

延伸查找 TMA swizzling 模式：

```bash
# 搜索 swizzle_mode 的相关说明
grep -r "swizzle_mode" skills/cuda-knowledge/references/ptx-docs/9-instruction-set/
```

### 4.2 场景二：FP8 量化推理的 cuBLAS 参数

vLLM 接入 FP8 量化，调用 `cublasGemmEx` 时需要确认：`CUDA_R_8F_E4M3` 作为 `Atype`，`computeType` 应该用 `CUBLAS_COMPUTE_32F` 还是 `CUBLAS_COMPUTE_32F_FAST_16F`？scale 因子怎么传？

```bash
# 查找 cublasGemmEx 函数签名及参数约束
grep -A 30 "cublasGemmEx" skills/cuda-knowledge/references/cublas-docs/2-using-the-cublas-api/
```

完整函数签名、每个参数的类型约束和合法组合，直接出来。

如果要用 `cuBLASLt` 实现带 bias 融合的 FP8 GEMM ：

```bash
# 查找 cublasLtMatmul 函数说明
grep -A 20 "cublasLtMatmul" skills/cuda-knowledge/references/cublas-docs/3-using-the-cublaslt-api/
```

epilogue 参数（ `CUBLASLT_EPILOGUE_RELU` 、 `CUBLASLT_EPILOGUE_BIAS` ）的说明也在里面。

### 4.3 场景三：Tensor Parallel 通信调优与 hang 排查

vLLM 跑 Tensor Parallel，all-reduce 延迟突然升高，或者直接 hang 住。

**调优**：先查算法和协议选项

```bash
# 查找 NCCL 算法与协议环境变量选项
grep -E "^## NCCL_(ALGO|PROTO|BUFFSIZE)" skills/cuda-knowledge/references/nccl-docs/env.md
```

`NCCL_ALGO=Ring` 还是 `Tree` ？ `NCCL_PROTO=LL128` 还是 `Simple` ？每个选项的适用场景在文档里写得很清楚。

**排查 hang**：开调试日志

```bash
# 查找 NCCL 调试日志环境变量
grep -A 8 "^## NCCL_DEBUG\b" skills/cuda-knowledge/references/nccl-docs/env.md
```

知道了 `NCCL_DEBUG=INFO` 和 `NCCL_DEBUG_SUBSYS=ALL` 的用法，再去看 RAS 排查指南：

```bash
# 查看 RAS 排查指南
cat skills/cuda-knowledge/references/nccl-docs/troubleshooting/ras.md
```

### 4.4 场景四：half/bfloat16 精度问题排查

vLLM custom kernel 里用了 `__hfma2` 做 FP16 fused multiply-add，结果和 FP32 路径有精度差。是舍入模式的问题？还是 operand 顺序导致的？

```bash
# 查找 __hfma2 函数签名与精度说明
grep -A 8 "__hfma2\b" skills/cuda-knowledge/references/cuda-math-docs/modules/group__cuda__math__intrinsic__half.md
```

函数签名、精度说明、舍入行为直接出来。

如果是 FP8 ↔ float 转换的精度问题（ vLLM KV cache 量化场景常见）：

```bash
# 查看 FP8 类型转换的语义
cat skills/cuda-knowledge/references/cuda-math-docs/modules/group__cuda__math__intrinsic__cast.md
```

`__nv_fp8_e4m3` 和 `__nv_fp8_e5m2` 之间的转换语义，以及和 float 互转时的截断行为，都在这里。

### 4.5 场景五：KV cache 内存分配的 Driver API 错误

vLLM 启动时报 `cuMemAddressReserve` 失败，或者 KV cache 分配时触发 `cudaErrorInvalidValue`，不确定是对齐要求没满足，还是参数范围问题。

```bash
# 查找 cuMemAddressReserve 的使用说明与约束
grep -A 20 "cuMemAddressReserve" skills/cuda-knowledge/references/cuda-driver-docs/modules/group__cuda__va.md
```

函数签名、 `size` 的对齐要求、 `addr` 的约束，以及返回值含义，全在里面。

Runtime 侧的错误码查询：

```bash
# 查找 cudaErrorInvalidValue 错误码说明
grep -A 10 "cudaErrorInvalidValue" skills/cuda-knowledge/references/cuda-runtime-docs/
```

### 4.6 让 AI IDE 直接回答 CUDA 问题

以上场景如果你在用 TRAE、Qoder 或 Claude Code 等支持 Skill 机制的 AI IDE 开发，可以更进一步——把这套文档装成一个 Skill，让 AI 直接从本地文档里检索答案，而不是靠训练数据（可能过时）或联网搜索（权限问题）。

**安装只需一行命令：**

```bash
# 将 cuda-knowledge skill 复制到目标目录
cp -r skills/cuda-knowledge ~/.trae/skills/
```

装好之后，直接问你的 AI 助手：

> "vLLM 里 FP8 WGMMA m64n16k16 的 D 矩阵寄存器布局是什么？"
> "cublasGemmEx 传 CUDA_R_8F_E4M3 时 computeType 该怎么选？"
> "NCCL_ALGO=Ring 和 Tree 分别适合什么拓扑？"

AI 会触发 Skill （关键词匹配 `SKILL.md` 的 description frontmatter ），从本地 `ptx-docs` / `cublas-docs` / `nccl-docs` 里检索相关章节，给出带文档引用的精准回答。

核心设计是**渐进式披露**： `SKILL.md` （~ 13KB ）常驻 context window ， 1000+ 个参考文件按需加载，不浪费 token 。

---

## 5. Agent Skill 矩阵

在基础的知识检索之外，本项目扩展出了一个完整的 Agent 技能矩阵（位于 `skills/` 目录），旨在为 Qoder 等 AI 助手提供一套自动化的 CUDA kernel 优化流水线。该矩阵包含以下五个核心技能：

- **`cuda-knowledge`**：知识库核心，包含 640+ Markdown 格式的官方文档，为其他技能提供严谨的 API 约束与底层知识。
- **`cuda-optimizer`**：核心调度技能，负责驱动“分析 - 优化 - 验证”的闭环迭代。
- **`cuda-code-generator`**：代码生成与修改技能，被严格指令必须基于 `cuda-knowledge` 查阅 API 细节，避免 AI 幻觉。
- **`ncu-rep-analyzer`**：NCU 性能分析技能，负责解读性能分析报告，识别 memory 瓶颈或计算瓶颈。
- **`kernel-benchmarker`**：编译与基准测试技能，负责代码的实际编译、正确性验证及性能测试。

通过这些技能的解耦与协同，AI 助手不再只是“文档检索引擎”，而是进阶为真正的“ CUDA 性能优化工程师”。

---

## 6. 怎么使用

以下提供三种使用方式，从最简单的直接检索到集成进 AI 工具，按需选择。

### 6.1 直接用现成文档

clone 仓库后， `skills/cuda-knowledge/references/` 目录下已经有完整的文档，直接 `grep` 即可：

```bash
# 查 PTX 指令
grep -r "mbarrier.init" skills/cuda-knowledge/references/ptx-docs/

# 查 cuBLAS 函数
grep -r "cublasSgemm" skills/cuda-knowledge/references/cublas-docs/

# 查 NCCL 环境变量
grep -E "^## NCCL_" skills/cuda-knowledge/references/nccl-docs/env.md

# 查 CUDA Math 内置函数
grep "^__device__" skills/cuda-knowledge/references/cuda-math-docs/modules/group__cuda__math__intrinsic__half.md
```

### 6.2 更新到最新版本

NVIDIA 文档版本更新时，用爬虫重新抓取：

```bash
# 需要先安装 uv（ https://github.com/astral-sh/uv ）
uv run nvidia_doc_sync/scrape_cuda_docs.py ptx
uv run nvidia_doc_sync/scrape_cuda_docs.py cublas
uv run nvidia_doc_sync/scrape_cuda_docs.py nccl
uv run nvidia_doc_sync/scrape_cuda_docs.py runtime
uv run nvidia_doc_sync/scrape_cuda_docs.py driver
uv run nvidia_doc_sync/scrape_cuda_docs.py math
```

`uv run` 会自动解析脚本头部的 PEP 723 依赖声明，无需单独 `pip install` ，也不需要 virtualenv 。

### 6.3 装进 AI IDE

你可以通过直接复制预构建的 Skill 目录，将整个自动化优化流水线集成到任何支持 Skill 机制的 AI IDE 中（如 TRAE 、 Qoder 、 Claude Code 等）：

```bash
# 以 TRAE 为例，将所有 skill 复制到其 skills 目录
cp -r skills/* ~/.trae/skills/

# 对于 Qoder 或 Claude Code，请参考对应工具的路径进行复制
```

重启你的 AI IDE，之后遇到 CUDA / PTX / cuBLAS / NCCL 相关问题，或者需要分析 NCU 报告、优化 Kernel 时，对应的 Skill 将自动激活并相互协作。

---

## 7. 设计上的几个决策

以下记录几个关键设计决策及其背后的考量。

### 7.1 为什么是单文件爬虫

NVIDIA 的文档格式并不统一： PTX ISA 和 cuBLAS 是 Sphinx 生成的**单页巨型 HTML** ； Runtime API 、 Driver API 和 Math API 是 Doxygen 生成的**多页站点** ； NCCL 是 Sphinx 的**多页站点**。 `scrape_cuda_docs.py` 内部实现了三种 scraper 类，对外统一一个入口，子命令决定走哪条路径。

### 7.2 为什么做两阶段清洗

API 文档（ Runtime / Driver / Math ）先下载原始 HTML 转 Markdown 存到 `*-raw/` 目录，再跑清洗 pass 去掉重复 TOC 、导航栏、冗余链接、版权声明，输出到最终目录。这样改清洗逻辑不需要重新下载——对于需要反复调整清洗规则的场景， `--skip-download` 可以把迭代时间从分钟级压到秒级。

### 7.3 渐进式 Skill 设计与多技能架构

`SKILL.md` 只有 ~ 13KB ，里面是 API 概述、触发关键词和检索指引，不包含原始文档内容。 AI 用关键词命中 Skill 后，再通过文件路径按需读取具体参考文件，确保 Token 消耗可控。此外，将优化、生成、分析与基准测试拆分为独立的子技能，极大地降低了单一 Agent 的提示词复杂度，提高了任务执行的可靠性。

### 7.4 可扩展架构

想加一套新文档（比如 cutlass 、 NVML ）？只需识别文档格式，实现 scraper ，注册 CLI ，下载验证，写搜索指南，最后更新对应的 `SKILL.md` 即可。

---

## 8. 结语

vLLM 等大模型推理框架的核心竞争力，往往源于 Kernel 级别极致的细节打磨——从寄存器布局的精准控制、 GEMM 中 epilogue 的深度融合，到 NCCL 算法的针对性选择，再到内存分配时对齐要求的严格满足。

而解答这些硬核细节的钥匙，正隐藏在 NVIDIA 浩如烟海的官方文档中。工欲善其事，必先利其器。一款顺手的文档检索工具与自动化优化技能矩阵，能让你从繁琐的翻阅中解放出来，将宝贵的精力聚焦于真正创造价值的性能优化上。

---

_文章中所有 grep 命令均在本地 `skills/cuda-knowledge/references/` 目录下执行。文档内容版权归 NVIDIA Corporation 所有，本项目为便于开发者查阅而进行的格式转换，仅供参考。_
