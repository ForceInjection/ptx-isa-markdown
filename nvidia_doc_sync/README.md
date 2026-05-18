# 使用指南

该脚本是一个统一的 NVIDIA CUDA 文档抓取工具，能够将 CUDA 的 HTML 文档转换为支持搜索的 Markdown 文件，目前支持 PTX ISA、CUDA Runtime API、CUDA Driver API、CUDA Math API、cuBLAS 以及 NCCL 的文档抓取。

---

## 1. 前置条件

本章节介绍运行抓取脚本所需的依赖环境及安装要求。

必须安装 [`uv`](https://github.com/astral-sh/uv) 。所有的 Python 依赖（例如 `beautifulsoup4` 、 `html2text` 、 `requests` ）均通过 PEP 723 内联元数据自动解析，无需单独执行 `pip install` 或配置虚拟环境。

---

## 2. 快速开始

本章节提供了一系列常用的快捷命令，帮助用户快速抓取各类 CUDA 文档。

```bash
# 抓取 PTX ISA 文档
uv run scrape_cuda_docs.py ptx

# 抓取 CUDA Runtime API 文档
uv run scrape_cuda_docs.py runtime

# 抓取 CUDA Driver API 文档
uv run scrape_cuda_docs.py driver

# 抓取 CUDA Math API 文档
uv run scrape_cuda_docs.py math

# 抓取 cuBLAS 文档
uv run scrape_cuda_docs.py cublas

# 抓取 NCCL 文档
uv run scrape_cuda_docs.py nccl
```

---

## 3. 命令参考

本章节详细列出了脚本支持的所有命令行参数及其功能说明。

```bash
# 脚本的命令行用法及可用参数列表
usage: scrape_cuda_docs.py [-h] [--output-dir OUTPUT_DIR] [--skip-download] [--force]
                           {ptx,runtime,driver,math,cublas,nccl}
```

### 3.1 位置参数

本小节介绍了脚本支持的核心位置参数，用于指定需要抓取的文档类型。

| 参数      | 描述                                                                    |
| --------- | ----------------------------------------------------------------------- |
| `ptx`     | 抓取 PTX ISA 文档（单页面结构，会被拆分为约 405 个文件）                |
| `runtime` | 抓取 CUDA Runtime API 文档（多页面结构：包含模块和数据结构）            |
| `driver`  | 抓取 CUDA Driver API 文档（多页面结构：包含模块和数据结构）             |
| `math`    | 抓取 CUDA Math API 文档（多页面结构：包含模块和数据结构，约 54 个文件） |
| `cublas`  | 抓取 cuBLAS 文档（单页面 Sphinx 结构，按章节拆分为约 319 个文件）       |
| `nccl`    | 抓取 NCCL 文档（多页面 Sphinx 站点，约 33 个文件，按主题组织）          |

### 3.2 选项参数

本小节说明了脚本支持的附加选项标志，用于控制输出路径和抓取行为。

| 标志                | 默认值                                        | 描述                                             |
| ------------------- | --------------------------------------------- | ------------------------------------------------ |
| `--output-dir PATH` | `skills/cuda-knowledge/references/<api>-docs` | 覆盖默认的输出目录                               |
| `--skip-download`   | `false`                                       | 跳过网络下载；仅对缓存的原始文件重新运行清理流程 |
| `--force`           | `false`                                       | 强制重新下载所有页面，即使文件已被缓存           |
| `-h, --help`        | —                                             | 显示帮助信息并退出                               |

---

## 4. 输出结构

本章节展示了各类文档抓取后的目录层级与文件组织形式。

### 4.1 PTX ISA (`ptx`)

本小节展示了 PTX ISA 文档抓取后的目录结构。

```text
# PTX ISA 文档输出结构示例
skills/cuda-knowledge/references/ptx-docs/
├── 1-introduction/
├── 2-programming-model/
├── ...
└── 9-instruction-set/      # 约 186 个文件
```

内容按照标题层级进行拆分，并组织在带编号的章节目录中。

### 4.2 cuBLAS (`cublas`)

本小节展示了 cuBLAS 文档抓取后的目录结构。

```text
# cuBLAS 文档输出结构示例
skills/cuda-knowledge/references/cublas-docs/
├── 1-introduction/
├── 2-using-the-cublas-api/
├── 3-using-the-cublaslt-api/
├── 4-using-the-cublasxt-api/
└── ...                     # 总计约 319 个文件
```

与 PTX ISA 采用相同的按标题拆分策略 —— cuBLAS 文档也是一个单页面的 Sphinx HTML 文件。

### 4.3 Runtime / Driver API (`runtime` / `driver`)

本小节展示了 CUDA 运行时和驱动 API 文档的输出格式。

```text
# Runtime 与 Driver API 文档输出结构示例
skills/cuda-knowledge/references/cuda-{runtime,driver}-docs/
├── modules/                # 每个 API 模块组对应一个文件
├── data-structures/        # 每个结构体/联合体对应一个文件
└── INDEX.md                # 自动生成的包含所有文件链接的索引
```

同级的 `cuda-{runtime,driver}-docs-raw/` 缓存目录用于存储清理前下载的原始文件。

### 4.4 CUDA Math API (`math`)

本小节展示了 CUDA 数学 API 文档的输出格式。

```text
# Math API 文档输出结构示例
skills/cuda-knowledge/references/cuda-math-docs/
├── modules/                # 14 个文件 —— 单/双精度、半精度、bfloat16、FP8/FP6/FP4、SIMD、类型转换、整数等
├── data-structures/        # 26 个文件 —— __half、__nv_bfloat16、__nv_fp8_*、__nv_fp6_*、__nv_fp4_*
└── INDEX.md
```

同级的 `cuda-math-docs-raw/` 缓存目录用于存储清理前下载的原始文件。

### 4.5 NCCL (`nccl`)

本小节展示了 NCCL 文档的输出组织方式。

```text
# NCCL 文档输出结构示例
skills/cuda-knowledge/references/nccl-docs/
├── usage/                  # 11 个文件 —— 通信器、集合通信、流、P2P、CUDA 图等
├── api/                    # 12 个文件 —— 集合通信、通信器、P2P、类型、操作、标志、设备 API 等
├── troubleshooting/        # 1 个文件  —— RAS 子系统
├── overview.md             # NCCL 概念与拓扑结构
├── setup.md                # 安装指南
├── env.md                  # 完整的环境变量参考
├── examples.md             # 代码示例
├── mpi.md                  # NCCL 与 MPI 集成
├── troubleshooting.md      # 挂起与错误诊断
├── nccl1.md                # NCCL 1 到 2 的迁移指南
└── INDEX.md
```

注意事项：

- 没有原始缓存目录（无需清理阶段 —— Sphinx 多页面已经是干净的格式）
- 页面被直接抓取到输出目录中（与 cuBLAS/PTX 相同）
- `--skip-download` 标志对此格式无效

---

## 5. 两阶段流水线 (Runtime / Driver)

本章节介绍了 API 抓取工具使用的下载与清理分离的两阶段工作流。

API 抓取工具使用了“先下载后清理”的流水线设计：

**第一阶段 — 下载**：获取每个页面并将原始的 Markdown 内容保存到 `*-raw/modules/` 和 `*-raw/data-structures/` 目录中。

**第二阶段 — 清理**：从缓存读取内容并写入清理后的输出，执行以下操作：

- 移除重复的函数目录 (TOC)
- 剥离页脚（NVIDIA 标志、版权声明、隐私政策）
- 移除仅含锚点的链接、零宽空格、 `[inherited]` 标签
- 移除 `See also:` 部分和模板化的提示信息
- 剥离超链接中的所有 URL（仅保留链接文本）
- 合并多余的空行

这种分离机制允许在不重新下载的情况下迭代清理逻辑：

```bash
# 在修改 clean_markdown_file() 函数后，仅重新运行清理步骤
uv run scrape_cuda_docs.py driver --skip-download
```

清理后的典型体积缩减率约为 **76–83%** 。

---

## 6. 常见工作流

本章节提供了一些常见的使用场景和对应的命令示例。

### 6.1 全新抓取所有文档集

本小节展示了如何从头抓取所有支持的文档。

```bash
# 逐个抓取所有支持的 API 文档
uv run scrape_cuda_docs.py ptx
uv run scrape_cuda_docs.py runtime
uv run scrape_cuda_docs.py driver
uv run scrape_cuda_docs.py math
uv run scrape_cuda_docs.py cublas
uv run scrape_cuda_docs.py nccl
```

### 6.2 从头重新下载所有内容（覆盖缓存）

本小节展示了如何强制刷新本地缓存。

```bash
# 强制忽略缓存并重新下载
uv run scrape_cuda_docs.py driver --force
```

### 6.3 抓取到自定义输出目录

本小节展示了如何指定文档输出的路径。

```bash
# 指定非默认的输出目录
uv run scrape_cuda_docs.py ptx --output-dir /path/to/my/output
```

### 6.4 在不重新下载的情况下迭代清理逻辑

本小节展示了如何高效测试文档清理逻辑。

```bash
# 在脚本中修改 clean_markdown_file() 后执行：
uv run scrape_cuda_docs.py driver --skip-download
```

---

## 7. 添加新的文档集

本章节详细说明了将新的 NVIDIA 文档集集成到技能库中的端到端完整工作流。所有步骤均基于添加 cuBLAS 和 CUDA Math API 的经验总结得出。

### 7.1 步骤 1 — 识别文档格式

本小节介绍了在编写代码前如何通过网络请求探测目标文档的 HTML 结构类型。

NVIDIA 的文档分为三种不同的 HTML 格式。在编写任何代码之前，请先探测目标 URL：

```bash
# 下载索引/登录页面并检查其大小和结构
curl -s "https://docs.nvidia.com/cuda/<new-api>/index.html" | wc -c
# > 1MB 单文件  →  Sphinx 单页面（类似 PTX ISA, cuBLAS）
# 50KB–200KB 索引页   →  Sphinx 多页面（类似 NCCL） — 包含 headerlinks, toctree
# < 50KB 索引页  →  Doxygen 多页面（类似 Runtime, Driver, Math API）

# 检查 HTML 中的 Sphinx 与 Doxygen 标识
curl -s "https://docs.nvidia.com/<path>/index.html" | grep -c "headerlink"   # Sphinx: >10
curl -s "https://docs.nvidia.com/<path>/modules.html" 2>/dev/null | wc -c   # Doxygen: 能找到
```

| 格式               | 示例                      | HTML 标识                                                      | 抓取器类                 |
| ------------------ | ------------------------- | -------------------------------------------------------------- | ------------------------ |
| **Sphinx 单页面**  | PTX ISA, cuBLAS           | `div.body`，内部包含 `<a class="headerlink">`，>1MB            | `SphinxScraper`          |
| **Sphinx 多页面**  | NCCL                      | `<a class="headerlink">`，包含 toctree 链接，50–200KB 的索引页 | `SphinxMultiPageScraper` |
| **Doxygen 多页面** | Runtime, Driver, Math API | `modules.html` 或 `index.html` 链接到各个模块页面              | `APIScraper`             |

### 7.2 步骤 2 — 实现抓取器

本小节说明了针对不同文档格式，如何在脚本中配置和添加新的抓取器规则。

**对于 Sphinx 单页面文档**，在 `scrape_cuda_docs.py` 的 `SphinxScraper.KNOWN_DOCS` 中添加一个条目：

```python
# 配置 Sphinx 单页面抓取规则
class SphinxScraper(DocumentationScraper):
    KNOWN_DOCS: dict[str, tuple[str, str]] = {
        "cublas": (
            "cuBLAS",
            "https://docs.nvidia.com/cuda/cublas/index.html",
        ),
        "new-api": (                                     # ← 新增条目
            "New API Display Name",
            "https://docs.nvidia.com/cuda/new-api/index.html",
        ),
    }
```

**对于 Sphinx 多页面文档**，在 `SphinxMultiPageScraper.KNOWN_DOCS` 中添加条目：

```python
# 配置 Sphinx 多页面抓取规则
class SphinxMultiPageScraper(DocumentationScraper):
    KNOWN_DOCS: dict[str, tuple[str, str, str]] = {
        "nccl": (
            "NCCL",
            "https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/",
            "index.html",
        ),
        "new-api": (                                     # ← 新增条目
            "New API Display Name",
            "https://docs.nvidia.com/<path>/",
            "index.html",                                # 包含 toctree 链接的发现页面
        ),
    }
```

该抓取器会从 toctree 或索引页中发现所有链接的 `.html` 页面，然后获取每个页面并转换为 Markdown，同时保留相对路径结构（例如 `usage/communicators.html` 转换为 `usage/communicators.md` ）。

**对于 Doxygen 多页面文档**，在 `APIScraper._CONFIG` 中添加条目：

```python
# 配置 Doxygen 多页面抓取规则
_CONFIG: dict[str, tuple[str, str, str, str, str]] = {
    # (base_url, modules_discovery_path, modules_href_pattern,
    #  structs_discovery_path, structs_href_pattern)
    "math": (
        "https://docs.nvidia.com/cuda/cuda-math-api/",
        "index.html",
        r"cuda_math_api/group__CUDA__MATH__.*\.html",
        "cuda_math_api/structs.html",
        r"struct.*\.html",
    ),
    "nvml": (                                           # ← 新增条目
        "https://docs.nvidia.com/deploy/nvml-api/",
        "modules.html",
        r"group__nvml.*\.html",
        "annotated.html",
        r"struct.*\.html",
    ),
}
```

> **发现路径提示**：如果模块链接使用了子目录前缀（例如 `cuda_math_api/group__...` ），发现 URL 和链接模式必须包含该前缀。抓取器使用 `Path(href).name` 作为文件名，并相对于发现页面的 URL 而非基准 URL 来解析链接。

### 7.3 步骤 3 — 注册 CLI 命令

本小节介绍了如何在脚本入口处暴露新添加的 API 选项。

在 `main()` 函数中，将新名称添加到两个位置：

```python
# 在 CLI 参数中注册新命令并配置默认输出路径
# 1. 添加到选项中
parser.add_argument(
    "api_type",
    choices=["ptx", "runtime", "driver", "math", "cublas", "nccl"],  # ← 在此处添加
)

# 2. 添加默认输出目录
default_dirs = {
    ...
    "nccl": "skills/cuda-knowledge/references/nccl-docs",  # ← 在此处添加
}

# 3. 对于 Sphinx 文档，在抓取器选择代码块中添加分发逻辑
#    注: 当前 main() 对 SphinxScraper 使用显式的 elif 分支（而非 KNOWN_DOCS 泛型分发）
elif args.api_type == "new-api":
    scraper = SphinxScraper.from_doc_type("new-api", args.output_dir)
```

### 7.4 步骤 4 — 下载文档

本小节说明了如何执行命令来验证新集成的抓取逻辑是否正常工作。

```bash
# 运行抓取脚本
uv run scrape_cuda_docs.py <new-api>
```

验证输出结果：

```bash
# 检查生成的文件数量和目录结构
find skills/cuda-knowledge/references/<new-api>-docs -name "*.md" | wc -l
ls skills/cuda-knowledge/references/<new-api>-docs/

# 抽查文件，确认标题和内容是否正确
head -10 skills/cuda-knowledge/references/<new-api>-docs/modules/<first-file>.md
```

如果标题看起来不正确（例如，显示为函数名如 `log()` 而不是章节标题），或缺少文件，请在修复发现模式后使用 `--force` 参数重新运行。

### 7.5 步骤 5 — 创建搜索指南

本小节指导如何为新文档创建配套的搜索指南，以帮助 Agent 更好地利用这些资料。

按照 `cublas.md` 或 `cuda-math.md` 的相同结构，创建 `skills/cuda-knowledge/references/<new-api>.md` 文件：

```text
# 搜索指南结构模板
# <New API> Reference

**Related guides:** ...

## Table of Contents
## Local Documentation       ← 文件数量、大小、质量标记
## When to Use               ← 编号的用例列表
## Quick Search Examples     ← 最常见查询的 grep 命令
## Documentation Structure   ← 带有注释的目录树
## Search Tips               ← 此 API 的命名模式
## Common Workflows          ← 逐步的 grep 工作流
## Troubleshooting           ← 常见错误及修复方法
## Version Information
```

### 7.6 步骤 6 — 更新 SKILL.md

本小节说明了如何更新技能配置文件，使 AI 能够识别并触发新文档的技能。

必须在 `skills/cuda-knowledge/SKILL.md` 中更新两个位置：

**1. 在“本地 API 文档”部分添加一个条目：**

```markdown
# 更新本地文档列表

**<New API>** — `references/<new-api>-docs/` (N 个文件, X.XMB)

- 搜索指南: `references/<new-api>.md`
- 适用场景: ...
```

**2. 在 description 的前置元数据中添加触发关键词**，以便在相关查询时激活该技能：

```yaml
# 更新技能触发关键词
description: "... Triggers on ..., <new-api>, <key-function-names>, ..."
```

如果不更新触发关键词，当用户询问新库时，Claude Code 将无法激活此技能。

---

## 8. 源 URL 列表

本章节汇总了目前所有已支持文档的官方原始链接。

| API              | URL                                                                 |
| ---------------- | ------------------------------------------------------------------- |
| PTX ISA          | `https://docs.nvidia.com/cuda/parallel-thread-execution/index.html` |
| CUDA Runtime API | `https://docs.nvidia.com/cuda/cuda-runtime-api/`                    |
| CUDA Driver API  | `https://docs.nvidia.com/cuda/cuda-driver-api/`                     |
| CUDA Math API    | `https://docs.nvidia.com/cuda/cuda-math-api/`                       |
| cuBLAS           | `https://docs.nvidia.com/cuda/cublas/index.html`                    |
| NCCL             | `https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/`        |
