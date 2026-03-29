# NCCL Reference

**Related guides:** cuda-runtime.md (streams, events), cuda-driver.md (context management), debugging-tools.md (compute-sanitizer)

## Table of Contents

- [Local Documentation](#local-documentation) — 33 markdown files, 0.5MB
- [When to Use NCCL Documentation](#when-to-use-nccl-documentation) — Collectives, P2P, multi-GPU, multi-node
- [Quick Search Examples](#quick-search-examples) — AllReduce, comm creation, env vars, errors
- [Key API Functions](#key-api-functions) — Communicators, collectives, P2P
- [Documentation Structure](#documentation-structure) — Page layout
- [Environment Variables Guide](#environment-variables-guide) — NCCL_DEBUG, NCCL_SOCKET_IFNAME, etc.
- [Search Tips](#search-tips) — Function naming patterns
- [Common Workflows](#common-workflows) — AllReduce, tensor parallel, pipeline parallel
- [Troubleshooting](#troubleshooting) — Hangs, NCCL_DEBUG, timeout errors

## Local Documentation

**Complete NCCL documentation is available locally at `nccl-docs/`**

The documentation has been converted to markdown with:

- ✅ All function signatures, parameters, and return values preserved
- ✅ 33 files organized by topic (0.5 MB)
- ✅ Full searchability with grep/ripgrep
- ✅ Environment variable reference
- ✅ Usage guide + API reference + troubleshooting sections

**Note:** Documentation is local and searchable with grep. Links to online resources provided for reference only.

## When to Use NCCL Documentation

Consult NCCL reference when:

1. **Collective operations** — `ncclAllReduce`, `ncclBroadcast`, `ncclReduceScatter`, `ncclAllGather` signatures and semantics
2. **Communicator creation** — `ncclCommInitRank`, `ncclCommInitAll`, `ncclGetUniqueId`, `ncclCommDestroy`
3. **Multi-GPU / multi-node** — Setting up communicators across nodes with `NCCL_SOCKET_IFNAME`, `NCCL_IB_*`
4. **Tensor parallel / pipeline parallel** — vLLM's `tp_group`, all-reduce on attention and MLP outputs
5. **Point-to-point** — `ncclSend` / `ncclRecv` for pipeline parallel bubble hiding
6. **Environment variable tuning** — `NCCL_ALGO`, `NCCL_PROTO`, `NCCL_NTHREADS`, `NCCL_BUFFSIZE`
7. **Error diagnosis** — `ncclResult_t` codes, `NCCL_DEBUG=INFO/WARN/TRACE`, RAS subsystem
8. **Device API** — Device-initiated communication kernels (GIN, remote reduce-and-copy)

## Quick Search Examples

```bash
# AllReduce signature and parameters
grep -r "ncclAllReduce" nccl-docs/api/

# All collective operations
grep -r "^## nccl" nccl-docs/api/colls.md

# Communicator creation functions
grep -r "ncclComm" nccl-docs/api/comms.md | head -30

# All environment variables
grep -r "^## NCCL_" nccl-docs/env.md

# Data types (ncclDataType_t values)
grep -r "ncclFloat\|ncclHalf\|ncclBfloat16" nccl-docs/api/types.md

# Reduction operators
grep -r "ncclSum\|ncclProd\|ncclMax\|ncclMin\|ncclAvg" nccl-docs/api/

# Error codes (ncclResult_t)
grep -r "ncclSuccess\|ncclUnhandledCudaError\|ncclSystemError" nccl-docs/api/types.md

# Point-to-point functions
grep -r "ncclSend\|ncclRecv" nccl-docs/api/p2p.md

# Group call semantics
grep -r "ncclGroupStart\|ncclGroupEnd" nccl-docs/

# Device API host setup
grep -r "ncclDevKernelConfig\|ncclDevChannelMask" nccl-docs/api/device_setup.md
```

## Key API Functions

| Category | Function | Description |
| -------- | -------- | ----------- |
| Communicators | `ncclCommInitRank` | Initialize rank in an nRanks communicator |
| Communicators | `ncclCommInitAll` | Initialize all ranks on current node |
| Communicators | `ncclGetUniqueId` | Generate unique ID for multi-process init |
| Communicators | `ncclCommCount` / `ncclCommUserRank` | Query communicator properties |
| Collectives | `ncclAllReduce` | All-to-all reduction (most common in DL) |
| Collectives | `ncclReduceScatter` | Reduce then scatter shards across ranks |
| Collectives | `ncclAllGather` | Gather shards from all ranks |
| Collectives | `ncclBroadcast` | Broadcast from root to all ranks |
| Collectives | `ncclReduce` | Reduce to single root rank |
| P2P | `ncclSend` / `ncclRecv` | Point-to-point (pipeline parallel) |
| Groups | `ncclGroupStart` / `ncclGroupEnd` | Batch multiple ops into one kernel launch |
| User-defined | `ncclRedOpCreatePreMulSum` | Custom pre-multiplied reduction operator |

## Documentation Structure

```text
nccl-docs/
├── overview.md                    # NCCL overview and concepts
├── setup.md                       # Installation and build
├── usage.md                       # Using NCCL index page
├── usage/
│   ├── communicators.md           # Creating and managing communicators
│   ├── collectives.md             # Collective operations overview
│   ├── data.md                    # Data pointers and memory layout
│   ├── streams.md                 # CUDA stream semantics
│   ├── groups.md                  # Group call semantics
│   ├── p2p.md                     # Point-to-point communication
│   ├── threadsafety.md            # Thread safety rules
│   ├── inplace.md                 # In-place operations
│   ├── cudagraph.md               # Using NCCL with CUDA Graphs
│   ├── bufferreg.md               # User buffer registration
│   └── deviceapi.md               # Device-initiated communication
├── api.md                         # API reference index page
├── api/
│   ├── colls.md                   # Collective functions (AllReduce, etc.)
│   ├── comms.md                   # Communicator creation/management
│   ├── group.md                   # Group call functions
│   ├── p2p.md                     # Point-to-point functions
│   ├── types.md                   # ncclDataType_t, ncclResult_t, ncclRedOp_t
│   ├── ops.md                     # User-defined reduction operators
│   ├── flags.md                   # API supported flags
│   ├── device.md                  # Device API overview
│   ├── device_setup.md            # Device API host-side setup
│   ├── device_memory.md           # Device API memory and LSA
│   ├── device_gin.md              # Device API GIN (GPU-Initiated NCCL)
│   └── device_reducecopy.md       # Remote reduce-and-copy primitives
├── nccl1.md                       # Migration from NCCL 1 to NCCL 2
├── examples.md                    # Code examples
├── mpi.md                         # NCCL + MPI integration
├── env.md                         # Environment variables (NCCL_DEBUG, NCCL_IB_*, etc.)
├── troubleshooting.md             # Troubleshooting guide
├── troubleshooting/ras.md         # RAS (Reliability, Availability, Serviceability)
└── INDEX.md
```

## Environment Variables Guide

Key variables for tuning and debugging (all documented in `env.md`):

| Variable | Common values | Purpose |
| -------- | ------------- | ------- |
| `NCCL_DEBUG` | `WARN`, `INFO`, `TRACE` | Verbosity — start with `INFO` to diagnose hangs |
| `NCCL_SOCKET_IFNAME` | `eth0`, `=ens1f0` | Network interface for host-to-host comms |
| `NCCL_IB_HCA` | `mlx5_0`, `^mlx5_2` | InfiniBand HCA selection |
| `NCCL_ALGO` | `Tree`, `Ring`, `CollNet` | Force collective algorithm |
| `NCCL_PROTO` | `Simple`, `LL`, `LL128` | Protocol selection (LL=low latency) |
| `NCCL_NTHREADS` | `256`, `512` | Threads per NCCL kernel |
| `NCCL_BUFFSIZE` | `4194304` | Per-channel buffer size (bytes) |
| `NCCL_P2P_DISABLE` | `1` | Disable P2P (force NVLink/PCIe paths) |
| `NCCL_TIMEOUT` | `1800` (seconds) | Comm timeout before error |

```bash
# Debug a hang: set before launching
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,COLL

# Find all env vars with their descriptions
grep -A3 "^## NCCL_" nccl-docs/env.md | head -60
```

## Search Tips

- **Collective functions**: all in `api/colls.md` — search `^## nccl` for function list
- **Types and enums**: `api/types.md` — `ncclDataType_t`, `ncclResult_t`, `ncclRedOp_t`
- **Communicator lifecycle**: `api/comms.md` — init, query, destroy
- **Environment variables**: `env.md` — each variable is a `## NCCL_*` heading
- **Troubleshooting patterns**: `troubleshooting.md` + `troubleshooting/ras.md`
- **Device API (GIN)**: `api/device_gin.md`, `api/device_reducecopy.md` — device-initiated comms

## Common Workflows

### Tensor Parallel AllReduce (vLLM pattern)

```bash
# Find AllReduce parameters (data types, ops)
grep -A20 "^## ncclAllReduce" nccl-docs/api/colls.md

# Check group call semantics for batching TP collectives
cat nccl-docs/api/group.md

# Supported data types for FP16/BF16 AllReduce
grep -r "ncclFloat16\|ncclBfloat16" nccl-docs/api/types.md
```

### Pipeline Parallel Send/Recv

```bash
# P2P function signatures
cat nccl-docs/api/p2p.md

# Group semantics for fused send+recv (avoid deadlock)
cat nccl-docs/usage/groups.md
```

### Multi-node Setup

```bash
# Communicator init across nodes (ncclGetUniqueId + ncclCommInitRank)
cat nccl-docs/usage/communicators.md

# Network env vars for multi-node
grep -E "^## NCCL_(SOCKET|IB|NET)" nccl-docs/env.md
```

### CUDA Graph Integration

```bash
# NCCL + CUDA Graph capture requirements
cat nccl-docs/usage/cudagraph.md
```

## Troubleshooting

| Symptom | Where to look | Key env var |
| ------- | ------------- | ----------- |
| Hang on init | `troubleshooting.md` → "Communicator Init" | `NCCL_DEBUG=INFO` |
| Hang during collective | `troubleshooting.md` → "Collective Hang" | `NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=COLL` |
| Timeout error | `troubleshooting.md` | `NCCL_TIMEOUT` |
| IB/RDMA not used | `env.md` NCCL_IB_* section | `NCCL_IB_HCA`, `NCCL_IB_DISABLE` |
| Wrong results | `usage/inplace.md`, `api/types.md` | Verify datatype/op match |
| Performance low | `env.md` NCCL_ALGO/NCCL_PROTO | `NCCL_ALGO=Ring NCCL_PROTO=LL128` |

```bash
# Diagnose with full trace
NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=ALL python your_script.py 2>&1 | grep -i "nccl\|error"

# Check RAS (health monitoring for NVLink)
cat nccl-docs/troubleshooting/ras.md
```
