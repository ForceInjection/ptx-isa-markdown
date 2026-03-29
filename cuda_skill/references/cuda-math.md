# CUDA Math API Reference

**Related guides:** cublas.md (GEMM/linear algebra), ptx-isa.md (instruction-level), cuda-runtime.md (device properties)

## Table of Contents

- [Local Documentation](#local-documentation) — 40 markdown files, 0.4MB
- [When to Use CUDA Math API Documentation](#when-to-use-cuda-math-api-documentation) — Device intrinsics, narrow-precision types, casting
- [Quick Search Examples](#quick-search-examples) — Fast intrinsics, FP8, half, bfloat16, SIMD
- [Precision Hierarchy](#precision-hierarchy) — Standard vs intrinsic functions
- [Documentation Structure](#documentation-structure) — 14 modules + 26 data structures
- [Narrow-Precision Type Guide](#narrow-precision-type-guide) — FP8/FP6/FP4 type selection
- [Search Tips](#search-tips) — Function naming patterns
- [Common Workflows](#common-workflows) — Activation functions, type casting, FP8 conversion

## Local Documentation

**Complete CUDA Math API documentation is available locally at `cuda-math-docs/`**

The documentation has been converted to markdown with:

- ✅ All device function signatures and descriptions preserved
- ✅ 40 files: 14 modules + 26 data structures (0.4 MB)
- ✅ Full searchability with grep/ripgrep
- ✅ Type layouts and bit-field definitions preserved
- ✅ Navigation, duplicate TOC, and footer removed (26.5% size reduction)

**Note:** All functions are `__device__` unless noted. For `__host__ __device__` variants see Standard C math.

## When to Use CUDA Math API Documentation

Consult CUDA Math API reference when:

1. **Fast device intrinsics** — `__expf`, `__logf`, `__sinf`, `__fmaf_rn` for reduced-latency math in kernels
2. **Half-precision arithmetic** — `__half` and `__half2` operations (`__hadd`, `__hmul`, `__hfma`, vector ops)
3. **Bfloat16 arithmetic** — `__nv_bfloat16` and `__nv_bfloat162` operations for BF16 training kernels
4. **FP8 type layouts** — Bit layout of `__nv_fp8_e4m3`, `__nv_fp8_e5m2`, `__nv_fp8_e8m0` (MX formats)
5. **FP6/FP4 types** — `__nv_fp6_e2m3`, `__nv_fp6_e3m2`, `__nv_fp4_e2m1` for ultra-low-precision
6. **Type casting intrinsics** — `__half2float`, `__float2half_rn`, FP8↔FP32 conversion
7. **SIMD byte/short operations** — Packed 8-bit/16-bit arithmetic in single 32-bit registers
8. **FP128 quad precision** — `__float128` operations for high-precision accumulation

## Quick Search Examples

### Fast single-precision intrinsics

```bash
# All __expf, __logf, __sinf, __cosf etc.
grep "__expf\|__logf\|__powf\|__sinf\|__cosf" cuda-math-docs/modules/group__cuda__math__intrinsic__single.md

# __fmaf_rn (fused multiply-add, round to nearest)
grep -A 10 "__fmaf_rn\b" cuda-math-docs/modules/group__cuda__math__intrinsic__single.md

# Standard sinf/cosf/expf (higher accuracy, slower)
grep "^__device__ float " cuda-math-docs/modules/group__cuda__math__single.md | head -20
```

### Half-precision (**half/**half2) operations

```bash
# All __half arithmetic functions
grep "^__device__.*__half\b" cuda-math-docs/modules/group__cuda__math__intrinsic__half.md | head -30

# __half2 vector operations (2 halfs in one register)
grep "__hadd2\|__hmul2\|__hfma2" cuda-math-docs/modules/group__cuda__math__intrinsic__half.md

# __half struct layout and constructors
cat cuda-math-docs/data-structures/struct____half.md
```

### Bfloat16 operations

```bash
# All __nv_bfloat16 functions
grep "^__device__.*bfloat16" cuda-math-docs/modules/group__cuda__math__intrinsic__bfloat16.md | head -20

# BF16 struct layout
cat cuda-math-docs/data-structures/struct____nv__bfloat16.md
```

### FP8 type layouts and conversion

```bash
# FP8 E4M3 and E5M2 struct definitions
cat cuda-math-docs/data-structures/struct____nv__fp8__e4m3.md
cat cuda-math-docs/data-structures/struct____nv__fp8__e5m2.md

# FP8 E8M0 (MX scale format)
cat cuda-math-docs/data-structures/struct____nv__fp8__e8m0.md

# FP8 intrinsics (conversion, arithmetic)
cat cuda-math-docs/modules/group__cuda__math__intrinsic__fp8.md
```

### Type casting intrinsics

```bash
# half <-> float conversions
grep "__half2float\|__float2half" cuda-math-docs/modules/group__cuda__math__intrinsic__cast.md

# All FP8 <-> float casting functions
grep "__nv_fp8\|__fp8" cuda-math-docs/modules/group__cuda__math__intrinsic__cast.md
```

### SIMD byte/short intrinsics

```bash
# SIMD packed arithmetic (vadd, vsub, vmax in 32-bit registers)
grep "^__device__" cuda-math-docs/modules/group__cuda__math__intrinsic__simd.md | head -20
```

## Precision Hierarchy

For each precision level CUDA offers two function sets:

| Level            | Module                                    | Accuracy                | Speed                              |
| ---------------- | ----------------------------------------- | ----------------------- | ---------------------------------- |
| Standard single  | `group__cuda__math__single.md`            | IEEE 754 compliant      | slower                             |
| Intrinsic single | `group__cuda__math__intrinsic__single.md` | ~1 ULP, device-specific | faster (`__expf`, `__logf`)        |
| Standard double  | `group__cuda__math__double.md`            | IEEE 754 compliant      | slower                             |
| Intrinsic double | `group__cuda__math__intrinsic__double.md` | reduced precision       | faster (`__drcp_rn`, `__dsqrt_rn`) |

**Rule of thumb for vLLM kernels:** Use intrinsic (`__expf`, `__logf`) for attention softmax and activation functions where small accuracy loss is acceptable. Use standard `expf`/`logf` for critical accumulation paths.

## Documentation Structure

```text
cuda-math-docs/
├── modules/                                           # 14 files
│   ├── group__cuda__math__single.md                  # 6. Single Precision Mathematical Functions
│   ├── group__cuda__math__intrinsic__single.md       # 7. Single Precision Intrinsics (__expf etc.)
│   ├── group__cuda__math__double.md                  # 8. Double Precision Mathematical Functions
│   ├── group__cuda__math__intrinsic__double.md       # 9. Double Precision Intrinsics
│   ├── group__cuda__math__intrinsic__half.md         # 4. Half Precision Intrinsics (__half ops)
│   ├── group__cuda__math__intrinsic__bfloat16.md     # 5. Bfloat16 Precision Intrinsics
│   ├── group__cuda__math__intrinsic__fp8.md          # 3. FP8 Intrinsics
│   ├── group__cuda__math__intrinsic__fp6.md          # 2. FP6 Intrinsics
│   ├── group__cuda__math__intrinsic__fp4.md          # 1. FP4 Intrinsics
│   ├── group__cuda__math__intrinsic__cast.md         # 11. Type Casting Intrinsics
│   ├── group__cuda__math__intrinsic__int.md          # 13. Integer Intrinsics
│   ├── group__cuda__math__int.md                    # 12. Integer Mathematical Functions
│   ├── group__cuda__math__intrinsic__simd.md        # 14. SIMD Intrinsics
│   └── group__cuda__math__quad.md                   # 10. FP128 Quad Precision
├── data-structures/                                   # 26 files
│   ├── struct____half.md                             # __half type layout + all member functions
│   ├── struct____half2.md                            # __half2 (packed 2x half)
│   ├── struct____nv__bfloat16.md                    # __nv_bfloat16 layout + ops
│   ├── struct____nv__bfloat162.md                   # __nv_bfloat162 (packed 2x bf16)
│   ├── struct____nv__fp8__e4m3.md                   # FP8 E4M3 (4-bit exp, 3-bit mantissa)
│   ├── struct____nv__fp8__e5m2.md                   # FP8 E5M2 (5-bit exp, 2-bit mantissa)
│   ├── struct____nv__fp8__e8m0.md                   # FP8 E8M0 (MX block scale format)
│   ├── struct____nv__fp8x2__e4m3.md                 # Packed 2x FP8 E4M3
│   ├── struct____nv__fp8x4__e4m3.md                 # Packed 4x FP8 E4M3
│   ├── struct____nv__fp6__e2m3.md                   # FP6 E2M3
│   ├── struct____nv__fp6__e3m2.md                   # FP6 E3M2
│   ├── struct____nv__fp4__e2m1.md                   # FP4 E2M1
│   └── ...                                          # raw variants and packed variants
└── INDEX.md
```

## Narrow-Precision Type Guide

| Type            | File                         | Format           | Use case                           |
| --------------- | ---------------------------- | ---------------- | ---------------------------------- |
| `__nv_fp8_e4m3` | `struct____nv__fp8__e4m3.md` | 1s + 4e + 3m     | Weights/activations (forward pass) |
| `__nv_fp8_e5m2` | `struct____nv__fp8__e5m2.md` | 1s + 5e + 2m     | Gradients (wider range)            |
| `__nv_fp8_e8m0` | `struct____nv__fp8__e8m0.md` | 8e (no mantissa) | MX block scale factors             |
| `__nv_fp6_e2m3` | `struct____nv__fp6__e2m3.md` | 1s + 2e + 3m     | Ultra-low-precision activations    |
| `__nv_fp6_e3m2` | `struct____nv__fp6__e3m2.md` | 1s + 3e + 2m     | Ultra-low-precision weights        |
| `__nv_fp4_e2m1` | `struct____nv__fp4__e2m1.md` | 1s + 2e + 1m     | MX FP4 block format                |
| `__half`        | `struct____half.md`          | FP16             | Standard half-precision            |
| `__nv_bfloat16` | `struct____nv__bfloat16.md`  | BF16             | Training (wider range than FP16)   |

## Search Tips

1. **Device function signatures**: all start with `__device__`

   ```bash
   grep "^__device__ float __expf" cuda-math-docs/modules/group__cuda__math__intrinsic__single.md
   ```

2. **Rounding mode suffixes**: `_rn` (nearest), `_rz` (zero), `_ru` (up), `_rd` (down)

   ```bash
   grep "__fmaf_rn\|__fmaf_rz\|__fmaf_ru\|__fmaf_rd" cuda-math-docs/modules/group__cuda__math__intrinsic__single.md
   ```

3. **Packed/vector types**: `2` suffix for 2-element, `x2`/`x4` in struct names

   ```bash
   grep "^__device__.*__half2\b" cuda-math-docs/modules/group__cuda__math__intrinsic__half.md
   ```

4. **FP8 conversion functions** follow pattern `__nv_cvt_*`

   ```bash
   grep "__nv_cvt_\|__float_as_fp8" cuda-math-docs/modules/group__cuda__math__intrinsic__cast.md
   ```

## Common Workflows

### Softmax kernel: fast exp + reduce

```bash
# Find __expf and max/min intrinsics for softmax numerator
grep "__expf\|__fmaxf\|__fminf" cuda-math-docs/modules/group__cuda__math__intrinsic__single.md

# Find __fdividef for softmax normalization
grep "__fdividef\|__frcp_rn" cuda-math-docs/modules/group__cuda__math__intrinsic__single.md
```

### FP8 ↔ float type conversion

```bash
# All type casting functions (float/half/bf16 <-> fp8)
cat cuda-math-docs/modules/group__cuda__math__intrinsic__cast.md

# FP8 struct constructors (implicit conversion from float)
grep "operator\|__nv_fp8_e4m3(" cuda-math-docs/data-structures/struct____nv__fp8__e4m3.md
```

### Half-precision fused multiply-add

```bash
# __hfma (fused multiply-add for __half)
grep -A 8 "__hfma\b" cuda-math-docs/modules/group__cuda__math__intrinsic__half.md

# __hfma2 (vector 2x half FMA)
grep -A 8 "__hfma2\b" cuda-math-docs/modules/group__cuda__math__intrinsic__half.md
```

### SIMD byte-packed ops (quantized kernels)

```bash
# Packed SIMD operations on 4x int8 packed in int32
grep "^__device__.*unsigned int\|vadd\|vmax\|vsub" cuda-math-docs/modules/group__cuda__math__intrinsic__simd.md | head -20
```

## Version Information

- **CUDA Math API Version**: 13.x
- **Total Size**: 0.4 MB (26.5% reduction from 0.6 MB raw)
- **Files**: 14 modules + 26 data structures + 1 index
- **Source**: https://docs.nvidia.com/cuda/cuda-math-api/
