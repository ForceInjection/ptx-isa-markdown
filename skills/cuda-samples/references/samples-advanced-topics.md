# Advanced Topics Samples

## 10.1 blockwiseSum (Python)

- **Path**: `python/2_CoreConcepts/blockwiseSum/blockwiseSum.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/blockwiseSum/blockwiseSum.py>
- **Pattern**: Three progressive kernel patterns: (1) simple indexing — one thread per element, (2) grid-stride loop — each thread processes multiple elements via `i += gridDim.x * blockDim.x`, (3) block partial sum — shared memory reduction per block
- **Arch**: All
- **Lines**: ~259

```cuda
// Pattern 2: Grid-stride loop
__global__ void strided_loop(const float* input, float* output, size_t N) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (size_t i = tid; i < N; i += stride) sum += input[i];
    output[tid] = sum;  // Each thread outputs its partial sum
}
```

## 10.2 prefixSum (Python)

- **Path**: `python/2_CoreConcepts/prefixSum/prefixSum.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/prefixSum/prefixSum.py>
- **Pattern**: Production-quality prefix sum (scan) using `cuda.compute.inclusive_scan` / `exclusive_scan` — wraps CUB DeviceScan behind a Pythonic API. Demonstrates cuda.core + cuda-cccl integration.
- **Arch**: All
- **Lines**: ~199

```python
from cuda.compute import inclusive_scan, exclusive_scan, OpKind

# Inclusive scan: output[i] = sum(input[0..i])
inclusive_scan(input_buf, output_buf, op=OpKind.PLUS, stream=stream)
# Exclusive scan: output[i] = sum(input[0..i-1]), output[0] = initial_value
exclusive_scan(input_buf, output_buf, op=OpKind.PLUS, stream=stream,
               initial_value=0)
```

## 10.3 jitLtoLinking

- **Path**: `python/2_CoreConcepts/jitLtoLinking/jitLtoLinking.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/jitLtoLinking/jitLtoLinking.py>
- **Pattern**: JIT link-time optimization linking — compile two translation units separately (one for the main kernel, one for a device function), then link with `Linker` at runtime. Supports PTX and LTO IR modes for plugin-style kernel architectures.
- **Arch**: All
- **Lines**: ~223

```python
# Compile device function as LTO IR
dev_opts = ProgramOptions(relocatable_device_code=True, link_time_optimization=True)
dev_module = Program(device_fn_src, code_type="c++").compile("ltoir", options=dev_opts)
# Compile main kernel as LTO IR
main_opts = ProgramOptions(relocatable_device_code=True, link_time_optimization=True)
main_module = Program(main_kernel_src, code_type="c++").compile("ltoir", options=main_opts)
# Link both modules with LTO
linker = Linker(dev_module, main_module, options=LinkerOptions(link_time_optimization=True))
cubin = linker.link("cubin")
kernel = cubin.get_kernel("apply_transform")
```

## 10.4 greenContext

- **Path**: `python/2_CoreConcepts/greenContext/greenContext.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/greenContext/greenContext.py>
- **Pattern**: Green context — static SM partitioning for latency-sensitive workloads on Hopper+. Splits GPU SMs into isolated groups so critical kernels get dedicated hardware without interference. Runs baseline vs contended vs green context scenarios.
- **Arch**: SM 9.0+ (Hopper)
- **Lines**: ~752

```python
total_sms = device.resources.sm
# Split SMs: group A gets dedicated SMs for latency-sensitive work
latency_group, shared_group = total_sms.split(
    SMResourceOptions(count=(dedicated_sm_count, total_sms.count - dedicated_sm_count))
)
latency_ctx = device.create_context(ContextOptions(resources=[latency_group]))
shared_ctx = device.create_context(ContextOptions(resources=[shared_group]))
# latency_ctx.is_green == True — dedicated hardware, no interference
latency_stream = latency_ctx.create_stream()
launch(latency_stream, config, latency_kernel, buf, n)
```
