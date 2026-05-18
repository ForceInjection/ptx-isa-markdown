# CUDA Graphs Samples

## 5.1 simpleCudaGraphs

- **Path**: `cpp/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/simpleCudaGraphs/simpleCudaGraphs.cu>
- **Pattern**: Both Graph API (manual node construction) and Stream Capture API for CUDA Graphs. Graph is instantiated once, launched repeatedly — eliminates per-launch CPU overhead.
- **Arch**: All (CUDA 10.0+)
- **Lines**: ~407

```cuda
// Stream Capture approach
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
kernel_A<<<grid, block, 0, stream>>>(d_a);
kernel_B<<<grid, block, 0, stream>>>(d_a, d_b);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
// Replay many times with zero per-launch overhead
for (int i = 0; i < N; i++) cudaGraphLaunch(instance, stream);
```

## 5.2 cudaGraphs (Python)

- **Path**: `python/2_CoreConcepts/cudaGraphs/cudaGraphs.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/cudaGraphs/cudaGraphs.py>
- **Pattern**: Python cuda.core GraphBuilder API — build graph, upload, launch, and update input data between replays
- **Arch**: All
- **Lines**: ~266

```python
builder = stream.create_graph_builder()
builder.begin_building()
launch(builder, config1, kernel1, buf_a)
launch(builder, config2, kernel2, buf_b)
builder.end_building()
graph = builder.complete()
graph.upload(stream)
graph.launch(stream)  # replay with cached launch
```

## 5.3 jacobiCudaGraphs

- **Path**: `cpp/3_CUDA_Features/jacobiCudaGraphs/jacobiCudaGraphs.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/jacobiCudaGraphs/jacobiCudaGraphs.cu>
- **Pattern**: Updating an instantiated graph with `cudaGraphExecKernelNodeSetParams` and `cudaGraphExecUpdate` — essential for iterative solvers that need to rewrite parameters without rebuilding the graph
- **Arch**: All (CUDA 10.0+)
- **Lines**: ~300

```cuda
// Update kernel node parameters without re-instantiating the graph
cudaGraphExecKernelNodeSetParams(graph_instance, node, &new_params);
// Or use graph update for structural changes
cudaGraphExecUpdate(graph_instance, new_graph, &error_log);
```

## 5.4 graphMemoryNodes

- **Path**: `cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/3_CUDA_Features/graphMemoryNodes/graphMemoryNodes.cu>
- **Pattern**: `cudaGraphAddMemAllocNode` / `cudaGraphAddMemFreeNode` — inline memory allocation within a CUDA Graph, eliminating runtime allocation jitter
- **Arch**: All (CUDA 11.4+)
- **Lines**: ~300

```cuda
cudaGraphAddMemAllocNode(&alloc_node, graph, NULL, 0, &mem_alloc_params);
cudaGraphAddKernelNode(&kernel_node, graph, &alloc_node, 1, &kernel_params);
cudaGraphAddMemFreeNode(&free_node, graph, &kernel_node, 1, d_ptr);
```
