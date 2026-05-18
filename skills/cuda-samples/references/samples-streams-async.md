# Streams, Events & Asynchronous Execution Samples

## 2.1 simpleStreams

- **Path**: `cpp/0_Introduction/simpleStreams/simpleStreams.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleStreams/simpleStreams.cu>
- **Pattern**: Overlapping kernel execution with H2D/D2H transfers across multiple streams. Creates N streams, then asynchronously pairs memcpy+compute in each.
- **Arch**: All (async overlap requires CC 1.1+)
- **Lines**: ~422

```cuda
cudaStream_t streams[N_STREAMS];
for (int i = 0; i < N_STREAMS; i++) cudaStreamCreate(&streams[i]);
for (int i = 0; i < N_STREAMS; i++) {
    cudaMemcpyAsync(d_data[i], h_data[i], size, cudaMemcpyHostToDevice, streams[i]);
    kernel<<<grid, block, 0, streams[i]>>>(d_data[i], factor[i]);
    cudaMemcpyAsync(h_result[i], d_data[i], size, cudaMemcpyDeviceToHost, streams[i]);
}
```

## 2.2 streamingCopyComputeOverlap (Python)

- **Path**: `python/2_CoreConcepts/streamingCopyComputeOverlap/streamingCopyComputeOverlap.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/2_CoreConcepts/streamingCopyComputeOverlap/streamingCopyComputeOverlap.py>
- **Pattern**: Pure Python demonstration of H2D/D2H overlap with kernel execution using cuda.core streams
- **Arch**: All
- **Lines**: ~312

```python
streams = [device.create_stream() for _ in range(num_streams)]
for s in range(num_streams):
    buf_in[s].copy_to(buf_dev[s], stream=streams[s])
    launch(streams[s], config, kernel, buf_dev[s], buf_out[s])
    buf_out[s].copy_to(buf_host[s], stream=streams[s])
```

## 2.3 simpleIPC

- **Path**: `cpp/0_Introduction/simpleIPC/simpleIPC.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleIPC/simpleIPC.cu>
- **Pattern**: Cross-process GPU memory sharing via `cudaIpcGetMemHandle` / `cudaIpcOpenMemHandle`. Each GPU is owned by a separate process.
- **Arch**: Linux or Windows TCC
- **Lines**: ~357

```cuda
// Process A: export handle
cudaIpcMemHandle_t handle;
cudaIpcGetMemHandle(&handle, d_data);
// Share handle via shared memory or socket to Process B
// Process B: import handle
cudaIpcOpenMemHandle(&d_remote, handle, cudaIpcMemLazyEnablePeerAccess);
// Now d_remote points to Process A's GPU memory
```

## 2.4 simpleP2P

- **Path**: `cpp/0_Introduction/simpleP2P/simpleP2P.cu`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/cpp/0_Introduction/simpleP2P/simpleP2P.cu>
- **Pattern**: Multi-GPU peer-to-peer access: enable P2P, launch kernels accessing remote GPU memory, P2P memcpy
- **Arch**: Multi-GPU with P2P support
- **Lines**: ~263

```cuda
cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);
// Kernel on GPU 0 reads from GPU 1's memory
cudaSetDevice(0);
kernel<<<grid, block>>>(d_data0, d_data1);  // d_data1 allocated on GPU 1
cudaDeviceSynchronize();
```
