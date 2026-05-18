# Multi-GPU & Distributed Computing Samples

## 9.1 simpleP2P (Python)

- **Path**: `python/4_DistributedComputing/simpleP2P/simpleP2P.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/4_DistributedComputing/simpleP2P/simpleP2P.py>
- **Pattern**: Multi-GPU peer-to-peer access with cuda.core — detect P2P capability, allocate `peer_accessible_by` buffers, launch kernels accessing remote GPU memory, measure P2P bandwidth
- **Arch**: Multi-GPU with P2P support
- **Lines**: ~377

```python
devices = system.get_devices()
for i in range(n_devs):
    for j in range(n_devs):
        if i != j and devices[i].can_access_peer(devices[j]):
            pairs.append((i, j))

mr0 = DeviceMemoryResource(dev0)
mr0.peer_accessible_by = [gpuid[1]]  # Grant GPU 1 access to GPU 0's memory
buf0 = mr0.allocate(buf_size)
mr1 = DeviceMemoryResource(dev1)
mr1.peer_accessible_by = [gpuid[0]]  # Grant GPU 0 access to GPU 1's memory
buf1 = mr1.allocate(buf_size)
# Launch kernel — each GPU reads from the other's buffer
launch(stream0, config, p2p_kernel, buf0, buf1, buf_size)
# Measure P2P bandwidth via CUDA events
start_event = dev0.create_event(options=EventOptions(enable_timing=True))
end_event = dev0.create_event(options=EventOptions(enable_timing=True))
```

## 9.2 ipcMemoryPool

- **Path**: `python/4_DistributedComputing/ipcMemoryPool/ipcMemoryPool.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/4_DistributedComputing/ipcMemoryPool/ipcMemoryPool.py>
- **Pattern**: Cross-process GPU memory sharing via CUDA IPC — parent creates IPC-enabled `DeviceMemoryResource`, child maps same physical GPU memory through `mp.Queue` pickle transport, round-trip verification
- **Arch**: Linux, GPU supporting POSIX FD-based IPC
- **Lines**: ~220

```python
import multiprocessing as mp

def parent():
    options = DeviceMemoryResourceOptions(ipc_enabled=True)
    mr = DeviceMemoryResource(device, options=options)
    buf = mr.allocate(size)  # IPC-enabled allocation
    buf.copy_from(host_data)  # Fill with known pattern
    queue.put(buf)  # Pickle transport — IPC handle + MR config
    result = queue.get()  # Wait for child
    buf.copy_to(host_verify)
    assert (host_verify == expected_new_pattern).all()

def child():
    buf = queue.get()  # Buffer reconstructed, memory mapped in child process
    assert buf.is_mapped
    buf.copy_to(host_data)
    assert (host_data == expected_parent_pattern).all()  # Verify parent's data
    buf.copy_from(new_pattern)  # Write new data — visible to parent
```

## 9.3 multiGPUGradientAverage

- **Path**: `python/4_DistributedComputing/multiGPUGradientAverage/multiGPUGradientAverage.py`
- **URL**: <https://github.com/NVIDIA/cuda-samples/blob/master/python/4_DistributedComputing/multiGPUGradientAverage/multiGPUGradientAverage.py>
- **Pattern**: Distributed gradient averaging via MPI Allreduce with host staging (GPU→CPU→MPI→CPU→GPU). Each MPI rank computes local gradients on its GPU, then averages across all ranks using CPU-hosted MPI. Works without CUDA-aware MPI.
- **Arch**: Multi-GPU, MPI (`mpi4py`)
- **Lines**: ~416

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Each GPU computes local gradients (value depends on rank for verification)
launch(stream, config, gradient_kernel, grad_buf, n, rank)
# Host staging: GPU → CPU
grad_buf.copy_to(host_grad)
# MPI Allreduce across all ranks
comm.Allreduce(MPI.IN_PLACE, host_grad, op=MPI.SUM)
host_grad /= size  # Average
# CPU → GPU for next iteration
grad_buf.copy_from(host_grad)
```
