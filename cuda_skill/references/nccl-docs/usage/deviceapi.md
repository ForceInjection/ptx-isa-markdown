# Device-Initiated Communication

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html

---

# Device-Initiated Communication[](#device-initiated-communication "Permalink to this heading")

Starting with version 2.28, NCCL provides a device-side communication API, making it possible to use communication primitives directly from user CUDA kernels.

## Device API[](#device-api "Permalink to this heading")

Device API consists of the following modules:

>   * **:ref:`LSA <device_api_lsa>` (Load/Store Accessible)** – for communication between devices accessible via memory load/store operations, using CUDA P2P. This includes devices connected over NVLink and some devices connected over PCIe, so long as they have P2P connectivity with each other (as indicated by `nvidia-smi topo -p2p p`). Up to NCCL 2.28.3, the availability of LSA was also subject to the [NCCL_P2P_LEVEL](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#env-nccl-p2p-level) distance check, but that is no longer the case with newer versions.
> 
>   * **Multimem** – for communication between devices using the hardware multicast feature provided by NVLink SHARP (available on some datacenter GPUs since the Hopper generation).
> 
>   * **GIN (GPU-Initiated Networking)** – for communication over the network (since NCCL 2.28.7).
> 
>   * **Reduce, Broadcast, and Fused Building Blocks** — Building Blocks for Computation-Fused Kernels: reduce, copy (broadcast), and reduce-then-copy (see [Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#device-api-reducecopy) in the API reference).
> 
> 


## Requirements[](#requirements "Permalink to this heading")

The device API relies on symmetric memory (see [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg)), which in turn depends on GPU virtual memory management (see [NCCL_CUMEM_ENABLE](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#env-nccl-cumem-enable)) and optionally – for multimem support – on NVLink SHARP (see [NCCL_NVLS_ENABLE](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#env-nccl-nvls-enable)).

GIN has the following requirements:

  * CUDA 12.2 or later when compiling the GPU code

  * NVIDIA GPUs: Volta or newer. NVIDIA GPU drivers >= 510.40.3

  * NVIDIA NICs: CX4 or newer. rdma-core >= 44.0

  * GPU Direct RDMA: GIN host proxy requires DMA-BUF or nvidia-peermem support. GIN GDAKI requires DMA-BUF with kernel version >= 6.1 or nvidia-peermem support

  * Network topology: Requires full NIC connectivity. Does not support topologies where NICs cannot communicate across rails. Also does not support `NCCL_CROSS_NIC=0`.

  * Fused NICs are not supported. To use GIN on dual-port NICs, set `NCCL_IB_MERGE_NICS=0`


Using the host RMA API requires CUDA 12.5 or greater.

Building with EMIT_LLVM_IR=1 (to generate readable LLVM intermediate representation code) requires CUDA 12.

## Cross-Version Compatibility[](#cross-version-compatibility "Permalink to this heading")

NCCL assumes the compile-time version of the device code is the same as the compile-time version of the corresponding host code (i.e., the call to [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate")). Starting with NCCL 2.29, the host-side structures are versioned, to enable cross-version compatibility checks. In general, the compile-time version cannot be newer than the runtime version (e.g., the version of `libnccl.so`). As of NCCL 2.29, backwards compatibility is supported for kernels utilizing LSA and multimem, i.e., a kernel compiled with NCCL 2.29.2/2.29.3 should continue to work when running with NCCL 2.29.7. Kernels utilizing GIN are currently not backwards compatible and need to be recompiled when NCCL is upgraded.

## Host-Side Setup[](#host-side-setup "Permalink to this heading")

To perform communication from the device kernel, a device communicator needs to be created first, using [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate"). Data transfer operations on buffers require symmetric memory windows (see [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg)). A custom communication kernel can then be launched using the standard CUDA syntax. The code excerpt below demonstrates these steps:
    
    
    int main() {
      [...]
      NCCLCHECK(ncclCommInitRank(&comm, nranks, id, rank));
    
      /* Buffer initialization and window creation */
      char* buffer;
      size_t size = 256*1048576;
      NCCLCHECK(ncclMemAlloc((void**)&buffer, size));
      ncclWindow_t win;
      NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));
    
      /* Get device communicator */
      ncclDevComm devComm;
      ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      int nCTAs = 16;
      reqs.lsaBarrierCount = nCTAs;
      NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
    
      /* Launch user kernel */
      customKernel<<<nCTAs, 512>>>(devComm, win);
      [...]
    }
    

Depending on the kernel and application requirements, the same window can be used for input and output, or multiple windows may be needed. When creating a device communicator, the resources that the kernel will need should be specified via the requirements list (see [`ncclDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommRequirements "ncclDevCommRequirements")). In the above example we specify just the number of barriers that our LSA kernel will need, in this case one for each CTA the kernel is to be launched on (16, each CTA running 512 threads).

## Simple LSA Kernel[](#simple-lsa-kernel "Permalink to this heading")
    
    
    template <typename T>
    __global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
      ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x };
      bar.sync(ncclCoopCta(), cuda::memory_order_relaxed);
    
      const int rank = devComm.lsaRank, nRanks = devComm.lsaSize;
      const int globalTid = threadIdx.x + blockDim.x * (rank + blockIdx.x * nRanks);
      const int globalNthreads = blockDim.x * gridDim.x * nRanks;
    
      for (size_t o = globalTid; o < count; o += globalNthreads) {
        T v = 0;
        for (int peer = 0; peer < nRanks; peer++) {
          T* inputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
          v += inputPtr[o];
        }
        for (int peer = 0; peer < nRanks; peer++) {
          T* outputPtr = (T*)ncclGetLsaPointer(win, offset, peer);
          outputPtr[o] = v;
        }
      }
    
      bar.sync(ncclCoopCta(), cuda::memory_order_release);
    }
    

The above code excerpt shows a simple device kernel – an in-place variant (the input buffer is reused for the output) of AllReduce, utilizing LSA support (data is transferred via memory load/store instructions).

The start of the buffer is specified as a (byte-based) _offset_ within the previously registered window _win_ (see [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg)); the buffer consists of _count_ elements of type _T_.

Before the kernel can start processing data, it needs to ensure that all participants are ready. It creates a memory barrier session _bar_ (see `ncclLsaBarrierSession`) and uses it to synchronize across all the threads of the CTA (_ncclCoopCta()_ ; see [Thread Groups](#devapi-coops)) and the ranks of the communicator (_devComm_). _ncclTeamTagLsa_ indicates the subset of ranks the barrier will apply to (see [Teams](#devapi-teams)) – this kernel assumes that all ranks are LSA-connected. _blockIdx.x_ is the CTA’s local index, used to select the barrier.

The kernel then calculates a globally unique index for each thread as well as the overall thread count, and can finally start processing data, using an all-to-all communication pattern. In each iteration of the outer loop, every participating thread loads a single input element from each communicator rank (the first inner loop). `ncclGetLsaPointer()` is used to calculate the locally-accessible address of the start of the buffer within each rank (remote device memory was previously mapped into the local address space – see [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-reg)). Extracted input data is accumulated and the result is stored back at each rank (the second inner loop). Before the kernel terminates, another memory synchronization needs to take place to ensure that all participants have finished processing their data.

Note that this simple implementation would likely fall short of achieving the peak bandwidth, as it utilizes neither vectorization nor loop unrolling. For optimized LSA reduce, copy, and fused reduce-then-copy building blocks (e.g. for AllReduce, AllGather, ReduceScatter), see [Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#device-api-reducecopy) in the Device API reference.

## Multimem Device Kernel[](#multimem-device-kernel "Permalink to this heading")
    
    
    int main() {
      [...]
      reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      int nCTAs = 16;
      reqs.lsaBarrierCount = nCTAs;
      reqs.lsaMultimem = true;
      NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
      [...]
    }
    
    template <typename T>
    __global__ void inPlaceAllReduceKernel(ncclDevComm devComm, ncclWindow_t win, size_t offset, size_t count) {
      ncclLsaBarrierSession<ncclCoopCta> bar { ncclCoopCta(), devComm, ncclTeamTagLsa(), blockIdx.x, /*multimem*/true };
      [...]
      T* mmPtr = (T*)ncclGetLsaMultimemPointer(win, offset, devComm);
      for (size_t o = globalTid; o < count; o += globalNthreads) {
        T v = multimem_sum(mmPtr+o);
        multimem_st(mmPtr+o, v);
      }
      [...]
    }
    

The above code excerpt demonstrates modifications needed to the earlier code segments to enable multimem support (the lines with critical changes are highlighted). On the host side, `lsaMultimem` needs to be set in the requirements prior to creating the device communicator ([`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate") will fail if the necessary hardware support is unavailable).

Within the device kernel, we can switch the memory barrier to a multimem-optimized variant by adding an extra argument to the constructor. The processing loop is actually simpler with multimem: `ncclGetLsaMultimemPointer()` needs to be invoked just once per kernel. The returned multicast memory pointer enables access to the device memory of all the ranks of the communicator without having to iterate over them, and the data can be reduced in hardware. To keep this example simple, the implementations of `multimem_sum` and `multimem_st` are not included; they need to be implemented using PTX, e.g., `multimem.ld_reduce.global.add` and `multimem.st.global`.

## Thread Groups[](#thread-groups "Permalink to this heading")

Many functions in the device API take a thread cooperative group as input to indicate which threads within the CTA will take part in the operation. NCCL provides three predefined ones: `ncclCoopThread()`, `ncclCoopWarp()`, and (the most commonly used) `ncclCoopCta()`.

Users may also pass CUDA cooperative groups, or any class which provides `thread_rank()`, `size()`, and `sync()` methods.

## Teams[](#teams "Permalink to this heading")

To address remote ranks or perform barriers, NCCL refers to subsets of ranks within a communicator as “teams”. NCCL provides three predefined ones:

>   * `ncclTeamWorld()` – the “world” team, encompassing all the ranks of a given communicator.
> 
>   * `ncclTeamLsa()` – all the peers accessible from the local rank using load/store operations.
> 
>   * `ncclTeamRail()` – the set of peers directly accessible from the local rank over the network, assuming that the network fabric is rail-optimized (see [NCCL_CROSS_NIC](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#env-nccl-cross-nic)).
> 
> 


The `ncclTeam` structure contains fairly self-explanatory elements `nRanks`, `rank`, and `stride`. The device API contains functions to verify team membership, convert rank numbers between teams, etc. The world and LSA teams are always contiguous (stride `1`), whereas the rail team is typically not – its stride equals the size of the LSA team (the assumption is thus that each rank _n_ within the local LSA team has direct network connectivity with corresponding ranks _n_ of all remote LSA teams).

## Host-Accessible Device Pointer Functions[](#host-accessible-device-pointer-functions "Permalink to this heading")

Starting with version 2.29, NCCL provides host-accessible functions that enable host code to obtain pointers to LSA memory regions.

The four functions are [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer") (multimem base pointer), [`ncclGetMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetMultimemDevicePointer "ncclGetMultimemDevicePointer") (multimem base pointer with custom handle), [`ncclGetLsaDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaDevicePointer "ncclGetLsaDevicePointer") (LSA peer pointer), and [`ncclGetPeerDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetPeerDevicePointer "ncclGetPeerDevicePointer") (world rank peer pointer). Functions automatically discover the associated communicator from the window object and return `ncclResult_t` error codes.

Usage Example:
    
    
    int main() {
      [...]
      // Allocate symmetric memory buffer
      char* buffer;
      size_t size = 256 * 1024 * 1024;  // 256 MB buffer
      NCCLCHECK(ncclMemAlloc((void**)&buffer, size));
    
      // Create window with the allocated buffer
      ncclWindow_t win;
      NCCLCHECK(ncclCommWindowRegister(comm, buffer, size, &win, NCCL_WIN_COLL_SYMMETRIC));
    
      // Get host-accessible pointers
      void* multimemPtr;
      void* lsaPtr;
      void* peerPtr;
    
      // Get multimem pointer (returns nullptr if multimem not supported)
      NCCLCHECK(ncclGetLsaMultimemDevicePointer(win, 0, &multimemPtr));
      if (multimemPtr == nullptr) {
          // Multimem not available, use fallback
      }
    
      // Get LSA pointer for peer 1
      NCCLCHECK(ncclGetLsaDevicePointer(win, 0, 1, &lsaPtr));
    
      // Get peer pointer for world rank 2
      NCCLCHECK(ncclGetPeerDevicePointer(win, 0, 2, &peerPtr));
    
      // Use pointers in custom kernels or legacy code
      customKernel<<<nCTAs, 256>>>(multimemPtr, lsaPtr, peerPtr);
    
      // Cleanup
      NCCLCHECK(ncclCommWindowDeregister(comm, &win));
      // Device pointers are invalidated after window deregistration
      NCCLCHECK(ncclMemFree(buffer));
      [...]
    }
    

Important notes: Pointer lifetime is limited to the shorter of Window and Communicator lifetime. Functions should be called once and pointers cached for reuse. For detailed function documentation, see [Host-Accessible Device Pointer Functions](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#device-api-host-functions).

## GIN Device Kernel[](#gin-device-kernel "Permalink to this heading")
    
    
    int main() {
      [...]
      reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
      int nCTAs = 1;
      reqs.railGinBarrierCount = nCTAs;
      reqs.ginSignalCount = 1;
      NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));
      [...]
    }
    
    template <typename T>
    __global__ void ginAlltoAllKernel(ncclDevComm devComm, ncclWindow_t win,
                                      size_t inputOffset, size_t outputOffset, size_t count) {
      int ginContext = 0;
      ncclGinSignal_t signalIndex = 0;
      ncclGin gin { devComm, ginContext };
      uint64_t signalValue = gin.readSignal(signalIndex);
    
      ncclGinBarrierSession<ncclCoopCta> bar { ncclCoopCta(), gin, ncclTeamWorld(devComm),
                                               devComm.railGinBarrier, blockIdx.x };
      bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);
    
      const int rank = devComm.rank, nRanks = devComm.nRanks;
      const int tid = threadIdx.x + blockIdx.x * blockDim.x;
      const int nThreads = blockDim.x * gridDim.x;
    
      const size_t size = count * sizeof(T);
      for (int peer = tid; peer < nRanks; peer += nThreads) {
        gin.put(ncclTeamWorld(devComm), peer, win, outputOffset + rank * size,
                win, inputOffset + peer * size, size, ncclGin_SignalInc{signalIndex});
      }
    
      gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + nRanks);
      gin.flush(ncclCoopCta());
    }
    

The above code excerpt demonstrates modifications needed to the earlier host code to enable GIN support, available since NCCL 2.28.7 (the lines with critical changes are highlighted), and also includes a GIN AlltoAll kernel. On the host side, compared to the LSA kernels, we request a launch on just a single CTA (because our kernel doesn’t have much to do) and we set `railGinBarrierCount` and `ginSignalCount` to request GIN-specific barriers and signals ([`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate") will fail if GIN support is unavailable). As with LSA barriers, we need as many of them as CTAs, but signals (used for completion notifications) can be shared between CTAs so, for this simple example, we’ll use just one per rank (for performance-oriented kernels, keeping signals exclusive to each CTA can improve performance).

On the device side, GIN API centers around the `ncclGin` object, initialized using the device communicator and a GIN context index (`0` will do for this simple example but, for performance-oriented kernels, using multiple contexts can provide a performance boost). To avoid race conditions, the initial value of the signal must be read _prior to_ the synchronizing barrier. GIN-specific barriers look much like their LSA counterparts, being local to each CTA, but communicating over the network, not memory. _ncclTeamWorld_ indicates all the ranks of a communicator (this kernel assumes that all the ranks can reach one another over the network, which in general need not be the case – see [NCCL_CROSS_NIC](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#env-nccl-cross-nic)).

Unlike with the AllReduce kernels, for AlltoAll the calculated thread index needs to be unique only locally within each rank. This is then used to determine the destination peer. The main GIN data transfer operation is the one-sided `put()`, here launched in parallel on all participating threads, one per each destination peer (the loop is needed merely if the total rank count exceeds the local thread count – this is why we launched on just a single CTA). `put()` takes the usual arguments such as the destination rank and buffer address, the source buffer, and the transfer size. It also accepts several optional arguments; the above example takes advantage of the _remoteAction_ , requesting that the destination peer increments the value of its local signal once the payload has been settled.

Once the local signal has been incremented by _nRanks_ , we know that every peer has deposited their data in this rank’s output buffer and thus that the buffer is ready; `waitSignal()` can be used to block until that happens. Before terminating, the kernel still needs to `flush()` all the previously initiated outgoing `put()` operations – while that does not guarantee remote completion, it does ensure that the local input buffer is safe to reuse. We can skip an explicit barrier at the end, since `waitSignal()` and `flush()` together ensure that nobody else is using this rank’s buffers.