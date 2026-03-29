# Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html

---

# Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels[](#device-api-remote-reduce-and-copy-building-blocks-for-custom-communication-kernels "Permalink to this heading")

**Device functions.** All functions on this page are callable from device (GPU) code only. They are **building blocks for computation-fused kernels** : they implement reduce, copy (broadcast), and fused reduce-then-copy operations, keeping communication and computation in a single kernel.

Key points:

  * **Communication patterns:** Sources and destinations can be on remote ranks (using [`ncclWindow_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclWindow_t "ncclWindow_t") as input and output of the API), enabling direct implementation of patterns such as [AllReduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce), [AllGather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather), and [ReduceScatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter).

  * **Building blocks:** Each function implements one peak-bandwidth **communication building block** , not a full algorithm. You can combine these blocks (and your own computation) in tandem to implement custom communication patterns. The three building blocks are:

    * [ReduceSum](#device-api-reducecopy-reducesum) — _reduce_ ; e.g. reduce phase of AllReduce or ReduceScatter

    * [Copy](#device-api-reducecopy-copy) — _broadcast/copy_ ; e.g. broadcast phase of AllReduce or copy in AllGather

    * [ReduceSumCopy](#device-api-reducecopy-reducesumcopy) — _fused reduce-then-copy_ ; e.g. one-step AllReduce or reduce-to-chunks for ReduceScatter

For non-sum reductions, see [Custom Reduction Operators](#device-api-reducecopy-custom-redop).

  * **API forms:** All functions are device-only (callable from `__device__` code) and come in two forms: **high-level convenience overloads** (the direct summation overloads described in the sections below; they work with NCCL windows, teams, and device communicators) and **lambda-based overloads** , which offer more flexibility for custom layouts (see [Lambda-Based (Custom Layouts)](#device-api-reducecopy-lambda)).

  * **GIN:** This API does not support [GIN](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#device-api-gin) (GPU-Initiated Networking) implicitly; use this API within the [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) domain and implement an explicit hierarchical design with NCCL GIN to exchange data between LSA domains.

  * **Invocation model (not rank-collective):** These functions are not rank-collective (unlike host API such as [`ncclAllReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllReduce "ncclAllReduce")). For a given memory region (e.g. a [`ncclWindow_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclWindow_t "ncclWindow_t"), offset, and count), only a single rank must issue the API call that uses that region. The per-operation sections specify whether each role is multi-rank (each rank issues for its own region) or single-rank (one rank issues for that region).

  * **Memory:** Source and destination regions must not overlap (except when exactly in-place: same buffer and same offset); otherwise behavior is undefined. The caller must ensure all arguments and buffer layouts meet the documented requirements; the API does not perform runtime checks.

  * **Alignment:** For best performance, use 16-byte aligned source and destination pointers.


## Compile-Time Requirements[](#compile-time-requirements "Permalink to this heading")

`NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE` must be defined to `1` before including the NCCL device headers so that all block sizes and type combinations are supported for **multimem** operations (see [multimem reduce](#ncclmultimemreducesum-symptr), [multimem copy](#ncclmultimemcopy-symptr), and related multimem APIs below). Without it, certain combinations of low-precision types, _count_ , and pointer alignment in multimem operations may hit runtime asserts. Because only one rank might trigger an assert, this can also lead to hangs. Defining it means the user acknowledges that they are willing to use cutting-edge APIs that might change between releases.

**Lambda-based overloads** The API uses device-side C++ lambda functions for overloads that take callables (e.g. lambdas) to describe source or destination layouts. The API also offers user-facing lambda-based overloads; see [Lambda-Based (Custom Layouts)](#device-api-reducecopy-lambda). Code that includes the NCCL device headers for this API must always be compiled with CUDA extended lambdas enabled (e.g. `--extended-lambda` with nvcc); otherwise you may get a compile-time static assert. See the [CUDA documentation for extended lambdas](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extended-lambdas).

## API Overview[](#api-overview "Permalink to this heading")

  * [ReduceSum](#device-api-reducecopy-reducesum) — Reduce building block.

  * [Copy](#device-api-reducecopy-copy) — Broadcast/copy building block.

  * [ReduceSumCopy](#device-api-reducecopy-reducesumcopy) — Fused reduce-then-copy building block.


**Common template parameters**

> **T**
>     
> 
> Element type. Supported types are: `float`, `double`, `half`, `int8`, `int16`, `int32`, `int64`; and, when available, the following low-precision types: `__nv_bfloat16`, `__nv_fp8_e4m3`, and `__nv_fp8_e5m2`. For low-precision types, sum reduction is accumulated in a wider type:
> 
> **T** | **Accumulation type**  
> ---|---  
> `half` | `float`  
> `__nv_bfloat16` | `float`  
> `__nv_fp8_e4m3` | `half`  
> `__nv_fp8_e5m2` | `half`  
>   
> For multimem reduce, this wider accumulation is performed on the NVLink Switch.
> 
> **Coop**
>     
> 
> Cooperation level (see [Thread Groups](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-coops)), e.g. `ncclCoopCta` or `ncclCoopThread`. All threads in the cooperative group defined by _Coop_ must participate in the call.
> 
> **IntCount**
>     
> 
> Type for the element count. The user can choose a 32-bit integer type (e.g. `unsigned int`) or a 64-bit integer type (e.g. `size_t`) depending on the size of the block region the API operates on.
> 
> **UNROLL**
>     
> 
> Optional; default `4*16/sizeof(T)`. UNROLL represents the tradeoff between register usage and achievable peak bandwidth; the optimal value depends on the register usage of the surrounding kernel. Higher _UNROLL_ allows vectorized load/store and more loop unrolling, which helps achieve peak bandwidth. High register usage can lower occupancy and may lead to register spilling; see the [CUDA Programming Guide section on kernel launch and occupancy](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html#kernel-launch-and-occupancy). The default is chosen to make good performance possible on most systems.
> 
> Example (ReduceSumCopy with _T_ = `float`, _Coop_ = `ncclCoopCta`, _IntCount_ = `size_t`, and _UNROLL_ set to the default for float, `4*16/sizeof(float)` = 16):
>
>> 
>>     size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
>>     size_t dstOffset = [...];  // byte offset into symmetric recv buffer on each peer
>>     ncclLsaReduceSumCopy<float, ncclCoopCta, size_t, 16>(ctaCoop, sendwin, srcOffset, recvwin, dstOffset, count, team);
>>     

## ReduceSum — N Sources to One Destination[](#reducesum-n-sources-to-one-destination "Permalink to this heading")

All ReduceSum variants reduce from N sources to one destination using sum. See [common template parameters](#device-api-reducecopy-common-params) (_T_ , _Coop_ , _IntCount_ , _UNROLL_).

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSum([Coop](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount13ncclDevComm_t "ncclLsaReduceSum::Coop") coop, ncclWindow_t window, size_t offset, [T](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount13ncclDevComm_t "ncclLsaReduceSum::T") *dstPtr, [IntCount](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount13ncclDevComm_t "ncclLsaReduceSum::IntCount") count, ncclDevComm_t devComm)[](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount13ncclDevComm_t "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Reduces from the symmetric buffer at _window_ \+ _offset_ on all [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) peers into local _dstPtr_. The reduction is over all LSA ranks in the communicator; pass _devComm_ (the device communicator).

_coop_ is the cooperative group (see [Thread Groups](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-coops)). _window_ is the window handle from a prior host-side `ncclCommWindowRegister` and must be the same window (and communicator) as _devComm_ ; the buffer region must remain registered for the duration of the call. _offset_ is the byte offset into _window_ where the source buffer starts on each peer; _offset_ \+ _count_ × `sizeof(T)` must not exceed the size of the registered window. _dstPtr_ is the local device pointer to the destination buffer; it must point to at least _count_ elements of type _T_ and must be accessible by all participating threads according to _coop_. _count_ is the number of elements to reduce; it must be the same on all [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) ranks, non-negative, and consistent with _IntCount_. _devComm_ is the device communicator.

**Barrier usage:** When using remote memory, synchronize before and after the call (see example below).

Example:
    
    
    ncclCoopCta ctaCoop;
    ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
    bar.sync(ctaCoop, cuda::memory_order_relaxed);
    
    size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
    size_t dstOffset = [...];  // byte offset into symmetric recv buffer on each peer
    T* dstPtr = (T*)ncclGetLocalPointer(recvwin, dstOffset);
    
    ncclLsaReduceSum<T, ncclCoopCta, size_t>(ctaCoop, sendwin, srcOffset, dstPtr, count, devComm);
    
    bar.sync(ctaCoop, cuda::memory_order_release);
    

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSum([Coop](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount8ncclTeam "ncclLsaReduceSum::Coop") coop, ncclWindow_t window, size_t offset, [T](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount8ncclTeam "ncclLsaReduceSum::T") *dstPtr, [IntCount](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount8ncclTeam "ncclLsaReduceSum::IntCount") count, ncclTeam team)[](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount8ncclTeam "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsareducesum-window-devcomm), except the user passes _team_ explicitly (e.g. `ncclTeamLsa(devComm)`) instead of _devComm_. All other parameters, invocation, and barrier usage are as documented for the overload above.

_team_ is the team of [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) ranks (see [Teams](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-teams)).

Example:
    
    
    ncclTeam team = ncclTeamLsa(devComm);
    ncclLsaReduceSum<T, ncclCoopCta, size_t>(ctaCoop, sendwin, srcOffset, dstPtr, count, team);
    

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSum([Coop](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount13ncclDevComm_t "ncclLsaReduceSum::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount13ncclDevComm_t "ncclLsaReduceSum::T")> src, [T](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount13ncclDevComm_t "ncclLsaReduceSum::T") *dstPtr, [IntCount](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount13ncclDevComm_t "ncclLsaReduceSum::IntCount") count, ncclDevComm_t devComm)[](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount13ncclDevComm_t "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsareducesum-window-devcomm), but the source is given by symmetric pointer _src_ instead of (window, offset). _dstPtr_ , _count_ , and _devComm_ are as for [ncclLsaReduceSum](#nccllsareducesum-window-devcomm). With `ncclSymPtr` you can construct with 0 offset and use `src + elementOffset` (offset in elements, no `sizeof(T)`).

Example:
    
    
    ncclSymPtr<float> src{sendwin, 0};
    ncclLsaReduceSum<float, ncclCoopCta, size_t>(ctaCoop, src + elementOffset, dstPtr, count, devComm);
    

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSum([Coop](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount8ncclTeam "ncclLsaReduceSum::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount8ncclTeam "ncclLsaReduceSum::T")> src, [T](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount8ncclTeam "ncclLsaReduceSum::T") *dstPtr, [IntCount](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount8ncclTeam "ncclLsaReduceSum::IntCount") count, ncclTeam team)[](#_CPPv4I000_iE16ncclLsaReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount8ncclTeam "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsareducesum-symptr-devcomm), except the user passes _team_ explicitly instead of _devComm_. The team is derived from the device communicator (e.g. `ncclTeamLsa(devComm)`).

Example:
    
    
    ncclSymPtr<float> src{sendwin, 0};
    ncclTeam team = ncclTeamLsa(devComm);
    ncclLsaReduceSum<float, ncclCoopCta, size_t>(ctaCoop, src + elementOffset, dstPtr, count, team);
    

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLocalReduceSum([Coop](#_CPPv4I000_iE18ncclLocalReduceSumv4CoopiP1T6size_tP1T8IntCount "ncclLocalReduceSum::Coop") coop, int nSrc, [T](#_CPPv4I000_iE18ncclLocalReduceSumv4CoopiP1T6size_tP1T8IntCount "ncclLocalReduceSum::T") *basePtr, size_t displ, [T](#_CPPv4I000_iE18ncclLocalReduceSumv4CoopiP1T6size_tP1T8IntCount "ncclLocalReduceSum::T") *dstPtr, [IntCount](#_CPPv4I000_iE18ncclLocalReduceSumv4CoopiP1T6size_tP1T8IntCount "ncclLocalReduceSum::IntCount") count)[](#_CPPv4I000_iE18ncclLocalReduceSumv4CoopiP1T6size_tP1T8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaReduceSum](#nccllsareducesum-window-team), but over **local** sources only (no other ranks, no remote memory). Sources are strided: the _i_ -th source is at `basePtr + i*displ` for _i_ = 0… _nSrc_ − 1. _basePtr_ is the base pointer, _displ_ is the stride in bytes; _dstPtr_ and _count_ are as for [ncclLsaReduceSum](#nccllsareducesum-window-team).

**Multimem reduce (ncclMultimemReduceSum)** — Multimem reduce uses NVLink SHARP (NVLS) multicast; the NVLink Switch performs the reduction from multimem sources. To query NVLS/multimem capability from the host, call [`ncclCommQueryProperties()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclCommQueryProperties "ncclCommQueryProperties") and check the `multimemSupport` field of [`ncclCommProperties_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclCommProperties_t "ncclCommProperties_t"). **Multimem restriction:** The local rank (self) must always be part of the multimem reduction or store; the multimem source or destination logically includes the calling rank. For multimem reduce, supported element types are `float`, `double`, `half`; the low-precision types when available: `__nv_bfloat16`, `__nv_fp8_e4m3`, `__nv_fp8_e5m2`; and `int32`, `uint32`, `int64`, `uint64`. The define described above (e.g. `NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1`) may need to be set for all type and block-size combinations.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemReduceSum([Coop](#_CPPv4I000_iE21ncclMultimemReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount18ncclMultimemHandle "ncclMultimemReduceSum::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE21ncclMultimemReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount18ncclMultimemHandle "ncclMultimemReduceSum::T")> src, [T](#_CPPv4I000_iE21ncclMultimemReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount18ncclMultimemHandle "ncclMultimemReduceSum::T") *dstPtr, [IntCount](#_CPPv4I000_iE21ncclMultimemReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount18ncclMultimemHandle "ncclMultimemReduceSum::IntCount") count, ncclMultimemHandle multimemHandle)[](#_CPPv4I000_iE21ncclMultimemReduceSumv4Coop10ncclSymPtrI1TEP1T8IntCount18ncclMultimemHandle "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Reduces from the multimem source _src_ (one logical buffer maps to all participating ranks) into local _dstPtr_. Invocation as for [ncclLsaReduceSum](#nccllsareducesum-window-devcomm). _src_ is the symmetric pointer to the multimem source; _dstPtr_ is the local destination; _count_ is the number of elements; _multimemHandle_ identifies the multimem context. To obtain it, set `lsaMultimem` to true in [`ncclDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommRequirements "ncclDevCommRequirements") when calling [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate"); the handle is then available from the device communicator in device code.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemReduceSum([Coop](#_CPPv4I000_iE21ncclMultimemReduceSumv4CoopP1TP1T8IntCount "ncclMultimemReduceSum::Coop") coop, [T](#_CPPv4I000_iE21ncclMultimemReduceSumv4CoopP1TP1T8IntCount "ncclMultimemReduceSum::T") *mcSrcPtr, [T](#_CPPv4I000_iE21ncclMultimemReduceSumv4CoopP1TP1T8IntCount "ncclMultimemReduceSum::T") *dstPtr, [IntCount](#_CPPv4I000_iE21ncclMultimemReduceSumv4CoopP1TP1T8IntCount "ncclMultimemReduceSum::IntCount") count)[](#_CPPv4I000_iE21ncclMultimemReduceSumv4CoopP1TP1T8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#ncclmultimemreducesum-symptr), but the source is given by raw multimem pointer _mcSrcPtr_ instead of `ncclSymPtr` \+ handle (e.g. from host-side [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer")). _dstPtr_ and _count_ are as for [ncclMultimemReduceSum](#ncclmultimemreducesum-symptr).

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemReduceSum([Coop](#_CPPv4I000_iE21ncclMultimemReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount18ncclMultimemHandle "ncclMultimemReduceSum::Coop") coop, ncclWindow_t window, size_t offset, [T](#_CPPv4I000_iE21ncclMultimemReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount18ncclMultimemHandle "ncclMultimemReduceSum::T") *dstPtr, [IntCount](#_CPPv4I000_iE21ncclMultimemReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount18ncclMultimemHandle "ncclMultimemReduceSum::IntCount") count, ncclMultimemHandle multimemHandle)[](#_CPPv4I000_iE21ncclMultimemReduceSumv4Coop12ncclWindow_t6size_tP1T8IntCount18ncclMultimemHandle "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#ncclmultimemreducesum-symptr), but the source is given by _window_ \+ _offset_ (byte offset) and _multimemHandle_ , analogous to the window-based LSA overload. _dstPtr_ and _count_ are as for [ncclMultimemReduceSum](#ncclmultimemreducesum-symptr).

## Copy (Broadcast) — One Source to N Destinations[](#copy-broadcast-one-source-to-n-destinations "Permalink to this heading")

All Copy variants copy from one source to N destinations. See [common template parameters](#device-api-reducecopy-common-params). Invocation as for [ncclLsaReduceSum](#nccllsareducesum-window-devcomm); for Copy, the source is local to the invoking rank.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaCopy([Coop](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T12ncclWindow_t6size_t8IntCount13ncclDevComm_t "ncclLsaCopy::Coop") coop, [T](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T12ncclWindow_t6size_t8IntCount13ncclDevComm_t "ncclLsaCopy::T") *srcPtr, ncclWindow_t window, size_t offset, [IntCount](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T12ncclWindow_t6size_t8IntCount13ncclDevComm_t "ncclLsaCopy::IntCount") count, ncclDevComm_t devComm)[](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T12ncclWindow_t6size_t8IntCount13ncclDevComm_t "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Copies from local _srcPtr_ into the symmetric buffer at _window_ \+ _offset_ on all [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) peers. Pass _devComm_ (the device communicator). _srcPtr_ is the local source; _window_ and _offset_ define the destination region on each peer; _count_ is the number of elements. Barrier usage: synchronize before and after when using remote memory (see [ncclLsaReduceSum](#nccllsareducesum-window-devcomm)).

Example (e.g. broadcast phase of AllGather):
    
    
    ncclCoopCta ctaCoop;
    ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
    bar.sync(ctaCoop, cuda::memory_order_relaxed);
    
    size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
    size_t dstOffset = [...];  // byte offset into symmetric recv buffer on each peer
    T* srcPtr = (T*)ncclGetLocalPointer(sendwin, srcOffset);
    ncclLsaCopy<T, ncclCoopCta, size_t>(ctaCoop, srcPtr, recvwin, dstOffset, count, devComm);
    
    bar.sync(ctaCoop, cuda::memory_order_release);
    

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaCopy([Coop](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T12ncclWindow_t6size_t8IntCount8ncclTeam "ncclLsaCopy::Coop") coop, [T](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T12ncclWindow_t6size_t8IntCount8ncclTeam "ncclLsaCopy::T") *srcPtr, ncclWindow_t window, size_t offset, [IntCount](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T12ncclWindow_t6size_t8IntCount8ncclTeam "ncclLsaCopy::IntCount") count, ncclTeam team)[](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T12ncclWindow_t6size_t8IntCount8ncclTeam "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsacopy-window-devcomm), except the user passes _team_ explicitly (e.g. `ncclTeamLsa(devComm)`) instead of _devComm_.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaCopy([Coop](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount13ncclDevComm_t "ncclLsaCopy::Coop") coop, [T](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount13ncclDevComm_t "ncclLsaCopy::T") *srcPtr, ncclSymPtr<[T](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount13ncclDevComm_t "ncclLsaCopy::T")> dst, [IntCount](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount13ncclDevComm_t "ncclLsaCopy::IntCount") count, ncclDevComm_t devComm)[](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount13ncclDevComm_t "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsacopy-window-devcomm), but the destination is given by symmetric pointer _dst_ instead of (window, offset). _srcPtr_ , _count_ , and _devComm_ are as for [ncclLsaCopy](#nccllsacopy-window-devcomm). You can construct _dst_ with 0 offset and use `dst + elementOffset` for element-based indexing.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaCopy([Coop](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount8ncclTeam "ncclLsaCopy::Coop") coop, [T](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount8ncclTeam "ncclLsaCopy::T") *srcPtr, ncclSymPtr<[T](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount8ncclTeam "ncclLsaCopy::T")> dst, [IntCount](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount8ncclTeam "ncclLsaCopy::IntCount") count, ncclTeam team)[](#_CPPv4I000_iE11ncclLsaCopyv4CoopP1T10ncclSymPtrI1TE8IntCount8ncclTeam "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsacopy-symptr-devcomm), except the user passes _team_ explicitly instead of _devComm_.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLocalCopy([Coop](#_CPPv4I000_iE13ncclLocalCopyv4CoopP1TiP1T6size_t8IntCount "ncclLocalCopy::Coop") coop, [T](#_CPPv4I000_iE13ncclLocalCopyv4CoopP1TiP1T6size_t8IntCount "ncclLocalCopy::T") *srcPtr, int nDst, [T](#_CPPv4I000_iE13ncclLocalCopyv4CoopP1TiP1T6size_t8IntCount "ncclLocalCopy::T") *basePtr, size_t displ, [IntCount](#_CPPv4I000_iE13ncclLocalCopyv4CoopP1TiP1T6size_t8IntCount "ncclLocalCopy::IntCount") count)[](#_CPPv4I000_iE13ncclLocalCopyv4CoopP1TiP1T6size_t8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaCopy](#nccllsacopy-window-team) for thread-cooperation behavior, but **local** only (no other ranks, no remote memory): copies from single source _srcPtr_ to _nDst_ strided destinations at `basePtr + i*displ` for _i_ = 0… _nDst_ − 1. _displ_ is the stride in bytes; _count_ is the number of elements copied to each destination.

**Multimem copy (ncclMultimemCopy)** — Copies from one local source to one multimem destination (one logical buffer over all ranks). Uses NVLink SHARP (NVLS) multicast. Query `multimemSupport` for capability; _multimemHandle_ as for [multimem reduce](#ncclmultimemreducesum-symptr). All element types are supported; for types less than 32 bits wide, the define described above (e.g. `NCCL_DEVICE_PERMIT_EXPERIMENTAL_CODE=1`) must be set for some count and pointer combinations.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemCopy([Coop](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1T10ncclSymPtrI1TE8IntCount18ncclMultimemHandle "ncclMultimemCopy::Coop") coop, [T](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1T10ncclSymPtrI1TE8IntCount18ncclMultimemHandle "ncclMultimemCopy::T") *srcPtr, ncclSymPtr<[T](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1T10ncclSymPtrI1TE8IntCount18ncclMultimemHandle "ncclMultimemCopy::T")> dst, [IntCount](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1T10ncclSymPtrI1TE8IntCount18ncclMultimemHandle "ncclMultimemCopy::IntCount") count, ncclMultimemHandle multimemHandle)[](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1T10ncclSymPtrI1TE8IntCount18ncclMultimemHandle "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Copies from local _srcPtr_ into the multimem destination _dst_ (one logical buffer maps to all participating ranks). Invocation as for [ncclLsaCopy](#nccllsacopy-window-devcomm). _multimemHandle_ as for [ncclMultimemReduceSum](#ncclmultimemreducesum-symptr).

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemCopy([Coop](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1TP1T8IntCount "ncclMultimemCopy::Coop") coop, [T](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1TP1T8IntCount "ncclMultimemCopy::T") *srcPtr, [T](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1TP1T8IntCount "ncclMultimemCopy::T") *mcDstPtr, [IntCount](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1TP1T8IntCount "ncclMultimemCopy::IntCount") count)[](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1TP1T8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#ncclmultimemcopy-symptr), but the destination is given by raw multimem pointer _mcDstPtr_ (e.g. from [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer")). _srcPtr_ and _count_ are as for [ncclMultimemCopy](#ncclmultimemcopy-symptr).

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemCopy([Coop](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1T12ncclWindow_t6size_t8IntCount18ncclMultimemHandle "ncclMultimemCopy::Coop") coop, [T](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1T12ncclWindow_t6size_t8IntCount18ncclMultimemHandle "ncclMultimemCopy::T") *srcPtr, ncclWindow_t window, size_t offset, [IntCount](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1T12ncclWindow_t6size_t8IntCount18ncclMultimemHandle "ncclMultimemCopy::IntCount") count, ncclMultimemHandle multimemHandle)[](#_CPPv4I000_iE16ncclMultimemCopyv4CoopP1T12ncclWindow_t6size_t8IntCount18ncclMultimemHandle "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#ncclmultimemcopy-symptr), but the destination is given by _window_ \+ _offset_ (byte offset) and _multimemHandle_. _srcPtr_ and _count_ are as for [ncclMultimemCopy](#ncclmultimemcopy-symptr).

## ReduceSumCopy[](#reducesumcopy "Permalink to this heading")

ReduceSumCopy combines reduction and copy into a single call. See [common template parameters](#device-api-reducecopy-common-params). Invocation as documented for the [ncclLsaReduceSum](#nccllsareducesum-window-devcomm) and [ncclLsaCopy](#nccllsacopy-window-devcomm) overloads above.

### LSA ReduceSumCopy (ncclLsaReduceSumCopy)[](#lsa-reducesumcopy-nccllsareducesumcopy "Permalink to this heading")

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSumCopy([Coop](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop12ncclWindow_t6size_t12ncclWindow_t6size_t8IntCount13ncclDevComm_t "ncclLsaReduceSumCopy::Coop") coop, ncclWindow_t srcWindow, size_t srcOffset, ncclWindow_t dstWindow, size_t dstOffset, [IntCount](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop12ncclWindow_t6size_t12ncclWindow_t6size_t8IntCount13ncclDevComm_t "ncclLsaReduceSumCopy::IntCount") count, ncclDevComm_t devComm)[](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop12ncclWindow_t6size_t12ncclWindow_t6size_t8IntCount13ncclDevComm_t "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Reduces from the [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) source at _srcWindow_ \+ _srcOffset_ (all LSA peers) and copies the result to the LSA destination at _dstWindow_ \+ _dstOffset_ (all LSA peers) in one call. Pass _devComm_ (the device communicator). _srcOffset_ and _dstOffset_ are byte offsets; _count_ is the number of elements. When using remote memory, barrier usage is the same as for [ncclLsaReduceSum](#nccllsareducesum-window-devcomm) and [ncclLsaCopy](#nccllsacopy-window-devcomm): synchronize before and after the call (see the examples there).

Example (e.g. LSA AllReduce; see `test/perf/all_reduce.cu` for block-parallel chunking):
    
    
    ncclCoopCta ctaCoop;
    ncclLsaBarrierSession<ncclCoopCta> bar { ctaCoop, devComm, ncclTeamLsa(devComm), devComm.lsaBarrier, blockIdx.x };
    bar.sync(ctaCoop, cuda::memory_order_relaxed);
    
    size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
    size_t dstOffset = [...];  // byte offset into symmetric recv buffer on each peer
    ncclLsaReduceSumCopy<T, ncclCoopCta, size_t>(ctaCoop, sendwin, srcOffset, recvwin, dstOffset, count, devComm);
    
    bar.sync(ctaCoop, cuda::memory_order_release);
    

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSumCopy([Coop](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop12ncclWindow_t6size_t12ncclWindow_t6size_t8IntCount8ncclTeam "ncclLsaReduceSumCopy::Coop") coop, ncclWindow_t srcWindow, size_t srcOffset, ncclWindow_t dstWindow, size_t dstOffset, [IntCount](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop12ncclWindow_t6size_t12ncclWindow_t6size_t8IntCount8ncclTeam "ncclLsaReduceSumCopy::IntCount") count, ncclTeam team)[](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop12ncclWindow_t6size_t12ncclWindow_t6size_t8IntCount8ncclTeam "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsareducesumcopy-window-devcomm), except the user passes _team_ explicitly (e.g. `ncclTeamLsa(devComm)`) instead of _devComm_.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSumCopy([Coop](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount13ncclDevComm_t "ncclLsaReduceSumCopy::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount13ncclDevComm_t "ncclLsaReduceSumCopy::T")> src, ncclSymPtr<[T](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount13ncclDevComm_t "ncclLsaReduceSumCopy::T")> dst, [IntCount](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount13ncclDevComm_t "ncclLsaReduceSumCopy::IntCount") count, ncclDevComm_t devComm)[](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount13ncclDevComm_t "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsareducesumcopy-window-devcomm), but the source is given by symmetric pointer _src_ and the destination by symmetric pointer _dst_ instead of (window, offset). _count_ and _devComm_ are as for [ncclLsaReduceSumCopy](#nccllsareducesumcopy-window-devcomm).

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSumCopy([Coop](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount8ncclTeam "ncclLsaReduceSumCopy::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount8ncclTeam "ncclLsaReduceSumCopy::T")> src, ncclSymPtr<[T](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount8ncclTeam "ncclLsaReduceSumCopy::T")> dst, [IntCount](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount8ncclTeam "ncclLsaReduceSumCopy::IntCount") count, ncclTeam team)[](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE10ncclSymPtrI1TE8IntCount8ncclTeam "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsareducesumcopy-symptr-devcomm), except the user passes _team_ explicitly instead of _devComm_.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSumCopy([Coop](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE8ncclTeam8IntCount "ncclLsaReduceSumCopy::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE8ncclTeam8IntCount "ncclLsaReduceSumCopy::T")> src, ncclTeam srcTeam, ncclSymPtr<[T](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE8ncclTeam8IntCount "ncclLsaReduceSumCopy::T")> dst, ncclTeam dstTeam, [IntCount](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE8ncclTeam8IntCount "ncclLsaReduceSumCopy::IntCount") count)[](#_CPPv4I000_iE20ncclLsaReduceSumCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE8ncclTeam8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaReduceSumCopy](#nccllsareducesumcopy-window-team), but source and destination use different teams (_srcTeam_ and _dstTeam_). Ranks in one team must still be load-store accessible (LSA) from ranks in the other (same LSA communicator; involved ranks must be able to access each other’s registered memory). _src_ is the source symmetric pointer over _srcTeam_ ; _dst_ is the destination symmetric pointer over _dstTeam_.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLocalReduceSumCopy([Coop](#_CPPv4I000_iE22ncclLocalReduceSumCopyv4CoopiP1T6size_tiP1T6size_t8IntCount "ncclLocalReduceSumCopy::Coop") coop, int nSrc, [T](#_CPPv4I000_iE22ncclLocalReduceSumCopyv4CoopiP1T6size_tiP1T6size_t8IntCount "ncclLocalReduceSumCopy::T") *srcBasePtr, size_t srcDispl, int nDst, [T](#_CPPv4I000_iE22ncclLocalReduceSumCopyv4CoopiP1T6size_tiP1T6size_t8IntCount "ncclLocalReduceSumCopy::T") *dstBasePtr, size_t dstDispl, [IntCount](#_CPPv4I000_iE22ncclLocalReduceSumCopyv4CoopiP1T6size_tiP1T6size_t8IntCount "ncclLocalReduceSumCopy::IntCount") count)[](#_CPPv4I000_iE22ncclLocalReduceSumCopyv4CoopiP1T6size_tiP1T6size_t8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaReduceSumCopy](#nccllsareducesumcopy-window-team) for thread-cooperation behavior, but **local** only (no other ranks, no remote memory). Reduces from _nSrc_ strided sources at `srcBasePtr + i*srcDispl` (i = 0… _nSrc_ − 1) and copies the result to _nDst_ strided destinations at `dstBasePtr + j*dstDispl` (j = 0… _nDst_ − 1). _srcDispl_ and _dstDispl_ are strides in bytes; _count_ is the number of elements per source/destination.

### Multimem ReduceSumCopy (ncclMultimemReduceSumCopy)[](#multimem-reducesumcopy-ncclmultimemreducesumcopy "Permalink to this heading")

Reduces from one multimem source and copies to one multimem destination (each one logical buffer maps to all participating ranks) in one call. To query multimem capability from the host, call [`ncclCommQueryProperties()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclCommQueryProperties "ncclCommQueryProperties") and check the `multimemSupport` field of [`ncclCommProperties_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclCommProperties_t "ncclCommProperties_t"). The multimem handle is obtained by setting `lsaMultimem` to true in [`ncclDevCommRequirements`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommRequirements "ncclDevCommRequirements") when calling [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate"); it is then available from the device communicator in device code.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemReduceSumCopy([Coop](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4Coop12ncclWindow_t6size_t18ncclMultimemHandle12ncclWindow_t6size_t18ncclMultimemHandle8IntCount "ncclMultimemReduceSumCopy::Coop") coop, ncclWindow_t srcWindow, size_t srcOffset, ncclMultimemHandle srcHandle, ncclWindow_t dstWindow, size_t dstOffset, ncclMultimemHandle dstHandle, [IntCount](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4Coop12ncclWindow_t6size_t18ncclMultimemHandle12ncclWindow_t6size_t18ncclMultimemHandle8IntCount "ncclMultimemReduceSumCopy::IntCount") count)[](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4Coop12ncclWindow_t6size_t18ncclMultimemHandle12ncclWindow_t6size_t18ncclMultimemHandle8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Reduces from the multimem source at _srcWindow_ \+ _srcOffset_ and copies to the multimem destination at _dstWindow_ \+ _dstOffset_ in one call. _srcHandle_ and _dstHandle_ identify the multimem contexts (may be the same or different). Invocation as documented for the [overload above](#nccllsareducesum-window-team).

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemReduceSumCopy([Coop](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "ncclMultimemReduceSumCopy::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "ncclMultimemReduceSumCopy::T")> src, ncclMultimemHandle srcHandle, ncclSymPtr<[T](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "ncclMultimemReduceSumCopy::T")> dst, ncclMultimemHandle dstHandle, [IntCount](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "ncclMultimemReduceSumCopy::IntCount") count)[](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#ncclmultimemreducesumcopy-window), but the source is given by symmetric pointer _src_ and the destination by symmetric pointer _dst_ instead of (window, offset). _srcHandle_ and _dstHandle_ are as for [ncclMultimemReduceSumCopy](#ncclmultimemreducesumcopy-window).

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemReduceSumCopy([Coop](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4CoopP1TP1T8IntCount "ncclMultimemReduceSumCopy::Coop") coop, [T](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4CoopP1TP1T8IntCount "ncclMultimemReduceSumCopy::T") *mcSrcPtr, [T](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4CoopP1TP1T8IntCount "ncclMultimemReduceSumCopy::T") *mcDstPtr, [IntCount](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4CoopP1TP1T8IntCount "ncclMultimemReduceSumCopy::IntCount") count)[](#_CPPv4I000_iE25ncclMultimemReduceSumCopyv4CoopP1TP1T8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#ncclmultimemreducesumcopy-symptr), but source and destination are given by raw multimem pointers _mcSrcPtr_ and _mcDstPtr_ (e.g. from [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer")).

### Mixed LSA and Multimem ReduceSumCopy[](#mixed-lsa-and-multimem-reducesumcopy "Permalink to this heading")

Reduce from [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) and write to multimem, or reduce from multimem and write to LSA, in one call.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSumMultimemCopy([Coop](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "ncclLsaReduceSumMultimemCopy::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "ncclLsaReduceSumMultimemCopy::T")> src, ncclTeam srcTeam, ncclSymPtr<[T](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "ncclLsaReduceSumMultimemCopy::T")> dst, ncclMultimemHandle dstHandle, [IntCount](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "ncclLsaReduceSumMultimemCopy::IntCount") count)[](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeam10ncclSymPtrI1TE18ncclMultimemHandle8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Reduces from [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) source _src_ over _srcTeam_ and copies to multimem destination _dst_ (one logical buffer maps to all participating ranks). _dstHandle_ as for [ncclMultimemCopy](#ncclmultimemcopy-symptr). Invocation as documented for [ncclLsaReduceSum](#nccllsareducesum-window-devcomm) and [ncclMultimemCopy](#ncclmultimemcopy-symptr).

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclLsaReduceSumMultimemCopy([Coop](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeamP1T8IntCount "ncclLsaReduceSumMultimemCopy::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeamP1T8IntCount "ncclLsaReduceSumMultimemCopy::T")> src, ncclTeam srcTeam, [T](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeamP1T8IntCount "ncclLsaReduceSumMultimemCopy::T") *mcDstPtr, [IntCount](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeamP1T8IntCount "ncclLsaReduceSumMultimemCopy::IntCount") count)[](#_CPPv4I000_iE28ncclLsaReduceSumMultimemCopyv4Coop10ncclSymPtrI1TE8ncclTeamP1T8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#nccllsareducesummultimemcopy-symptr), but the multimem destination is given by raw pointer _mcDstPtr_ instead of `ncclSymPtr` \+ handle.

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemReduceSumLsaCopy([Coop](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE8ncclTeam8IntCount "ncclMultimemReduceSumLsaCopy::Coop") coop, ncclSymPtr<[T](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE8ncclTeam8IntCount "ncclMultimemReduceSumLsaCopy::T")> src, ncclMultimemHandle srcHandle, ncclSymPtr<[T](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE8ncclTeam8IntCount "ncclMultimemReduceSumLsaCopy::T")> dst, ncclTeam dstTeam, [IntCount](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE8ncclTeam8IntCount "ncclMultimemReduceSumLsaCopy::IntCount") count)[](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4Coop10ncclSymPtrI1TE18ncclMultimemHandle10ncclSymPtrI1TE8ncclTeam8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Reduces from multimem source _src_ and copies to [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) destination _dst_ over _dstTeam_. _srcHandle_ as for [ncclMultimemReduceSum](#ncclmultimemreducesum-symptr). Invocation as documented for [ncclMultimemReduceSum](#ncclmultimemreducesum-symptr) and [ncclLsaCopy](#nccllsacopy-window-devcomm).

template<typename T, typename Coop, typename IntCount, int UNROLL>  
void ncclMultimemReduceSumLsaCopy([Coop](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4CoopP1T10ncclSymPtrI1TE8ncclTeam8IntCount "ncclMultimemReduceSumLsaCopy::Coop") coop, [T](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4CoopP1T10ncclSymPtrI1TE8ncclTeam8IntCount "ncclMultimemReduceSumLsaCopy::T") *mcSrcPtr, ncclSymPtr<[T](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4CoopP1T10ncclSymPtrI1TE8ncclTeam8IntCount "ncclMultimemReduceSumLsaCopy::T")> dst, ncclTeam dstTeam, [IntCount](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4CoopP1T10ncclSymPtrI1TE8ncclTeam8IntCount "ncclMultimemReduceSumLsaCopy::IntCount") count)[](#_CPPv4I000_iE28ncclMultimemReduceSumLsaCopyv4CoopP1T10ncclSymPtrI1TE8ncclTeam8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [above](#ncclmultimemreducesumlsacopy-symptr), but the multimem source is given by raw pointer _mcSrcPtr_ instead of `ncclSymPtr` \+ handle.

## Lambda-Based (Custom Layouts)[](#lambda-based-custom-layouts "Permalink to this heading")

Lambda-based overloads give more flexibility and allow custom memory layouts for reduce and/or copy. They can mix local and [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa)-remote sources or destinations (e.g. one source from local memory, others from LSA windows), and can express non-contiguous or index-dependent addressing that the fixed window/symptr overloads do not support.

**Conditions for the lambda.** The callable (e.g. lambda) must return `T*` and be invocable from device code with a single index argument. When the callable is a lambda, it must be qualified with `__device__` and the build must have CUDA extended lambdas enabled (see [Compile-Time Requirements](#device-api-reducecopy-compile)). Let _n_ be the associated count (_nSrc_ or _nDst_ depending on the API).

  1. _n_ > 0\. Otherwise behavior is undefined.

  2. For every index _i_ in [0, _n_), the call _lambda*(*i_) must return a pointer to the start of a valid region of at least _count_ contiguous elements of type _T_. The same restrictions apply as for the corresponding non-lambda API: e.g. for [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) sources/destinations the region must be in registered LSA memory and remain valid for the duration of the call; for local pointers they must be accessible to all threads in _coop_.

  3. When the API designates source or destination as **multimem** , every pointer returned by the lambda for that side must be a **multimem pointer** (e.g. from [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer")). Using LSA or local pointers for the multimem side is undefined behavior. Conversely, multimem pointers cannot be used where LSA or local pointers are accepted.

  4. The relationship between _n_ and the logical set of sources or destinations is as documented for each API (e.g. _nSrc_ sources meaning _nSrc_ distinct source regions, or _nDst_ destinations meaning _nDst_ distinct destination regions).


Violating any of these conditions is undefined behavior; the API does not perform runtime checks.

Example: reduce from a window over a team, but **exclude the local rank** (e.g. reduce only from remote peers). Use a source lambda that maps index _i_ to the _i_ -th _other_ rank and pass _nSrc_ = _team_`.nRanks` − 1:
    
    
    size_t srcOffset = [...];  // byte offset into symmetric send buffer on each peer
    ncclTeam team = ncclTeamLsa(devComm);
    int myRank = devComm.rank;
    int nSrc = team.nRanks - 1;   // all ranks except this one
    
    auto srcLambda = [=] __device__ (int i) -> T* {
      int peer = (i < myRank) ? i : i + 1;   // skip myRank
      return (T*)ncclGetLsaPointer(sendwin, srcOffset, peer);
    };
    
    ncclLsaReduceSum<T, ncclCoopCta, size_t>(ctaCoop, srcLambda, nSrc, dstPtr, count);
    

template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>  
void ncclLsaReduceSum([Coop](#_CPPv4I0000_iE16ncclLsaReduceSumv4Coop9SrcLambdaiP1T8IntCount "ncclLsaReduceSum::Coop") coop, [SrcLambda](#_CPPv4I0000_iE16ncclLsaReduceSumv4Coop9SrcLambdaiP1T8IntCount "ncclLsaReduceSum::SrcLambda") srcLambda, int nSrc, [T](#_CPPv4I0000_iE16ncclLsaReduceSumv4Coop9SrcLambdaiP1T8IntCount "ncclLsaReduceSum::T") *dstPtr, [IntCount](#_CPPv4I0000_iE16ncclLsaReduceSumv4Coop9SrcLambdaiP1T8IntCount "ncclLsaReduceSum::IntCount") count)[](#_CPPv4I0000_iE16ncclLsaReduceSumv4Coop9SrcLambdaiP1T8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaReduceSum](#nccllsareducesum-window-team), but the source layout is given by _srcLambda*(index) returning ``T*`` for each of *nSrc_ sources; result to local _dstPtr_. _coop_ and _count_ as for [ncclLsaReduceSum](#nccllsareducesum-window-team). _srcLambda_ is called with indices 0 to _nSrc_ − 1.

template<typename T, typename Coop, typename SrcLambda, typename IntCount, int UNROLL>  
void ncclLocalReduceSum([Coop](#_CPPv4I0000_iE18ncclLocalReduceSumv4Coop9SrcLambdaiP1T8IntCount "ncclLocalReduceSum::Coop") coop, [SrcLambda](#_CPPv4I0000_iE18ncclLocalReduceSumv4Coop9SrcLambdaiP1T8IntCount "ncclLocalReduceSum::SrcLambda") srcLambda, int nSrc, [T](#_CPPv4I0000_iE18ncclLocalReduceSumv4Coop9SrcLambdaiP1T8IntCount "ncclLocalReduceSum::T") *dstPtr, [IntCount](#_CPPv4I0000_iE18ncclLocalReduceSumv4Coop9SrcLambdaiP1T8IntCount "ncclLocalReduceSum::IntCount") count)[](#_CPPv4I0000_iE18ncclLocalReduceSumv4Coop9SrcLambdaiP1T8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaReduceSum](#nccllsareducesum-window-team), but over **local** sources only (no other ranks, no remote memory). _srcLambda*(index) returns ``T*`` for each of *nSrc_ sources; result to _dstPtr_. _coop_ and _count_ as for [ncclLsaReduceSum](#nccllsareducesum-window-team). _srcLambda_ is called with indices 0 to _nSrc_ − 1.

template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>  
void ncclLsaCopy([Coop](#_CPPv4I0000_iE11ncclLsaCopyv4CoopP1T9DstLambdai8IntCount "ncclLsaCopy::Coop") coop, [T](#_CPPv4I0000_iE11ncclLsaCopyv4CoopP1T9DstLambdai8IntCount "ncclLsaCopy::T") *srcPtr, [DstLambda](#_CPPv4I0000_iE11ncclLsaCopyv4CoopP1T9DstLambdai8IntCount "ncclLsaCopy::DstLambda") dstLambda, int nDst, [IntCount](#_CPPv4I0000_iE11ncclLsaCopyv4CoopP1T9DstLambdai8IntCount "ncclLsaCopy::IntCount") count)[](#_CPPv4I0000_iE11ncclLsaCopyv4CoopP1T9DstLambdai8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaCopy](#nccllsacopy-window-team), but the destination layout is given by _dstLambda*(index) returning ``T*`` for each of *nDst_ destinations. _srcPtr_ is the local source; _coop_ and _count_ as for [ncclLsaCopy](#nccllsacopy-window-team). _dstLambda_ is called with indices 0 to _nDst_ − 1.

template<typename T, typename Coop, typename DstLambda, typename IntCount, int UNROLL>  
void ncclLocalCopy([Coop](#_CPPv4I0000_iE13ncclLocalCopyv4CoopP1T9DstLambdai8IntCount "ncclLocalCopy::Coop") coop, [T](#_CPPv4I0000_iE13ncclLocalCopyv4CoopP1T9DstLambdai8IntCount "ncclLocalCopy::T") *srcPtr, [DstLambda](#_CPPv4I0000_iE13ncclLocalCopyv4CoopP1T9DstLambdai8IntCount "ncclLocalCopy::DstLambda") dstLambda, int nDst, [IntCount](#_CPPv4I0000_iE13ncclLocalCopyv4CoopP1T9DstLambdai8IntCount "ncclLocalCopy::IntCount") count)[](#_CPPv4I0000_iE13ncclLocalCopyv4CoopP1T9DstLambdai8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaCopy](#nccllsacopy-window-team), but **local** only (no other ranks, no remote memory). _dstLambda*(index) returns ``T*`` for each of *nDst_ destinations; _srcPtr_ , _coop_ , and _count_ as for [ncclLsaCopy](#nccllsacopy-window-team). _dstLambda_ is called with indices 0 to _nDst_ − 1.

The following four overloads are the lambda-based forms of ReduceSumCopy. They differ by whether the **source** and **destination** are **:ref:`LSA <device_api_lsa>`** or **multimem**. For any **multimem** side, the common case is a single multimem pointer (one already represents multiple remote spaces). Multiple multimem pointers are also supported; the API then initiates multicast to or from all of them.

Warning

Multicast always includes the self rank. With more than one multimem source or destination, this creates overlapping ranks. The user must ensure correctness.

template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>  
void ncclLsaReduceSumLsaCopy([Coop](#_CPPv4I00000_iE23ncclLsaReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclLsaReduceSumLsaCopy::Coop") coop, [SrcLambda](#_CPPv4I00000_iE23ncclLsaReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclLsaReduceSumLsaCopy::SrcLambda") srcLambda, int nSrc, [DstLambda](#_CPPv4I00000_iE23ncclLsaReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclLsaReduceSumLsaCopy::DstLambda") dstLambda, int nDst, [IntCount](#_CPPv4I00000_iE23ncclLsaReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclLsaReduceSumLsaCopy::IntCount") count)[](#_CPPv4I00000_iE23ncclLsaReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

**Source: :ref:`LSA <device_api_lsa>`. Destination: LSA.** Same as [ncclLsaReduceSumCopy](#nccllsareducesumcopy-window-team), but the source layout is given by _srcLambda*(*i_) returning `T*` for each of _nSrc_ sources (_i_ = 0 to _nSrc_ − 1) and the destination layout by _dstLambda*(*j_) for each of _nDst_ destinations (_j_ = 0 to _nDst_ − 1). _coop_ and _count_ as for [ncclLsaReduceSumCopy](#nccllsareducesumcopy-window-team). When using remote memory, barrier usage is as for [ncclLsaReduceSumCopy](#nccllsareducesumcopy-window-devcomm).

template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>  
void ncclLsaReduceSumMultimemCopy([Coop](#_CPPv4I00000_iE28ncclLsaReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclLsaReduceSumMultimemCopy::Coop") coop, [SrcLambda](#_CPPv4I00000_iE28ncclLsaReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclLsaReduceSumMultimemCopy::SrcLambda") srcLambda, int nSrc, [DstLambda](#_CPPv4I00000_iE28ncclLsaReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclLsaReduceSumMultimemCopy::DstLambda") dstLambda, int nDst, [IntCount](#_CPPv4I00000_iE28ncclLsaReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclLsaReduceSumMultimemCopy::IntCount") count)[](#_CPPv4I00000_iE28ncclLsaReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

**Source: :ref:`LSA <device_api_lsa>`. Destination: multimem.** Same as [ncclLsaReduceSumMultimemCopy](#nccllsareducesummultimemcopy-symptr), but the source layout is given by _srcLambda*(*i_) for _i_ = 0 to _nSrc_ − 1 and the destination layout by _dstLambda*(*j_) for _j_ = 0 to _nDst_ − 1. _srcLambda_ returns [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) pointers; _dstLambda_ must return multimem pointers (e.g. from [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer")). The common case is _nDst_ = 1. _coop_ and _count_ as for [ncclLsaReduceSumCopy](#nccllsareducesumcopy-window-team). Invocation as documented for [ncclLsaReduceSum](#nccllsareducesum-window-devcomm) and [ncclMultimemCopy](#ncclmultimemcopy-symptr).

template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>  
void ncclMultimemReduceSumLsaCopy([Coop](#_CPPv4I00000_iE28ncclMultimemReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclMultimemReduceSumLsaCopy::Coop") coop, [SrcLambda](#_CPPv4I00000_iE28ncclMultimemReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclMultimemReduceSumLsaCopy::SrcLambda") srcLambda, int nSrc, [DstLambda](#_CPPv4I00000_iE28ncclMultimemReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclMultimemReduceSumLsaCopy::DstLambda") dstLambda, int nDst, [IntCount](#_CPPv4I00000_iE28ncclMultimemReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclMultimemReduceSumLsaCopy::IntCount") count)[](#_CPPv4I00000_iE28ncclMultimemReduceSumLsaCopyv4Coop9SrcLambdai9DstLambdai8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

**Source: multimem. Destination: :ref:`LSA <device_api_lsa>`.** Same as [ncclMultimemReduceSumLsaCopy](#ncclmultimemreducesumlsacopy-symptr), but the source layout is given by _srcLambda*(*i_) for _i_ = 0 to _nSrc_ − 1 and the destination layout by _dstLambda*(*j_) for _j_ = 0 to _nDst_ − 1. _srcLambda_ must return multimem pointers (e.g. from [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer")); _dstLambda_ returns [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) pointers. The common case is _nSrc_ = 1. _coop_ and _count_ as for [ncclMultimemReduceSumLsaCopy](#ncclmultimemreducesumlsacopy-symptr). Invocation as documented for [ncclMultimemReduceSum](#ncclmultimemreducesum-symptr) and [ncclLsaCopy](#nccllsacopy-window-devcomm).

template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename IntCount, int UNROLL>  
void ncclMultimemReduceSumMultimemCopy([Coop](#_CPPv4I00000_iE33ncclMultimemReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclMultimemReduceSumMultimemCopy::Coop") coop, [SrcLambda](#_CPPv4I00000_iE33ncclMultimemReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclMultimemReduceSumMultimemCopy::SrcLambda") srcLambda, int nSrc, [DstLambda](#_CPPv4I00000_iE33ncclMultimemReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclMultimemReduceSumMultimemCopy::DstLambda") dstLambda, int nDst, [IntCount](#_CPPv4I00000_iE33ncclMultimemReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "ncclMultimemReduceSumMultimemCopy::IntCount") count)[](#_CPPv4I00000_iE33ncclMultimemReduceSumMultimemCopyv4Coop9SrcLambdai9DstLambdai8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

**Source: multimem. Destination: multimem.** Same as [ncclMultimemReduceSumCopy](#ncclmultimemreducesumcopy-window), but the source layout is given by _srcLambda*(*i_) for _i_ = 0 to _nSrc_ − 1 and the destination layout by _dstLambda*(*j_) for _j_ = 0 to _nDst_ − 1. _srcLambda_ and _dstLambda_ must each return multimem pointers (e.g. from [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer")). The common case is a single multimem source and a single multimem destination. _coop_ and _count_ as for [ncclMultimemReduceSumCopy](#ncclmultimemreducesumcopy-window). Invocation as documented for [ncclLsaReduceSum](#nccllsareducesum-window-team).

## Custom Reduction Operators[](#custom-reduction-operators "Permalink to this heading")

The APIs below take an explicit reduction operator (_redOp_) instead of a fixed sum, enabling custom reductions (e.g. min, max, product). **Restrictions for *redOp*:**

  * _redOp_ is a callable (e.g. functor or lambda) that takes two arguments of type _T_ (or `const T&`) and returns _T_ (the combined value).

  * No order is guaranteed in which elements are combined; the reduction may be applied in any order across the sources.

  * The callable must be **const** : it must not modify internal state. If _redOp_ is a functor, its `operator()` must be `const`; stateless lambdas satisfy this by default. Violating this is undefined behavior.


template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount, int UNROLL>  
void ncclLsaReduceLsaCopy([Coop](#_CPPv4I000000_iE20ncclLsaReduceLsaCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceLsaCopy::Coop") coop, [SrcLambda](#_CPPv4I000000_iE20ncclLsaReduceLsaCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceLsaCopy::SrcLambda") srcLambda, int nSrc, [DstLambda](#_CPPv4I000000_iE20ncclLsaReduceLsaCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceLsaCopy::DstLambda") dstLambda, int nDst, [RedOp](#_CPPv4I000000_iE20ncclLsaReduceLsaCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceLsaCopy::RedOp") const &redOp, [IntCount](#_CPPv4I000000_iE20ncclLsaReduceLsaCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceLsaCopy::IntCount") count)[](#_CPPv4I000000_iE20ncclLsaReduceLsaCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaReduceSumLsaCopy](#nccllsareducesumlsacopy-lambda), but the reduction is performed with the explicit _redOp_ callable instead of sum. _redOp_ must satisfy the restrictions above. Sources and destinations are [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa). _srcLambda_ and _dstLambda_ are as for the [lambda-based ReduceSumCopy APIs](#nccllsareducesumlsacopy-lambda); _coop_ , _count_ , and barrier usage as for [ncclLsaReduceSumCopy](#nccllsareducesumcopy-window-team).

template<typename T, typename Coop, typename SrcLambda, typename DstLambda, typename RedOp, typename IntCount, int UNROLL>  
void ncclLsaReduceMultimemCopy([Coop](#_CPPv4I000000_iE25ncclLsaReduceMultimemCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceMultimemCopy::Coop") coop, [SrcLambda](#_CPPv4I000000_iE25ncclLsaReduceMultimemCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceMultimemCopy::SrcLambda") srcLambda, int nSrc, [DstLambda](#_CPPv4I000000_iE25ncclLsaReduceMultimemCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceMultimemCopy::DstLambda") dstLambda, int nDst, [RedOp](#_CPPv4I000000_iE25ncclLsaReduceMultimemCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceMultimemCopy::RedOp") const &redOp, [IntCount](#_CPPv4I000000_iE25ncclLsaReduceMultimemCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "ncclLsaReduceMultimemCopy::IntCount") count)[](#_CPPv4I000000_iE25ncclLsaReduceMultimemCopyv4Coop9SrcLambdai9DstLambdaiRK5RedOp8IntCount "Permalink to this definition")  

    

For shared requirements (invocation model, memory, alignment), see the [introduction](#device-api-reducecopy).

Same as [ncclLsaReduceSumMultimemCopy](#nccllsareducesummultimemcopy-lambda), but the reduction is performed with the explicit _redOp_ callable instead of sum. _redOp_ must satisfy the restrictions above. Sources are [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa); destinations are multimem (one logical buffer maps to all participating ranks). _dstLambda_ must return multimem pointers (e.g. from [`ncclGetLsaMultimemDevicePointer()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclGetLsaMultimemDevicePointer "ncclGetLsaMultimemDevicePointer")). For custom reduction operators, use this API with [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) sources and multimem destinations; the multimem hardware path supports only sum. _coop_ and _count_ as for [ncclLsaReduceSumMultimemCopy](#nccllsareducesummultimemcopy-lambda). Invocation as documented for [ncclLsaReduceSum](#nccllsareducesum-window-devcomm) and [ncclMultimemCopy](#ncclmultimemcopy-symptr).