# Using NCCL

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage.html

---

# Using NCCL[](#using-nccl "Permalink to this heading")

Using NCCL is similar to using any other library in your code:

  1. Install the NCCL library on your system

  2. Modify your application to link to that library

  3. Include the header file nccl.h in your application

  4. Create a communicator (see [Creating a Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#communicator-label))

  5. Use NCCL collective communication primitives to perform data communication. You can familiarize yourself with the [NCCL API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api.html#api-label) documentation to maximize your usage performance.


Collective communication primitives are common patterns of data transfer among a group of CUDA devices. A communication algorithm involves many processors that are communicating together. Each CUDA device is identified within the communication group by a zero-based index or rank. Each rank uses a communicator object to refer to the collection of GPUs that are intended to work together. The creation of a communicator is the first step needed before launching any communication operation.

  * [Creating a Communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)
    * [Creating a communicator with options](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#creating-a-communicator-with-options)
    * [Creating a communicator using multiple ncclUniqueIds](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#creating-a-communicator-using-multiple-nccluniqueids)
    * [Shrinking a communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#shrinking-a-communicator)
    * [Growing a communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#growing-a-communicator)
    * [Creating more communicators](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#creating-more-communicators)
    * [Using multiple NCCL communicators concurrently](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)
    * [Finalizing a communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#finalizing-a-communicator)
    * [Destroying a communicator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#destroying-a-communicator)
  * [Error handling and communicator abort](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#error-handling-and-communicator-abort)
    * [Asynchronous errors and error handling](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#asynchronous-errors-and-error-handling)
  * [Fault Tolerance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#fault-tolerance)
  * [Quality of Service](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#quality-of-service)
  * [Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
    * [AllReduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce)
    * [Broadcast](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast)
    * [Reduce](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reduce)
    * [AllGather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather)
    * [ReduceScatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter)
    * [AlltoAll](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#alltoall)
    * [Gather](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#gather)
    * [Scatter](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#scatter)
  * [Data Pointers](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/data.html)
  * [CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)
    * [Mixing Multiple Streams within the same ncclGroupStart/End() group](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html#mixing-multiple-streams-within-the-same-ncclgroupstart-end-group)
  * [Group Calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html)
    * [Management Of Multiple GPUs From One Thread](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html#management-of-multiple-gpus-from-one-thread)
    * [Aggregated Operations (2.2 and later)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html#aggregated-operations-2-2-and-later)
    * [Group Operation Ordering Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html#group-operation-ordering-semantics)
    * [Nonblocking Group Operation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html#nonblocking-group-operation)
  * [Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html)
    * [Two-sided communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#two-sided-communication)
      * [Sendrecv](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#sendrecv)
      * [One-to-all (scatter)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#one-to-all-scatter)
      * [All-to-one (gather)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#all-to-one-gather)
      * [All-to-all](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#all-to-all)
      * [Neighbor exchange](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#neighbor-exchange)
    * [One-sided communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#one-sided-communication)
      * [PutSignal and WaitSignal](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#putsignal-and-waitsignal)
      * [Barrier](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#barrier)
      * [All-to-all](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#id1)
  * [Thread Safety](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html)
  * [In-place Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html)
  * [Using NCCL with CUDA Graphs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)
  * [User Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html)
    * [NVLink Sharp Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#nvlink-sharp-buffer-registration)
    * [IB Sharp Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#ib-sharp-buffer-registration)
    * [General Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#general-buffer-registration)
    * [Buffer Registration and PXN](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#buffer-registration-and-pxn)
    * [Memory Allocator](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#memory-allocator)
    * [Window Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#window-registration)
    * [Zero-CTA Optimization](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#zero-cta-optimization)
  * [Device-Initiated Communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html)
    * [Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#device-api)
    * [Requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#requirements)
    * [Cross-Version Compatibility](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#cross-version-compatibility)
    * [Host-Side Setup](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#host-side-setup)
    * [Simple LSA Kernel](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#simple-lsa-kernel)
    * [Multimem Device Kernel](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#multimem-device-kernel)
    * [Thread Groups](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#thread-groups)
    * [Teams](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#teams)
    * [Host-Accessible Device Pointer Functions](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#host-accessible-device-pointer-functions)
    * [GIN Device Kernel](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#gin-device-kernel)