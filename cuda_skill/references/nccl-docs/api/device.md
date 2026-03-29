# Device API

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device.html

---

# Device API[](#device-api "Permalink to this heading")

The Device API allows communication to be initiated and performed from device (GPU) code. It is organized into the following areas:

  * **Host-Side Setup** — Creating and configuring device communicators, querying properties, host-accessible device pointer functions, and related types.

  * **Memory and LSA** — Load/store accessible (LSA) memory, barriers, pointer accessors, and multimem.

  * **GIN (GPU-Initiated Networking)** — One-sided transfers, signals, counters, and network barriers.

  * **Reduce, Broadcast, and Fused Building Blocks** — Building blocks for computation-fused kernels: reduce, copy (broadcast), and reduce-then-copy; used to implement algorithms such as AllReduce, AllGather, and ReduceScatter.


For an introduction and usage examples, see [Device-Initiated Communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html).

  * [Device API – Host-Side Setup](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html)
    * [Host-Side Setup](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#host-side-setup)
    * [Host-Accessible Device Pointer Functions](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#host-accessible-device-pointer-functions)
  * [Device API – Memory and LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html)
    * [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html#lsa)
    * [Multimem](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html#multimem)
  * [Device API – GIN](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html)
    * [GIN](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#gin)
  * [Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html)
    * [Compile-Time Requirements](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#compile-time-requirements)
    * [API Overview](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#api-overview)
    * [ReduceSum — N Sources to One Destination](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#reducesum-n-sources-to-one-destination)
    * [Copy (Broadcast) — One Source to N Destinations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#copy-broadcast-one-source-to-n-destinations)
    * [ReduceSumCopy](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#reducesumcopy)
    * [Lambda-Based (Custom Layouts)](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#lambda-based-custom-layouts)
    * [Custom Reduction Operators](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#custom-reduction-operators)