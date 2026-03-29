# NCCL API Supported Flags

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html

---

# NCCL API Supported Flags[](#nccl-api-supported-flags "Permalink to this heading")

The following show all flags which are supported by NCCL APIs.

## Window Registration Flags[](#window-registration-flags "Permalink to this heading")

NCCL_WIN_DEFAULT[](#c.NCCL_WIN_DEFAULT "Permalink to this definition")  

    

Register buffer into NCCL window with default behavior. The default behavior allows users to pass any offset to the buffer head address as the input of NCCL collective operations. However, this behavior can cause suboptimal performance in NCCL due to the asymmetric buffer usage.

NCCL_WIN_COLL_SYMMETRIC[](#c.NCCL_WIN_COLL_SYMMETRIC "Permalink to this definition")  

    

Register buffer into NCCL window, and users need to guarantee the offset to the buffer head address from all ranks must be equal when calling NCCL collective operations. It allows NCCL to operate buffer in a symmetric way and provide the best performance.

NCCL_WIN_COLL_STRICT_ORDERING[](#c.NCCL_WIN_COLL_STRICT_ORDERING "Permalink to this definition")  

    

Register buffer into NCCL window while ensuring strict ordering for window operations using the IB Verbs transport. This flag is mostly intended for buffers used for GIN VA Signals (see [Signals and Counters](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#devapi-signals)).

## NCCL Communicator CTA Policy Flags[](#nccl-communicator-cta-policy-flags "Permalink to this heading")

NCCL_CTA_POLICY_DEFAULT[](#c.NCCL_CTA_POLICY_DEFAULT "Permalink to this definition")  

    

Use the default CTA policy for NCCL communicator. In this policy, NCCL will automatically adjust resource usage and achieve maximal performance. This policy is suitable for most applications.

NCCL_CTA_POLICY_EFFICIENCY[](#c.NCCL_CTA_POLICY_EFFICIENCY "Permalink to this definition")  

    

Use the CTA efficiency policy for NCCL communicator. In this policy, NCCL will optimize CTA usage and use minimal number of CTAs to achieve the decent performance when possible. This policy is suitable for applications which require better compute and communication overlap.

NCCL_CTA_POLICY_ZERO[](#c.NCCL_CTA_POLICY_ZERO "Permalink to this definition")  

    

Use the Zero-CTA policy for NCCL communicator. In this policy, NCCL will use zero CTA whenever it can, even when that choice may sacrifice some performance. Select this mode when your application must preserve the maximum number of CTAs for compute kernels.

## Communicator Shrink Flags[](#communicator-shrink-flags "Permalink to this heading")

These flags modify the behavior of the `ncclCommShrink` operation.

NCCL_SHRINK_DEFAULT[](#c.NCCL_SHRINK_DEFAULT "Permalink to this definition")  

    

Default behavior. Shrink the parent communicator without affecting ongoing operations. Value: `0x00`.

NCCL_SHRINK_ABORT[](#c.NCCL_SHRINK_ABORT "Permalink to this definition")  

    

First, terminate ongoing parent communicator operations, and then proceed with shrinking the communicator. This is used for error recovery scenarios where the parent communicator might be in a hung state. Resources of parent comm are still not freed, users should decide whether to call ncclCommAbort on the parent communicator after shrink. Value: `0x01`.