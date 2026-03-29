# Types

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html

---

# Types[](#types "Permalink to this heading")

The following types are used by the NCCL library.

## ncclComm_t[](#ncclcomm-t "Permalink to this heading")

type ncclComm_t[](#c.ncclComm_t "Permalink to this definition")  

    

NCCL communicator. Points to an opaque structure inside NCCL.

## ncclResult_t[](#ncclresult-t "Permalink to this heading")

type ncclResult_t[](#c.ncclResult_t "Permalink to this definition")  

    

Return values for all NCCL functions. Possible values are:

ncclSuccess[](#c.ncclResult_t.ncclSuccess "Permalink to this definition")  

    

(`0`) Function succeeded.

ncclUnhandledCudaError[](#c.ncclResult_t.ncclUnhandledCudaError "Permalink to this definition")  

    

(`1`) A call to a CUDA function failed.

ncclSystemError[](#c.ncclResult_t.ncclSystemError "Permalink to this definition")  

    

(`2`) A call to the system failed.

ncclInternalError[](#c.ncclResult_t.ncclInternalError "Permalink to this definition")  

    

(`3`) An internal check failed. This is due to either a bug in NCCL or a memory corruption.

ncclInvalidArgument[](#c.ncclResult_t.ncclInvalidArgument "Permalink to this definition")  

    

(`4`) An argument has an invalid value.

ncclInvalidUsage[](#c.ncclResult_t.ncclInvalidUsage "Permalink to this definition")  

    

(`5`) The call to NCCL is incorrect. This is usually reflecting a programming error.

ncclRemoteError[](#c.ncclResult_t.ncclRemoteError "Permalink to this definition")  

    

(`6`) A call failed possibly due to a network error or a remote process exiting prematurely.

ncclInProgress[](#c.ncclResult_t.ncclInProgress "Permalink to this definition")  

    

(`7`) A NCCL operation on the communicator is being enqueued and is being progressed in the background.

Whenever a function returns an error (neither ncclSuccess nor ncclInProgress), NCCL should print a more detailed message when the environment variable [NCCL_DEBUG](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-debug) is set to “WARN”.

## ncclDataType_t[](#nccldatatype-t "Permalink to this heading")

type ncclDataType_t[](#c.ncclDataType_t "Permalink to this definition")  

    

NCCL defines the following integral and floating data-types.

ncclInt8[](#c.ncclDataType_t.ncclInt8 "Permalink to this definition")  

    

Signed 8-bits integer

ncclChar[](#c.ncclDataType_t.ncclChar "Permalink to this definition")  

    

Signed 8-bits integer

ncclUint8[](#c.ncclDataType_t.ncclUint8 "Permalink to this definition")  

    

Unsigned 8-bits integer

ncclInt32[](#c.ncclDataType_t.ncclInt32 "Permalink to this definition")  

    

Signed 32-bits integer

ncclInt[](#c.ncclDataType_t.ncclInt "Permalink to this definition")  

    

Signed 32-bits integer

ncclUint32[](#c.ncclDataType_t.ncclUint32 "Permalink to this definition")  

    

Unsigned 32-bits integer

ncclInt64[](#c.ncclDataType_t.ncclInt64 "Permalink to this definition")  

    

Signed 64-bits integer

ncclUint64[](#c.ncclDataType_t.ncclUint64 "Permalink to this definition")  

    

Unsigned 64-bits integer

ncclFloat16[](#c.ncclDataType_t.ncclFloat16 "Permalink to this definition")  

    

16-bits floating point number (half precision)

ncclHalf[](#c.ncclDataType_t.ncclHalf "Permalink to this definition")  

    

16-bits floating point number (half precision)

ncclFloat32[](#c.ncclDataType_t.ncclFloat32 "Permalink to this definition")  

    

32-bits floating point number (single precision)

ncclFloat[](#c.ncclDataType_t.ncclFloat "Permalink to this definition")  

    

32-bits floating point number (single precision)

ncclFloat64[](#c.ncclDataType_t.ncclFloat64 "Permalink to this definition")  

    

64-bits floating point number (double precision)

ncclDouble[](#c.ncclDataType_t.ncclDouble "Permalink to this definition")  

    

64-bits floating point number (double precision)

ncclBfloat16[](#c.ncclDataType_t.ncclBfloat16 "Permalink to this definition")  

    

16-bits floating point number (truncated precision in bfloat16 format, CUDA 11 or later)

ncclFloat8e4m3[](#c.ncclDataType_t.ncclFloat8e4m3 "Permalink to this definition")  

    

8-bits floating point number, 4 exponent bits, 3 mantissa bits (CUDA >= 11.8 and SM >= 90)

ncclFloat8e5m2[](#c.ncclDataType_t.ncclFloat8e5m2 "Permalink to this definition")  

    

8-bits floating point number, 5 exponent bits, 2 mantissa bits (CUDA >= 11.8 and SM >= 90)

## ncclRedOp_t[](#ncclredop-t "Permalink to this heading")

type ncclRedOp_t[](#c.ncclRedOp_t "Permalink to this definition")  

    

Defines the reduction operation.

ncclSum[](#c.ncclRedOp_t.ncclSum "Permalink to this definition")  

    

Perform a sum (+) operation

ncclProd[](#c.ncclRedOp_t.ncclProd "Permalink to this definition")  

    

Perform a product (*) operation

ncclMin[](#c.ncclRedOp_t.ncclMin "Permalink to this definition")  

    

Perform a min operation

ncclMax[](#c.ncclRedOp_t.ncclMax "Permalink to this definition")  

    

Perform a max operation

ncclAvg[](#c.ncclRedOp_t.ncclAvg "Permalink to this definition")  

    

Perform an average operation, i.e. a sum across all ranks, divided by the number of ranks.

## ncclScalarResidence_t[](#ncclscalarresidence-t "Permalink to this heading")

type ncclScalarResidence_t[](#c.ncclScalarResidence_t "Permalink to this definition")  

    

Indicates where (memory space) scalar arguments reside and when they can be dereferenced.

ncclScalarHostImmediate[](#c.ncclScalarResidence_t.ncclScalarHostImmediate "Permalink to this definition")  

    

The scalar resides in host memory and should be dereferenced in the most immediate way.

ncclScalarDevice[](#c.ncclScalarResidence_t.ncclScalarDevice "Permalink to this definition")  

    

The scalar resides on device visible memory and should be dereferenced once needed.

## ncclConfig_t[](#ncclconfig-t "Permalink to this heading")

type ncclConfig_t[](#c.ncclConfig_t "Permalink to this definition")  

    

A structure-based configuration users can set to initialize a communicator; a newly created configuration must be initialized by NCCL_CONFIG_INITIALIZER.

NCCL_CONFIG_INITIALIZER[](#c.ncclConfig_t.NCCL_CONFIG_INITIALIZER "Permalink to this definition")  

    

A configuration macro initializer which must be assigned to a newly created configuration.

blocking[](#c.ncclConfig_t.blocking "Permalink to this definition")  

    

This attribute can be set as integer 0 or 1 to indicate nonblocking or blocking communicator behavior correspondingly. Blocking is the default behavior.

cgaClusterSize[](#c.ncclConfig_t.cgaClusterSize "Permalink to this definition")  

    

Set Cooperative Group Array (CGA) size of kernels launched by NCCL. This attribute can be set between 0 and 8, and the default value is 4 since sm90 architecture and 0 for older architectures.

minCTAs[](#c.ncclConfig_t.minCTAs "Permalink to this definition")  

    

Set the minimal number of CTAs NCCL should use for each kernel. Set to a positive integer value, up to 32. The default value is 1.

maxCTAs[](#c.ncclConfig_t.maxCTAs "Permalink to this definition")  

    

Set the maximal number of CTAs NCCL should use for each kernel. Set to a positive integer value, up to 32. The default value is 32.

netName[](#c.ncclConfig_t.netName "Permalink to this definition")  

    

Specify the network module name NCCL should use for network communication. The value of netName must match exactly the name of the network module (case-insensitive). NCCL internal network module names are “IB” (generic IB verbs) and “Socket” (TCP/IP sockets). External network plugins define their own names. The default value is undefined, and NCCL will choose the network module automatically.

splitShare[](#c.ncclConfig_t.splitShare "Permalink to this definition")  

    

Specify whether to share resources with child communicator during communicator split. Set the value of splitShare to 0 or 1. The default value is 0. When the parent communicator is created with splitShare=1 during ncclCommInitRankConfig, the child communicator can share internal resources of the parent during communicator split. Split communicators are in the same family. When resources are shared, aborting any communicator can result in other communicators in the same family becoming unusable. Irrespective of whether sharing resources or not, users should always abort/destroy all no longer needed communicators to free up resources. Note: when the parent communicator has been revoked, resource sharing during split is disabled regardless of this flag.

shrinkShare[](#c.ncclConfig_t.shrinkShare "Permalink to this definition")  

    

Specify whether to share resources with child communicator during communicator shrink. Set the value of shrinkShare to 0 or 1. The default value is 0. Note: when shrink is used with NCCL_SHRINK_ABORT, the value of shrinkShare is ignored and no resources are shared. When the parent communicator has been revoked, resource sharing is also disabled. The behavior of this flag is similar to splitShare, see above.

trafficClass[](#c.ncclConfig_t.trafficClass "Permalink to this definition")  

    

Set the traffic class (TC) to use for network operations on the communicator. The meaning of TC is specific to the network plugin in use by the communicator (e.g. IB networks use service level, RoCE networks use type of service). Assigning different TCs to each communicator can benefit workloads which overlap communication. TCs are defined by the system configuration and should be greater than or equal to 0. Note that environment variables, such as NCCL_IB_SL and NCCL_IB_TC, take precedence over user-specified TC values. To utilize user-defined TCs, ensure that these environment variables are unset.

collnetEnable[](#c.ncclConfig_t.collnetEnable "Permalink to this definition")  

    

Set 1/0 to enable/disable IB SHARP on the communicator. The default value is 0 (disabled).

CTAPolicy[](#c.ncclConfig_t.CTAPolicy "Permalink to this definition")  

    

Set the policy for the communicator. The full list of supported policies can be found in [NCCL Communicator CTA Policy Flags](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html#cta-policy-flags). The default value is NCCL_CTA_POLICY_DEFAULT.

nvlsCTAs[](#c.ncclConfig_t.nvlsCTAs "Permalink to this definition")  

    

Set the total number of CTAs NCCL should use for NVLS kernels. Set to a positive integer value. By default, NCCL will automatically determine the best number of CTAs based on the system configuration.

commName[](#c.ncclConfig_t.commName "Permalink to this definition")  

    

Specify the user defined name for the communicator. The communicator name can be used by NCCL to enrich logging and profiling.

nChannelsPerNetPeer[](#c.ncclConfig_t.nChannelsPerNetPeer "Permalink to this definition")  

    

Set the number of network channels to be used for pairwise communication. The value must be a positive integer and will be round up to the next power of 2. The default value is optimized for the AlltoAll communication pattern. Consider increasing the value to increase the bandwidth for send/recv communication.

graphUsageMode[](#c.ncclConfig_t.graphUsageMode "Permalink to this definition")  

    

Set the graph usage mode for the communicator. It support three possible values: 0 (no graphs), 1 (one graph) and 2 (either multiple graphs or mix of graph and non-graph). The default value is 2.

## ncclSimInfo_t[](#ncclsiminfo-t "Permalink to this heading")

type ncclSimInfo_t[](#c.ncclSimInfo_t "Permalink to this definition")  

    

This struct will be used by ncclGroupSimulateEnd() to return information about the calls.

NCCL_SIM_INFO_INITIALIZER[](#c.ncclSimInfo_t.NCCL_SIM_INFO_INITIALIZER "Permalink to this definition")  

    

NCCL_SIM_INFO_INITIALIZER is a configuration macro initializer which must be assigned to a newly created ncclSimInfo_t struct.

estimatedTime[](#c.ncclSimInfo_t.estimatedTime "Permalink to this definition")  

    

Estimated time for the operation(s) in the group call will be returned in this attribute.

## ncclCommMemStat_t[](#ncclcommmemstat-t "Permalink to this heading")

type ncclCommMemStat_t[](#c.ncclCommMemStat_t "Permalink to this definition")  

    

Memory statistic selectors for [`ncclCommMemStats()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommMemStats "ncclCommMemStats").

ncclStatGpuMemSuspend[](#c.ncclCommMemStat_t.ncclStatGpuMemSuspend "Permalink to this definition")  

    

Communicator allocated GPU memory that can be released via suspend (bytes).

ncclStatGpuMemSuspended[](#c.ncclCommMemStat_t.ncclStatGpuMemSuspended "Permalink to this definition")  

    

Whether communicator allocated GPU memory is currently suspended (`0` = active, `1` = suspended).

ncclStatGpuMemPersist[](#c.ncclCommMemStat_t.ncclStatGpuMemPersist "Permalink to this definition")  

    

Communicator allocated GPU memory that cannot be suspended (bytes).

ncclStatGpuMemTotal[](#c.ncclCommMemStat_t.ncclStatGpuMemTotal "Permalink to this definition")  

    

Total communicator allocated GPU memory that is tracked by NCCL (bytes).

## ncclWindow_t[](#ncclwindow-t "Permalink to this heading")

type ncclWindow_t[](#c.ncclWindow_t "Permalink to this definition")  

    

NCCL window object for window registration and deregistration.