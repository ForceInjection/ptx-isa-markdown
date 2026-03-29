# Device API – Host-Side Setup

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html

---

# Device API – Host-Side Setup[](#device-api-host-side-setup "Permalink to this heading")

## Host-Side Setup[](#host-side-setup "Permalink to this heading")

**Host functions and types.** The following are for use in host code: creating and destroying device communicators, querying properties, and the requirement and property types. The [`ncclDevComm`](#c.ncclDevComm "ncclDevComm") structure is then passed to device code.

### ncclDevComm[](#nccldevcomm "Permalink to this heading")

type ncclDevComm[](#c.ncclDevComm "Permalink to this definition")  

    

A structure describing a device communicator, as created on the host side using [`ncclDevCommCreate()`](#c.ncclDevCommCreate "ncclDevCommCreate"). The structure is used primarily on the device side. In general, fields in this struct are considered internal and should not be accessed by users. An exception is made for the following fields, which are guaranteed to be stable across NCCL versions:

int rank[](#c.ncclDevComm.rank "Permalink to this definition")  

    

The rank within the communicator.

int nRanks[](#c.ncclDevComm.nRanks "Permalink to this definition")  

    

The size of the communicator.

int lsaRank[](#c.ncclDevComm.lsaRank "Permalink to this definition")  

    

int lsaSize[](#c.ncclDevComm.lsaSize "Permalink to this definition")  

    

Rank within the local [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) team and its size (see [Teams](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-teams)).

uint8_t ginContextCount[](#c.ncclDevComm.ginContextCount "Permalink to this definition")  

    

The number of supported GIN contexts (see [`ncclGin`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv47ncclGin "ncclGin"); available since NCCL 2.28.7).

### ncclDevCommCreate[](#nccldevcommcreate "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclDevCommCreate([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, struct [ncclDevCommRequirements](#c.ncclDevCommRequirements "ncclDevCommRequirements") const *reqs, struct [ncclDevComm](#c.ncclDevComm "ncclDevComm") *outDevComm)[](#c.ncclDevCommCreate "Permalink to this definition")  

    

Creates a new device communicator (see [`ncclDevComm`](#c.ncclDevComm "ncclDevComm")) corresponding to the supplied host-side communicator _comm_. The result is returned in the _outDevComm_ buffer (which needs to be supplied by the caller). The caller needs to also provide a filled-in list of requirements via the _reqs_ argument (see [`ncclDevCommRequirements`](#c.ncclDevCommRequirements "ncclDevCommRequirements")); the function will allocate any necessary resources to meet them. It is recommended to call [`ncclCommQueryProperties()`](#c.ncclCommQueryProperties "ncclCommQueryProperties") before calling the function; the function will fail if the specified requirements are not supported. Since this is a collective call, every rank in the communicator needs to participate. If called within a group, _outDevComm_ may not be filled in until `ncclGroupEnd()` has completed.

Note that this is a _host-side_ function.

### ncclDevCommDestroy[](#nccldevcommdestroy "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclDevCommDestroy([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, struct [ncclDevComm](#c.ncclDevComm "ncclDevComm") const *devComm)[](#c.ncclDevCommDestroy "Permalink to this definition")  

    

Destroys a device communicator (see [`ncclDevComm`](#c.ncclDevComm "ncclDevComm")) previously created using [`ncclDevCommCreate()`](#c.ncclDevCommCreate "ncclDevCommCreate") and releases any allocated resources. The caller must ensure that no device kernel that uses this device communicator could be running at the time this function is invoked.

Note that this is a _host-side_ function.

### ncclDevCommRequirements[](#nccldevcommrequirements "Permalink to this heading")

type ncclDevCommRequirements[](#c.ncclDevCommRequirements "Permalink to this definition")  

    

A host-side structure specifying the list of requirements when creating device communicators (see [`ncclDevComm`](#c.ncclDevComm "ncclDevComm")). Since NCCL 2.29, this struct must be initialized using `NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER`.

bool lsaMultimem[](#c.ncclDevCommRequirements.lsaMultimem "Permalink to this definition")  

    

Specifies whether multimem support is required for all [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) ranks.

int lsaBarrierCount[](#c.ncclDevCommRequirements.lsaBarrierCount "Permalink to this definition")  

    

Specifies the number of memory barriers to allocate (see [`ncclLsaBarrierSession`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession")). These barriers are necessary to write fused kernel and may be required by building blocks such as those in [Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#device-api-reducecopy).

int railGinBarrierCount[](#c.ncclDevCommRequirements.railGinBarrierCount "Permalink to this definition")  

    

Specifies the number of network barriers to allocate (see [`ncclGinBarrierSession`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession"); available since NCCL 2.28.7).

int barrierCount[](#c.ncclDevCommRequirements.barrierCount "Permalink to this definition")  

    

Specifies the minimum number for both the memory and network barriers (see above; available since NCCL 2.28.7).

int ginSignalCount[](#c.ncclDevCommRequirements.ginSignalCount "Permalink to this definition")  

    

Specifies the number of network signals to allocate (see [`ncclGinSignal_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv415ncclGinSignal_t "ncclGinSignal_t"); available since NCCL 2.28.7).

int ginCounterCount[](#c.ncclDevCommRequirements.ginCounterCount "Permalink to this definition")  

    

Specifies the number of network counters to allocate (see [`ncclGinCounter_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html#_CPPv416ncclGinCounter_t "ncclGinCounter_t"); available since NCCL 2.28.7).

bool ginForceEnable[](#c.ncclDevCommRequirements.ginForceEnable "Permalink to this definition")  

    

**Deprecated.** Forces GIN (GPU-Initiated Networking) support to be enabled by automatically setting `ginConnectionType` to `NCCL_GIN_CONNECTION_FULL`. This field is deprecated in favor of explicitly setting [`ginConnectionType`](#c.ncclDevCommRequirements.ginConnectionType "ginConnectionType") to the desired value. When set to `true`, it overrides the `ginConnectionType` field. New code should use [`ginConnectionType`](#c.ncclDevCommRequirements.ginConnectionType "ginConnectionType") directly instead of this field. Available since NCCL 2.28.7, deprecated since NCCL 2.29.4.

[ncclGinConnectionType_t](#c.ncclGinConnectionType_t "ncclGinConnectionType_t") ginConnectionType[](#c.ncclDevCommRequirements.ginConnectionType "Permalink to this definition")  

    

Specifies the type of GIN (GPU-Initiated Networking) connection to establish for the device communicator. This field controls whether GIN is enabled and how it is configured. When set to `NCCL_GIN_CONNECTION_FULL`, GIN is initialized and all ranks connect to all other ranks in the communicator. When set to `NCCL_GIN_CONNECTION_RAIL`, GIN is initialized and each rank connects to other ranks in the same rail team. If GIN resources are requested via `ginSignalCount`, `ginCounterCount`, `barrierCount`, or `railGinBarrierCount` while this field is set to `NCCL_GIN_CONNECTION_NONE`, device communicator creation will fail with `ncclInvalidArgument`. Available since NCCL 2.29.4.

See [`ncclGinConnectionType_t`](#c.ncclGinConnectionType_t "ncclGinConnectionType_t") for possible values.

ncclDevResourceRequirements_t *resourceRequirementsList[](#c.ncclDevCommRequirements.resourceRequirementsList "Permalink to this definition")  

    

Specifies a list of resource requirements. This is best set to NULL for now.

ncclTeamRequirements_t *teamRequirementsList[](#c.ncclDevCommRequirements.teamRequirementsList "Permalink to this definition")  

    

Specifies a list of requirements for particular teams. This is best set to NULL for now.

### ncclCommQueryProperties[](#ncclcommqueryproperties "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommQueryProperties([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, [ncclCommProperties_t](#c.ncclCommProperties_t "ncclCommProperties_t") *props)[](#c.ncclCommQueryProperties "Permalink to this definition")  

    

Exposes communicator properties by filling in _props_. Before calling this function, _props_ must be initialized using `NCCL_COMM_PROPERTIES_INITIALIZER`. Introduced in NCCL 2.29.

Note that this is a _host-side_ function.

### ncclCommProperties_t[](#ncclcommproperties-t "Permalink to this heading")

type ncclCommProperties_t[](#c.ncclCommProperties_t "Permalink to this definition")  

    

A structure describing the properties of the communicator. Introduced in NCCL 2.29. Properties include:

int rank[](#c.ncclCommProperties_t.rank "Permalink to this definition")  

    

Rank within the communicator.

int nRanks[](#c.ncclCommProperties_t.nRanks "Permalink to this definition")  

    

Size of the communicator.

int cudaDev[](#c.ncclCommProperties_t.cudaDev "Permalink to this definition")  

    

CUDA device index.

int nvmlDev[](#c.ncclCommProperties_t.nvmlDev "Permalink to this definition")  

    

NVML device index.

bool deviceApiSupport[](#c.ncclCommProperties_t.deviceApiSupport "Permalink to this definition")  

    

Whether the device API is supported. If false, a [`ncclDevComm`](#c.ncclDevComm "ncclDevComm") cannot be created.

bool multimemSupport[](#c.ncclCommProperties_t.multimemSupport "Permalink to this definition")  

    

Whether ranks in the same [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) team can communicate using multimem. If false, a [`ncclDevComm`](#c.ncclDevComm "ncclDevComm") cannot be created with multimem resources.

[ncclGinType_t](#c.ncclGinType_t "ncclGinType_t") ginType[](#c.ncclCommProperties_t.ginType "Permalink to this definition")  

    

The GIN type supported by the communicator. If equal to `NCCL_GIN_TYPE_NONE`, a [`ncclDevComm`](#c.ncclDevComm "ncclDevComm") cannot be created with GIN connection type `NCCL_GIN_CONNECTION_FULL`.

int nLsaTeams[](#c.ncclCommProperties_t.nLsaTeams "Permalink to this definition")  

    

The number of [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) teams across the entire communicator. Available since NCCL 2.29.4.

[ncclGinType_t](#c.ncclGinType_t "ncclGinType_t") railedGinType[](#c.ncclCommProperties_t.railedGinType "Permalink to this definition")  

    

The railed GIN type supported by the communicator. If equal to `NCCL_GIN_TYPE_NONE`, a [`ncclDevComm`](#c.ncclDevComm "ncclDevComm") cannot be created with GIN connection type `NCCL_GIN_CONNECTION_RAIL`. Available since NCCL 2.29.4.

### ncclGinType_t[](#ncclgintype-t "Permalink to this heading")

type ncclGinType_t[](#c.ncclGinType_t "Permalink to this definition")  

    

GIN type. Communication between different GIN types is not supported. Possible values include:

NCCL_GIN_TYPE_NONE[](#c.ncclGinType_t.NCCL_GIN_TYPE_NONE "Permalink to this definition")  

    

GIN is not supported.

NCCL_GIN_TYPE_PROXY[](#c.ncclGinType_t.NCCL_GIN_TYPE_PROXY "Permalink to this definition")  

    

Host Proxy GIN type.

NCCL_GIN_TYPE_GDAKI[](#c.ncclGinType_t.NCCL_GIN_TYPE_GDAKI "Permalink to this definition")  

    

GPUDirect Async Kernel-Initiated (GDAKI) GIN type.

### ncclGinConnectionType_t[](#ncclginconnectiontype-t "Permalink to this heading")

type ncclGinConnectionType_t[](#c.ncclGinConnectionType_t "Permalink to this definition")  

    

Specifies the type of GIN connection for device communicators. This enum controls whether GIN (GPU-Initiated Networking) resources should be allocated and what connection type to use. Used in [`ncclDevCommRequirements`](#c.ncclDevCommRequirements "ncclDevCommRequirements") when creating device communicators. Available since NCCL 2.29.4.

NCCL_GIN_CONNECTION_NONE[](#c.ncclGinConnectionType_t.NCCL_GIN_CONNECTION_NONE "Permalink to this definition")  

    

No GIN connectivity.

NCCL_GIN_CONNECTION_FULL[](#c.ncclGinConnectionType_t.NCCL_GIN_CONNECTION_FULL "Permalink to this definition")  

    

Full GIN connectivity. Each rank is connected to all other ranks.

NCCL_GIN_CONNECTION_RAIL[](#c.ncclGinConnectionType_t.NCCL_GIN_CONNECTION_RAIL "Permalink to this definition")  

    

Railed GIN connectivity. Each rank is connected to other ranks in the same rail team.

## Host-Accessible Device Pointer Functions[](#host-accessible-device-pointer-functions "Permalink to this heading")

**Host functions.** The following are callable from host code only. They provide host-side access to device pointer functionality, enabling host code to obtain pointers to [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) memory regions.

All functions return `ncclResult_t` error codes. On success, `ncclSuccess` is returned. On failure, appropriate error codes are returned (e.g., `ncclInvalidArgument` for invalid parameters, `ncclInternalError` for internal failures), unless otherwise specified.

The returned pointers are valid for the lifetime of the window. Pointers should not be used after either the window or communicator is destroyed. Obtained pointers are device pointers.

### ncclGetLsaMultimemDevicePointer[](#ncclgetlsamultimemdevicepointer "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclGetLsaMultimemDevicePointer([ncclWindow_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclWindow_t "ncclWindow_t") window, size_t offset, void **outPtr)[](#c.ncclGetLsaMultimemDevicePointer "Permalink to this definition")  

    

Returns a multimem base pointer for the [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) team associated with the given window. This function provides host-side access to the multimem memory functionality.

_window_ is the NCCL window object (must not be NULL). _offset_ is the byte offset within the window. _outPtr_ is the output parameter for the multimem pointer (must not be NULL).

This function requires [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) multimem support (multicast capability on the system). The window must be registered with a communicator that supports symmetric memory, and the hardware must support NVLink SHARP multicast functionality.

Note

If the system does not support multimem, the function returns `ncclSuccess` with `*outPtr` set to `nullptr`. This allows applications to gracefully detect and handle the absence of multimem support without breaking the communicator. Users should check if the returned pointer is `nullptr` to determine availability.

Example:
    
    
    void* multimemPtr;
    ncclResult_t result = ncclGetLsaMultimemDevicePointer(window, 0, &multimemPtr);
    if (result == ncclSuccess) {
        if (multimemPtr != nullptr) {
            // Use multimemPtr for multimem operations
        } else {
            // Multimem not supported, use fallback approach
        }
    }
    

### ncclGetMultimemDevicePointer[](#ncclgetmultimemdevicepointer "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclGetMultimemDevicePointer([ncclWindow_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclWindow_t "ncclWindow_t") window, size_t offset, ncclMultimemHandle multimem, void **outPtr)[](#c.ncclGetMultimemDevicePointer "Permalink to this definition")  

    

Returns a multimem base pointer using a provided multimem handle instead of the window’s internal multimem. This function enables using external or custom multimem handles for pointer calculation.

_window_ is the NCCL window object (must not be NULL). _offset_ is the byte offset within the window. _multimem_ is the multimem handle containing the multimem base pointer (multimem.mcBasePtr must not be NULL). _outPtr_ is the output parameter for the multimem pointer (must not be NULL).

This function requires [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) multimem support (multicast capability on the system).

Note

If the system does not support multimem, the function returns `ncclSuccess` with `*outPtr` set to `nullptr`. The function validates that `multimem.mcBasePtr` is not nullptr before proceeding.

Example:
    
    
    // Get multimem handle from device communicator setup
    ncclMultimemHandle customHandle;
    // ... (obtain handle)
    
    void* multimemPtr;
    ncclResult_t result = ncclGetMultimemDevicePointer(window, 0, customHandle, &multimemPtr);
    if (result == ncclSuccess) {
        if (multimemPtr != nullptr) {
            // Use multimemPtr for multimem operations with custom handle
        } else {
            // Multimem not supported, use fallback approach
        }
    }
    

### ncclGetLsaDevicePointer[](#ncclgetlsadevicepointer "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclGetLsaDevicePointer([ncclWindow_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclWindow_t "ncclWindow_t") window, size_t offset, int lsaRank, void **outPtr)[](#c.ncclGetLsaDevicePointer "Permalink to this definition")  

    

Returns a load/store accessible pointer to the memory buffer of a specific [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) peer within the window. This function provides host-side access to LSA pointer functionality using LSA rank directly.

_window_ is the NCCL window object (must not be NULL). _offset_ is the byte offset within the window (must be >= 0 and < window size). _lsaRank_ is the LSA rank of the target peer (must be >= 0 and < LSA team size). _outPtr_ is the output parameter for the LSA pointer (must not be NULL).

On success, `ncclSuccess` is returned and the LSA pointer is returned in `outPtr`.

The window must be registered with a communicator that supports LSA. The LSA rank must be within the valid range for the LSA team, and the target peer must be load/store accessible (P2P connectivity required).

Example:
    
    
    void* lsaPtr;
    ncclResult_t result = ncclGetLsaDevicePointer(window, 0, 1, &lsaPtr);
    if (result == ncclSuccess) {
        // Use lsaPtr to access LSA peer 1's memory
    }
    

### ncclGetPeerDevicePointer[](#ncclgetpeerdevicepointer "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclGetPeerDevicePointer([ncclWindow_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclWindow_t "ncclWindow_t") window, size_t offset, int peer, void **outPtr)[](#c.ncclGetPeerDevicePointer "Permalink to this definition")  

    

Returns a load/store accessible pointer to the memory buffer of a specific world rank peer within the window. This function converts world rank to [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) rank internally and provides host-side access to peer pointer functionality.

_window_ is the NCCL window object (must not be NULL). _offset_ is the byte offset within the window. _peer_ is the world rank of the target peer (must be >= 0 and < communicator size). _outPtr_ is the output parameter for the peer pointer (must not be NULL).

On success, `ncclSuccess` is returned and the peer pointer is returned in `outPtr`.

If the peer is not reachable via [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) (not in LSA team), `outPtr` is set to NULL and `ncclSuccess` is returned. This matches the behavior of the device-side `ncclGetPeerPointer` function.

The window must be registered with a communicator that supports LSA. The peer rank must be within the valid range for the communicator, and the target peer must be load/store accessible (P2P connectivity required).

Example:
    
    
    void* peerPtr;
    ncclResult_t result = ncclGetPeerDevicePointer(window, 0, 2, &peerPtr);
    if (result == ncclSuccess) {
        if (peerPtr != NULL) {
            // Use peerPtr to access world rank 2's memory
        } else {
            // Peer 2 is not reachable via LSA
        }
    }