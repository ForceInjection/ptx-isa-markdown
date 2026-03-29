# Communicator Creation and Management Functions

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html

---

# Communicator Creation and Management Functions[](#communicator-creation-and-management-functions "Permalink to this heading")

The following functions are public APIs exposed by NCCL to create and manage the collective communication operations.

## ncclGetLastError[](#ncclgetlasterror "Permalink to this heading")

const char *ncclGetLastError([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm)[](#c.ncclGetLastError "Permalink to this definition")  

    

Returns a human-readable string corresponding to the last error that occurred in NCCL. Note: The error is not cleared by calling this function. Please note that the string returned by ncclGetLastError could be unrelated to the current call and can be a result of previously launched asynchronous operations, if any.

## ncclGetErrorString[](#ncclgeterrorstring "Permalink to this heading")

const char *ncclGetErrorString([ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") result)[](#c.ncclGetErrorString "Permalink to this definition")  

    

Returns a human-readable string corresponding to the passed error code.

## ncclGetVersion[](#ncclgetversion "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclGetVersion(int *version)[](#c.ncclGetVersion "Permalink to this definition")  

    

The ncclGetVersion function returns the version number of the currently linked NCCL library. The NCCL version number is returned in _version_ and encoded as an integer which includes the `NCCL_MAJOR`, `NCCL_MINOR` and `NCCL_PATCH` levels. The version number returned will be the same as the `NCCL_VERSION_CODE` defined in _nccl.h_. NCCL version numbers can be compared using the supplied macro `NCCL_VERSION` as `NCCL_VERSION(MAJOR,MINOR,PATCH)`

## ncclGetUniqueId[](#ncclgetuniqueid "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclGetUniqueId(ncclUniqueId *uniqueId)[](#c.ncclGetUniqueId "Permalink to this definition")  

    

Generates an Id to be used in ncclCommInitRank. ncclGetUniqueId should be called once when creating a communicator and the Id should be distributed to all ranks in the communicator before calling ncclCommInitRank. _uniqueId_ should point to a ncclUniqueId object allocated by the user.

## ncclCommInitRank[](#ncclcomminitrank "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommInitRank([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") *comm, int nranks, ncclUniqueId commId, int rank)[](#c.ncclCommInitRank "Permalink to this definition")  

    

Creates a new communicator (multi thread/process version). _rank_ must be between 0 and _nranks_ -1 and unique within a communicator clique. Each rank is associated to a CUDA device, which has to be set before calling ncclCommInitRank. ncclCommInitRank implicitly synchronizes with other ranks, hence it must be called by different threads/processes or used within ncclGroupStart/ncclGroupEnd.

## ncclCommInitAll[](#ncclcomminitall "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommInitAll([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") *comms, int ndev, const int *devlist)[](#c.ncclCommInitAll "Permalink to this definition")  

    

Creates a clique of communicators (single process version) in a blocking way. This is a convenience function to create a single-process communicator clique. Returns an array of _ndev_ newly initialized communicators in _comms_. _comms_ should be pre-allocated with size at least ndev*sizeof([`ncclComm_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t")). _devlist_ defines the CUDA devices associated with each rank. If _devlist_ is NULL, the first _ndev_ CUDA devices are used, in order.

## ncclCommInitRankConfig[](#ncclcomminitrankconfig "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommInitRankConfig([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") *comm, int nranks, ncclUniqueId commId, int rank, [ncclConfig_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclConfig_t "ncclConfig_t") *config)[](#c.ncclCommInitRankConfig "Permalink to this definition")  

    

This function works the same way as _ncclCommInitRank_ but accepts a configuration argument of extra attributes for the communicator. If config is passed as NULL, the communicator will have the default behavior, as if ncclCommInitRank was called.

See the [Creating a communicator with options](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#init-rank-config) section for details on configuration options.

## ncclCommInitRankScalable[](#ncclcomminitrankscalable "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommInitRankScalable([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") *newcomm, int nranks, int myrank, int nId, ncclUniqueId *commIds, [ncclConfig_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclConfig_t "ncclConfig_t") *config)[](#c.ncclCommInitRankScalable "Permalink to this definition")  

    

This function works the same way as _ncclCommInitRankConfig_ but accepts a list of ncclUniqueIds instead of a single one. If only one ncclUniqueId is passed, the communicator will be initialized as if ncclCommInitRankConfig was called. The provided ncclUniqueIds will all be used to initialize the single communicator given in argument.

See the [Creating a communicator with options](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#init-rank-config) section for details on how to create and distribute the list of ncclUniqueIds.

## ncclCommSplit[](#ncclcommsplit "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommSplit([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, int color, int key, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") *newcomm, [ncclConfig_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclConfig_t "ncclConfig_t") *config)[](#c.ncclCommSplit "Permalink to this definition")  

    

The _ncclCommSplit_ is a collective function and creates a set of new communicators from an existing one. Ranks which pass the same _color_ value will be part of the same group; color must be a non-negative value. If it is passed as _NCCL_SPLIT_NOCOLOR_ , it means that the rank will not be part of any group, therefore returning NULL as newcomm. The value of key will determine the rank order, and the smaller key means the smaller rank in new communicator. If keys are equal between ranks, then the rank in the original communicator will be used to order ranks. If the new communicator needs to have a special configuration, it can be passed as _config_ , otherwise setting config to NULL will make the new communicator inherit the original communicator’s configuration. When split, there should not be any outstanding NCCL operations on the _comm_. Otherwise, it might cause a deadlock.

## ncclCommShrink[](#ncclcommshrink "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommShrink([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, int *excludeRanksList, int excludeRanksCount, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") *newcomm, [ncclConfig_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclConfig_t "ncclConfig_t") *config, int shrinkFlags)[](#c.ncclCommShrink "Permalink to this definition")  

    

The _ncclCommShrink_ function creates a new communicator by removing specified ranks from an existing communicator. It is a collective function that must be called by all participating ranks in the newly created communicator. Ranks that are part of _excludeRanksList_ should not call this function. The original ranks listed in _excludeRanksList_ (of size _excludeRanksCount_) will be excluded from the new communicator. Within the new communicator, ranks will be updated to maintain a contiguous set of ids. If the new communicator needs a special configuration, it can be passed as _config_ ; otherwise, setting config to NULL will make the new communicator inherit the configuration of the parent communicator.

The _shrinkFlags_ parameter controls the behavior of the operation. Use _NCCL_SHRINK_DEFAULT_ (or _0_) for normal operation, or _NCCL_SHRINK_ABORT_ when shrinking after an error on the parent communicator. Specifically, when using _NCCL_SHRINK_DEFAULT_ , there should not be any outstanding NCCL operations on the _comm_ to avoid potential deadlocks. Further, if the parent communicator has the flag config.shrinkShare set to 1, NCCL will reuse the parent communicator resources. On the other hand, when using _NCCL_SHRINK_ABORT_ , NCCL will automatically abort any outstanding operations on the parent communicator, and no resources will be shared between the parent and the newly created communicator.

## ncclCommGetUniqueId[](#ncclcommgetuniqueid "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommGetUniqueId([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, ncclUniqueId *uniqueId)[](#c.ncclCommGetUniqueId "Permalink to this definition")  

    

The _ncclCommGetUniqueId_ function generates a unique identifier for growing an existing communicator exactly once. This function must be called by only one rank (the coordinator) before each grow operation on the existing communicator. The coordinator is responsible for distributing the _uniqueId_ to all new ranks before they join the communicator via _ncclCommGrow_. This function should only be called when there are no outstanding NCCL operations on the communicator.

## ncclCommGrow[](#ncclcommgrow "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommGrow([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, int nRanks, const ncclUniqueId *uniqueId, int rank, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") *newcomm, [ncclConfig_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclConfig_t "ncclConfig_t") *config)[](#c.ncclCommGrow "Permalink to this definition")  

    

The _ncclCommGrow_ function creates a new communicator by adding new ranks to an existing communicator. It must be called by both existing ranks (from the parent communicator) and new ranks (joining the communicator).

**For existing ranks:**

  * _comm_ should be the parent communicator

  * _rank_ must be set to _-1_ (existing ranks retain their original rank in the new communicator)

  * _uniqueId_ should be _NULL_ (existing ranks receive coordination information internally)

  * The function creates _newcomm_ with the same rank as in the parent communicator


**For new ranks:**

  * _comm_ should be _NULL_

  * _rank_ must be set to the desired rank in the new communicator (must be >= parent communicator size)

  * _uniqueId_ must be the unique identifier obtained from _ncclCommGetUniqueId_ called by the coordinator


The _nRanks_ parameter specifies the total number of ranks in the new communicator and must be greater than the size of the parent communicator. If the new communicator needs a special configuration, it can be passed as _config_ ; otherwise, setting config to NULL will make the new communicator inherit the configuration of the parent communicator (for existing ranks) or use default configuration (for new ranks).

There should not be any outstanding NCCL operations on the parent communicator when calling this function to avoid potential deadlocks. After the grow operation completes, the parent communicator should be destroyed using _ncclCommDestroy_ to free resources.

**Example workflow:**

  1. Coordinator rank calls _ncclCommGetUniqueId_ to generate the grow identifier

  2. Coordinator distributes the _uniqueId_ to all new ranks (out-of-band)

  3. All existing ranks call _ncclCommGrow_ with _comm_ =parent, _rank_ =-1, _uniqueId_ =NULL (except for Coordinator rank which passes the _uniqueId_)

  4. All new ranks call _ncclCommGrow_ with _comm_ =NULL, _rank_ =new_rank, _uniqueId_ =received_id


## ncclCommRevoke[](#ncclcommrevoke "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommRevoke([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, int revokeFlags)[](#c.ncclCommRevoke "Permalink to this definition")  

    

Revokes in-flight operations on a communicator without destroying resources. Successful return may be _ncclInProgress_ (non-blocking) while revocation completes asynchronously; applications can query _ncclCommGetAsyncError_ until it returns _ncclSuccess_.

_revokeFlags_ must be set to _NCCL_REVOKE_DEFAULT_ (0). Other values are reserved for future use.

After revoke completes, the communicator is quiesced and safe for destroy, split, and shrink. Launching new collectives on a revoked communicator returns _ncclInvalidUsage_. Calling _ncclCommFinalize_ after revoke is not supported. Resource sharing via _splitShare_ /_shrinkShare_ is disabled when the parent communicator is revoked.

## ncclCommFinalize[](#ncclcommfinalize "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommFinalize([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm)[](#c.ncclCommFinalize "Permalink to this definition")  

    

Finalize a communicator object _comm_. When the communicator is marked as nonblocking, _ncclCommFinalize_ is a nonblocking function. Successful return from it will set communicator state as _ncclInProgress_ and indicates the communicator is under finalization where all uncompleted operations and the network-related resources are being flushed and freed. Once all NCCL operations are complete, the communicator will transition to the _ncclSuccess_ state. Users can query that state with _ncclCommGetAsyncError_.

## ncclCommDestroy[](#ncclcommdestroy "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommDestroy([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm)[](#c.ncclCommDestroy "Permalink to this definition")  

    

Destroy a communicator object _comm_. If _ncclCommFinalize_ is called by users, users should guarantee that the state of the communicator becomes _ncclSuccess_ before calling _ncclCommDestroy_. In all cases, the communicator should no longer be accessed after _ncclCommDestroy_ returns. It is recommended that users call _ncclCommFinalize_ and then _ncclCommDestroy_.

_ncclCommDestroy_ will call _ncclCommFinalize_ internally, unless _ncclCommFinalize_ was previously called on the communicator. If _ncclCommFinalize_ was previously called on the communicator object _comm_ , then _ncclCommDestroy_ is a purely local operation.

This function is an intra-node collective call, which all ranks on the same node should call to avoid a hang.

## ncclCommAbort[](#ncclcommabort "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommAbort([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm)[](#c.ncclCommAbort "Permalink to this definition")  

    

_ncclCommAbort_ frees resources that are allocated to a communicator object _comm_ and aborts any uncompleted operations before destroying the communicator. All active ranks are required to call this function in order to abort the NCCL communicator successfully. For more use cases, please check [Fault Tolerance](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html#ft).

## ncclCommGetAsyncError[](#ncclcommgetasyncerror "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommGetAsyncError([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, [ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") *asyncError)[](#c.ncclCommGetAsyncError "Permalink to this definition")  

    

Queries the progress and potential errors of asynchronous NCCL operations. Operations which do not require a stream argument (e.g. ncclCommFinalize) can be considered complete as soon as the function returns _ncclSuccess_ ; operations with a stream argument (e.g. ncclAllReduce) will return _ncclSuccess_ as soon as the operation is posted on the stream but may also report errors through ncclCommGetAsyncError() until they are completed. If the return code of any NCCL function is _ncclInProgress_ , it means the operation is in the process of being enqueued in the background, and users must query the states of the communicators until all the states become _ncclSuccess_ before calling another NCCL function. Before the states change into _ncclSuccess_ , users are not allowed to issue CUDA kernel to the streams being used by NCCL. If there has been an error on the communicator, user should destroy the communicator with [`ncclCommAbort()`](#c.ncclCommAbort "ncclCommAbort"). If an error occurs on the communicator, nothing can be assumed about the completion or correctness of operations enqueued on that communicator.

## ncclCommCount[](#ncclcommcount "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommCount(const [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, int *count)[](#c.ncclCommCount "Permalink to this definition")  

    

Returns in _count_ the number of ranks in the NCCL communicator _comm_.

## ncclCommCuDevice[](#ncclcommcudevice "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommCuDevice(const [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, int *device)[](#c.ncclCommCuDevice "Permalink to this definition")  

    

Returns in _device_ the CUDA device associated with the NCCL communicator _comm_.

## ncclCommUserRank[](#ncclcommuserrank "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommUserRank(const [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, int *rank)[](#c.ncclCommUserRank "Permalink to this definition")  

    

Returns in _rank_ the rank of the caller in the NCCL communicator _comm_.

## ncclCommRegister[](#ncclcommregister "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommRegister(const [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, void *buff, size_t size, void **handle)[](#c.ncclCommRegister "Permalink to this definition")  

    

Registers the buffer _buff_ with _size_ under communicator _comm_ for zero-copy communication; _handle_ is returned for future deregistration. See _buff_ and _size_ requirements and more instructions in [User Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#user-buffer-reg).

## ncclCommDeregister[](#ncclcommderegister "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommDeregister(const [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, void *handle)[](#c.ncclCommDeregister "Permalink to this definition")  

    

Deregister buffer represented by _handle_ under communicator _comm_.

## ncclCommWindowRegister[](#ncclcommwindowregister "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommWindowRegister([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, void *buff, size_t size, [ncclWindow_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclWindow_t "ncclWindow_t") *win, int winFlags)[](#c.ncclCommWindowRegister "Permalink to this definition")  

    

Collectively register local buffer _buff_ with _size_ under communicator _comm_ into NCCL window. Since this is a collective call, every rank in the communicator needs to participate in the registration, and _size_ by default needs to be equal among the ranks. _win_ is returned for future deregistration (if called within a group, the value may not be filled in until ncclGroupEnd() has completed). See _buff_ requirement and more instructions in [User Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#user-buffer-reg). User can also pass different win flags to control the registration behavior. For more win flags information, please refer to [Window Registration Flags](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html#win-flags).

## ncclCommWindowDeregister[](#ncclcommwindowderegister "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommWindowDeregister([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, [ncclWindow_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclWindow_t "ncclWindow_t") win)[](#c.ncclCommWindowDeregister "Permalink to this definition")  

    

Deregister NCCL window represented by _win_ under communicator _comm_. Deregistration is local to the rank, and caller needs to make sure the corresponding buffer within the window is not being accessed by any NCCL operation.

## ncclMemAlloc[](#ncclmemalloc "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclMemAlloc(void **ptr, size_t size)[](#c.ncclMemAlloc "Permalink to this definition")  

    

Allocate a GPU buffer with _size_. Allocated buffer head address will be returned by _ptr_ , and the actual allocated size can be larger than requested because of the buffer granularity requirements from all types of NCCL optimizations.

## ncclMemFree[](#ncclmemfree "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclMemFree(void *ptr)[](#c.ncclMemFree "Permalink to this definition")  

    

Free memory allocated by _ncclMemAlloc()_.

## ncclCommSuspend[](#ncclcommsuspend "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommSuspend([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, int flags)[](#c.ncclCommSuspend "Permalink to this definition")  

    

Suspend communicator operations to free resources. The communicator cannot be used for any NCCL operations while suspended. There should be no outstanding NCCL operations on _comm_ when this function is called.

The _flags_ parameter controls which resources are released:

  * _NCCL_SUSPEND_MEM_ (`0x01`) – Release dynamic GPU memory allocations held by the communicator.


A suspended communicator can be restored to an active state by calling _ncclCommResume_.

## ncclCommResume[](#ncclcommresume "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommResume([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm)[](#c.ncclCommResume "Permalink to this definition")  

    

Resume all previously suspended resources on communicator _comm_. After this call returns successfully, the communicator is fully operational and can be used for NCCL operations again.

## ncclCommMemStats[](#ncclcommmemstats "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclCommMemStats([ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, [ncclCommMemStat_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclCommMemStat_t "ncclCommMemStat_t") stat, uint64_t *value)[](#c.ncclCommMemStats "Permalink to this definition")  

    

Query communicator memory statistics. The _stat_ parameter selects which statistic to retrieve, and the result is written to _*value_. See [`ncclCommMemStat_t`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclCommMemStat_t "ncclCommMemStat_t") for the list of available statistics.