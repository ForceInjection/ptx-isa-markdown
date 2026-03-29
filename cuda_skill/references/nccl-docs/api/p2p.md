# Point To Point Communication Functions

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html

---

# Point To Point Communication Functions[](#point-to-point-communication-functions "Permalink to this heading")

NCCL provides two types of point-to-point communication primitives: two-sided operations and one-sided operations.

## Two-Sided Point-to-Point Operations[](#two-sided-point-to-point-operations "Permalink to this heading")

(Since NCCL 2.7) Two-sided point-to-point communication primitives need to be used when ranks need to send and receive arbitrary data from each other, which cannot be expressed as a broadcast or allgather, i.e. when all data sent and received is different. Both sender and receiver must explicitly participate.

### ncclSend[](#ncclsend "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclSend(const void *sendbuff, size_t count, [ncclDataType_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t "ncclDataType_t") datatype, int peer, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream_t stream)[](#c.ncclSend "Permalink to this definition")  

    

Send data from `sendbuff` to rank `peer`.

Rank `peer` needs to call ncclRecv with the same `datatype` and the same `count` as this rank.

This operation is blocking for the GPU. If multiple [`ncclSend()`](#c.ncclSend "ncclSend") and [`ncclRecv()`](#c.ncclRecv "ncclRecv") operations need to progress concurrently to complete, they must be fused within a [`ncclGroupStart()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupStart "ncclGroupStart")/ [`ncclGroupEnd()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupEnd "ncclGroupEnd") section.

Related links: [Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#point-to-point).

### ncclRecv[](#ncclrecv "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclRecv(void *recvbuff, size_t count, [ncclDataType_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t "ncclDataType_t") datatype, int peer, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream_t stream)[](#c.ncclRecv "Permalink to this definition")  

    

Receive data from rank `peer` into `recvbuff`.

Rank `peer` needs to call ncclSend with the same `datatype` and the same `count` as this rank.

This operation is blocking for the GPU. If multiple [`ncclSend()`](#c.ncclSend "ncclSend") and [`ncclRecv()`](#c.ncclRecv "ncclRecv") operations need to progress concurrently to complete, they must be fused within a [`ncclGroupStart()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupStart "ncclGroupStart")/ [`ncclGroupEnd()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupEnd "ncclGroupEnd") section.

Related links: [Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#point-to-point).

## One-Sided Point-to-Point Operations (RMA)[](#one-sided-point-to-point-operations-rma "Permalink to this heading")

One-sided Remote Memory Access (RMA) operations enable ranks to directly access remote memory without explicit participation from the target process. These operations require the target memory to be pre-registered within a symmetric memory window using [`ncclCommWindowRegister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister").

### ncclPutSignal[](#ncclputsignal "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclPutSignal(const void *localbuff, size_t count, [ncclDataType_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t "ncclDataType_t") datatype, int peer, [ncclWindow_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclWindow_t "ncclWindow_t") peerWin, size_t peerWinOffset, int sigIdx, int ctx, unsigned int flags, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream_t stream)[](#c.ncclPutSignal "Permalink to this definition")  

    

Write data from `localbuff` to rank `peer`’s registered memory window `peerWin` at offset `peerWinOffset` and subsequently updating a remote signal.

The target memory window `peerWin` must be registered using [`ncclCommWindowRegister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister").

The `sigIdx` is the signal index identifier for the operation. It must be set to 0 for now.

The `ctx` is the context identifier for the operation. It must be set to 0 for now.

The `flags` parameter is reserved for future use. It must be set to 0 for now.

The return of [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") to the CPU thread indicates that the operation has been successfully enqueued to the CUDA stream. At the completion of [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") on the CUDA stream, the `localbuff` is safe to reuse or modify. When a signal is updated on the remote peer, it guarantees that the data from the corresponding [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") operation has been delivered to the remote memory. All prior [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") and [`ncclSignal()`](#c.ncclSignal "ncclSignal") operations to the same peer and context have also completed their signal updates.

Related links: [Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#point-to-point).

### ncclSignal[](#ncclsignal "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclSignal(int peer, int sigIdx, int ctx, unsigned int flags, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream_t stream)[](#c.ncclSignal "Permalink to this definition")  

    

Send a signal to rank `peer` without transferring data.

The `sigIdx` is the signal index identifier for the operation. It must be set to 0 for now.

The `ctx` is the context identifier for the operation. It must be set to 0 for now.

The `flags` parameter is reserved for future use. It must be set to 0 for now.

When a signal is updated on the remote peer, all prior [`ncclPutSignal()`](#c.ncclPutSignal "ncclPutSignal") and [`ncclSignal()`](#c.ncclSignal "ncclSignal") operations to the same peer and context have also completed their signal updates.

Related links: [Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#point-to-point).

### ncclWaitSignal[](#ncclwaitsignal "Permalink to this heading")

type ncclWaitSignalDesc_t[](#c.ncclWaitSignalDesc_t "Permalink to this definition")  

    

Descriptor that specifies how many signal operations to wait for from a particular rank on a given signal index and context.

int opCnt[](#c.ncclWaitSignalDesc_t.opCnt "Permalink to this definition")  

    

Number of signal operations to wait for.

int peer[](#c.ncclWaitSignalDesc_t.peer "Permalink to this definition")  

    

Target peer to wait for signals from.

int sigIdx[](#c.ncclWaitSignalDesc_t.sigIdx "Permalink to this definition")  

    

Signal index identifier. Must be set to 0 for now.

int ctx[](#c.ncclWaitSignalDesc_t.ctx "Permalink to this definition")  

    

Context identifier. Must be set to 0 for now.

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclWaitSignal(int nDesc, [ncclWaitSignalDesc_t](#c.ncclWaitSignalDesc_t "ncclWaitSignalDesc_t") *signalDescs, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm, cudaStream_t stream)[](#c.ncclWaitSignal "Permalink to this definition")  

    

Wait for signals as described in the signal descriptor array.

The `nDesc` parameter specifies the number of signal descriptors in the `signalDescs` array. Each descriptor indicates how many signals (`opCnt`) to expect from a specific `peer` on a particular signal index (`sigIdx`) and context (`ctx`).

The return of [`ncclWaitSignal()`](#c.ncclWaitSignal "ncclWaitSignal") to the CPU thread indicates that the operation has been successfully enqueued to the CUDA stream. At the completion of [`ncclWaitSignal()`](#c.ncclWaitSignal "ncclWaitSignal") on the CUDA stream, all specified signal operations have been received and the corresponding data is visible in local memory.

Related links: [Point-to-point communication](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#point-to-point).