# Point-to-point communication

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html

---

# Point-to-point communication[](#point-to-point-communication "Permalink to this heading")

## Two-sided communication[](#two-sided-communication "Permalink to this heading")

(Since NCCL 2.7) Point-to-point communication can be used to express any communication pattern between ranks. Any point-to-point communication needs two NCCL calls: a call to [`ncclSend()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclSend "ncclSend") on one rank and a corresponding [`ncclRecv()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclRecv "ncclRecv") on the other rank, with the same count and data type.

Multiple calls to [`ncclSend()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclSend "ncclSend") and [`ncclRecv()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclRecv "ncclRecv") targeting different peers can be fused together with [`ncclGroupStart()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupStart "ncclGroupStart") and [`ncclGroupEnd()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupEnd "ncclGroupEnd") to form more complex communication patterns such as one-to-all (scatter), all-to-one (gather), all-to-all or communication with neighbors in an N-dimensional space.

Point-to-point calls within a group will be blocking until that group of calls completes, but calls within a group can be seen as progressing independently, hence should never block each other. It is therefore important to merge calls that need to progress concurrently to avoid deadlocks. The only exception is point-to-point calls within a group targeting the _same_ peer, which are executed in order.

Below are a few examples of classic point-to-point communication patterns used by parallel applications. NCCL semantics allow for all variants with different sizes, datatypes, and buffers, per rank.

### Sendrecv[](#sendrecv "Permalink to this heading")

In MPI terms, a sendrecv operation is when two ranks exchange data, both sending and receiving at the same time. This can be done by merging both ncclSend and ncclRecv calls into one:
    
    
    ncclGroupStart();
    ncclSend(sendbuff, sendcount, sendtype, peer, comm, stream);
    ncclRecv(recvbuff, recvcount, recvtype, peer, comm, stream);
    ncclGroupEnd();
    

### One-to-all (scatter)[](#one-to-all-scatter "Permalink to this heading")

A one-to-all operation from a `root` rank can be expressed by merging all send and receive operations in a group:
    
    
    ncclGroupStart();
    if (rank == root) {
      for (int r=0; r<nranks; r++)
        ncclSend(sendbuff[r], size, type, r, comm, stream);
    }
    ncclRecv(recvbuff, size, type, root, comm, stream);
    ncclGroupEnd();
    

### All-to-one (gather)[](#all-to-one-gather "Permalink to this heading")

Similarly, an all-to-one operation to a `root` rank would be implemented this way:
    
    
    ncclGroupStart();
    if (rank == root) {
      for (int r=0; r<nranks; r++)
        ncclRecv(recvbuff[r], size, type, r, comm, stream);
    }
    ncclSend(sendbuff, size, type, root, comm, stream);
    ncclGroupEnd();
    

### All-to-all[](#all-to-all "Permalink to this heading")

An all-to-all operation would be a merged loop of send/recv operations to/from all peers:
    
    
    ncclGroupStart();
    for (int r=0; r<nranks; r++) {
      ncclSend(sendbuff[r], sendcount, sendtype, r, comm, stream);
      ncclRecv(recvbuff[r], recvcount, recvtype, r, comm, stream);
    }
    ncclGroupEnd();
    

### Neighbor exchange[](#neighbor-exchange "Permalink to this heading")

Finally, exchanging data with neighbors in an N-dimensional space could be done with:
    
    
    ncclGroupStart();
    for (int d=0; d<ndims; d++) {
      ncclSend(sendbuff[d], sendcount, sendtype, next[d], comm, stream);
      ncclRecv(recvbuff[d], recvcount, recvtype, prev[d], comm, stream);
    }
    ncclGroupEnd();
    

## One-sided communication[](#one-sided-communication "Permalink to this heading")

(Since NCCL 2.29) One-sided communication enables a rank to write data to remote memory using [`ncclPutSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclPutSignal "ncclPutSignal") without requiring the target rank to issue a matching operation. The target memory must be pre-registered using [`ncclCommWindowRegister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister"). Point-to-point synchronization can be achieved by having the target rank call [`ncclWaitSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclWaitSignal "ncclWaitSignal") to wait for signals.

Multiple [`ncclPutSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclPutSignal "ncclPutSignal") calls can be grouped using [`ncclGroupStart()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupStart "ncclGroupStart") and [`ncclGroupEnd()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupEnd "ncclGroupEnd"). Operations to different peers or contexts within a group may execute concurrently and complete in any order. The completion of [`ncclGroupEnd()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html#c.ncclGroupEnd "ncclGroupEnd") guarantees that all operations in the group have achieved completion. Operations to the same peer and context are executed in order: both data delivery and signal updates on the remote peer follow the program order.

Below are a few examples of classic one-sided communication patterns used by parallel applications.

### PutSignal and WaitSignal[](#putsignal-and-waitsignal "Permalink to this heading")

A ping-pong pattern using [`ncclPutSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclPutSignal "ncclPutSignal") and [`ncclWaitSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclWaitSignal "ncclWaitSignal"). This example shows the full setup including memory allocation and window registration:
    
    
    // Allocate symmetric memory for RMA operations
    void *sendbuff, *recvbuff;
    NCCLCHECK(ncclMemAlloc((void**)&sendbuff, size));
    NCCLCHECK(ncclMemAlloc((void**)&recvbuff, size));
    
    // Register buffers as symmetric windows
    ncclWindow_t sendWindow, recvWindow;
    NCCLCHECK(ncclCommWindowRegister(comm, sendbuff, size, &sendWindow, NCCL_WIN_COLL_SYMMETRIC));
    NCCLCHECK(ncclCommWindowRegister(comm, recvbuff, size, &recvWindow, NCCL_WIN_COLL_SYMMETRIC));
    
    int peer = (rank == 0) ? 1 : 0;
    ncclWaitSignalDesc_t waitDesc = {.opCnt = 1, .peer = peer, .sigIdx = 0, .ctx = ctx};
    
    if (rank == 0) {
      // Rank 0: wait then put
      NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));
      NCCLCHECK(ncclPutSignal(sendbuff, count, datatype, peer, recvWindow, 0,
                        0, 0, 0, comm, stream));
    } else {
      // Rank 1: put then wait
      NCCLCHECK(ncclPutSignal(sendbuff, count, datatype, peer, recvWindow, 0,
                        0, 0, 0, comm, stream));
      NCCLCHECK(ncclWaitSignal(1, &waitDesc, comm, stream));
    }
    
    CUDACHECK(cudaStreamSynchronize(stream));
    
    // Cleanup
    NCCLCHECK(ncclCommWindowDeregister(comm, sendWindow));
    NCCLCHECK(ncclCommWindowDeregister(comm, recvWindow));
    NCCLCHECK(ncclMemFree(sendbuff));
    NCCLCHECK(ncclMemFree(recvbuff));
    

### Barrier[](#barrier "Permalink to this heading")

A barrier pattern using [`ncclSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclSignal "ncclSignal") and [`ncclWaitSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclWaitSignal "ncclWaitSignal"). Each rank signals to all other ranks and waits for signals from all ranks:
    
    
    ncclWaitSignalDesc_t *waitDescs = malloc(nranks * sizeof(ncclWaitSignalDesc_t));
    for (int r = 0; r < nranks; r++) {
      waitDescs[r].opCnt = 1;
      waitDescs[r].peer = r;
      waitDescs[r].sigIdx = 0;
      waitDescs[r].ctx = 0;
    }
    
    ncclGroupStart();
    for (int r = 0; r < nranks; r++) {
      ncclSignal(r, 0, 0, 0, comm, stream);
    }
    ncclGroupEnd();
    
    ncclWaitSignal(nranks, waitDescs, comm, stream);
    

### All-to-all[](#id1 "Permalink to this heading")

An all-to-all operation using [`ncclPutSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclPutSignal "ncclPutSignal"). Each rank sends data to all other ranks and waits for signals from all ranks. User needs to register the memory window for each peer using [`ncclCommWindowRegister()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/comms.html#c.ncclCommWindowRegister "ncclCommWindowRegister") in advance. User needs to guarantee the buffers are ready before calling [`ncclPutSignal()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#c.ncclPutSignal "ncclPutSignal"). This could be done with the barrier shown above.
    
    
    size_t offset[nranks];
    ncclWaitSignalDesc_t *waitDescs = malloc(nranks * sizeof(ncclWaitSignalDesc_t));
    for (int r = 0; r < nranks; r++) {
      offset[r] = r * count * wordSize(datatype);
      waitDescs[r].opCnt = 1;
      waitDescs[r].peer = r;
      waitDescs[r].sigIdx = 0;
      waitDescs[r].ctx = 0;
    }
    
    ncclGroupStart();
    for (int r = 0; r < nranks; r++) {
      ncclPutSignal(sendbuff[r], count, datatype, r, window, offset[r],
              0, 0, 0, comm, stream);
    }
    ncclGroupEnd();
    
    ncclWaitSignal(nranks, waitDescs, comm, stream);