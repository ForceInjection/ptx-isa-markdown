# NCCL and MPI

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/mpi.html

---

# NCCL and MPI[](#nccl-and-mpi "Permalink to this heading")

## API[](#api "Permalink to this heading")

The NCCL API and usage are similar to MPI but there are many minor differences. The following list summarizes these differences:

### Using multiple devices per process[](#using-multiple-devices-per-process "Permalink to this heading")

Similarly to the concept of MPI endpoints, NCCL does not require ranks to be mapped 1:1 to processes. A NCCL communicator may have many ranks (and, thus, multiple devices) associated to a single process. Hence, if used with MPI, a single MPI rank (a NCCL process) may have multiple devices associated with it.

### ReduceScatter operation[](#reducescatter-operation "Permalink to this heading")

The ncclReduceScatter operation is similar to the MPI_Reduce_scatter_block operation, not the MPI_Reduce_scatter operation. The MPI_Reduce_scatter function is intrinsically a “vector” function, while MPI_Reduce_scatter_block (defined later to fill the missing semantics) provides regular counts similarly to the mirror function MPI_Allgather. This is an oddity of MPI which has not been fixed for legitimate retro-compatibility reasons and that NCCL does not follow.

### Send and Receive counts[](#send-and-receive-counts "Permalink to this heading")

In many collective operations, MPI allows for different send and receive counts and types, as long as sendcount*sizeof(sendtype) == recvcount*sizeof(recvtype). NCCL does not allow that, defining a single count and a single data-type.

For AllGather and ReduceScatter operations, the count is equal to the per-rank size, which is the smallest size; the other count being equal to nranks*count. The function prototype clearly shows which count is provided. ncclAllGather has a sendcount as argument, while ncclReduceScatter has a recvcount as argument.

Note: When performing or comparing AllReduce operations using a combination of ReduceScatter and AllGather, define the sendcount and recvcount as the total count divided by the number of ranks, with the correct count rounding-up, if it is not a perfect multiple of the number of ranks.

### Other collectives and point-to-point operations[](#other-collectives-and-point-to-point-operations "Permalink to this heading")

NCCL does not define specific verbs for sendrecv, gather, gatherv, scatter, scatterv, alltoall, alltoallv, alltoallw, nor neighbor collectives. All those operations can be simply expressed using a combination of ncclSend, ncclRecv, and ncclGroupStart/ncclGroupEnd, similarly to how they can be expressed with MPI_Isend, MPI_Irecv and MPI_Waitall.

ncclRecv does not support the equivalent of MPI_ANY_SOURCE; a specific source rank must always be provided. Similarly, the provided receive count must match the send count. Further, there is no concept of message tags.

### In-place operations[](#in-place-operations "Permalink to this heading")

For more information, see [In-place Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/inplace.html#in-place-operations).

## Using NCCL within an MPI Program[](#using-nccl-within-an-mpi-program "Permalink to this heading")

NCCL can be easily used in conjunction with MPI. NCCL collectives are similar to MPI collectives, therefore, creating a NCCL communicator out of an MPI communicator is straightforward. It is therefore easy to use MPI for CPU-to-CPU communication and NCCL for GPU-to-GPU communication.

However, some implementation details in MPI can lead to issues when using NCCL inside an MPI program.

### MPI Progress[](#mpi-progress "Permalink to this heading")

MPI defines a notion of progress which means that MPI operations need the program to call MPI functions (potentially multiple times) to make progress and eventually complete.

In some implementations, progress on one rank may need MPI to be called on another rank. While this is usually bad for performance, it can be argued that this is a valid MPI implementation.

As a result, blocking on a NCCL collective operation, for example calling cudaStreamSynchronize, may create a deadlock in some cases because not calling MPI on one rank could block other ranks, preventing them from reaching the NCCL call that would unblock the NCCL collective on the first rank.

In that case, the cudaStreamSynchronize call should be replaced by a loop like the following:
    
    
    cudaError_t err = cudaErrorNotReady;
    int flag;
    while (err == cudaErrorNotReady) {
      err = cudaStreamQuery(args->streams[i]);
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
    }
    

### Inter-GPU Communication with CUDA-aware MPI[](#inter-gpu-communication-with-cuda-aware-mpi "Permalink to this heading")

Using NCCL to perform inter-GPU communication concurrently with CUDA-aware MPI may create deadlocks.

NCCL creates inter-device dependencies, meaning that after it has been launched, a NCCL kernel will wait (and potentially block the CUDA device) until all ranks in the communicator launch their NCCL kernel. CUDA-aware MPI may also create such dependencies between devices depending on the MPI implementation.

Using both MPI and NCCL to perform transfers between the same sets of CUDA devices concurrently is therefore not guaranteed to be safe.