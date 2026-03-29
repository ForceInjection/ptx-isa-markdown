# Collective Operations

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

---

# Collective Operations[](#collective-operations "Permalink to this heading")

Collective operations have to be called for each rank (hence CUDA device), using the same count and the same datatype, to form a complete collective operation. Failure to do so will result in undefined behavior, including hangs, crashes, or data corruption.

## AllReduce[](#allreduce "Permalink to this heading")

The AllReduce operation performs reductions on data (for example, sum, min, max) across devices and stores the result in the receive buffer of every rank.

In a _sum_ allreduce operation between _k_ ranks, each rank will provide an array in of N values, and receive identical results in array out of N values, where out[i] = in0[i]+in1[i]+…+in(k-1)[i].

![../_images/allreduce.png](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allreduce.png)

All-Reduce operation: each rank receives the reduction of input values across ranks.[](#id9 "Permalink to this image")

Related links: [`ncclAllReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllReduce "ncclAllReduce").

## Broadcast[](#broadcast "Permalink to this heading")

The Broadcast operation copies an N-element buffer from the root rank to all the ranks.

![../_images/broadcast.png](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/broadcast.png)

Broadcast operation: all ranks receive data from a “root” rank.[](#id10 "Permalink to this image")

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

Related links: [`ncclBroadcast()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclBroadcast "ncclBroadcast").

## Reduce[](#reduce "Permalink to this heading")

The Reduce operation performs the same operation as AllReduce, but stores the result only in the receive buffer of a specified root rank.

![../_images/reduce.png](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reduce.png)

Reduce operation: one rank receives the reduction of input values across ranks.[](#id11 "Permalink to this image")

Important note: The root argument is one of the ranks (not a device number), and is therefore impacted by a different rank to device mapping.

Note: A Reduce, followed by a Broadcast, is equivalent to the AllReduce operation.

Related links: [`ncclReduce()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduce "ncclReduce").

## AllGather[](#allgather "Permalink to this heading")

The AllGather operation gathers N values from k ranks into an output buffer of size k*N, and distributes that result to all ranks.

The output is ordered by the rank index. The AllGather operation is therefore impacted by a different rank to device mapping.

![../_images/allgather.png](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/allgather.png)

AllGather operation: each rank receives the aggregation of data from all ranks in the order of the ranks.[](#id12 "Permalink to this image")

Note: Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation.

Related links: [`ncclAllGather()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAllGather "ncclAllGather").

## ReduceScatter[](#reducescatter "Permalink to this heading")

The ReduceScatter operation performs the same operation as Reduce, except that the result is scattered in equal-sized blocks between ranks, each rank getting a chunk of data based on its rank index.

The ReduceScatter operation is impacted by a different rank to device mapping since the ranks determine the data layout.

![../_images/reducescatter.png](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/reducescatter.png)

Reduce-Scatter operation: input values are reduced across ranks, with each rank receiving a subpart of the result.[](#id13 "Permalink to this image")

Related links: [`ncclReduceScatter()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclReduceScatter "ncclReduceScatter")

## AlltoAll[](#alltoall "Permalink to this heading")

In an AlltoAll operation between k ranks, each rank provides an input buffer of size k*N values, where the j-th chunk of N values is sent to destination rank j. Each rank receives an output buffer of size k*N values, where the i-th chunk of N values comes from source rank i.

![../_images/alltoall.png](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/alltoall.png)

AlltoAll operation: exchanges data between all ranks, where each rank sends different data to every other rank and receives different data from every other rank.[](#id14 "Permalink to this image")

Related links: [`ncclAlltoAll()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclAlltoAll "ncclAlltoAll").

## Gather[](#gather "Permalink to this heading")

The Gather operation gathers N values from k ranks into an output buffer on the root rank of size k*N.

![../_images/gather.png](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/gather.png)

Gather operation: root rank receives data from all ranks.[](#id15 "Permalink to this image")

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

Related links: [`ncclGather()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclGather "ncclGather").

## Scatter[](#scatter "Permalink to this heading")

The Scatter operation distributes a total of N*k values from the root rank to k ranks, each rank receiving N values.

![../_images/scatter.png](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/_images/scatter.png)

Scatter operation: root rank distributes data to all ranks.[](#id16 "Permalink to this image")

Important note: The root argument is one of the ranks, not a device number, and is therefore impacted by a different rank to device mapping.

Related links: [`ncclScatter()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/colls.html#c.ncclScatter "ncclScatter").