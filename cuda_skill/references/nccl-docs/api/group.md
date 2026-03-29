# Group Calls

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/group.html

---

# Group Calls[](#group-calls "Permalink to this heading")

Group primitives define the behavior of the current thread to avoid blocking. They can therefore be used from multiple threads independently.

Related links: [Group Calls](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/groups.html#group-calls).

## ncclGroupStart[](#ncclgroupstart "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclGroupStart()[](#c.ncclGroupStart "Permalink to this definition")  

    

Start a group call.

All subsequent calls to NCCL until ncclGroupEnd will not block due to inter-CPU synchronization.

## ncclGroupEnd[](#ncclgroupend "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclGroupEnd()[](#c.ncclGroupEnd "Permalink to this definition")  

    

End a group call.

Returns when all operations since ncclGroupStart have been processed. This means the communication primitives have been enqueued to the provided streams, but are not necessarily complete.

When used with the ncclCommInitRank call, the ncclGroupEnd call waits for all communicators to be initialized.

## ncclGroupSimulateEnd[](#ncclgroupsimulateend "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclGroupSimulateEnd([ncclSimInfo_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclSimInfo_t "ncclSimInfo_t") *simInfo)[](#c.ncclGroupSimulateEnd "Permalink to this definition")  

    

Simulate a ncclGroupEnd() call and return NCCL’s simulation info in a structure passed as an argument.