# User Defined Reduction Operators

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/ops.html

---

# User Defined Reduction Operators[](#user-defined-reduction-operators "Permalink to this heading")

The following functions are public APIs exposed by NCCL to create and destroy custom reduction operators for use in reduction collectives.

## ncclRedOpCreatePreMulSum[](#ncclredopcreatepremulsum "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclRedOpCreatePreMulSum([ncclRedOp_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclRedOp_t "ncclRedOp_t") *op, void *scalar, [ncclDataType_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclDataType_t "ncclDataType_t") datatype, [ncclScalarResidence_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclScalarResidence_t "ncclScalarResidence_t") residence, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm)[](#c.ncclRedOpCreatePreMulSum "Permalink to this definition")  

    

Creates a new reduction operator which pre-multiplies input values by a given scalar locally before reducing them with peer values via summation. Both the input values and the scalar are of type _datatype_. For use only with collectives launched against _comm_ and _datatype_. The _residence_ argument indicates whether the memory pointed to by _scalar_ should be dereferenced immediately by the host before this function returns (ncclScalarHostImmediate), or by the device during execution of the reduction collective (ncclScalarDevice). Upon return, the newly created operator’s handle is stored in _op_.

## ncclRedOpDestroy[](#ncclredopdestroy "Permalink to this heading")

[ncclResult_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclResult_t "ncclResult_t") ncclRedOpDestroy([ncclRedOp_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclRedOp_t "ncclRedOp_t") op, [ncclComm_t](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html#c.ncclComm_t "ncclComm_t") comm)[](#c.ncclRedOpDestroy "Permalink to this definition")  

    

Destroys the reduction operator _op_. The operator must have been created by ncclRedOpCreatePreMul with the matching communicator _comm_. An operator may be destroyed as soon as the last NCCL function which is given that operator returns.