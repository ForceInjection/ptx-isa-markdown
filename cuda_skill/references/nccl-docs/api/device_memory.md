# Device API – Memory and LSA

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_memory.html

---

# Device API – Memory and LSA[](#device-api-memory-and-lsa "Permalink to this heading")

This page documents device-side memory and LSA (load/store accessible) functionality. For host-accessible device pointer functions, see [Host-Accessible Device Pointer Functions](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#device-api-host-functions) in the setup guide.

## LSA[](#lsa "Permalink to this heading")

**Device functions.** The following are callable from device (GPU) code only. LSA is used by the pointer accessors below.

### ncclLsaBarrierSession[](#nccllsabarriersession "Permalink to this heading")

template<typename Coop>  
class ncclLsaBarrierSession[](#_CPPv4I0E21ncclLsaBarrierSession "Permalink to this definition")  

    

A class representing a memory barrier session.

ncclLsaBarrierSession([Coop](#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession::Coop") coop, ncclDevComm const &comm, ncclTeamTagLsa tag, uint32_t index, bool multimem = false)[](#_CPPv4N21ncclLsaBarrierSession21ncclLsaBarrierSessionE4CoopRK11ncclDevComm14ncclTeamTagLsa8uint32_tb "Permalink to this definition")  

    

Initializes a new memory barrier session. _coop_ represents a cooperative group (see [Thread Groups](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-coops)). _comm_ is the device communicator created using [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate"). _ncclTeamTagLsa_ is here to indicate which subset of ranks the barrier will apply to. The identifier of the underlying barrier to use is provided by _index_ (it should be different for each _coop_ ; typically set to `blockIdx.x` to ensure uniqueness between CTAs). _multimem_ requests a hardware-accelerated implementation using memory multicast.

void arrive([Coop](#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession::Coop"), cuda::memory_order order)[](#_CPPv4N21ncclLsaBarrierSession6arriveE4CoopN4cuda12memory_orderE "Permalink to this definition")  

    

Signals the arrival of the thread at the barrier session.

void wait([Coop](#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession::Coop"), cuda::memory_order order)[](#_CPPv4N21ncclLsaBarrierSession4waitE4CoopN4cuda12memory_orderE "Permalink to this definition")  

    

Blocks until all threads of all team members arrive at the barrier session.

void sync([Coop](#_CPPv4I0E21ncclLsaBarrierSession "ncclLsaBarrierSession::Coop"), cuda::memory_order order)[](#_CPPv4N21ncclLsaBarrierSession4syncE4CoopN4cuda12memory_orderE "Permalink to this definition")  

    

Synchronizes all threads of all team members that participate in the barrier session (combines `arrive` and `wait`).

### ncclGetPeerPointer[](#ncclgetpeerpointer "Permalink to this heading")

void *ncclGetPeerPointer(ncclWindow_t w, size_t offset, int peer)[](#_CPPv418ncclGetPeerPointer12ncclWindow_t6size_ti "Permalink to this definition")  

    

Returns a load/store accessible pointer to the memory buffer of device _peer_ within the window _w_. _offset_ is byte-based. _peer_ is a rank index within the world team (see [Teams](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-teams)). This function will return NULL if the _peer_ is not within the [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) team.

### ncclGetLsaPointer[](#ncclgetlsapointer "Permalink to this heading")

void *ncclGetLsaPointer(ncclWindow_t w, size_t offset, int lsaPeer)[](#_CPPv417ncclGetLsaPointer12ncclWindow_t6size_ti "Permalink to this definition")  

    

Returns a load/store accessible pointer to the memory buffer of device _lsaPeer_ within the window _w_. _offset_ is byte-based. This is similar to `ncclGetPeerPointer`, but here _lsaPeer_ is a rank index within the [LSA](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#device-api-lsa) team (see [Teams](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-teams)). For high-level reduce and copy operations over LSA memory, see [Device API – Remote Reduce and Copy: Building Blocks for Custom Communication Kernels](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_reducecopy.html#device-api-reducecopy).

### ncclGetLocalPointer[](#ncclgetlocalpointer "Permalink to this heading")

void *ncclGetLocalPointer(ncclWindow_t w, size_t offset)[](#_CPPv419ncclGetLocalPointer12ncclWindow_t6size_t "Permalink to this definition")  

    

Returns a load-store accessible pointer to the memory buffer of the current device within the window _w_. _offset_ is byte-based. This is just a shortcut version of `ncclGetPeerPointer` with _devComm.rank_ as _peer_ , or `ncclGetLsaPointer` with _devComm.lsaRank_ as _lsaPeer_.

## Multimem[](#multimem "Permalink to this heading")

### ncclGetLsaMultimemPointer[](#ncclgetlsamultimempointer "Permalink to this heading")

void *ncclGetLsaMultimemPointer(ncclWindow_t w, size_t offset, ncclDevComm const &devComm)[](#_CPPv425ncclGetLsaMultimemPointer12ncclWindow_t6size_tRK11ncclDevComm "Permalink to this definition")  

    

Returns a multicast memory pointer associated with the window _w_ and device communicator _devComm_. _offset_ is byte-based. Availability of multicast memory is hardware-dependent.