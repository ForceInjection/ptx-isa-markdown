# Device API – GIN

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html

---

# Device API – GIN[](#device-api-gin "Permalink to this heading")

## GIN[](#gin "Permalink to this heading")

**Device functions.** The following are callable from device (GPU) code only. GIN is supported since NCCL 2.28.7.

### ncclGin[](#ncclgin "Permalink to this heading")

class ncclGin[](#_CPPv47ncclGin "Permalink to this definition")  

    

A class encompassing major elements of the GIN support.

ncclGin(ncclDevComm const &comm, int contextIndex)[](#_CPPv4N7ncclGin7ncclGinERK11ncclDevCommi "Permalink to this definition")  

    

Initializes a new `ncclGin` object. _comm_ is the device communicator created using [`ncclDevCommCreate()`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_setup.html#c.ncclDevCommCreate "ncclDevCommCreate"). _contextIndex_ is the index of the GIN context – a network communication channel. Using multiple GIN contexts allows the implementation to spread traffic onto multiple connections, avoiding locking and bottlenecks. Therefore, performance-oriented kernels should cycle among the available contexts to improve resource utilization (the number of available contexts is available via `ginContextCount`).

void put(ncclTeam team, int peer, ncclWindow_t dstWnd, size_t dstOffset, ncclWindow_t srcWnd, size_t srcOffset, size_t bytes, RemoteAction remoteAction, LocalAction localAction, Coop coop, DescriptorSmem descriptor, cuda::thread_scope alreadyReleased, cuda::thread_scope expected_scope)[](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "Permalink to this definition")  

    

Schedules a device-initiated, one-sided data transfer operation from a local buffer to a remote buffer on a peer.

_peer_ is a rank within _team_ (see [Teams](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-teams)); it may refer to the local rank (a loopback). The destination and source buffers are each specified using the window (_dstWnd_ , _srcWnd_) and a byte-based offset (_dstOffset_ , _srcOffset_). _bytes_ specifies the data transfer count in bytes. If GIN is initialized with connection type `NCCL_GIN_CONNECTION_RAIL`, _peer_ must be within the same rail team as the local rank.

Arguments beyond the first seven are optional. _remoteAction_ and _localAction_ specify actions to undertake on the destination peer and on the local rank when the payload has been settled and the input has been consumed (respectively). They default to `ncclGin_None` (no action); other options include `ncclGin_Signal{Inc|Add}` (for _remoteAction_) and `ncclGin_CounterInc` (for _localAction_); see [Signals and Counters](#devapi-signals) below for more details. _coop_ indicates the set of threads participating in this operation (see [Thread Groups](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-coops)); it defaults to `ncclCoopThread` (a single device thread), which is the recommended model.

The visibility of the signal on the destination peer implies the visibility of the put data it is attached to _and all the preceding puts to the same peer, provided that they were issued using the same GIN context_.

The API also defines an alternative, “convenience” variant of this method that uses `ncclSymPtr` types to specify the buffers and expects size to be conveyed in terms of the number of elements instead of the byte count. There are also two `putValue` variants that take a single element at a time (no greater than eight bytes), passed by value.

void flush(Coop coop, cuda::memory_order ord = cuda::memory_order_acquire)[](#_CPPv4N7ncclGin5flushE4CoopN4cuda12memory_orderE "Permalink to this definition")  

    

Ensures that all the pending transfer operations scheduled by any threads of _coop_ are locally consumed, meaning that their source buffers are safe to reuse. Makes no claims regarding the completion status on the remote peer(s).

### Signals and Counters[](#signals-and-counters "Permalink to this heading")

type ncclGinSignal_t[](#_CPPv415ncclGinSignal_t "Permalink to this definition")  

    

Signals are used to trigger actions on remote peers, most commonly on the completion of a [`ncclGin::put()`](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::put") operation. They each have a 64-bit integer value associated with them that can be manipulated atomically.

struct ncclGin_SignalAdd[](#_CPPv417ncclGin_SignalAdd "Permalink to this definition")  

    

[ncclGinSignal_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal[](#_CPPv4N17ncclGin_SignalAdd6signalE "Permalink to this definition")  

    

uint64_t value[](#_CPPv4N17ncclGin_SignalAdd5valueE "Permalink to this definition")  

    

struct ncclGin_SignalInc[](#_CPPv417ncclGin_SignalInc "Permalink to this definition")  

    

[ncclGinSignal_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal[](#_CPPv4N17ncclGin_SignalInc6signalE "Permalink to this definition")  

    

These objects can be passed as the _remoteAction_ arguments of methods such as [`ncclGin::put()`](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::put") and [`ncclGin::signal()`](#_CPPv4N7ncclGin6signalE8ncclTeami12RemoteAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::signal") to describe the actions to perform on the peer on receipt – in this case, increase the value of a _signal_ specified by index. `ncclGin_SignalInc{signalIdx}` is functionally equivalent to `ncclGin_SignalAdd{signalIdx, 1}`; however, it may not be mixed with other signal-modifying operations without an intervening signal reset (see below). Signal values use “rolling” comparison logic to ensure that an unsigned overflow maintains the property of `x < x + 1`.

struct ncclGin_VASignalInc[](#_CPPv419ncclGin_VASignalInc "Permalink to this definition")  

    

ncclWindow_t signalWindow[](#_CPPv4N19ncclGin_VASignalInc12signalWindowE "Permalink to this definition")  

    

size_t signalOffset[](#_CPPv4N19ncclGin_VASignalInc12signalOffsetE "Permalink to this definition")  

    

struct ncclGin_VASignalAdd[](#_CPPv419ncclGin_VASignalAdd "Permalink to this definition")  

    

ncclWindow_t signalWindow[](#_CPPv4N19ncclGin_VASignalAdd12signalWindowE "Permalink to this definition")  

    

size_t signalOffset[](#_CPPv4N19ncclGin_VASignalAdd12signalOffsetE "Permalink to this definition")  

    

uint64_t value[](#_CPPv4N19ncclGin_VASignalAdd5valueE "Permalink to this definition")  

    

These objects represent “VA signals”: signals that are located at an arbitrary VA (window and offset pair) instead of a pre-allocated signal index. Like the `ncclGin_SignalInc` and `ncclGinSignalAdd` objects, these objects can be passed as the _remoteAction_ arguments of methods such as [`ncclGin::put()`](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::put") and [`ncclGin::signal()`](#_CPPv4N7ncclGin6signalE8ncclTeami12RemoteAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::signal") to increment a signal on the peer. To use a VA signal, the window must be registered with flags [`NCCL_WIN_COLL_STRICT_ORDERING`](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/flags.html#c.NCCL_WIN_COLL_STRICT_ORDERING "NCCL_WIN_COLL_STRICT_ORDERING"). When an address is used as a signal, all reads and writes to the address must be issued via GIN (i.e., a `RemoteAction` or GIN signal method).

**Signal methods of ncclGin:**

void [ncclGin](#_CPPv47ncclGin "ncclGin")::signal(ncclTeam team, int peer, RemoteAction remoteAction, Coop coop, DescriptorSmem descriptor, cuda::thread_scope alreadyReleased, cuda::thread_scope expected_scope)[](#_CPPv4N7ncclGin6signalE8ncclTeami12RemoteAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "Permalink to this definition")  

    

uint64_t [ncclGin](#_CPPv47ncclGin "ncclGin")::readSignal([ncclGinSignal_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal, int bits = 64, cuda::memory_order ord = cuda::memory_order_acquire)[](#_CPPv4N7ncclGin10readSignalE15ncclGinSignal_tiN4cuda12memory_orderE "Permalink to this definition")  

    

void [ncclGin](#_CPPv47ncclGin "ncclGin")::waitSignal(Coop coop, [ncclGinSignal_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal, uint64_t least, int bits = 64, cuda::memory_order ord = cuda::memory_order_acquire)[](#_CPPv4N7ncclGin10waitSignalE4Coop15ncclGinSignal_t8uint64_tiN4cuda12memory_orderE "Permalink to this definition")  

    

void [ncclGin](#_CPPv47ncclGin "ncclGin")::resetSignal([ncclGinSignal_t](#_CPPv415ncclGinSignal_t "ncclGinSignal_t") signal)[](#_CPPv4N7ncclGin11resetSignalE15ncclGinSignal_t "Permalink to this definition")  

    

These are signal-specific methods of [`ncclGin`](#_CPPv47ncclGin "ncclGin"). [`ncclGin::signal()`](#_CPPv4N7ncclGin6signalE8ncclTeami12RemoteAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::signal") implements an explicit signal notification without an accompanying data transfer operation; it takes a subset of arguments of [`ncclGin::put()`](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::put"). [`ncclGin::readSignal()`](#_CPPv4N7ncclGin10readSignalE15ncclGinSignal_tiN4cuda12memory_orderE "ncclGin::readSignal") returns the bottom _bits_ of the value of the _signal_. [`ncclGin::waitSignal()`](#_CPPv4N7ncclGin10waitSignalE4Coop15ncclGinSignal_t8uint64_tiN4cuda12memory_orderE "ncclGin::waitSignal") waits for the bottom _bits_ of the _signal_ value to meet or exceed _least_. Finally, [`ncclGin::resetSignal()`](#_CPPv4N7ncclGin11resetSignalE15ncclGinSignal_t "ncclGin::resetSignal") resets the _signal_ value to `0` (this method may not race with concurrent modifications to the signal).

uint64_t [ncclGin](#_CPPv47ncclGin "ncclGin")::readSignal(ncclWindow_t signalWindow, size_t signalOffset, int bits = 64, cuda::memory_order ord = cuda::memory_order_acquire)[](#_CPPv4N7ncclGin10readSignalE12ncclWindow_t6size_tiN4cuda12memory_orderE "Permalink to this definition")  

    

void [ncclGin](#_CPPv47ncclGin "ncclGin")::waitSignal(Coop coop, ncclWindow_t signalWindow, size_t signalOffset, uint64_t least, int bits = 64, cuda::memory_order ord = cuda::memory_order_acquire)[](#_CPPv4N7ncclGin10waitSignalE4Coop12ncclWindow_t6size_t8uint64_tiN4cuda12memory_orderE "Permalink to this definition")  

    

void [ncclGin](#_CPPv47ncclGin "ncclGin")::resetSignal(ncclWindow_t signalWindow, size_t signalOffset)[](#_CPPv4N7ncclGin11resetSignalE12ncclWindow_t6size_t "Permalink to this definition")  

    

These are VA signal-specific methods of [`ncclGin`](#_CPPv47ncclGin "ncclGin").

type ncclGinCounter_t[](#_CPPv416ncclGinCounter_t "Permalink to this definition")  

    

Counters are used to trigger actions on the local rank; as such, they are complementary to signals, which are meant for remote actions. Like signals, they use “rolling” comparison logic, but they are limited to storing values of at most 56 bits.

struct ncclGin_CounterInc[](#_CPPv418ncclGin_CounterInc "Permalink to this definition")  

    

[ncclGinCounter_t](#_CPPv416ncclGinCounter_t "ncclGinCounter_t") counter[](#_CPPv4N18ncclGin_CounterInc7counterE "Permalink to this definition")  

    

This object can be passed as the _localAction_ argument of methods such as [`ncclGin::put()`](#_CPPv4N7ncclGin3putE8ncclTeami12ncclWindow_t6size_t12ncclWindow_t6size_t6size_t12RemoteAction11LocalAction4Coop14DescriptorSmemN4cuda12thread_scopeEN4cuda12thread_scopeE "ncclGin::put"). It is the only action defined for counters.

**Counter methods of ncclGin:**

uint64_t [ncclGin](#_CPPv47ncclGin "ncclGin")::readCounter([ncclGinCounter_t](#_CPPv416ncclGinCounter_t "ncclGinCounter_t") counter, int bits = 56, cuda::memory_order ord = cuda::memory_order_acquire)[](#_CPPv4N7ncclGin11readCounterE16ncclGinCounter_tiN4cuda12memory_orderE "Permalink to this definition")  

    

void [ncclGin](#_CPPv47ncclGin "ncclGin")::waitCounter(Coop coop, [ncclGinCounter_t](#_CPPv416ncclGinCounter_t "ncclGinCounter_t") counter, uint64_t least, int bits = 56, cuda::memory_order ord = cuda::memory_order_acquire)[](#_CPPv4N7ncclGin11waitCounterE4Coop16ncclGinCounter_t8uint64_tiN4cuda12memory_orderE "Permalink to this definition")  

    

void [ncclGin](#_CPPv47ncclGin "ncclGin")::resetCounter([ncclGinCounter_t](#_CPPv416ncclGinCounter_t "ncclGinCounter_t") counter)[](#_CPPv4N7ncclGin12resetCounterE16ncclGinCounter_t "Permalink to this definition")  

    

These are counter-specific methods of [`ncclGin`](#_CPPv47ncclGin "ncclGin") and they are functionally equivalent to their signal counterparts discussed above.

### ncclGinBarrierSession[](#ncclginbarriersession "Permalink to this heading")

template<typename Coop>  
class ncclGinBarrierSession[](#_CPPv4I0E21ncclGinBarrierSession "Permalink to this definition")  

    

A class representing a network barrier session.

ncclGinBarrierSession([Coop](#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession::Coop") coop, [ncclGin](#_CPPv47ncclGin "ncclGin") gin, ncclTeamTagRail tag, uint32_t index)[](#_CPPv4N21ncclGinBarrierSession21ncclGinBarrierSessionE4Coop7ncclGin15ncclTeamTagRail8uint32_t "Permalink to this definition")  

    

Initializes a new network barrier session. _coop_ represents a cooperative group (see [Thread Groups](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-coops)). _gin_ is a previously initialized [`ncclGin`](#_CPPv47ncclGin "ncclGin") object. _ncclTeamTagRail_ indicates that the barrier will apply to all peers on the same rail as the local rank (see [Teams](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-teams)). _index_ identifies the underlying barrier to use (it should be different for each _coop_ ; typically set to `blockIdx.x` to ensure uniqueness between CTAs).

ncclGinBarrierSession([Coop](#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession::Coop") coop, [ncclGin](#_CPPv47ncclGin "ncclGin") gin, ncclTeam team, ncclGinBarrierHandle handle, uint32_t index)[](#_CPPv4N21ncclGinBarrierSession21ncclGinBarrierSessionE4Coop7ncclGin8ncclTeam20ncclGinBarrierHandle8uint32_t "Permalink to this definition")  

    

Initializes a new network barrier session. This is the general-purpose variant to be used, e.g., when communicating with ranks from the world team (see [Teams](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/deviceapi.html#devapi-teams)), whereas the previous variant was specific to the rail team. This variant expects _team_ to be passed as an argument, and also takes an extra _handle_ argument indicating the location of the underlying barriers (typically set to the `railGinBarrier` field of the device communicator).

void sync([Coop](#_CPPv4I0E21ncclGinBarrierSession "ncclGinBarrierSession::Coop") coop, cuda::memory_order order, ncclGinFenceLevel fence)[](#_CPPv4N21ncclGinBarrierSession4syncE4CoopN4cuda12memory_orderE17ncclGinFenceLevel "Permalink to this definition")  

    

Synchronizes all threads of all team members that participate in the barrier session. `ncclGinFenceLevel::Relaxed` is the only defined value for _fence_ for now.