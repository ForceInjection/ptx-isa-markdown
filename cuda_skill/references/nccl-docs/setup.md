# Setup

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/setup.html

---

# Setup[](#setup "Permalink to this heading")

NCCL is a communication library providing optimized GPU-to-GPU communication for high-performance applications. It is not, like MPI, providing a parallel environment including a process launcher and manager. NCCL relies therefore on the application’s process management system and CPU-side communication system for its own bootstrap.

Similarly to MPI and other libraries which are optimized for performance, NCCL does not provide secure network communication between GPUs. It is therefore the responsibility of the user to ensure NCCL operates over a secure network, both for bootstrap (controlled by [NCCL_SOCKET_IFNAME](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname)) and for high-speed communication.