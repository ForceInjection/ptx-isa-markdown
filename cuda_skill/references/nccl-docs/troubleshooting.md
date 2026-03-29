# Troubleshooting

**Source:** https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html

---

# Troubleshooting[](#troubleshooting "Permalink to this heading")

Ensure you are familiar with the following known issues and useful debugging strategies.

## Errors[](#errors "Permalink to this heading")

NCCL calls may return a variety of return codes. Ensure that the return codes are always equal to ncclSuccess. If any call fails and returns a value different from ncclSuccess, setting NCCL_DEBUG to “WARN” will make NCCL print an explicit warning message before returning the error.

Errors are grouped into different categories.

  * ncclUnhandledCudaError and ncclSystemError indicate that a call to an external library failed.

  * ncclInvalidArgument and ncclInvalidUsage indicate there was a programming error in the application using NCCL.


In either case, refer to the NCCL warning message to understand how to resolve the problem.

## RAS[](#ras "Permalink to this heading")

Starting with version 2.24, NCCL includes a reliability, availability, and serviceability (RAS) subsystem to help with the diagnosis and debugging of crashes and hangs.

  * [RAS](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html)
    * [Principle of Operation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html#principle-of-operation)
    * [RAS Queries](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html#ras-queries)
    * [Sample Output](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html#sample-output)
    * [JSON Output](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html#json-output)
    * [Monitoring Mode](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/ras.html#monitoring-mode)


## GPU Direct[](#gpu-direct "Permalink to this heading")

NCCL heavily relies on GPU Direct for inter-GPU communication. This refers to the ability for a GPU to directly communicate with another device, such as another GPU or a network card, using direct point-to-point PCI messages.

Direct point-to-point PCI messages can fail or perform poorly for a variety of reasons, like missing components, a bad configuration of a virtual machine or a container, or some BIOS settings.

### GPU-to-GPU communication[](#gpu-to-gpu-communication "Permalink to this heading")

To make sure GPU-to-GPU communication is working correctly, look for the `p2pBandwidthLatencyTest` from the CUDA samples found here: <https://github.com/nvidia/cuda-samples>
    
    
    cd cuda-samples/Samples/5_Domain_Specific/p2pBandwidthLatencyTest
    make
    ./p2pBandwidthLatencyTest
    

The test should run to completion and report good performance between GPUs.

Another tool for checking GPU-to-GPU performance is called `nvbandwidth`. This can be downloaded and built from the code and instructions found here: <https://github.com/NVIDIA/nvbandwidth>

### GPU-to-NIC communication[](#gpu-to-nic-communication "Permalink to this heading")

GPUs can also communicate directly with network cards using GPU Direct RDMA (GDRDMA). This requires having compatible network cards and drivers, plus loading an extra kernel module called `nvidia-peermem`. The `nvidia-peermem` module is now supplied with the CUDA drivers, however it must be loaded on each node boot with:
    
    
    sudo modprobe nvidia-peermem
    

GDRDMA can also be enabled by using the DMA-BUF feature of recent Linux kernels combined with the Open Source Nvidia GPU driver. In this case, NCCL will automatically detect and enable DMA-BUF so the nvidia-peermem module will not be necessary.

### PCI Access Control Services (ACS)[](#pci-access-control-services-acs "Permalink to this heading")

**Baremetal systems**

IO virtualization (also known as VT-d or IOMMU) can interfere with GPU Direct by redirecting all PCI point-to-point traffic to the CPU root complex, causing a significant performance reduction or even a hang. You can check whether ACS is enabled on PCI bridges by running:
    
    
    sudo lspci -vvv | grep ACSCtl
    

If lines show “SrcValid+”, then ACS might be enabled. Looking at the full output of lspci, one can check if a PCI bridge has ACS enabled.
    
    
    sudo lspci -vvv
    

If PCI switches have ACS enabled, it needs to be disabled. On some systems this can be done from the BIOS by disabling IO virtualization or VT-d. For Broadcom PLX devices, it can be done from the OS but needs to be done again after each reboot.

Use the command below to find the PCI bus IDs of PLX PCI bridges:
    
    
    sudo lspci | grep PLX
    

Next, use setpci to disable ACS with the command below, replacing 03:00.0 by the PCI bus ID of each PCI bridge.
    
    
    sudo setpci -s 03:00.0 ECAP_ACS+0x6.w=0000
    

Or you can use a script similar to this:
    
    
    for BDF in `lspci -d "*:*:*" | awk '{print $1}'`; do
      # skip if it doesn't support ACS
      sudo setpci -v -s ${BDF} ECAP_ACS+0x6.w > /dev/null 2>&1
      if [ $? -ne 0 ]; then
        continue
      fi
      sudo setpci -v -s ${BDF} ECAP_ACS+0x6.w=0000
    done
    

**Virtual machines**

Virtual machines require ACS to function, hence disabling ACS is not an option. To run with maximum performance inside virtual machines, ATS needs to be enabled in network adapters.

## Topology detection[](#topology-detection "Permalink to this heading")

NCCL relies on /sys to discover the PCI topology of GPUs and network cards. When running inside a virtual machine or container, make sure /sys is properly mounted. Having /sys expose a virtual PCI topology can result in sub-optimal performance.

## Memory issues[](#memory-issues "Permalink to this heading")

### Shared memory[](#shared-memory "Permalink to this heading")

To communicate between processes and even between threads of a process, NCCL creates shared memory segments, traditionally in /dev/shm. The operating system’s limits on these resources may need to be increased accordingly. Please see your system’s documentation for details.

If insufficient shared memory is available, NCCL will fail to initialize. Running with NCCL_DEBUG=WARN will show a message similar to this:
    
    
    NCCL WARN Error: failed to extend /dev/shm/nccl-03v824 to 4194660 bytes
    

**Docker**

In particular, Docker containers default to limited shared and pinned memory resources. When using NCCL inside a container, please make sure to adjust the shared memory size inside the container, for example by adding the following arguments to the docker launch command line:
    
    
    --shm-size=1g --ulimit memlock=-1
    

**Systemd**

When running jobs using mpirun or SLURM, systemd may remove files in shared memory when it detects that the corresponding user is not logged in, in an attempt to clean up old temporary files. This can cause NCCL to crash during init with an error like:
    
    
    NCCL WARN unlink shared memory /dev/shm/nccl-d5rTd0 failed, error: No such file or directory
    

Given mpirun and SLURM jobs can run on the node without the user being seen as logged in by systemd, system administrators need to disable that clean-up mechanism, which can be performed by SLURM epilogue scripts instead. To do this, the following line needs to be set in /etc/systemd/logind.conf:
    
    
    RemoveIPC=no
    

Once updated, the daemons should be restarted with:
    
    
    sudo systemctl restart systemd-logind
    

**cuMem host allocations**

Starting with version 2.23, NCCL supports an alternative shared memory mechanism using cuMem host allocations. From NCCL 2.24, if CUDA driver >= 12.6 and CUDA runtime >= 12.2, it is enabled by default in favor of /dev/shm.

However, cuMem host allocations rely on correctly configured and working NUMA support, which may not be available in some VM and containerization scenarios. In particular, Docker by default disables NUMA support (it can be enabled by invoking Docker with `--cap-add SYS_NICE`). From version 2.26.5, NCCL checks if cuMem host allocations work and, if needed, automatically falls back to the /dev/shm code. In prior versions, the same outcome can be achieved by manually specifying `NCCL_CUMEM_HOST_ENABLE=0`. We still recommend configuring the underlying system to ensure that cuMem host allocations work, as they provide improved reliability during communicator aborts.

cuMem host allocations may fail on systems without CUDA P2P connectivity if CUDA driver version prior to 13.0 is being used. Furthermore, [CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html) feature can affect NCCL’s ability to accurately determine the current driver version, resulting in cuMem host allocations being enabled on older drivers than intended. We continue to investigate additional mechanisms to detect such circumstances; in the meantime, use `NCCL_CUMEM_HOST_ENABLE=0` to deactivate this feature if it causes issues.

### Stack size[](#stack-size "Permalink to this heading")

NCCL’s graph search algorithm is highly recursive and, especially on MNNVL systems where many ranks are reachable via CUDA P2P, may temporarily require more than 2 MB of thread stack during communicator creation. While the default Linux stack size limit (8 MB) is known to be sufficient, we’ve seen crashes if the limit is changed to `unlimited`. Due to an idiosyncrasy of GNU libc (see the man page of `pthread_create(3)`), such a setting results in a _decrease_ of the stack size of NCCL’s background threads to just 2 MB, which may not be sufficiently large. Use `ulimit -s` in bash to print the current limit; if needed, reset it to 8192 KB using `ulimit -s 8192` (one also needs to ensure that the new setting is propagated to other nodes when launching a multi-node NCCL job). Starting with version 2.28, NCCL queries the default stack size for newly launched threads and, if necessary, changes it to a safe value for the current job. We still recommend that users on affected systems attempt to get the system-wide setting fixed as – however well intentioned – it is a potentially serious misconfiguration that could have negative effects extending beyond NCCL jobs.

### Unified Memory (UVM)[](#unified-memory-uvm "Permalink to this heading")

Starting with version 2.23, NCCL utilizes CUDA memory pools to optimize graph capturing. This feature relies on UVM being available. While UVM may not be on by default in some virtual machine (VM) setups, it can typically be enabled through a configuration change.

## Networking issues[](#networking-issues "Permalink to this heading")

### IP Network Interfaces[](#ip-network-interfaces "Permalink to this heading")

NCCL auto-detects which network interfaces to use for inter-node communication. If some interfaces are in the UP state but are not able to communicate between nodes, NCCL may try to use them anyway and therefore fail during the init functions or even hang.

For information about how to specify which interfaces to use, see the Environment Variables section, particularly the `NCCL_SOCKET_IFNAME` environment variable.

### IP Ports[](#ip-ports "Permalink to this heading")

NCCL opens TCP ports to connect processes together and exchange connection information. To restrict the range of ports used by NCCL, one can set the `net.ipv4.ip_local_port_range` property of the Linux kernel.

This example shows how to restrict NCCL ports to 50000-51000:
    
    
    echo 50000 51000 > /proc/sys/net/ipv4/ip_local_port_range
    

Or to make this permanent, add a line to /etc/sysctl.conf:
    
    
    echo "net.ipv4.ip_local_port_range = 50000 51000" >> /etc/sysctl.conf
    

Restricting the port range can be useful to open a corresponding range in the firewall, for example on Google Cloud:
    
    
    gcloud compute --project=myproject firewall-rules create ncclnet0-ingress --direction=INGRESS --priority=1 --network=ncclnet --action=ALLOW --rules=tcp:50000-51000,22,1024-1039 --destination-ranges=0.0.0.0/0 --target-tags=ncclnet
    

### InfiniBand[](#infiniband "Permalink to this heading")

Before running NCCL on InfiniBand, running low-level InfiniBand tests (and in particular the ib_write_bw test) can help verify whether the nodes are able to communicate properly.

A common issue seen with InfiniBand is the library not being able to register sufficient pinned memory. In such cases you may see an error like:
    
    
    NCCL WARN Call to ibv_create_qp failed
    

or
    
    
    NCCL WARN Call to ibv_reg_mr failed
    

The solution is to remove the user limits on registering pinned memory. This can be done by adding these lines:
    
    
    * soft memlock unlimited
    * hard memlock unlimited
    

To the `/etc/security/limits.conf` configuration file or equivalent on your Linux distribution.

### RDMA over Converged Ethernet (RoCE)[](#rdma-over-converged-ethernet-roce "Permalink to this heading")

Before running NCCL on RoCE, running low-level RDMA tests (and in particular the `ib_write_bw` test) can help verify whether the nodes are able to communicate properly.

A common issue seen with RoCE is the incorrect GID Index being selected for the RoCE v2 NICs. This can result in the following error:
    
    
    NCCL WARN Call to ibv_modify_qp failed with error Invalid argument
    

With NCCL 2.21 and later the GID index is dynamically selected, but with prior versions the user would need to run:
    
    
    show_gids
    

And then set `NCCL_IB_GID_INDEX` to the GID INDEX for the RoCE v2 VER GID. With NCCL 2.21 and later releases, this environment variable should _not_ be set.

Users may also need to set `NCCL_IB_TC` when using RoCE based networks. Refer to your vendor’s documentation for the values this should be set to.

### MPI[](#mpi "Permalink to this heading")

Before running NCCL with MPI (e.g. `mpirun <my_application>`), running a simple MPI test can help verify whether the nodes are able to communicate properly.

You can do this is two steps. First make sure an application can be launched in parallel:
    
    
    # Open MPI based MPIs:
    mpirun -np <number of processes> -N <processes per node> "hostname"
    
    # MPICH based MPIs:
    mpirun -np <number of processes> -ppn <processes per node> "hostname"
    

Second, make sure MPI can be initialized and run a simple reduction:
    
    
    wget https://raw.githubusercontent.com/pmodels/mpich/main/examples/cpi.c
    mpicc -o cpi cpi.c
    mpirun -np <number of processes> -N <processes per node> ./cpi
    

### Open MPI based MPIs (e.g. NVIDIA HPC-X)[](#open-mpi-based-mpis-e-g-nvidia-hpc-x "Permalink to this heading")

Many NCCL-based applications are compiled with MPI to utilize its parallel launcher and broadcast mechanisms during startup. In cluster environments, if MPI is not correctly configured, the `mpirun` command may fail to start applications, hang, or produce errors. The following guidelines will help you troubleshoot common MPI-related startup and connectivity issues. These settings assume an environment in which variables are automatically forwarded to each MPI rank (e.g. SLURM cluster). If you are unsure you can explicitly forward the variables through `mpirun -x VARIABLE_NAME=<variable_value>` instead of `export VARIABLE_NAME=<variable_value>`.

These settings will not have any impact on NCCL performance, but if MPI is used frequently for communications, then application performance may be impacted.

#### Network interface selection[](#network-interface-selection "Permalink to this heading")

If the application hangs at startup or displays a segmentation fault in `libmpi.so`, MPI may be selecting an incorrect network interface. You can list active and connected interfaces with:
    
    
    ip -br link | grep LOWER_UP | grep ' UP '
    

Usually, only a subset of interfaces (such as `eth*`, `en*`, or `ib*`) are connected to the network. Loopback (`lo`) and container-related interfaces are typically not suitable. If your administrator has specified `NCCL_SOCKET_IFNAME`, use the same interface with MPI by setting:
    
    
    export OMPI_MCA_btl_tcp_if_include=<interface-name>
    

Alternatively, to exclude interfaces that are usually not connected to the network (used for loopback or containers):
    
    
    export OMPI_MCA_btl_tcp_if_exclude=lo,docker0,virbr0
    

Note: Do not use include and exclude options simultaneously.

#### PMIx Data Store selection[](#pmix-data-store-selection "Permalink to this heading")

There has been an issue (see <https://github.com/open-mpi/ompi/issues/7516>) with an PMIx component in Open MPI in the past. This has since been fixed, but can still occur if you MPI is based on an odler version. If the application reports an error similar to
    
    
    PMIX ERROR: ERROR in file gds_ds12_lock_pthread.c
    

You can force a different GDS component through `export PMIX_MCA_gds=hash`.

#### UCX and HPC-X considerations[](#ucx-and-hpc-x-considerations "Permalink to this heading")

HPC-X commonly utilizes the Unified Communication X (UCX) library. If you encounter UCX warnings such as:
    
    
    UCX  WARN  network device 'XXX' is not available, please use one or more of: YYY, ...
    

set the device explicitly:
    
    
    export UCX_NET_DEVICES=YYY
    

For UCX error messages like:
    
    
    UCX  ERROR   no active messages transport to <no debug data>: Unsupported operation
    Error: Failed to resolve UCX endpoint
    

try simplifying the UCX transport selection:
    
    
    export UCX_TLS=self,sm,tcp
    

If necessary, you can disable UCX components and revert to basic TCP communication:
    
    
    export OMPI_MCA_pml=^ucx
    export OMPI_MCA_coll_hcoll_enable=0
    export OMPI_MCA_coll=^ucc
    export OMPI_MCA_btl=self,tcp