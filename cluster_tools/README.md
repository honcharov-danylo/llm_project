## Cluster tools

Directory for useful commands and examples.

---
- show_nodes.sh - provides snapshot of what’s available in the cluster.
  Full command is:
  
  ```$ condor_status -af CUDADeviceName CUDADriverVersion CUDACapability CUDAGlobalMemoryMb | sort -u```
    
- Tested output (might change over time)        
  -     NVIDIA A100 80GB PCIe 12.0 8.0 81085
        NVIDIA A40 12.0 8.6 45466
        NVIDIA GeForce GTX 1080 Ti 12.0 6.1 11172
        Quadro RTX 5000 11.6 7.5 16125
        Quadro RTX 6000 11.6 7.5 22699
        
  - You can specify requirements on any of these fields, including the device name if you really need something specific.  However, as always, the more restrictive your requirements are the longer it may take for those resources to be available.  