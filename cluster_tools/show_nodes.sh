# script to show available devices on nodes / cuda driver version / amount of memory on the gpu
condor_status -af CUDADeviceName CUDADriverVersion CUDACapability CUDAGlobalMemoryMb | sort -u
