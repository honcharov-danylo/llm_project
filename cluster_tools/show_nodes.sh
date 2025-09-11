# script to show available devices on nodes / cuda driver version / amount of memory on the gpu
condor_status -constraint 'CUDADriverVersion >= 4.0' -af Name CUDADeviceName CUDADriverVersion CUDACapability CUDAGlobalMemoryMb GPUs TotalGPUs TotalSlotGPUs | awk 'BEGIN {
    # ANSI color codes
    BOLD="\033[1m"
    BLUE="\033[34m"
    GREEN="\033[32m"
    YELLOW="\033[33m"
    CYAN="\033[36m"
    MAGENTA="\033[35m"
    RESET="\033[0m"
}
{
    # Parse fields directly by position
    name = $1
    
    # Handle undefined values
    for (i = 2; i <= 8; i++) {
        if ($i == "undefined" || $i == "") $i = "N/A"
    }
    
    # Extract GPU name - more robust approach
    gpu_name = ""
    for (i = 2; i <= NF; i++) {
        # Check if we hit a numeric field (driver version)
        if ($i ~ /^[0-9]+\.[0-9]+$/) {
            # Everything from field 2 to i-1 is the GPU name
            for (j = 2; j < i; j++) {
                gpu_name = gpu_name (j == 2 ? "" : " ") $j
            }
            driver = $i
            capability = $(i+1)
            memory = $(i+2)
            gpus = $(i+3)
            total_gpus = $(i+4)
            total_slot_gpus = $(i+5)
            break
        }
    }
    
    # Fallback if no driver version found
    if (gpu_name == "") {
        gpu_name = $2
        driver = $3
        capability = $4
        memory = $5
        gpus = $6
        total_gpus = $7
        total_slot_gpus = $8
    }
    
    # Store data
    n++
    names[n] = name
    gpu_names[n] = gpu_name
    drivers[n] = driver
    capabilities[n] = capability
    memories[n] = memory
    gpus_list[n] = gpus
    total_gpus_list[n] = total_gpus
    total_slot_gpus_list[n] = total_slot_gpus
    
    # Track max widths for dynamic formatting
    if (length(name) > max_name) max_name = length(name)
    if (length(gpu_name) > max_gpu) max_gpu = length(gpu_name)
}
END {
    # Set minimum column widths
    if (max_name < 20) max_name = 20
    if (max_gpu < 20) max_gpu = 20
    
    # Print header with dynamic widths
    printf BOLD BLUE "%-*s %-*s %-10s %-12s %-10s %-5s %-9s %-12s\n" RESET, 
        max_name, "Name", max_gpu, "CUDADeviceName", "Driver", "Capability", "MemoryMb", "GPUs", "TotalGPUs", "SlotGPUs"
    
    # Print separator
    for (i = 1; i <= max_name; i++) printf "="
    printf " "
    for (i = 1; i <= max_gpu; i++) printf "="
    printf " %-10s %-12s %-10s %-5s %-9s %-12s\n",
        "==========", "============", "==========", "=====", "=========", "============"
    
    # Bubble sort by capability (descending) and GPUs (descending within same capability)
    for (i = 1; i <= n; i++) {
        for (j = 1; j <= n - i; j++) {
            # Convert capability to numeric for comparison
            cap1 = capabilities[j]
            cap2 = capabilities[j+1]
            gsub(/[^0-9.]/, "", cap1)  # Remove non-numeric chars
            gsub(/[^0-9.]/, "", cap2)
            
            # Convert GPUs to numeric (handle N/A)
            gpu1 = gpus_list[j]
            gpu2 = gpus_list[j+1]
            if (gpu1 == "N/A") gpu1 = -1
            if (gpu2 == "N/A") gpu2 = -1
            
            # Sort logic: 
            # 1. Primary: capability descending (cap1 < cap2 means swap)
            # 2. Secondary: if capabilities equal, sort by GPUs descending
            swap = 0
            if (cap1 + 0 < cap2 + 0) {
                swap = 1
            } else if (cap1 + 0 == cap2 + 0 && gpu1 + 0 < gpu2 + 0) {
                swap = 1
            }
            
            if (swap) {
                # Swap all arrays
                temp = names[j]; names[j] = names[j+1]; names[j+1] = temp
                temp = gpu_names[j]; gpu_names[j] = gpu_names[j+1]; gpu_names[j+1] = temp
                temp = drivers[j]; drivers[j] = drivers[j+1]; drivers[j+1] = temp
                temp = capabilities[j]; capabilities[j] = capabilities[j+1]; capabilities[j+1] = temp
                temp = memories[j]; memories[j] = memories[j+1]; memories[j+1] = temp
                temp = gpus_list[j]; gpus_list[j] = gpus_list[j+1]; gpus_list[j+1] = temp
                temp = total_gpus_list[j]; total_gpus_list[j] = total_gpus_list[j+1]; total_gpus_list[j+1] = temp
                temp = total_slot_gpus_list[j]; total_slot_gpus_list[j] = total_slot_gpus_list[j+1]; total_slot_gpus_list[j+1] = temp
            }
        }
    }
    
    # Print sorted data
    for (i = 1; i <= n; i++) {
        gpu_name = gpu_names[i]
        
        # Enhanced color coding
        if (gpu_name ~ /A100/) color = GREEN
        else if (gpu_name ~ /H100/) color = BOLD GREEN
        else if (gpu_name ~ /A40/) color = CYAN
        else if (gpu_name ~ /V100/) color = MAGENTA
        else if (gpu_name ~ /RTX/) color = BLUE
        else if (gpu_name ~ /GTX/) color = YELLOW
        else color = RESET
        
        printf color "%-*s %-*s %-10s %-12s %-10s %-5s %-9s %-12s\n" RESET,
            max_name, names[i], max_gpu, gpu_name, drivers[i], capabilities[i], 
            memories[i], gpus_list[i], total_gpus_list[i], total_slot_gpus_list[i]
    }
    
    # Print summary
    printf "\n" BOLD "Total nodes: %d\n" RESET, n
}' | less -R