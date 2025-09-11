## Cluster tools

Directory for useful commands and examples.

---
- show_nodes.sh - outputs a table of what’s available in the cluster, sorted by CUDA capability and number of GPUs
    
- bash.sub - template submission file to start an interactive bash shell at a node
    - use with `condor_submit -interactive bash.sub`
    - can use, for example, to test a node's properties, for to see how multiple GPUs on one node are set up, run `nvidia-smi topo -m`