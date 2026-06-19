#!/bin/bash
#SBATCH --job-name=netcheck
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --output=netcheck.%J.out
#SBATCH --error=netcheck.%J.err
# Add the right CPU partition for Prajna, e.g.:
#SBATCH --partition=debug
#SBATCH --qos=debug

# Checks whether compute nodes can reach the NIM API.
echo "Node: $(hostname)"
curl -s -o /dev/null -w "HTTP %{http_code} in %{time_total}s\n" \
  --max-time 20 https://integrate.api.nvidia.com/v1/models \
  -H "Authorization: Bearer ${NVIDIA_API_KEY}" \
  || echo "FAILED: compute node likely has no outbound internet. Ask hpc@iitb.ac.in about a proxy (then set https_proxy in job scripts)."
