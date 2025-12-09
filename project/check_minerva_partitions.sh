#!/bin/bash
# Quick script to check available partitions on Minerva

echo "=== Available Partitions on Minerva ==="
sinfo

echo ""
echo "=== Partitions with GPU support ==="
sinfo -o "%P %G %l" | grep -i gpu

echo ""
echo "=== To see detailed info about a specific partition ==="
echo "sinfo -p <partition_name>"

