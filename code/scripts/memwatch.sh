#!/usr/bin/env bash
# Diagnostic memory/GPU watcher for the STORY_01_02 Step 13 pod-kill investigation.
#
# Logs one line every 2 s with: timestamp, the POD's cgroup memory usage vs its
# limit (the number that actually matters in a container — `free` shows the whole
# host node, not the pod's cap), host free RAM, and GPU memory used.
#
# Usage (log to the PERSISTENT /home volume so the trace survives a pod kill):
#   bash scripts/memwatch.sh > /home/maia-user/Andre/logs/memwatch.log 2>&1 &
#   echo "memwatch PID: $!"
# Stop it with: kill <that PID>   (or it dies automatically if the pod is killed)
set -u

# cgroup v2 first, fall back to v1.
cg_cur=/sys/fs/cgroup/memory.current
cg_max=/sys/fs/cgroup/memory.max
if [ ! -f "$cg_cur" ]; then
  cg_cur=/sys/fs/cgroup/memory/memory.usage_in_bytes
  cg_max=/sys/fs/cgroup/memory/memory.limit_in_bytes
fi

limit_raw=$(cat "$cg_max" 2>/dev/null || echo "unknown")
limit_mb=$(awk -v b="$limit_raw" 'BEGIN{ if (b+0>0) printf "%d", b/1048576; else print b }')
echo "cgroup_current_file=$cg_cur"
echo "pod_cgroup_limit=${limit_raw} (~${limit_mb} MB)"
echo "----"

while true; do
  ts=$(date +%T)
  cur=$(cat "$cg_cur" 2>/dev/null || echo 0)
  cur_mb=$(awk -v b="$cur" 'BEGIN{ printf "%d", b/1048576 }')
  host_used=$(free -m | awk '/Mem:/{print $3}')
  host_avail=$(free -m | awk '/Mem:/{print $7}')
  gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "$ts  pod_used_MB=$cur_mb  host_used_MB=$host_used  host_avail_MB=$host_avail  gpu_used_MB=${gpu:-NA}"
  sleep 2
done
