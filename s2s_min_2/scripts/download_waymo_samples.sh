#!/usr/bin/env bash
# Pull a small number of Waymo Open Dataset v2.0.1 segments (LiDAR-only components).
# Picks first N alphabetically from each split — change SEG_LIMIT_TRAIN / SEG_LIMIT_VAL to taste.
#
# Requires gsutil. Activate the project env first:
#   source /home/satya/anaconda3/etc/profile.d/conda.sh
#   conda activate /home/satya/conda_envs/selfocc
set -euo pipefail

BUCKET="gs://waymo_open_dataset_v_2_0_1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE="${WAYMO_BASE:-$REPO_ROOT/s2s_min/data/waymo}"
# LiDAR-only: enough to train the LiDAR encoder. Drop camera_* and lidar_box (detection labels)
# and lidar_camera_projection (only useful when pairing LiDAR with images).
COMPONENTS=(
  lidar lidar_calibration lidar_pose
  vehicle_pose stats
)

SEG_LIMIT_TRAIN=${SEG_LIMIT_TRAIN:-20}
SEG_LIMIT_VAL=${SEG_LIMIT_VAL:-5}

# Existing segment(s) we should skip (kept for back-compat with the older sample dir).
EXISTING=""

download_split () {
  local split="$1" limit="$2"
  echo ""
  echo "============================================================"
  echo "  Split: $split  (target: $limit segments)"
  echo "============================================================"

  # Get segment IDs from the camera_image component (canonical list).
  mapfile -t all_segs < <(gsutil ls "$BUCKET/$split/camera_image/" | grep -oE '[0-9]+_[0-9]+_[0-9]+_[0-9]+_[0-9]+\.parquet$' | sort -u)
  echo "  Available: ${#all_segs[@]} segments"

  # Filter out segment we already have (for training) and take first N.
  local picked=()
  for s in "${all_segs[@]}"; do
    if [[ -n "$EXISTING" && "$split" == "training" && "$s" == "$EXISTING" ]]; then continue; fi
    picked+=("$s")
    [[ ${#picked[@]} -ge $limit ]] && break
  done
  echo "  Picking: ${#picked[@]} segments"
  printf '    %s\n' "${picked[@]}"

  # Ensure directories
  for comp in "${COMPONENTS[@]}"; do
    mkdir -p "$BASE/$split/$comp"
  done

  # Download each component for each picked segment (parallel per-call).
  for seg in "${picked[@]}"; do
    echo ""
    echo "  >>> $seg"
    for comp in "${COMPONENTS[@]}"; do
      local dst="$BASE/$split/$comp/$seg"
      if [[ -f "$dst" ]]; then
        echo "    [skip] $comp (already present)"
        continue
      fi
      gsutil cp "$BUCKET/$split/$comp/$seg" "$dst" 2>&1 | tail -1 | sed "s/^/    [$comp] /" &
    done
    wait
  done
}

download_split training "$SEG_LIMIT_TRAIN"
download_split validation "$SEG_LIMIT_VAL"

echo ""
echo "============================================================"
echo "  Done. Final layout:"
echo "============================================================"
du -sh "$BASE"/*/ 2>/dev/null || du -sh "$BASE"
echo ""
find "$BASE" -name '*.parquet' | awk -F/ '{print $(NF-2)"/"$(NF-1)}' | sort | uniq -c
