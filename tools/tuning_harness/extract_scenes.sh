#!/usr/bin/env bash
# Extract the tuning + held-out scene segments: first 200 frames (20 s @ 10 fps),
# keep SVS_FV, SVS_RV, AD_FL, AD_RR only (disk budget ~9 GB/scene).
set -u
ROOT=~/bins/13-10-2026_Limerick_Tunnel_night
OUT=~/isp_tuning/scenes
N=200
mkdir -p "$OUT"
cd ~/raw-dataset-converter
while read -r name rec; do
  [ -z "$name" ] && continue
  [ -f "$OUT/$name/.done" ] && continue
  if [ ! -d "$OUT/$name/SVS_FV/raw" ] || [ "$(ls "$OUT/$name/SVS_FV/raw" | wc -l)" -lt "$N" ]; then
    rm -rf "$OUT/$name"
    python3 -m car_converter --recording-dir "$ROOT/$rec" --output "$OUT/$name" --sensors svs,ad \
      --stitch-ad-raw --max-frames "$N" > "$OUT/$name.log" 2>&1 || { echo "FAIL $name"; continue; }
  fi
  rm -rf "$OUT/$name/SVS_MVL" "$OUT/$name/SVS_MVR" "$OUT/$name/AD_FR" "$OUT/$name/AD_RL"
  for cam in "$OUT/$name"/*/raw; do
    ls "$cam" | sort | tail -n +$((N + 1)) | sed "s|^|$cam/|" | xargs -r rm -f
  done
  echo "$rec" > "$OUT/$name/.source"
  touch "$OUT/$name/.done"
  echo "done $name"
done <<'EOF'
T1_urban_day       6ACE64DC/NUIG_20261013_181553_009
T2_canopy_static   6ACE62A9/NUIG_20261013_175936_003
T3_bridge_sky      6ACE64DC/NUIG_20261013_181422_006
T4_motorway_dusk   6ACE6AF3/NUIG_20261013_183849_003
T5_twilight_rural  6ACE6CE5/NUIG_20261013_185443_000
T6_night_lights    6ACE7227/NUIG_20261013_190544_002
H1_urban_day       6ACE64DC/NUIG_20261013_181723_012
H2_night_lights    6ACE7084/NUIG_20261013_190215_001
H3_night_dark      6ACE72F8/NUIG_20261013_190708_001
EOF
echo ALLDONE
