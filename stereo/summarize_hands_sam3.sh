#!/bin/bash
# Per-episode summary of the SAM 3 hand stage (jq only, safe on a login node).
# Usage: summarize_hands_sam3.sh <run> [...]
R=${V2D_ROOT:-/scratch/kghasemz/v2d}/runs
printf "%-22s %7s %7s %7s %7s %7s %7s %6s %6s %6s\n" run covA_L covA_R covC_L covC_R covB_L covB_R far nodist reasso
for n in "$@"; do
  f=$R/$n/stereo/hands_sam3/summary.json
  [ -f "$f" ] || { printf "%-22s (no summary)\n" "$n"; continue; }
  jq -r --arg n "$n" '[$n,
    (.coverage_frac.a.left//0), (.coverage_frac.a.right//0),
    (.coverage_frac.c.left//0), (.coverage_frac.c.right//0),
    (.coverage_frac.b.left//0), (.coverage_frac.b.right//0),
    .rejected_far, .rejected_no_distance, .n_reassociation]
    | @tsv' "$f" | awk -F'\t' '{printf "%-22s %7s %7s %7s %7s %7s %7s %6s %6s %6s\n",$1,$2,$3,$4,$5,$6,$7,$8,$9,$10}'
done
echo
printf "%-22s %-6s %8s %8s %8s %8s %8s\n" run side frames mpjpe_tri jitter reproj tri_jit
for n in "$@"; do
  for tag in base q sam3; do
    f=$R/$n/stereo/mano_fit_$tag/report.json
    [ -f "$f" ] || continue
    jq -r --arg n "$n" --arg tag "$tag" '.sides | to_entries[] |
      [$n+"("+$tag+")", .key, (.value.mano_fit.frames//0),
       (.value.mano_fit.mpjpe_tri_mm_mean//0), (.value.mano_fit.jitter_all_frames_mm_per_frame2//0),
       (.value.mano_fit.reproj_wilor2d_px_mean//0),
       (.value.triangulated_X_smooth.jitter_on_tri_frames_mm_per_frame2//0)] | @tsv' "$f" \
      | awk -F'\t' '{printf "%-22s %-6s %8s %8.2f %8.2f %8.2f %8.2f\n",$1,$2,$3,$4,$5,$6,$7}'
  done
done
