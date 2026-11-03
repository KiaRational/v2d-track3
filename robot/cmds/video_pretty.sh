# Report-quality MP4 of the LAST policy for every "<run> <motion-data dir>" line of $LIST -> viz/robot_pretty/<run>[_SUFFIX].mp4
# Camera aimed at the episode's objects (mean object position over the episode, from its sequence parquet); warm studio
# background, light floor, walnut table, shadows. Env: LIST, CAM (front34 | top | low), NFRAMES (600), SUFFIX.
# Run via run_fc.sbatch (cwd = flash_chord source).
export PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true
O2=/scratch/kghasemz/v2d/viz/robot_pretty; mkdir -p $O2
case ${CAM:-front34} in            # camera offset from the action centre (m)
  front34) OFF="0.42,-0.56,0.40" ;;
  top)     OFF="0.04,-0.22,0.55" ;;
  low)     OFF="0.32,-0.36,0.16" ;;
esac
LOOK="viewer.background_color=[0.95,0.94,0.92] viewer.support_color=[0.55,0.37,0.22] \
 viewer.ambient_sky=[0.60,0.60,0.62] viewer.ambient_ground=[0.35,0.33,0.30] viewer.draw_shadows=true viewer.spotlight_enabled=true \
 viewer.exposure=1.0"
while read -r NAME HMDD; do
  O=/scratch/kghasemz/v2d/robot/fc_out/${NAME}; mkdir -p $O/video
  LAST=$(ls $O/policy_*.safetensors 2>/dev/null | sort -V | tail -1); [ -z "$LAST" ] && { echo "$NAME: no policy"; continue; }
  SEQ=$(echo $NAME | sed -E 's/_task2.*//'); case $SEQ in t10_e20) SEQ=t7_e20;; t10_*) SEQ=t6_${SEQ#t10_};; esac
  PQ=$(ls $HMDD/ego_recon/processed/sequence_id=$SEQ/robot_name=sharpa_wave/*.parquet 2>/dev/null | head -1)
  read -r TGT POS <<< "$(python - "$PQ" "$OFF" <<'PY'
import sys, numpy as np, pyarrow.parquet as pq
off = np.array([float(x) for x in sys.argv[2].split(",")])
try:
    P = np.array(pq.read_table(sys.argv[1], columns=["object_body_position"]).column(0)[0].as_py(), float).reshape(-1, 3)
    c = np.r_[np.median(P[:, 0]), np.median(P[:, 1]), 0.10]
except Exception:
    c = np.array([0.0, 0.0, 0.07])
t, p = c, c + off
print(f"[{t[0]:.3f},{t[1]:.3f},{t[2]:.3f}] [{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]")
PY
)"
  export HUMAN_MOTION_DATA_DIR=$HMDD
  OUT=$O2/${NAME}${SUFFIX:+_$SUFFIX}.mp4
  python /home/kghasemz/v2d-vulcan/robot/cmds/pretty_view.py evaluation.checkpoint=$LAST viewer.backend=mp4 viewer.headless=true \
    viewer.realtime=false viewer.output_path=$OUT viewer.num_frames=${NFRAMES:-600} viewer.video_fps=30 viewer.width=1920 \
    viewer.height=1080 markers.enabled=false "viewer.camera_position=$POS" "viewer.camera_target=$TGT" $LOOK 2>&1 \
    | grep -E "Video|rror" | tail -3
  [ -f $OUT ] && echo "VIDEO $OUT target $TGT camera $POS"
done < $LIST
