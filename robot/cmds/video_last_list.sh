# MP4 of the LAST policy (unassisted rollout) for every "<run name> <motion-data dir>" line of $LIST; copies to
# viz/robot/<name>.mp4. Env: LIST (file). Run via run_fc.sbatch.
export PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true
mkdir -p /scratch/kghasemz/v2d/viz/robot
while read -r NAME HMDD; do
  O=/scratch/kghasemz/v2d/robot/fc_out/${NAME}; mkdir -p $O/video
  LAST=$(ls $O/policy_*.safetensors 2>/dev/null | sort -V | tail -1); [ -z "$LAST" ] && { echo "$NAME: no policy"; continue; }
  s=$(basename $LAST .safetensors | cut -d_ -f2)
  export HUMAN_MOTION_DATA_DIR=$HMDD
  python scripts/view_policy.py evaluation.checkpoint=$LAST viewer.backend=mp4 viewer.headless=true viewer.realtime=false \
    viewer.output_path=$O/video/policy_$s.mp4 viewer.num_frames=600 viewer.video_fps=30 viewer.width=1280 viewer.height=720 \
    markers.enabled=false "viewer.camera_position=[0.55,-0.75,0.45]" "viewer.camera_target=[0.0,-0.12,0.05]" 2>&1 | grep -E "Video|Error" | tail -2
  cp $O/video/policy_$s.mp4 /scratch/kghasemz/v2d/viz/robot/${NAME}.mp4 && echo "VIDEO $NAME"
done < $LIST
