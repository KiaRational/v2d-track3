# Physical zero-residual replay of a converted sequence (hands follow the retargeted reference through their PD
# gains, contact on), rendered headless to MP4 -- a cheap scene check before training. Env (APPTAINERENV_*):
#   SEQ (sequence id), TAG (output name), VOC (object assist scale, default 0 = no assist: true physics),
#   TABLE (1 default: our table prim replaces the infinite ground plane), NF (frames), EXTRA (Hydra overrides).
# Output: /scratch/kghasemz/v2d/viz/16_sim_table/<SEQ>_<TAG>.mp4
P=/scratch/kghasemz/v2d/robot/hmd/ego_recon/processed/sequence_id=${SEQ}/robot_name=sharpa_wave
U=/scratch/kghasemz/v2d/robot/hmd/ego_recon/reconstructed_stage/${SEQ}_support.usda
O=/scratch/kghasemz/v2d/viz/16_sim_table; mkdir -p $O
SUP=false; [ -s $U ] && grep -qE "Cylinder|Cube" $U && SUP=true
[ "${TABLE:-1}" = 0 ] && SUP=false  # ground plane + our table overlap
TBL=""; [ "$SUP" = true ] && [ "${TABLE:-1}" = 1 ] && grep -q '"table"' $U && TBL="collision.ground=false"
echo "support=$SUP overrides: ${TBL:-none}"
export PYOPENGL_PLATFORM=egl PYGLET_HEADLESS=true
python scripts/debug/replay.py "task.parquet='${P}'" scene.support=$SUP ${TBL} replay.voc_scale=${VOC:-0.0} \
  viewer.backend=mp4 viewer.headless=true viewer.realtime=false viewer.output_path=$O/${SEQ}_${TAG:-replay}.mp4 \
  viewer.num_frames=${NF:-400} viewer.video_fps=30 viewer.width=1280 viewer.height=720 markers.enabled=false \
  "viewer.camera_position=${CAM:-[0.55,-0.75,0.45]}" "viewer.camera_target=${TGT:-[0.0,-0.05,0.0]}" ${EXTRA} \
  2>&1 | grep -v "^Module\|nefc" | tail -25
ls -la $O/${SEQ}_${TAG:-replay}.mp4
