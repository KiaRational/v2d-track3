# t3: object poses + robot policies of the Track 3 submission

Object poses (`tpf_recon`), per episode, our meshes only (`meshes_recon`), SAM 3 masks (`objects_recon/masks`):
1. `fph_arr.sbatch` -- `stereo/fp_hyps.py`: every FoundationPose candidate (~252) per frame, 60-frame chunks
   (task list: one line `run object start end` per chunk).
2. `te_arr.sbatch` -- `stereo/te_obj_tracks.py`: TrackEverything 3D tracks on object surfaces (SLAM world).
3. `tp_arr.sbatch` -- `stereo/track_pose.py` (Viterbi over candidates: hand-aware mask IoU x score, tracked points on
   the surface, pairwise agreement with tracked motion; long-range track anchoring) -> `tp_recon`;
   `stereo/pose_filter.py` (symmetry unwrap, speed-limit flip/reject Viterbi, smoothing) -> `tpf_recon`.
4. Robot: `robot/cmds/submit_c3.sh` with `TAG=t3 OV=tpf_recon HV=hands_c RECIPES=task2:tracking_250m_task2`
   (convert -> physics assets -> walls-only collision -> FlashSAC -> official evaluator).
5. `pack_t3.sbatch`: pack + preflight (rollouts with all worlds valid; meshes = the simulator's visual meshes).

## t4-t7 (2026-10-30): fixes found by reviewing the t3 robot rollouts
- `stereo/rest_hold.py`, `stereo/spin_guard.py` (c19): no rotation without a grasp; held objects rotate with the hand.
- `stereo/twist_unwrap.py` (`twist.sbatch`): exact minimal twist of rotationally symmetric objects (cups) about their axis
  (the snapped twist was noise the simulator reward made the robot chase) -> `tpfsu_recon`.
- `stereo/hand_depen.py` (`depen.sbatch`): reference hands pushed out of objects (cups as solids) -> `stereo/hands_dp`.
- `stereo/rest_table.py` (`rest_table.sbatch`): untouched objects rest exactly on the table -> `tpfsut_recon`.
- `stereo/extend_mesh.py` (`pitcher_stretch.sbatch`): water pitcher mesh lengthened at the bottom (pose fit the body,
  the mesh was 36 mm short, so the pitcher floated).
- `robot/carve_collision.py` (`t6_carve.sbatch`): dish-rack collision pieces removed where another object really sat.
- `robot/cmds/train_v2d_reward_hmd.sh`: SEED env (second training seed).
