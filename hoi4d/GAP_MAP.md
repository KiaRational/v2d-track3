# Hand-Object-4D README -> existing pipeline (gap map, 2026-10-06)

Spec: /scratch/kghasemz/v2d/Hand-Object-4D-README.md. Existing outputs stay the "initial" reconstruction;
only gaps are built. Standing user rulings: hand corrections temporally smooth (knot-based), detected hand
frames kept close to observations (GeneOH / per-frame s06f/s06g rejected), no mocap inside the pipeline
(public mocap = external validation only), every new hand variant shown on video before it becomes default.

| README stage | Existing | Status | Gap to build |
|---|---|---|---|
| 1 Ingest/validate | stages/s00-s02, s01 calib (calib/ego_calib_resolved.json), stereo/table_plane.py | done | scene_manifest.json (units, gravity, transforms, quaternion wxyz, entity registry) |
| 2 VLM task | - | missing | semantics: frame sampling + VLM -> task.json (phases, evidence, alternatives) |
| 3 SAM 3 | stages/s05_objects_v2.py (+ s06d hands) | done | entity registry, cross-view association flags |
| 4 Stereo depth | s03_stereo_depth (FoundationStereo) -> runs/<r>/depth_stereo | done | validity/uncertainty map (LR consistency) |
| 5 Init geometry/hands | meshes_obj, collision parts, FoundationPose, HOPformer hand branch (hop_fit20 -> mano_fit_hop) | partial | HOPformer object hypotheses vs FoundationPose comparison |
| 6 Track 4D | object_icp, fuse_objects/se3_fusion, symmetry.py, s06b/s06c | partial | CoTracker correspondences (object track support + hand) |
| 7 Diagnostics | refine_agent/verify.py, audit_contact, wrench_feasibility, physics_check, motion_quality | partial | unified anomalies.json in README schema |
| 8 GPT-Astra review | refine_agent/agent.py (rule-based) | missing | tool-calling agent, 9 tools, budgets, review_log.jsonl |
| 9 Physics/contact opt | s06h/s06i/wrench (hands), se3_fusion (objects) | partial | staged objective (geometric -> contact selection -> physical), contacts.json |
| 10 Validate/loop | refine_agent accept/rollback | partial | compare_versions, acceptance, rollback, termination |
| 11 Export | robot/export_result_bundle.py | partial | trajectories_initial/refined.npz, contacts.json, metrics.json, run manifest |

First milestone clip: public ep11 (blue-cup stacking; mocap for external validation).
