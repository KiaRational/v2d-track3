# Motion refinement changes

The default path now uses cubic B-spline priors alongside observation, contact,
penetration and acceleration terms. Smoothing is a soft constraint: the final
trajectory remains a fitted hand/object motion, rather than an unconstrained
spline replacement.

## What changed

- `stereo/motion_quality.py` provides time-aware acceleration, cubic spline
  projection, bounded-gap interpolation, sticking-contact windows and centroid
  symmetry transforms.
- `fuse_objects.py` enables a weak cubic pose prior (`--spline_weight 0.2`,
  `--spline_knot_seconds 0.15`). Contact changes split spline spans. Rotations use
  a local SO(3) chart and skip spline rotation fitting near its branch cut.
  Both hands may contribute contact. A static-motion prior now requires observed
  slow motion and support-plane proximity; a missing hand alone is insufficient.
- `s06h_grasp_plausible.py` defaults to `--smoothing spline`. It regularizes the
  complete world joint trajectories while optimizing MANO parameters, preserving
  hand kinematics. Source frame spacing determines acceleration. Contact changes
  and long gaps split the spline spans. Corrections are bounded to 25 mm wrist
  translation, 20 degrees wrist rotation, and 20 degrees per finger axis-angle
  correction. These are configurable trust limits, not measured actuator limits.
- Contact anchors are restricted to observed, near-surface intervals with limited
  slip and ramped at their boundaries. They do not persist for an entire grasp
  after a finger releases. Penetration and table losses remain active.
- Candidate selection writes `mano_fit_selected` (recording each grasp's chosen
  `squeeze_mm`/`contact_mm`); the launchers then run the final refinement
  `s06h --ref_hands mano_fit --hands mano_fit_selected --segment_params mano_fit_selected`
  to produce `mano_fit_phys`. It starts from the selected hands, but the data term,
  trust limits and anchors are measured against the original stereo fit, and each
  grasp keeps its selected squeeze. Final outputs recompute joints from MANO.
- The final report checks finite geometry and a robust penetration statistic: per
  frame maximum penetration, excluding frames whose object pose is suspect (initial
  penetration > 25 mm), with the table checked only on contact frames or near an
  object's footprint; it passes when the 95th percentile is <= 5 mm and <= 5% of
  frames exceed 10 mm. Export refuses mismatched object variants. A failed or
  missing quality check is a loud warning plus manifest `quality.passed = false`
  unless `--strict` (`V2D_EXPORT_STRICT=1` in `robot/run_convert.sbatch`) is given.
  This is a geometric check, not proof of robot grasp stability.
- Export writes source `fps` and `timestamps` to `result.npz` and `manifest.json`.
  The upstream loader ignores `hand_*_is_valid`, so once a side has any observation
  export writes no zero frames: interior gaps of any length are filled in the world
  frame (wrist-joint lerp, wrist and finger slerp, then back to each frame's camera
  frame) and leading/trailing gaps hold the first/last observed world pose. Filled
  frames are `is_valid` True and `hand_*_is_interpolated` True; the manifest counts
  `n_interpolated_short` (<= `--max_gap_s`, 0.15 s), `n_interpolated_long`,
  `n_held_lead` and `n_held_trail` per side. `--no_fill_long` fills only short gaps
  (the rest stay zeros, `is_valid` False). A side with no observations stays zeros,
  `is_valid` False, with a loud warning and `no_observations` in the manifest.
- Invalid camera/object frames fail export rather than becoming identity poses.
- Grasp selection requires both wrench feasibility and a mean hold displacement
  below the configured threshold for its `held` flag. The wrench model also includes
  the angular momentum term. Actuator and force limits are still not modeled.

## Use

These source changes do not regenerate existing reconstructions or policies.
Run the modified stages through Slurm, following the existing job launchers:

1. Regenerate object fusion with `V2D_OBJ_DIR=objects_sam3` so that the object
   variant includes the updated contact/static logic and spline prior.
2. Run `stereo/run_grasp_full_array.sbatch` for the desired episode list.
3. Run `robot/run_convert.sbatch` for the resulting reference.

The default shared object variant is `fused_all_sam3`. To use a refined variant,
set `V2D_OBJECT_VARIANT` to the same value for both the full grasp job and the
conversion job. Regenerate the hand refinement when object poses change; a
matching directory name alone does not establish that old outputs are current.

For ablations, use `--spline_weight 0` in object fusion and
`--smoothing acceleration` in hand refinement. Larger spline knot spacing
smooths more strongly; check short contact transitions before increasing it.

The toolkit loader at
`$V2D_ROOT/fc_src/reconstruction/modules/v2d_task_library_loader/lib/ego_recon_loader.py`
is upstream (commit aee95f25) plus one minimal patch, kept in
`patches/ego_recon_loader.py`: `get_fps()` returns the bundle's fps (`result.npz`
`fps`, else `manifest.json` `fps`) and otherwise the original 30.0. The challenge
source videos are 20 fps; the hard-coded 30 played parquets 1.5x fast. Nothing else
differs (no timing requirement, gap handling or imports of this repository).
`git -C $V2D_ROOT/fc_src diff` must show only that change. No container rebuild is
needed for the existing launcher, which mounts the source checkout.

## Validation and limits

`tests/run_motion_tests.sbatch` runs synthetic CPU tests in Slurm. It covers
timestamp validation, world-frame hand gap filling under a moving camera, the
loader's bundle fps, the final-pass trust limits, the robust quality statistic,
rotation wraparound, bimanual contacts, centroid symmetry, jitter reduction,
spline phase/gap boundaries, export gating, and an optimization integration
test using analytic geometry and a synthetic differentiable hand model.

No existing run data or checkpoints are needed by these tests. Actual visual
quality and robot execution still require regenerated data and evaluation in the
target simulator. This change does not implement unknown-object reconstruction,
multi-object export, actuator-constrained retargeting, or full coupled camera,
hand and object optimization.
