# Reconstructed-mesh pose recovery and hand-aware refinement

`stereo/evidence_fusion.py` repairs large pose errors before continuous joint
optimization. It reads only reconstructed meshes, predicted hands, estimated
camera poses, object masks and stereo depth. It does not load public object
meshes, mocap poses, or evaluation ground truth. The existing source variants
and mesh files are preserved.

Track 3 requires participants' reconstructed meshes, including when a matching
public mesh exists. Mesh vertices and output object poses must use the same body
frame and metre units. See the [official FAQ](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_challenge_2026_faqs.pdf)
and the local submission kit's `tracks/track_3/README.md`. These outputs are
reconstruction references; a complete challenge submission still requires
learned-policy rollouts. This stage does not submit anything.

## Method

1. Every fifth frame, extract masked stereo-depth points, excluding other objects
   and hands and eroding segmentation boundaries. Score the source pose, raw FP
   alternatives and 24 principal-axis orientation hypotheses. Refine the best
   four with bounded-iteration ICP. Accept a replacement only if it improves
   geometric fit by at least 1 mm and 12%, while staying within 12 cm of the
   original centroid. These rotations are hypotheses, not assumed symmetries:
   a handle or brush bristles must be explained by the observed geometry.
2. Infer carry intervals from the existing carry report plus hand proximity and
   airborne/co-motion evidence. Missing hand observations split intervals.
   Two hands can constrain the same object. No hand observation alone is not
   evidence that the object is stationary.
3. Estimate a robust object-in-palm transform from independent geometric
   observations within each interval. Use medoid initialization and Cauchy
   reweighting to resist half-turn outliers. Optimize the world-frame SE(3)
   trajectory with visual measurements, hand-relative motion and acceleration
   penalties. Supported, observed slow-motion intervals get a rest constraint.
4. Optionally run `s06j_joint_hoi.py --evidence_variant ...` for simultaneous
   bounded hand/object adjustment, hand-object contact and table penetration.
   This uses the recovered observations rather than pulling toward the original
   flipped FP observations. It requires complete evidence, matching timing and
   mesh hashes, and a passing geometry gate. The joint solver's grasp seed is
   now robust to mixed rotation branches as well.

All rotation changes are made about the mesh centroid and converted back to its
original body frame before writing camera-frame `R`/`T` files. The source FPS is
used for temporal penalties. Long missing camera trajectories are rejected.

## Vulcan execution

Submit from scratch; never run these Python programs on a login node. The
launchers load the verified Apptainer module and use the existing `agile.sif`.
The CPU recovery stage requests one CPU, 12 GB and one hour. Joint refinement
requests one `l40s`, four CPUs, 32 GB and one hour. All output/cache I/O is on
scratch. Account and modules were verified on Vulcan during implementation.

```bash
cd /scratch/kghasemz/v2d
sbatch ~/v2d-vulcan/stereo/run_evidence_fusion.sbatch --test
sbatch ~/v2d-vulcan/stereo/run_evidence_fusion.sbatch \
  evaluation_ep000003 evaluation_ep000019 evaluation_ep000030
```

Recovery writes `runs/<run>/fused_evidence_recon/<object>/poses/` and
`evidence_report.json`. It refuses to overwrite an existing output directory.
Use `--out <fresh_variant>` for another experiment. An incomplete run has
`complete: false`; downstream joint refinement refuses it. Reports include
per-frame geometry evidence, actual recovered observation transforms, mesh and
source hashes, solver diagnostics and before/after motion metrics.

After inspecting the geometry gates:

```bash
sbatch ~/v2d-vulcan/stereo/run_evidence_fusion.sbatch --preview \
  evaluation_ep000003 evaluation_ep000019 evaluation_ep000030
sbatch ~/v2d-vulcan/stereo/run_evidence_joint.sbatch \
  evaluation_ep000003 evaluation_ep000019 evaluation_ep000030
sbatch ~/v2d-vulcan/stereo/run_evidence_fusion.sbatch --preview \
  --variant fused_evidence_joint \
  evaluation_ep000003 evaluation_ep000019 evaluation_ep000030
```

The joint output pair is `fused_evidence_joint` / `stereo/hands_evidence_joint`.
Previews and `validation.json` go to `viz/<variant>/<run>/`. Contact sheets show
the original image, source-mesh overlay and refined-mesh overlay. Validation
reports depth-fit changes, angular steps, centroid acceleration and table
penetration. Review the joint hand quality report too before exporting a new
robot reference. Existing policy inputs are not automatically switched.

## Validation and limits

The CPU regression suite checks non-origin-centred meshes, minority half-turn
outliers, geometric recovery of a reversed asymmetric shape, preservation of
real hand rotation, suppression of false relative spin, unsupported motion,
and missing-hand interval boundaries. The existing joint suite covers contact,
handovers, wrist correction, symmetry, round-table geometry and output I/O.

A geometry gate is an observed-depth consistency check, not a ground-truth
accuracy score. Occlusion, a wrong reconstructed shape, an inaccurate table or
camera trajectory, and incorrect hands can still make the pose ambiguous.
Single rigid-grasp intervals do not model arbitrary in-hand manipulation; splits
and soft losses limit but do not eliminate that issue. The joint stage does not
provide a full rigid-body dynamics or actuator-feasibility guarantee. Report
failed checks and inspect close-ups instead of calling a smoother trajectory
physically correct. Regenerate downstream hand references and policy training
when choosing a new accepted object variant.
