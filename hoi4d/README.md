# hoi4d: physics-aware 4D hand-object reconstruction with an agentic review loop

Implementation of `/scratch/kghasemz/v2d/Hand-Object-4D-README.md` on top of the existing V2D stereo pipeline
(the existing outputs are the *initial* reconstruction; see `GAP_MAP.md` for what was reused vs built).

## Run

```bash
sbatch run_pipeline.sbatch <run>                         # one episode, all stages (3 x L40S, ~25 min)
STAGES="diagnose refine" sbatch run_pipeline.sbatch <run> # a subset
sbatch --array=0-39%6 run_pipeline_array.sbatch /scratch/kghasemz/v2d/hoi4d_runs.txt   # all public + eval
```
GPUs 0,1 serve Qwen3-VL-32B-Instruct-FP8 with vLLM (tensor parallel 2): the VLM (stage 2) and the reviewer
(stage 8, the "GPT-Astra" role; `reviewer.provider: openai` switches to the OpenAI API with a user-exported
`OPENAI_API_KEY`). GPU 2 runs everything else in the `agile` image.

## Stages -> code

| README stage | Code | Output (runs/<r>/hoi4d/) |
|---|---|---|
| 1 ingest, world frame, conventions | `ingest/manifest.py`, `conventions.py` | `scene_manifest.json` |
| 2 VLM task understanding | `semantics/task.py`, `llm.py`, `visualization/overlay.py` | `task.json`, `semantics_log.jsonl` |
| 3-5 segmentation, depth, poses, hands | existing pipeline (SAM 3, FoundationStereo, FoundationPose+ICP fusion, stereo MANO + HOPformer prior) via adapter | `trajectories_initial.npz` |
| 5.6 collision assets (cavity-aware) | `geometry/assets.py` (CoACD convex parts) | - |
| 6.2 temporal correspondence | `tracking/cotracker.py` (CoTracker3, depth-lifted, symmetry-aware rigidity) | `tracks.npz` |
| 7 deterministic diagnostics | `diagnostics/{motion,contacts,run}.py` + imported `refine_agent/verify.py`, `wrench_feasibility.py` | `anomalies_<v>.json`, `contacts_<v>.json`, `diag_<v>.npz` |
| 8 agent review (11 tools, budgets, log) | `agents/{review,tools,tool_specs}.py` | `review_log.jsonl`, `review_summary.json` |
| 9 physics + contact optimisation (staged, 9 terms) | `physics/optimize.py`, `physics/wrench.py` | `trajectories_<v>.npz`, `physics_report_<v>.json` |
| 10 validate, accept / roll back | `agents/tools.py: compare, accept, accept_entities`; MuJoCo hold test; 4D videos | `versions.json` (every version kept) |
| 11 export | `export/final.py`, `export/versions.py` | `trajectories_refined.npz`, `contacts.json`, `anomalies.json`, `metrics.json`, `run_manifest.json`, `report.md` |

## Decisions (where the README leaves choices open)
- World frame: z up = table normal, origin at the table-disk centre on the top surface (`A @ DROID-SLAM`).
- All corrections are cubic B-spline knots every 4 frames (no per-frame corrections: earlier per-frame hand
  fixes and GeneOH were rejected for jitter / twisted fingers).
- Acceptance is per entity: a repair is kept only for the objects / hands whose own checks improve, then the
  combination is re-validated. Hands must also stay within 1 cm (p95) of their observed joints and add no
  fingertip jitter (p95 relative acceleration +5 % max).
- Rotationally symmetric objects: CoTracker rigidity uses radius + height along the axis (spin is unobservable).
- Mocap (public episodes) is never used inside the pipeline; only for external validation.
- Licences: CoTracker3 weights CC-BY-NC-4.0; Qwen3-VL Apache-2.0.

## Tests
`PYTHONPATH=src python -m pytest tests` (agile image): transforms, world alignment, timestamped derivatives,
symmetry-aware rotation distance, jump residual vs fast motion, NaN-not-identity for missing poses, cavity collision.
