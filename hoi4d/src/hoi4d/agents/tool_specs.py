"""JSON tool schemas for the review agent (README stage 8 tool interface + record_finding/finish)."""

INTERVAL = {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2,
            "description": "[t_start_s, t_end_s] in seconds"}


def _fn(name, desc, props, req):
    return {"type": "function", "function": {"name": name, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


TOOL_SPECS = [
    _fn("get_task_hypotheses", "Semantic interpretation (VLM task.json): phases, contact hypotheses, uncertainty.", {}, []),
    _fn("get_trajectory", "Poses of one entity in a window (subsampled to <= 40 rows): position (m, world z-up), "
        "rotation (axis-angle deg relative to the window's first frame), validity, provenance source, uncertainty.",
        {"entity_id": {"type": "string"}, "interval": INTERVAL}, ["entity_id", "interval"]),
    _fn("inspect_frames", "Show original colour frames of a window (<= 4 frames) with the current reconstruction "
        "overlaid (object mesh outlines, hand skeletons), labelled with entity IDs.",
        {"interval": INTERVAL, "views": {"type": "array", "items": {"type": "string", "enum": ["colour", "overlay", "stereo_left"]}},
         "max_frames": {"type": "integer", "minimum": 1, "maximum": 4}}, ["interval"]),
    _fn("compute_motion_metrics", "Step sizes, constant-velocity prediction residuals (jump evidence), speed, "
        "acceleration, jerk (symmetry-aware rotation) for an entity in a window, plus CoTracker point-track "
        "agreement (rigid fit residual of the object's tracked surface points vs its pose).",
        {"entity_id": {"type": "string"}, "interval": INTERVAL}, ["entity_id", "interval"]),
    _fn("check_contacts", "Hand-object / object-object / object-support distances, penetration depth, slip of the "
        "hand relative to a grasped object, and support hypotheses in a window.",
        {"interval": INTERVAL, "entity_id": {"type": "string"}}, ["interval"]),
    _fn("check_physics", "Quasi-static feasibility in a window over plausible parameter ranges: can the observed "
        "contacts (friction cones) balance gravity + inertia of the object? Reports feasible fraction per setting.",
        {"interval": INTERVAL, "entity_id": {"type": "string"},
         "parameter_ranges": {"type": "object", "properties": {"mu": {"type": "array", "items": {"type": "number"}},
                                                               "mass_kg": {"type": "array", "items": {"type": "number"}}}}},
        ["interval", "entity_id"]),
    _fn("rerun_stage", "Queue a bounded re-estimation in a window (applied after this round, validated, rolled back "
        "if worse). stage: object_fusion (overrides: reicp | no_fp | fp_low | hold) or hand_fit (overrides: "
        "squeeze_mm 1.5|3|5, contact_mm 15|30).",
        {"stage": {"type": "string", "enum": ["object_fusion", "hand_fit"]}, "entity_id": {"type": "string"},
         "interval": INTERVAL, "overrides": {"type": "object"}, "reason": {"type": "string"}},
        ["stage", "entity_id", "interval", "overrides", "reason"]),
    _fn("propose_refinement", "Queue an evidence-linked constraint proposal for the physics/contact optimizer in a "
        "window: contact pairs to activate/deactivate (grasp/support/release), objects that must be static, "
        "symmetric-rotation ambiguity to keep. Never pose values.",
        {"interval": INTERVAL, "constraints": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["contact_on", "contact_off", "static", "support", "keep_symmetric"]},
            "entity_a": {"type": "string"}, "entity_b": {"type": "string"}}, "required": ["kind", "entity_a"]}},
         "evidence": {"type": "string"}}, ["interval", "constraints", "evidence"]),
    _fn("compare_versions", "Compare two trajectory versions: visual (depth/mask) fit, contacts, penetration, "
        "physics feasibility, jitter, uncertainty, globally and in an optional window.",
        {"before": {"type": "string"}, "after": {"type": "string"}, "interval": INTERVAL}, ["before", "after"]),
    _fn("record_finding", "Record the conclusion for one anomaly (required for every anomaly you examine).",
        {"anomaly_id": {"type": "string"}, "entities": {"type": "array", "items": {"type": "string"}},
         "interval": INTERVAL, "status": {"type": "string", "enum": ["confirmed_violation", "probable_tracking_error",
                                                                      "semantic_mismatch", "insufficient_evidence", "dismissed"]},
         "measurements": {"type": "string"}, "frames": {"type": "array", "items": {"type": "integer"}},
         "likely_cause": {"type": "string"}, "alternatives": {"type": "string"},
         "proposed_repair": {"type": "string"}, "validation_criteria": {"type": "string"}},
        ["anomaly_id", "entities", "interval", "status", "likely_cause", "alternatives", "proposed_repair",
         "validation_criteria"]),
    _fn("finish", "End this review round.", {"summary": {"type": "string"}}, ["summary"]),
]
