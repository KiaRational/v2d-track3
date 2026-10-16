"""Conventions shared by every hoi4d artifact (README "Core data contracts": define them once, here).

Units           metres, seconds, radians (degrees only in human-readable reports, suffixed _deg).
Quaternion      (w, x, y, z), unit norm, w >= 0 not enforced (q and -q are the same rotation).
Transforms      T_a_b is a 4x4 homogeneous matrix mapping points expressed in frame b into frame a:
                p_a = T_a_b @ p_b (column vectors). T_world_obj = object pose in the world.
World frame     z up (opposite gravity), origin on the support surface (table top centre), x/y in the table
                plane; built from the existing pipeline's world + table_plane.json (see ingest.manifest).
Camera frame    OpenCV: x right, y down, z forward.
Time            t_s = frame_index / fps from the recording (episode.json); frame_index is the index of the
                colour/stereo frame in the stored videos.
Hands           MANO, right/left as separate entities ("hand_right", "hand_left"); global_orient (3, axis-angle),
                hand_pose (45, axis-angle, MANO joint order, no PCA), betas (10, constant per sequence),
                transl (3) in the world frame; joints (21, 3) in MANO/OpenPose order
                (wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4).
Missing values  NaN plus valid=False. Never an identity pose or zeros for a failed estimate.
Provenance      every per-frame value carries a source code (SOURCE_CODES) saying where it came from.
"""
from __future__ import annotations

QUAT_ORDER = "wxyz"
UNITS = {"length": "m", "time": "s", "angle": "rad"}
TRANSFORM_CONVENTION = "T_a_b maps points in frame b to frame a: p_a = T_a_b @ p_b"

# per-frame provenance of a pose value
SOURCE_CODES = {
    0: "missing",          # no estimate (value is NaN)
    1: "observed",         # direct estimator output, visible in the frame
    2: "fused",            # robust fusion of several observations (e.g. FoundationPose + stereo ICP)
    3: "interpolated",     # filled between observed frames (occlusion / not detected)
    4: "held_static",      # bridged as static (object resting, occluded)
    5: "refined",          # changed by an accepted refinement (geometric / contact / physics stage)
    6: "hand_follow",      # object carried by a hand while occluded
}
SOURCE_IDS = {v: k for k, v in SOURCE_CODES.items()}

MANO_JOINT_NAMES = ["wrist",
                    "thumb1", "thumb2", "thumb3", "thumb4",
                    "index1", "index2", "index3", "index4",
                    "middle1", "middle2", "middle3", "middle4",
                    "ring1", "ring2", "ring3", "ring4",
                    "pinky1", "pinky2", "pinky3", "pinky4"]

HAND_ENTITIES = ("hand_right", "hand_left")

# README stage 9: objective terms, in this order everywhere (configs, residual reports)
LOSS_TERMS = ("visual", "depth", "temporal", "anatomy", "contact", "collision", "dynamics", "task", "prior")

ANOMALY_TYPES = (
    "translation_jump", "rotation_jump", "high_acceleration", "high_jerk",
    "reprojection", "silhouette", "depth_disagreement", "cross_view_disagreement",
    "identity_switch", "tracking_loss", "hand_anatomy",
    "hand_object_penetration", "object_object_penetration", "object_support_penetration", "separation",
    "unsupported_object", "contact_slip", "unstable_rest", "wrench_infeasible",
    "semantic_mismatch",
)
ANOMALY_STATUS = ("candidate", "confirmed_violation", "probable_tracking_error", "semantic_mismatch",
                  "insufficient_evidence", "repaired", "unresolved", "dismissed")
CONTACT_MODES = ("grasp", "support", "sticking", "sliding", "release", "none")
