"""URDF for an object with no provided Track 3 scan (wooden_spoon, mesh from SAM 3D: meshes_obj/<name>/<name>.obj),
written like the dataset's data/track_3/public/urdf/<name>_rigid.urdf: 0.3 kg placeholder mass (what every
Track 3 object, and the policy simulator's generate_rigid_urdfs, uses) and COM/inertia as a uniform
triangle-surface estimate over the mesh. Goes to $V2D_ROOT/urdf_extra (the dataset folder stays untouched);
wrench_feasibility / physics_check / mj_track fall back to it.
Usage: python make_extra_urdf.py wooden_spoon [--mass 0.3]
"""
import argparse
import os
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(os.environ.get("V2D_ROOT", "/scratch/kghasemz/v2d"))


def shell_inertia(m, mass, n=200000, seed=0):
    """COM and inertia about the COM of a uniform-density surface shell (area-weighted samples)."""
    p, _ = trimesh.sample.sample_surface_even(m, n, seed=seed) if hasattr(trimesh.sample, "sample_surface_even") \
        else trimesh.sample.sample_surface(m, n)
    com = p.mean(0)
    q = p - com
    I = mass / len(q) * (np.eye(3) * (q ** 2).sum() - q.T @ q)
    return com, I


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--mass", type=float, default=0.3)
    a = ap.parse_args()
    src = ROOT / "meshes_obj" / a.name / f"{a.name}.obj"
    m = trimesh.load(src, force="mesh")
    com, I = shell_inertia(m, a.mass)
    out = ROOT / "urdf_extra"
    out.mkdir(parents=True, exist_ok=True)
    x = f"""<?xml version="1.0"?>
<robot name="track3_{a.name}">
  <!-- Ours, not from the dataset: mesh {src} (SAM 3D Objects, metric from stereo). {a.mass} kg placeholder mass
       like every Track 3 object; COM/inertia are a uniform triangle-surface estimate over the mesh. -->
  <mujoco><compiler balanceinertia="true" discardvisual="false"/></mujoco>
  <link name="object">
    <inertial>
      <origin xyz="{com[0]:.9g} {com[1]:.9g} {com[2]:.9g}" rpy="0 0 0"/>
      <mass value="{a.mass}"/>
      <inertia ixx="{I[0, 0]:.9g}" ixy="{I[0, 1]:.9g}" ixz="{I[0, 2]:.9g}" iyy="{I[1, 1]:.9g}" iyz="{I[1, 2]:.9g}" izz="{I[2, 2]:.9g}"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{src}" scale="1 1 1"/></geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><mesh filename="{src}" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
"""
    (out / f"{a.name}_rigid.urdf").write_text(x)
    print(x)
    print(f"extent cm {np.round((m.bounds[1] - m.bounds[0]) * 100, 1)}, COM mm {np.round(com * 1000, 2)}")


if __name__ == "__main__":
    main()
