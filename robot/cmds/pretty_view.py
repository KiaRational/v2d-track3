"""Report-quality rendering wrapper around FlashCHORD's scripts/view_policy.py (rendering only - no physics change).
1. The viewer paints every support shape with one support_color, so our table and the floor box below it came out the
   same colour; here the table (first support shape) keeps support_color and the rest (floor) get PRETTY_FLOOR.
2. Camera aimed at the action: PRETTY_TARGET / PRETTY_POS are computed by the caller from the episode's objects.
Usage (cwd = flash_chord source): PRETTY_FLOOR="0.86,0.85,0.82" python pretty_view.py <view_policy overrides...>"""
import os
import runpy
import sys

import warp as wp

import flash_chord.visualization.viewer as V

_orig = V.configure_render_style
FLOOR = tuple(float(x) for x in os.environ.get("PRETTY_FLOOR", "0.86,0.85,0.82").split(","))


def styled(viewer, config, scene=None):
    _orig(viewer, config, scene)
    if scene is None or getattr(scene, "support", None) is None or scene.model.shape_color is None:
        return
    per_world = scene.collision_layout.shape_count
    s0, s1 = scene.support.shapes.start, scene.support.shapes.stop
    for w in range(scene.world_count):
        a, b = w * per_world + s0 + 1, w * per_world + s1   # shape 0 of the support = the table cylinder
        if b > a:
            scene.model.shape_color[a:b].fill_(wp.vec3(*FLOOR))


V.configure_render_style = styled
sys.argv = ["scripts/view_policy.py"] + sys.argv[1:]
runpy.run_path("scripts/view_policy.py", run_name="__main__")
