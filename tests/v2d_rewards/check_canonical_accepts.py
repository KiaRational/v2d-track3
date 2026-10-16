"""Run with the UNMODIFIED FlashCHORD on PYTHONPATH: the stripped v2d checkpoint config must instantiate
(env config incl. objective, curriculum) and its stage weights must match the canonical term names."""
import json
import sys

import flash_chord
from omegaconf import OmegaConf

from flash_chord.configuration import instantiate_typed
from flash_chord.envs.rl import RLEnvConfig
from flash_chord.lifecycle.curriculum import FixedCurriculum
from flash_chord.objectives.config import OBJECTIVE_TERM_NAMES

print("flash_chord from", flash_chord.__file__)
cfg = OmegaConf.create(json.load(open(sys.argv[1])))
env = instantiate_typed(cfg.env, RLEnvConfig)
curriculum = instantiate_typed(cfg.training.curriculum, FixedCurriculum)
for stage in curriculum.stages:
    stage.weights_for(OBJECTIVE_TERM_NAMES)
print("OK: canonical code instantiates the stripped config; objective weights", env.objective.weights)
