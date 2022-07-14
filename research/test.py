"""Testing script
"""
import torch
import json
import os
import pickle
import re
import numpy as np
from datetime import datetime
from torchvision.transforms import Lambda

from datasets import DecodeDataset
from models import MDND
from trainers import Trainer

from torch.optim import Adam
from aihwkit.optim import AnalogOptimizer
from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.inference import RRAMLikeNoiseModel
from aihwkit.simulator.configs.utils import (
    MappingParameter, WeightClipParameter, WeightClipType,
    WeightModifierParameter, WeightNoiseType, WeightModifierType
)

MDND_LOAD_PATH = 'research/saves/fp-mdnd/fp_trained_mdnd_model_d3_p01_nU16_nR3-2022-07-11 12:25:29.338821.pth'

# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16

# resistive processing unit (for hwa training)
rpu_config = InferenceRPUConfig()
rpu_config.drift_compensation = None
rpu_config.mapping = MappingParameter(digital_bias=False, # bias term is handled by the analog tile (crossbar)
                                      max_input_size=512,
                                      max_output_size=512)
# training
rpu_config.clip = WeightClipParameter(sigma=2.5, type=WeightClipType.LAYER_GAUSSIAN) # weight clipping
# training and inference
rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
rpu_config.forward.out_res = 1/256.  # 8-bit ADC discretization.
rpu_config.forward.out_noise = 0.
rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=60.0, prog_noise_scale=1.) # rram noise
rpu_config.modifier = WeightModifierParameter(pdrop=0.1, # defective device probability
                                              enable_during_test=True,
                                              std_dev=0.005, # training noise
                                              type=WeightModifierType.REL_NORMAL)

device = "cuda" if torch.cuda.is_available() else "cpu"
# memristive deep neural decoder
analog_model = MDND(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    rpu_config=rpu_config
).to(device)
# load weights (but use the current RPU config)
analog_model.load_state_dict(torch.load(MDND_LOAD_PATH))

analog_model.load_rpu_config(rpu_config)

# conductances = analog_model.get_conductances()
print("")