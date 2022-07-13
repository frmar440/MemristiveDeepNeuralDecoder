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

# resistive processing unit
rpu_config = InferenceRPUConfig()
rpu_config.mapping = MappingParameter(digital_bias=False) # bias term is handled by the analog tile (crossbar)

device = "cuda" if torch.cuda.is_available() else "cpu"
# memristive deep neural decoder
analog_model = MDND(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    rpu_config=rpu_config
).to(device)
# load weights (but use the current RPU config)
analog_model.load_state_dict(torch.load(MDND_LOAD_PATH), load_rpu_config=False)

conductances = analog_model.get_conductances()
print("")