"""Test probe
"""
import torch
import json
import os
import pickle
import re
import numpy as np
import pandas as pd
from datetime import datetime
from torchvision.transforms import Lambda
from collections import OrderedDict

from datasets import DecodeDataset
from models import MDND
from trainers import Trainer, Tester

from torch.optim import Adam
from aihwkit.optim import AnalogOptimizer
from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.inference import RRAMLikeNoiseModel
from aihwkit.simulator.configs.utils import (
    MappingParameter, WeightClipParameter, WeightClipType,
    WeightModifierParameter, WeightNoiseType, WeightModifierType
)


DATA_PATHS = [
    'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p0015_Nt1M_rnnData_aT1651078684.txt',
    'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p0035_Nt1M_rnnData_aT1651078734.txt',
    'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p005_Nt1M_rnnData_aT1651078773.txt',
    'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p006_Nt1M_rnnData_aT1651078797.txt',
    'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p0065_Nt1M_rnnData_aT1651078809.txt',
    'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p007_Nt1M_rnnData_aT1651078820.txt',
    'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p0085_Nt1M_rnnData_aT1651078854.txt',
    'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'
]

LOAD_PATH = 'research/saves/hhwa-mdnd/hhwa_mdnd_model_d3_p01_nU16_pdrop0.100-2022-07-19 14:37:58.964200.pth'

# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16

# hwa training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 10

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
analog_model.load_state_dict(torch.load(LOAD_PATH), load_rpu_config=False)
print("")
