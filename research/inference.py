"""Inference script
"""
import torch
import numpy as np
import json
import pickle
from datetime import datetime
from torchvision.transforms import Lambda

from datasets import DecodeDataset
from models import DND, MDND
from trainers import Tester

from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.inference import RRAMLikeNoiseModel
from aihwkit.simulator.configs.utils import (
    MappingParameter, WeightClipParameter, WeightClipType,
    WeightModifierParameter, WeightNoiseType, WeightModifierType
)

DATA_PATH = 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'
LOAD_PATH = 'research/saves/mdnd/fp_trained_mdnd_model_d3_p01_nU16_nR3-2022-07-05 18:40:45.209994.pth'

# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16

# inference parameters
BATCH_SIZE = 1024


# load test dataset
with open(DATA_PATH, 'rb') as f:
    dico = pickle.loads(f.read())

test_decode_data = DecodeDataset(
    dico=dico,
    train=False,
    transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
    target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
)


# resistive processing unit
rpu_config = InferenceRPUConfig()
rpu_config.drift_compensation = None
rpu_config.mapping = MappingParameter(digital_bias=False, # bias term is handled by the analog tile (crossbar)
                                      max_input_size=512,
                                      max_output_size=512)
rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
rpu_config.forward.out_res = 1/256.  # 8-bit ADC discretization.
rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
rpu_config.forward.w_noise = 0.0

# training
rpu_config.clip = WeightClipParameter(sigma=2.5, type=WeightClipType.LAYER_GAUSSIAN)
# inference
rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=66.0, prog_noise_scale=1.) # rram noise
# training and inference
rpu_config.modifier = WeightModifierParameter(pdrop=0.1) # defective device probability
# rpu_config.modifier.type = WeightModifierType.ADD_NORMAL
# rpu_config.modifier.std_dev = 0.0


device = "cuda" if torch.cuda.is_available() else "cpu"
# loss function
loss_fn = torch.nn.CrossEntropyLoss()
# tester
tester = Tester(
    test_data=test_decode_data,
    batch_size=BATCH_SIZE,
    loss_fn=loss_fn
)

# for w_noise in np.linspace(50., 100., 10):
#     rpu_config.forward.w_noise = w_noise
# memristive deep neural decoder
analog_model = MDND(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    rpu_config=rpu_config
).to(device)
# load weights (but use the current RPU config)
analog_model.load_state_dict(torch.load(LOAD_PATH), load_rpu_config=False)
# inference
tester(analog_model)
