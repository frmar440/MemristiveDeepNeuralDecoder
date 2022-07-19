"""Convert script
"""
import torch
import json
import pickle
import re
import os
from datetime import datetime

from datasets import DecodeDataset
from models import DND, MDND
from trainers import Trainer

from torch.nn import Linear, Conv1d, Conv2d, Conv3d, Sequential, RNN
from torch.optim import Adam
from torchvision.transforms import Lambda

from aihwkit.nn import AnalogLinear, AnalogConv1d, AnalogConv2d, AnalogConv3d, AnalogRNN, AnalogSequential
from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs.utils import MappingParameter
from aihwkit.nn.conversion import convert_to_analog


CONVERSION_MAP = {Linear: AnalogLinear,
                  Conv1d: AnalogConv1d,
                  Conv2d: AnalogConv2d,
                  Conv3d: AnalogConv3d,
                  RNN: AnalogRNN,
                  Sequential: AnalogSequential,
                  DND: MDND}

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

DND_LOAD_PATHS = list(filter(lambda s: s[-3:] == 'pth', os.listdir('research/saves/dnd')))

# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16


for DATA_PATH in DATA_PATHS:

    # regex
    pfr = re.search('p[0-9]*', DATA_PATH).group(0)
    re_pfr = re.compile(pfr)

    DND_LOAD_PATH = list(filter(re_pfr.search, DND_LOAD_PATHS))[0]


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # deep neural decoder
    model = DND(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=HIDDEN_SIZE
    ).to(device)
    
    model.load_state_dict(torch.load(f'research/saves/dnd/{DND_LOAD_PATH}'))


    time = datetime.now()

    # resistive processing unit
    rpu_config = InferenceRPUConfig()
    rpu_config.mapping = MappingParameter(digital_bias=False) # bias term is handled by the analog tile (crossbar)
    # convert dnd to mdnd
    analog_model = convert_to_analog(model, rpu_config, conversion_map=CONVERSION_MAP)
    # save fp-mdnd
    torch.save(analog_model.state_dict(),
               f'research/saves/fp-mdnd/fp_mdnd_model_d3_{pfr}_nU{HIDDEN_SIZE}-{time}.pth')
