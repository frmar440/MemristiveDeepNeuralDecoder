"""Inference script
"""
import torch
import json
import pickle
from datetime import datetime
from torchvision.transforms import Lambda

from datasets import DecodeDataset
from models import AnalogRNN, RNN
from trainers import Tester

from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference import RRAMLikeNoiseModel
from aihwkit.simulator.configs.utils import MappingParameter
from aihwkit.nn.conversion import convert_to_analog

DATA_PATH = 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'
LOAD_PATH = 'research/saves/fp_trained_arnn_model_d3_p01_nU16_nR3-2022-07-01 20:44:52.048965.pth'

USE_HWA_TRAINING = False

# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16
NUM_LAYERS = 1

BATCH_SIZE = 512

# load test datasets
with open(DATA_PATH, 'rb') as f:
    dico = pickle.loads(f.read())


test_decode_data = DecodeDataset(
    dico=dico,
    train=False,
    transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
    target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
)

# resistive processing unit
mapping = MappingParameter(
    digital_bias=False, # bias term is handled by the analog tile (crossbar)
    max_input_size=512,
    max_output_size=512
)
rpu_config = InferenceRPUConfig(mapping=mapping)
rpu_config.forward.inp_res = 1/256. # 8-bit DAC discretization.
rpu_config.forward.out_res = 1/256. # 8-bit ADC discretization.
rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=66.0) # the model described

device = "cuda" if torch.cuda.is_available() else "cpu"

if USE_HWA_TRAINING:
    ID = f'hwa_trained_arnn_model_d3_p01_nU{HIDDEN_SIZE}_nR3'
    model = AnalogRNN(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        rpu_config=rpu_config,
        id=ID
    ).to(device)
    model.load_state_dict(torch.load(LOAD_PATH))
else:
    ID = f'fp_trained_arnn_model_d3_p01_nU{HIDDEN_SIZE}_nR3'
    model = RNN(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        id=ID
    ).to(device)
    model.load_state_dict(torch.load(LOAD_PATH))
    model = convert_to_analog(model, rpu_config)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

tester = Tester(
    test_data=test_decode_data,
    batch_size=BATCH_SIZE,
    loss_fn=loss_fn,
    batch_first=False
)
tester(model)

# t_inference = 0.
# for _ in range(10):
#     model.drift_analog_weights(t_inference) # update read noise
#     tester.test_loop()
