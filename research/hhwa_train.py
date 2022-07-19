"""HHWA training script
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
MDND_LOAD_PATHS = os.listdir('research/saves/fp-mdnd')

# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16

# hwa training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
EPOCHS = 10


def get_probes(pdrop):
    probe_rnn = torch.ones((HIDDEN_SIZE, INPUT_SIZE+HIDDEN_SIZE+1))
    probe_linear = torch.ones((OUTPUT_SIZE, HIDDEN_SIZE+1))

    # probe rnn layer
    for i in range(HIDDEN_SIZE):
        for j in range(INPUT_SIZE+HIDDEN_SIZE+1):
            if np.random.uniform() < pdrop:
                probe_rnn[i, j] = 0
    
    # probe linear layer
    for i in range(OUTPUT_SIZE):
        for j in range(HIDDEN_SIZE+1):
            if np.random.uniform() < pdrop:
                probe_linear[i, j] = 0
    
    return OrderedDict({"rnn": probe_rnn, "linear": probe_linear})


device = "cuda" if torch.cuda.is_available() else "cpu"
# loss function
loss_fn = torch.nn.CrossEntropyLoss()


# resistive processing unit (TRAINING)
training_rpu_config = InferenceRPUConfig()
training_rpu_config.drift_compensation = None
training_rpu_config.mapping = MappingParameter(digital_bias=False, # bias term is handled by the analog tile (crossbar)
                                               max_input_size=512,
                                               max_output_size=512)
training_rpu_config.clip = WeightClipParameter(sigma=2.5, type=WeightClipType.LAYER_GAUSSIAN) # weight clipping
training_rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
training_rpu_config.forward.out_res = 1/256.  # 8-bit ADC discretization.
training_rpu_config.forward.out_noise = 0.
training_rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=60.0, prog_noise_scale=1.) # rram noise (no weights programmation in training mode)
training_rpu_config.modifier = WeightModifierParameter(enable_during_test=True,
                                                       std_dev=0.005, # training noise
                                                       type=WeightModifierType.REL_NORMAL)


# resistive processing unit (TEST)
test_rpu_config = InferenceRPUConfig()
test_rpu_config.drift_compensation = None
test_rpu_config.mapping = MappingParameter(digital_bias=False, # bias term is handled by the analog tile (crossbar)
                                           max_input_size=512,
                                           max_output_size=512)
test_rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
test_rpu_config.forward.out_res = 1/256.  # 8-bit ADC discretization.
test_rpu_config.forward.out_noise = 0.
test_rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=60.0, prog_noise_scale=1.) # rram noise (weights programmation in test mode)
test_rpu_config.modifier = WeightModifierParameter(enable_during_test=True)

##########
# TRAIN FOR DIFFERENT PDROP @ PFR = 0.01

DATA_PATH = DATA_PATHS[-1]

for pdrop in np.linspace(0.0, 0.1, 6): # iterate for different pdrop

    # probes with given defective device probability
    probes = get_probes(pdrop)

    # regex
    pfr = re.search('p[0-9]*', DATA_PATH).group(0)
    re_pfr = re.compile(pfr)

    MDND_LOAD_PATH = list(filter(re_pfr.search, MDND_LOAD_PATHS))[0]

    # load training and test datasets
    with open(DATA_PATH, 'rb') as f:
        dico = pickle.loads(f.read())

    training_decode_data = DecodeDataset(
        dico=dico,
        train=True,
        transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
        target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
    )

    test_decode_data = DecodeDataset(
        dico=dico,
        train=False,
        transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
        target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # memristive deep neural decoder
    analog_model = MDND(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        rpu_config=training_rpu_config
    ).to(device)
    # load weights (but use the current RPU config)
    analog_model.load_state_dict(torch.load(f'research/saves/fp-mdnd/{MDND_LOAD_PATH}'), load_rpu_config=False)
    # load probes
    analog_model.load_probes(probes)

    # analog optimizer
    optimizer = AnalogOptimizer(Adam, analog_model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups(analog_model)


    # hwa training
    trainer = Trainer(
        training_data=training_decode_data,
        test_data=test_decode_data,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_rpu_config=training_rpu_config,
        test_rpu_config=test_rpu_config,
        max_accuracy=True
    )
    # hwa-training
    trainer(analog_model, n_step=4)


    time = datetime.now()
    # save ha-mdnd
    torch.save(trainer.max_state_dict,
               f'research/saves/hhwa-mdnd/hhwa_mdnd_model_d3_{pfr}_nU{HIDDEN_SIZE}_pdrop{pdrop:.3f}-{time}.pth')
    # save hwa training parameters
    with open(f'research/saves/hhwa-mdnd/hhwa_mdnd_model_d3_{pfr}_nU{HIDDEN_SIZE}_pdrop{pdrop:.3f}-{time}.json', 'w') as file:
        file.write(json.dumps(trainer.training_state_dict()))

##########
# TRAIN FOR DIFFERENT PFR @ PDROP = 0.1

pdrop = 0.1

for DATA_PATH in DATA_PATHS[:-1]: # iterate for different pfr

    # probes with given defective device probability
    probes = get_probes(pdrop)

    # regex
    pfr = re.search('p[0-9]*', DATA_PATH).group(0)
    re_pfr = re.compile(pfr)

    MDND_LOAD_PATH = list(filter(re_pfr.search, MDND_LOAD_PATHS))[0]

    # load training and test datasets
    with open(DATA_PATH, 'rb') as f:
        dico = pickle.loads(f.read())

    training_decode_data = DecodeDataset(
        dico=dico,
        train=True,
        transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
        target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
    )

    test_decode_data = DecodeDataset(
        dico=dico,
        train=False,
        transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
        target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
    )


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # memristive deep neural decoder
    analog_model = MDND(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        output_size=OUTPUT_SIZE,
        rpu_config=training_rpu_config
    ).to(device)
    # load weights (but use the current RPU config)
    analog_model.load_state_dict(torch.load(f'research/saves/fp-mdnd/{MDND_LOAD_PATH}'), load_rpu_config=False)
    # load probes
    analog_model.load_probes(probes)

    # analog optimizer
    optimizer = AnalogOptimizer(Adam, analog_model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups(analog_model)


    # hwa training
    trainer = Trainer(
        training_data=training_decode_data,
        test_data=test_decode_data,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        loss_fn=loss_fn,
        optimizer=optimizer,
        training_rpu_config=training_rpu_config,
        test_rpu_config=test_rpu_config,
        max_accuracy=True
    )
    # hwa-training
    trainer(analog_model, n_step=4)


    time = datetime.now()
    # save ha-mdnd
    torch.save(trainer.max_state_dict,
               f'research/saves/hhwa-mdnd/hhwa_mdnd_model_d3_{pfr}_nU{HIDDEN_SIZE}_pdrop{pdrop:.3f}-{time}.pth')
    # save hwa training parameters
    with open(f'research/saves/hhwa-mdnd/hhwa_mdnd_model_d3_{pfr}_nU{HIDDEN_SIZE}_pdrop{pdrop:.3f}-{time}.json', 'w') as file:
        file.write(json.dumps(trainer.training_state_dict()))
