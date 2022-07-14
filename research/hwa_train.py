"""HWA training script
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
LEARNING_RATES = [1e-5, 1e-4, 1e-3, 1e-2]
BATCH_SIZE = 32
EPOCHS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"
# loss function
loss_fn = torch.nn.CrossEntropyLoss()

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

DATA_PATH = DATA_PATHS[-1]

# for std in np.linspace(0.0, 0.05, 11): # iterate for different std

for i, learning_rate in enumerate(LEARNING_RATES):
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
        rpu_config=rpu_config
    ).to(device)
    # load weights (but use the current RPU config)
    analog_model.load_state_dict(torch.load(f'research/saves/fp-mdnd/{MDND_LOAD_PATH}'), load_rpu_config=False)


    # analog optimizer
    optimizer = AnalogOptimizer(Adam, analog_model.parameters(), lr=learning_rate)
    optimizer.regroup_param_groups(analog_model)
    # hwa training
    trainer = Trainer(
        training_data=training_decode_data,
        test_data=test_decode_data,
        learning_rate=learning_rate,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        loss_fn=loss_fn,
        optimizer=optimizer
    )
    trainer(analog_model)

    if i == 0:
        # dataframe init
        df = pd.DataFrame(index=trainer.train_batches, columns=LEARNING_RATES, dtype='float64')
    
    df[learning_rate] = trainer.train_losses

df.to_pickle('research/experiments/results/training/hwa_lr_losses.pkl')

# time = datetime.now()
# # save ha-mdnd
# torch.save(analog_model.state_dict(),
#             f'research/saves/hwa-mdnd/hwa_trained_mdnd_model_d3_{pfr}_nU{HIDDEN_SIZE}_std{std:.3f}-{time}.pth')
# # save hwa training parameters
# with open(f'research/saves/hwa-mdnd/hwa_trained_mdnd_model_d3_{pfr}_nU{HIDDEN_SIZE}_std{std:.3f}-{time}.json', 'w') as file:
#     file.write(json.dumps(trainer.training_state_dict()))



# rpu_config.modifier.std_dev = 0.005 # expected value

# for pdrop in np.linspace(0.0, 0.3, 7): # iterate for different pdrop
    
#     # regex
#     pfr = re.search('p[0-9]*', DATA_PATH).group(0)
#     re_pfr = re.compile(pfr)

#     MDND_LOAD_PATH = list(filter(re_pfr.search, MDND_LOAD_PATHS))[0]

#     # load training and test datasets
#     with open(DATA_PATH, 'rb') as f:
#         dico = pickle.loads(f.read())

#     training_decode_data = DecodeDataset(
#         dico=dico,
#         train=True,
#         transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
#         target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
#     )

#     test_decode_data = DecodeDataset(
#         dico=dico,
#         train=False,
#         transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
#         target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
#     )


#     rpu_config.modifier.pdrop = pdrop

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # memristive deep neural decoder
#     analog_model = MDND(
#         input_size=INPUT_SIZE,
#         hidden_size=HIDDEN_SIZE,
#         output_size=OUTPUT_SIZE,
#         rpu_config=rpu_config
#     ).to(device)
#     # load weights (but use the current RPU config)
#     analog_model.load_state_dict(torch.load(f'research/saves/fp-mdnd/{MDND_LOAD_PATH}'), load_rpu_config=False)


#     # analog optimizer
#     optimizer = AnalogOptimizer(Adam, analog_model.parameters(), lr=LEARNING_RATE)
#     optimizer.regroup_param_groups(analog_model)
#     # hwa training
#     trainer = Trainer(
#         training_data=training_decode_data,
#         test_data=test_decode_data,
#         learning_rate=LEARNING_RATE,
#         batch_size=BATCH_SIZE,
#         epochs=EPOCHS,
#         loss_fn=loss_fn,
#         optimizer=optimizer
#     )
#     # hwa training
#     trainer(analog_model)


#     time = datetime.now()
#     # save ha-mdnd
#     torch.save(analog_model.state_dict(),
#                f'research/saves/hwa-mdnd/hwa_trained_mdnd_model_d3_{pfr}_nU{HIDDEN_SIZE}_pdrop{pdrop:.3f}-{time}.pth')
#     # save hwa training parameters
#     with open(f'research/saves/hwa-mdnd/hwa_trained_mdnd_model_d3_{pfr}_nU{HIDDEN_SIZE}_pdrop{pdrop:.3f}-{time}.json', 'w') as file:
#         file.write(json.dumps(trainer.training_state_dict()))
