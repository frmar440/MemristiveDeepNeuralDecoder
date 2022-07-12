"""Performance experiment
"""
import torch
import numpy as np
import pandas as pd
import re
import json
import os
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

def decoder_performance_run():
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
    MDND_LOAD_PATHS = os.listdir('research/saves/fp-mdnd')

    # model parameters
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    HIDDEN_SIZE = 16

    # inference parameters
    BATCH_SIZE = 32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # resistive processing unit
    rpu_config = InferenceRPUConfig()
    rpu_config.mapping = MappingParameter(digital_bias=False) # bias term is handled by the analog tile (crossbar)
    rpu_config.drift_compensation = None
    # non-idealities
    rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
    rpu_config.forward.out_res = 1/256.  # 8-bit ADC discretization.
    rpu_config.forward.out_noise = 0.
    rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=60.0, prog_noise_scale=1.) # rram noise
    rpu_config.modifier = WeightModifierParameter(pdrop=0.1, # defective device probability
                                                  enable_during_test=True)

    # dataframe init
    columns = pd.MultiIndex.from_product([["baseline", "fp-mdnd"],
                                          ["mean", "std"]])
    df = pd.DataFrame(index=[0.15, 0.35, 0.5, 0.6, 0.65, 0.7, 0.85, 1.0], columns=columns, dtype='float64')

    # models iteration
    for model_label in ["baseline", "fp-mdnd"]:

        data = []
        # iterate through physical fault rate
        for DATA_PATH in DATA_PATHS:
            
            # regex
            pfr = re.search('p[0-9]*', DATA_PATH).group(0)
            re_pfr = re.compile(pfr)
            # load test dataset
            with open(DATA_PATH, 'rb') as f:
                dico = pickle.loads(f.read())

            test_decode_data = DecodeDataset(
                dico=dico,
                train=False,
                transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
                target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
            )

            # testers
            tester= Tester(
                test_data=test_decode_data,
                batch_size=BATCH_SIZE,
                loss_fn=loss_fn
            )

            if model_label == "baseline":
                model = DND(
                    input_size=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    output_size=OUTPUT_SIZE
                ).to(device)
                # load weights
                DND_LOAD_PATH = 'research/saves/dnd/' + list(filter(re_pfr.search, DND_LOAD_PATHS))[0]
                model.load_state_dict(torch.load(DND_LOAD_PATH))

            elif model_label == "fp-mdnd":
                model = MDND(
                    input_size=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    output_size=OUTPUT_SIZE,
                    rpu_config=rpu_config
                ).to(device)
                # load weights (but use the current RPU config)
                MDND_LOAD_PATH = 'research/saves/fp-mdnd/' + list(filter(re_pfr.search, MDND_LOAD_PATHS))[0]
                model.load_state_dict(torch.load(MDND_LOAD_PATH), load_rpu_config=False)

            # statistics iteration
            for _ in range(10):
                tester(model, inference=True)

            data.append(tester.accuracies)
            
        df[model_label, "mean"] = np.mean(data, axis=1)
        df[model_label, "std"] = np.std(data, axis=1)

    df.to_pickle('research/experiments/results/decoder_performance.pkl')