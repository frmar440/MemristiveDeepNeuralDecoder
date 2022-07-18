"""Inference (after hwa) pdrop experiment
"""
import torch
import numpy as np
import pandas as pd
import re
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

def hwa_inference_pdrop_run():
    PDROPS = np.linspace(0.0, 0.1, 11)

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
    MDND_LOAD_PATH = 'research/saves/hwa-mdnd/hwa_trained_mdnd_model_d3_p01_nU16_pdrop0.100-2022-07-16 13:46:45.439212.pth'

    DATA_PATH = DATA_PATHS[-1]

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
    test_rpu_config = InferenceRPUConfig()
    test_rpu_config.mapping = MappingParameter(digital_bias=False) # bias term is handled by the analog tile (crossbar)
    test_rpu_config.drift_compensation = None
    # non-idealities
    test_rpu_config.forward.inp_res = 1/256.  # infinite steps.
    test_rpu_config.forward.out_res = 1/256.  # infinite steps.
    test_rpu_config.forward.out_noise = 0.
    test_rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=60.0, prog_noise_scale=1.) # rram noise
    test_rpu_config.modifier = WeightModifierParameter(pdrop=0.0, # defective device probability
                                                    enable_during_test=True)

    # dataframe init
    columns = pd.MultiIndex.from_product([["p01"],
                                          ["mean", "std"]])
    df = pd.DataFrame(index=PDROPS, columns=columns, dtype='float64')


    # regex
    pfr = re.search('p[0-9]*', DATA_PATH).group(0)

    # load test dataset
    with open(DATA_PATH, 'rb') as f:
        dico = pickle.loads(f.read())

    test_decode_data = DecodeDataset(
        dico=dico,
        train=False,
        transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
        target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
    )

    data = []
    # prog noise scale iteration
    for pdrop in PDROPS:

        test_rpu_config.modifier.pdrop = pdrop

        # tester
        tester = Tester(
            test_data=test_decode_data,
            batch_size=BATCH_SIZE,
            loss_fn=loss_fn,
            test_rpu_config=test_rpu_config
        )

        model = MDND(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE,
            rpu_config=test_rpu_config
        ).to(device)
        # load weights (but use the current RPU config)
        model.load_state_dict(torch.load(f'research/saves/fp-mdnd/{MDND_LOAD_PATH}'))
        
        # statistics iteration
        for _ in range(10):
            tester(model, inference=True)
        
        data.append(tester.accuracies)
    
    df[pfr, "mean"] = np.mean(data, axis=1)
    df[pfr, "std"] = np.std(data, axis=1)

    # save data experiment
    df.to_pickle('research/experiments/results/hwa_inference_pdrop.pkl')