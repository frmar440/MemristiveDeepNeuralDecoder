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
    FP_MDND_LOAD_PATHS = os.listdir('research/saves/fp-mdnd')
    HWA_MDND_LOAD_PATHS = list(filter(lambda s: s[-3:] == 'pth', os.listdir('research/saves/hwa-mdnd')))
    HHWA_MDND_LOAD_PATHS = list(filter(lambda s: s[-3:] == 'pth', os.listdir('research/saves/hhwa-mdnd')))

    # model parameters
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    HIDDEN_SIZE = 16

    # inference parameters
    BATCH_SIZE = 32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()


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
    test_rpu_config.modifier = WeightModifierParameter(pdrop=0.0, # defective device probability
                                                       enable_during_test=True)

    # dataframe init
    columns = pd.MultiIndex.from_product([["baseline", "fp-mdnd", "hwa-mdnd", "hhwa-mdnd"],
                                          ["mean", "std"]])
    df = pd.DataFrame(index=[0.15, 0.35, 0.5, 0.6, 0.65, 0.7, 0.85, 1.0], columns=columns, dtype='float64')

    # training scheme iteration (curves)
    for training_scheme in ["baseline", "fp-mdnd", "hwa-mdnd", "hhwa-mdnd"]:

        data = []
        # test pdrop iteration (x-axis)
        for DATA_PATH in DATA_PATHS:
            
            # load test dataset
            with open(DATA_PATH, 'rb') as f:
                dico = pickle.loads(f.read())

            test_decode_data = DecodeDataset(
                dico=dico,
                train=False,
                transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
                target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
            )

            # regex
            pfr = re.search('p[0-9]*', DATA_PATH).group(0)
            pdrop = 'pdrop0.100'
            pfr_pdrop = f'{pfr}.*{pdrop}'

            re_pfr = re.compile(pfr)
            re_pfr_pdrop = re.compile(pfr_pdrop)

            if training_scheme == "baseline":
                DND_LOAD_PATH = list(filter(re_pfr.search, DND_LOAD_PATHS))[0]

                model = DND(
                    input_size=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    output_size=OUTPUT_SIZE,
                ).to(device)
                # load weights (but use the current RPU config)
                model.load_state_dict(torch.load(f'research/saves/dnd/{DND_LOAD_PATH}'))

            else:
                if training_scheme == "fp-mdnd":
                    MDND_LOAD_PATH = list(filter(re_pfr.search, FP_MDND_LOAD_PATHS))[0]
                    test_rpu_config.modifier.pdrop = 0.1
                elif training_scheme == "hwa-mdnd":
                    MDND_LOAD_PATH = list(filter(re_pfr_pdrop.search, HWA_MDND_LOAD_PATHS))[0]
                    test_rpu_config.modifier.pdrop = 0.1
                elif training_scheme == "hhwa-mdnd":
                    MDND_LOAD_PATH = list(filter(re_pfr_pdrop.search, HHWA_MDND_LOAD_PATHS))[0]
                    # pdrop is integrated by probe during training
                    test_rpu_config.modifier.pdrop = 0.


                model = MDND(
                    input_size=INPUT_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    output_size=OUTPUT_SIZE,
                    rpu_config=test_rpu_config
                ).to(device)
                # load weights (but use the current RPU config)
                model.load_state_dict(torch.load(f'research/saves/{training_scheme}/{MDND_LOAD_PATH}'), load_rpu_config=False)

            # tester
            tester = Tester(
                test_data=test_decode_data,
                batch_size=BATCH_SIZE,
                loss_fn=loss_fn,
                test_rpu_config=test_rpu_config
            )

            # statistics iteration
            for _ in range(10):
                tester(model)
            
            data.append(tester.accuracies)
        
        df[training_scheme, "mean"] = np.mean(data, axis=1)
        df[training_scheme, "std"] = np.std(data, axis=1)

    # save data experiment
    df.to_pickle('research/experiments/results/decoder_performance.pkl')
