"""Prog noise scale resolution experiment
"""
import torch
import numpy as np
import pandas as pd
import re
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

def prog_noise_scale_run():
    PROG_NOISE_SCALES = np.linspace(1, 24, 8)

    DATA_PATHS = [
        # 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p0015_Nt1M_rnnData_aT1651078684.txt',
        'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p0035_Nt1M_rnnData_aT1651078734.txt',
        # 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p005_Nt1M_rnnData_aT1651078773.txt',
        'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p006_Nt1M_rnnData_aT1651078797.txt',
        # 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p0065_Nt1M_rnnData_aT1651078809.txt',
        'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p007_Nt1M_rnnData_aT1651078820.txt',
        # 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p0085_Nt1M_rnnData_aT1651078854.txt',
        'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'
    ]
    MDND_LOAD_PATHS = [
        'research/saves/mdnd/fp_trained_mdnd_model_d3_p0035_nU16_nR3-2022-07-06 08:46:36.818726.pth',
        'research/saves/mdnd/fp_trained_mdnd_model_d3_p006_nU16_nR3-2022-07-06 09:14:58.713649.pth',
        'research/saves/mdnd/fp_trained_mdnd_model_d3_p007_nU16_nR3-2022-07-06 09:43:27.510154.pth',
        'research/saves/mdnd/fp_trained_mdnd_model_d3_p01_nU16_nR3-2022-07-06 10:12:12.403292.pth'
    ]

    # model parameters
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    HIDDEN_SIZE = 16

    # inference parameters
    BATCH_SIZE = 1024

    # resistive processing unit
    rpu_config = InferenceRPUConfig()
    rpu_config.drift_compensation = None
    rpu_config.mapping = MappingParameter(digital_bias=False, # bias term is handled by the analog tile (crossbar)
                                        max_input_size=512,
                                        max_output_size=512)
    rpu_config.forward.inp_res = -1.  # infinite steps.
    rpu_config.forward.out_res = -1.  # infinite steps.
    rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=66.0, prog_noise_scale=0.) # rram noise
    rpu_config.modifier = WeightModifierParameter(pdrop=0.0, # defective device probability
                                                  enable_during_test=True)

    # dataframe init
    columns = pd.MultiIndex.from_product([["p0035", "p006", "p007", "p01"], ["mean", "std"]])
    df = pd.DataFrame(index=PROG_NOISE_SCALES, columns=columns, dtype='float64')

    # physical fault rate iteration
    for DATA_PATH, MDND_LOAD_PATH in zip(DATA_PATHS, MDND_LOAD_PATHS):

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

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # loss function
        loss_fn = torch.nn.CrossEntropyLoss()
        # tester
        tester = Tester(
            test_data=test_decode_data,
            batch_size=BATCH_SIZE,
            loss_fn=loss_fn
        )

        data = []
        # prog noise scale iteration
        for prog_noise_scale in PROG_NOISE_SCALES:

            rpu_config.noise_model.prog_noise_scale = prog_noise_scale

            analog_model = MDND(
                input_size=INPUT_SIZE,
                hidden_size=HIDDEN_SIZE,
                output_size=OUTPUT_SIZE,
                rpu_config=rpu_config
            ).to(device)
            # load weights (but use the current RPU config)
            analog_model.load_state_dict(torch.load(MDND_LOAD_PATH), load_rpu_config=False)
            
            # statistics iteration
            for _ in range(10):
                tester(analog_model, inference=True)
            
            data.append(tester.accuracies)
            tester.reset_stats()
        
        df[pfr, "mean"] = np.mean(data, axis=1)
        df[pfr, "std"] = np.std(data, axis=1)

    # save data experiment
    df.to_pickle('research/experiments/results/prog_noise_scale.pkl')
