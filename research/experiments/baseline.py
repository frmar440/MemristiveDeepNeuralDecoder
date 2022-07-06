"""Baseline experiment
"""
import torch
import numpy as np
import pandas as pd
import re
import json
import pickle
from datetime import datetime
from torchvision.transforms import Lambda

from datasets import DecodeDataset
from models import DND, MDND
from trainers import Tester

def baseline_run():
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
    DND_LOAD_PATHS = [
        'research/saves/dnd/fp_trained_dnd_model_d3_p0035_nU16_nR3-2022-07-06 08:46:36.818726.pth',
        'research/saves/dnd/fp_trained_dnd_model_d3_p006_nU16_nR3-2022-07-06 09:14:58.713649.pth',
        'research/saves/dnd/fp_trained_dnd_model_d3_p007_nU16_nR3-2022-07-06 09:43:27.510154.pth',
        'research/saves/dnd/fp_trained_dnd_model_d3_p01_nU16_nR3-2022-07-06 10:12:12.403292.pth'
    ]

    # model parameters
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    HIDDEN_SIZE = 16

    # inference parameters
    BATCH_SIZE = 1024

    se = pd.Series(index=["p0035", "p006", "p007", "p01"], dtype='float64')
    # iterate through physical fault rate
    for DATA_PATH, DND_LOAD_PATH, in zip(DATA_PATHS, DND_LOAD_PATHS):

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

        model = DND(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            output_size=OUTPUT_SIZE
        ).to(device)
        # load weights
        model.load_state_dict(torch.load(DND_LOAD_PATH))

        for _ in range(10):
            tester(model, inference=True)
        
        se[pfr] = np.mean(tester.accuracies)

        se.to_pickle('research/experiments/results/baseline.pkl')
