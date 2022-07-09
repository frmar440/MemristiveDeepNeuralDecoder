"""FP learning rate experiment
"""
import torch
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

from datasets import DecodeDataset
from models import DND, MDND
from trainers import Trainer

from torch.nn import Linear, Conv1d, Conv2d, Conv3d, Sequential, RNN
from torch.optim import Adam
from torchvision.transforms import Lambda

def fp_learning_rate_run():
    DATA_PATH = 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'

    # model parameters
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    HIDDEN_SIZE = 16

    # training parameters
    LEARNING_RATES = [1e-4, 1e-3, 1e-2]
    BATCH_SIZES = [32, 128, 512]
    EPOCHS = 50

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

    # dataframe init
    columns = pd.MultiIndex.from_product([BATCH_SIZES, LEARNING_RATES])
    df = pd.DataFrame(index=np.arange(1, EPOCHS+1),
                    columns=columns,
                    dtype='float64')

    for batch_size in BATCH_SIZES:

        for learning_rate in LEARNING_RATES:

            device = "cuda" if torch.cuda.is_available() else "cpu"
            # deep neural decoder
            model = DND(
                input_size=INPUT_SIZE,
                output_size=OUTPUT_SIZE,
                hidden_size=HIDDEN_SIZE
            ).to(device)

            # loss function
            loss_fn = torch.nn.CrossEntropyLoss()
            # optimizer
            optimizer = Adam(model.parameters(), lr=learning_rate)
            # trainer
            trainer = Trainer(
                training_data=training_decode_data,
                test_data=test_decode_data,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=EPOCHS,
                loss_fn=loss_fn,
                optimizer=optimizer
            )
            # floating-point training
            trainer(model)

            df[batch_size, learning_rate] = trainer.accuracies

    df.to_pickle('research/experiments/results/fp_batchsize_lr.pkl')
