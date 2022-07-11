"""FP batch size - learning rate experiment
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

from torch.optim import Adam
from torchvision.transforms import Lambda

from aihwkit.optim import AnalogOptimizer
from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.inference import RRAMLikeNoiseModel
from aihwkit.simulator.configs.utils import (
    MappingParameter, WeightClipParameter, WeightClipType,
    WeightModifierParameter, WeightNoiseType, WeightModifierType
)

def fp_batchsize_lr_run():
    DATA_PATH = 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'
    LOAD_PATH = ''

    # model parameters
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    HIDDEN_SIZE = 16

    # training parameters
    LEARNING_RATES = [1e-4, 1e-3, 1e-2]
    BATCH_SIZES = [32, 128, 512]
    EPOCHS = 30

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # loss function
    loss_fn = torch.nn.CrossEntropyLoss()

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

    # resistive processing unit
    rpu_config = InferenceRPUConfig()
    rpu_config.drift_compensation = None
    rpu_config.mapping = MappingParameter(digital_bias=False, # bias term is handled by the analog tile (crossbar)
                                          max_input_size=512,
                                          max_output_size=512)
    # training
    # rpu_config.clip = WeightClipParameter(sigma=2.5, type=WeightClipType.LAYER_GAUSSIAN)
    # training and inference
    rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
    rpu_config.forward.out_res = 1/256.  # 8-bit ADC discretization.
    rpu_config.forward.out_noise = 0.
    rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=60.0, prog_noise_scale=1.) # rram noise
    rpu_config.modifier = WeightModifierParameter(pdrop=0.1, # defective device probability
                                                  enable_during_test=True,
                                                  std_dev=0.005, # 0.5% training noise
                                                  type=WeightModifierType.ADD_NORMAL)

    # dataframe init
    columns = pd.MultiIndex.from_product([BATCH_SIZES, LEARNING_RATES])
    df = pd.DataFrame(index=np.arange(1, EPOCHS+1),
                      columns=columns,
                      dtype='float64')

    for batch_size in BATCH_SIZES:

        for learning_rate in LEARNING_RATES:

            # analog optimizer
            optimizer = AnalogOptimizer(Adam, analog_model.parameters(), lr=learning_rate)
            optimizer.regroup_param_groups(analog_model)

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

            # memristive deep neural decoder
            analog_model = DND(
                input_size=INPUT_SIZE,
                output_size=OUTPUT_SIZE,
                hidden_size=HIDDEN_SIZE
            ).to(device)
            # load weights (but use the current RPU config)
            analog_model.load_state_dict(torch.load(LOAD_PATH), load_rpu_config=False)

            # hwa training
            trainer(analog_model)

            df[batch_size, learning_rate] = trainer.accuracies

    df.to_pickle('research/experiments/results/hwa_batchsize_lr.pkl')
