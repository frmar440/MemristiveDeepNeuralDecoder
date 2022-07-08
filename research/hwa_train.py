"""HWA training script
"""
import torch
import json
import pickle
from datetime import datetime
from torchvision.transforms import Lambda

from datasets import DecodeDataset
from models import MDND
from trainers import Trainer

from torch.optim import Adam
from aihwkit.optim import AnalogOptimizer
from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.inference import RRAMLikeNoiseModel
from aihwkit.simulator.configs.utils import (
    MappingParameter, WeightClipParameter, WeightClipType,
    WeightModifierParameter, WeightNoiseType, WeightModifierType
)


DATA_PATH = 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'
LOAD_PATH = 'research/saves/mdnd/fp_trained_mdnd_model_d3_p01_nU16_nR3-2022-07-05 18:40:45.209994.pth'

# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16

# hwa training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
EPOCHS = 20


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
rpu_config.clip = WeightClipParameter(sigma=2.5, type=WeightClipType.LAYER_GAUSSIAN)
# training and inference
rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
rpu_config.forward.out_res = 1/256.  # 8-bit ADC discretization.
rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=66.0, prog_noise_scale=1.) # rram noise
rpu_config.modifier = WeightModifierParameter(pdrop=0.0) # defective device probability


device = "cuda" if torch.cuda.is_available() else "cpu"
# memristive deep neural decoder
analog_model = MDND(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    rpu_config=rpu_config
).to(device)
# load weights (but use the current RPU config)
analog_model.load_state_dict(torch.load(LOAD_PATH), load_rpu_config=False)


# loss function
loss_fn = torch.nn.CrossEntropyLoss()
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
    optimizer=optimizer
)
# hwa training
trainer(analog_model)


ID = f'hwa_trained_mdnd_model_d3_p01_nU16_nR3'

time = datetime.now()
# save mdnd
torch.save(analog_model.state_dict(), f'research/saves/mdnd/{ID}-{time}.pth')
# save hwa training parameters
with open(f'research/saves/mdnd/{ID}-{time}.json', 'w') as file:
    file.write(json.dumps(trainer.training_state_dict()))
