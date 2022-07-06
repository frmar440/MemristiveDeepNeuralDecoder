"""Training script
"""
import torch
import json
import pickle
from datetime import datetime

from datasets import DecodeDataset
from models import DND, MDND
from trainers import Trainer

from torch.nn import Linear, Conv1d, Conv2d, Conv3d, Sequential, RNN
from torch.optim import Adam
from torchvision.transforms import Lambda

from aihwkit.nn import AnalogLinear, AnalogConv1d, AnalogConv2d, AnalogConv3d, AnalogRNN, AnalogSequential
from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.simulator.configs.utils import MappingParameter
from aihwkit.nn.conversion import convert_to_analog


CONVERSION_MAP = {Linear: AnalogLinear,
                  Conv1d: AnalogConv1d,
                  Conv2d: AnalogConv2d,
                  Conv3d: AnalogConv3d,
                  RNN: AnalogRNN,
                  Sequential: AnalogSequential,
                  DND: MDND}

DATA_PATH = 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'

# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16

# training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
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
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
# trainer
trainer = Trainer(
    training_data=training_decode_data,
    test_data=test_decode_data,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    loss_fn=loss_fn,
    optimizer=optimizer
)
# floating-point training
trainer(model)


ID = f'fp_trained_dnd_model_d3_p01_nU16_nR3'

time = datetime.now()
# save dnd
torch.save(model.state_dict(), f'research/saves/dnd/{ID}-{time}.pth')
# save fp training parameters
with open(f'research/saves/dnd/{ID}-{time}.json', 'w') as file:
    file.write(json.dumps(trainer.training_state_dict()))


ID = f'fp_trained_mdnd_model_d3_p01_nU16_nR3'

# resistive processing unit
rpu_config = InferenceRPUConfig()
rpu_config.mapping = MappingParameter(digital_bias=False, # bias term is handled by the analog tile (crossbar)
                                      max_input_size=512,
                                      max_output_size=512)
# convert dnd to mdnd
analog_model = convert_to_analog(model, rpu_config, conversion_map=CONVERSION_MAP)
# save mdnd
torch.save(analog_model.state_dict(), f'research/saves/mdnd/{ID}-{time}.pth')
