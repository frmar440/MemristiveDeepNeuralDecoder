"""Training script
"""
import torch
import json
import pickle
from datetime import datetime
from torchvision.transforms import Lambda
from torch.optim import SGD, Adam

from datasets import DecodeDataset
from models import MDND, DND
from trainers import Trainer

from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference import RRAMLikeNoiseModel
from aihwkit.simulator.configs.utils import MappingParameter
from aihwkit.simulator.configs.utils import WeightNoiseType
from aihwkit.nn.conversion import convert_to_analog


DATA_PATH = 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'

USE_HWA_TRAINING = False
# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16
NUM_LAYERS = 1

# training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
EPOCHS = 1


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
rpu_config.mapping = MappingParameter(
    digital_bias=False, # bias term is handled by the analog tile (crossbar)
    max_input_size=512,
    max_output_size=512
)
rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
rpu_config.forward.out_res = 1/256. # 8-bit ADC discretization.
rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=66.0) # the model described

device = "cuda" if torch.cuda.is_available() else "cpu"


# if USE_HWA_TRAINING:
#     ID = f'hwa_trained_arnn_model_d3_p01_nU{HIDDEN_SIZE}_nR3'
#     # hardware-aware training
#     model = MDND(
#         input_size=INPUT_SIZE,
#         hidden_size=HIDDEN_SIZE,
#         output_size=OUTPUT_SIZE,
#         num_layers=NUM_LAYERS,
#         rpu_config=rpu_config,
#         id=ID
#     ).to(device)

#     # loss function
#     loss_fn = torch.nn.CrossEntropyLoss()
#     # optimizer
#     optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
#     optimizer.regroup_param_groups(model)

# else:
ID = f'fp_trained_arnn_model_d3_p01_nU{HIDDEN_SIZE}_nR3'
# floating-point training
model = DND(
    input_size=INPUT_SIZE,
    output_size=OUTPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    id=ID
).to(device)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

if __name__ == '__main__':
    trainer = Trainer(
        training_data=training_decode_data,
        test_data=test_decode_data,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        loss_fn=loss_fn,
        optimizer=optimizer,
        batch_first=False
    )
    trainer(model)

    time = datetime.now()
    # save model
    torch.save(model.state_dict(),
            f'research/saves/{model.id}-{time}.pth')
    # save training parameters
    with open(f'research/saves/{model.id}-{time}.json', 'w') as file:
        file.write(json.dumps(trainer.training_state_dict()))
