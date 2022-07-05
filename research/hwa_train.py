"""HWA training script
"""
import torch
import json
import pickle
from datetime import datetime
from torchvision.transforms import Lambda

from datasets import DecodeDataset
from models import DND, MDND
from trainers import Trainer, Tester

from torch.nn import Linear, Conv1d, Conv2d, Conv3d, Sequential, RNN
from aihwkit.nn import AnalogLinear, AnalogConv1d, AnalogConv2d, AnalogConv3d, AnalogRNN, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import InferenceRPUConfig, FloatingPointRPUConfig
from aihwkit.inference import RRAMLikeNoiseModel
from aihwkit.simulator.configs.utils import MappingParameter, WeightClipParameter, WeightClipType, WeightModifierParameter, WeightNoiseType
from aihwkit.nn.conversion import convert_to_analog


DATA_PATH = 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'
LOAD_PATH = 'research/saves/fp_trained_dnd_model_d3_p01_nU16_nR3-2022-07-04 19:44:15.612018.pth'

USE_HWA_TRAINING = False

# hwa training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
EPOCHS = 1

ID = f'{"hwa" if USE_HWA_TRAINING else "fp"}_trained_mdnd_model_d3_p01_nU16_nR3'

_CONVERSION_MAP = {Linear: AnalogLinear,
                   Conv1d: AnalogConv1d,
                   Conv2d: AnalogConv2d,
                   Conv3d: AnalogConv3d,
                   RNN: AnalogRNN,
                   Sequential: AnalogSequential,
                   DND: MDND}


# resistive processing unit
rpu_config = InferenceRPUConfig()
rpu_config.drift_compensation = None
rpu_config.mapping = MappingParameter(digital_bias=False, # bias term is handled by the analog tile (crossbar)
                                      max_input_size=512,
                                      max_output_size=512)
rpu_config.forward.inp_res = 1/256.  # 8-bit DAC discretization.
rpu_config.forward.out_res = 1/256.  # 8-bit ADC discretization.
rpu_config.forward.w_noise_type = WeightNoiseType.ADDITIVE_CONSTANT
rpu_config.forward.w_noise = 0.0

# training
rpu_config.clip = WeightClipParameter(sigma=2.5, type=WeightClipType.LAYER_GAUSSIAN)
# inference
rpu_config.noise_model = RRAMLikeNoiseModel(g_max=200.0, g_min=66.0, prog_noise_scale=1.) # rram noise
# training and inference
rpu_config.modifier = WeightModifierParameter(pdrop=0.1) # defective device probability

# load the dnd model (pretrained baseline network)
model = torch.load(LOAD_PATH)
# convert dnd to mdnd
model = convert_to_analog(model, rpu_config, conversion_map=_CONVERSION_MAP)
rpu_config.forward.w_noise = 0.1
model.load_rpu_config(rpu_config)
rpu_config.forward.w_noise = 0.2
model.load_rpu_config(rpu_config)


with open(DATA_PATH, 'rb') as f:
    dico = pickle.loads(f.read())

test_decode_data = DecodeDataset(
    dico=dico,
    train=False,
    transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
    target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()


if USE_HWA_TRAINING:
    # load training and test datasets
    training_decode_data = DecodeDataset(
        dico=dico,
        train=True,
        transform=Lambda(lambda y: torch.tensor(y, dtype=torch.float)),
        target_transform=Lambda(lambda y: torch.zeros(2, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot encoding
    )

    # analog optimizer
    optimizer = AnalogSGD(model.parameters(), lr=LEARNING_RATE)
    optimizer.regroup_param_groups(model)

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
    trainer(model)

# inference test
tester = Tester(
    test_data=test_decode_data,
    batch_size=BATCH_SIZE,
    loss_fn=loss_fn
)
tester(model)

time = datetime.now()
# save mdnd
# torch.save(model, f'research/saves/{ID}-{time}.pth')
# if USE_HWA_TRAINING:
#     # save hwa training parameters
#     with open(f'research/saves/{ID}-{time}.json', 'w') as file:
#         file.write(json.dumps(trainer.training_state_dict()))
