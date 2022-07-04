"""Inference script
"""
import torch

from trainers import Tester
from train import test_decode_data, rpu_config, model, loss_fn, BATCH_SIZE

from aihwkit.nn.conversion import convert_to_analog


LOAD_PATH = 'research/saves/fp_trained_arnn_model_d3_p01_nU16_nR3-2022-07-04 11:21:50.965605.pth'

model.load_state_dict(torch.load(LOAD_PATH))
model = convert_to_analog(model, rpu_config)

# loss function
loss_fn = torch.nn.CrossEntropyLoss()

tester = Tester(
    test_data=test_decode_data,
    batch_size=BATCH_SIZE,
    loss_fn=loss_fn,
    batch_first=False
)
tester(model)

# t_inference = 0.
# for _ in range(10):
#     model.drift_analog_weights(t_inference) # update read noise
#     tester.test_loop()
