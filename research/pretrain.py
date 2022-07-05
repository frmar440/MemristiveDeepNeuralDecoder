"""Training script
"""
import torch
import json
import pickle
from datetime import datetime
from torchvision.transforms import Lambda
from torch.optim import Adam

from datasets import DecodeDataset
from models import DND
from trainers import Trainer


DATA_PATH = 'research/1QBit/test_data_d3/surfaceCodeRMX_d3_p01_Nt1M_rnnData_aT1651079378.txt'

# model parameters
INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 16

# training parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
EPOCHS = 1

ID = f'fp_trained_dnd_model_d3_p01_nU{HIDDEN_SIZE}_nR3'

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

# floating-point training
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

time = datetime.now()
# save dnd
torch.save(model, f'research/saves/{ID}-{time}.pth')
# save training parameters
with open(f'research/saves/{ID}-{time}.json', 'w') as file:
    file.write(json.dumps(trainer.training_state_dict()))
