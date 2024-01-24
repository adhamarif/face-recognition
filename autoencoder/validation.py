import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from device import DEVICE
from load_dataset import CustomImageDataset, IMAGE_FOLDER, LABEL_FILE
import numpy as np
from autoencoder_net import Network, Encoder, Decoder
from train import Model
from tqdm import tqdm
import os


# import the netwrok
net = Network().to(DEVICE)

# checkpoint_path
checkpoint_path = r'autoencoder\model_checkpoints\new_model\dennis\best_model_checkpoint.pth'

# load the checkpoint
try:
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
except:
    print('Could not find model from the local disk')
    exit()

transform = torch.nn.Sequential(
    transforms.Resize((320, 320), antialias=True),
    transforms.ConvertImageDtype(torch.float32)
)

# load the datasets
dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, 
                                transform=transform, target_label='adham')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# import the base model
model = Model(net, loss_function=nn.MSELoss())

model.predict(dataloader)




