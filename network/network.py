import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from device import DEVICE
from load_dataset import CustomImageDataset, LABEL_FILE, IMAGE_FOLDER
import numpy as np
from transform import training_transform,validation_transform


class Down(nn.Module):
    def __init__(self, in_features, out_features):
        super(Down, self).__init__()
        
        self.seq  = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=(5,5), padding="same"),
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            nn.BatchNorm2d(num_features = out_features),
            nn.ReLU()
            )

    def forward(self, x):
        return self.seq(x)

class NeuralNetwork(torch.nn.Module):
    def __init__(self,num_labels):
        super(NeuralNetwork, self).__init__()

        self.num_labels = num_labels

        self.seq = nn.Sequential(
            Down(in_features =   3, out_features =  16), # 32x64x64 
            nn.Dropout2d(0.2), # Compare https://arxiv.org/abs/1411.4280
            Down(in_features =  16, out_features =  32), #  64x32x32
            nn.Dropout2d(0.2),
            Down(in_features =  32, out_features =  64), #  128x16x16
            nn.Dropout2d(0.2),
            Down(in_features =  64, out_features =  128), # 256x8x8
            nn.Dropout2d(0.2),
            Down(in_features =  128, out_features =  256), # 512x4x4
            nn.Dropout2d(0.2),
            nn.Flatten(), # 4096 dimensional
            nn.Linear(16384, 512), # 16384 to 512 dimensional
            nn.ReLU(), # Another ReLU
            nn.Dropout(0.5),
            nn.Linear(512, num_labels),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        for i,layer in enumerate(self.seq):
            x = layer(x)
            print(f"Size after layer {i}:{x.size()}")
        return x

# model = NeuralNetwork().to(DEVICE)
# print(model)

if __name__ == "__main__":
    
    dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,transform=training_transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_labels = dataset.img_labels.iloc[:, 1].nunique()
    net = NeuralNetwork(num_labels).to(DEVICE)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Network has {params} total parameters")

    batch, labels = dataloader.__iter__().__next__()
    x = net(batch)
    print(x.shape)


    # RESIZE- error happening because image is not resized before feeding!!!!!!