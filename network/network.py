import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from .device import DEVICE
from .load_dataset import CustomImageDataset, LABEL_FILE, IMAGE_FOLDER
import numpy as np
from .transform import training_transform,validation_transform
from sklearn.model_selection import train_test_split


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
            Down(in_features =   3, out_features =  16), # 16x128x128 
            nn.Dropout2d(0.2), 
            Down(in_features =  16, out_features =  32), #  32x64x64
            nn.Dropout2d(0.2),
            Down(in_features =  32, out_features =  64), #  64x32x32
            nn.Dropout2d(0.2),
            Down(in_features =  64, out_features =  128), # 128x16x16
            nn.Dropout2d(0.2),
            Down(in_features =  128, out_features =  256), # 256x8x8
            nn.Dropout2d(0.2),
            Down(in_features =  256, out_features = 512), # 512x4x4
            nn.Dropout2d(0.2),
            nn.Flatten(), # 4096 dimensional
            nn.Linear(8192, 512), # 8192 to 512 dimensional
            nn.ReLU(), # Another ReLU
            nn.Dropout(0.5),
            nn.Linear(512, num_labels)
            # nn.Softmax(dim=-1)- Commented out because CrossEntropyLoss already apply  softmax
        )

    def forward(self, x):
        for i,layer in enumerate(self.seq):
            x = layer(x)
        return x


if __name__ == "__main__":
    
    train_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='train',balance_class=True,transform=training_transform)
    val_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='val',transform=validation_transform)
    dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=True)
    num_labels = 12
    net = NeuralNetwork(num_labels).to(DEVICE)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Network has {params} total parameters")
    loss_fn = torch.nn.CrossEntropyLoss()
    batch, labels = dataloader_train.__iter__().__next__()
    batch = batch.to(DEVICE)
    labels = labels.to(DEVICE)
    x = net(batch)
    loss = loss_fn(x,labels)
    print(x.shape,loss)

