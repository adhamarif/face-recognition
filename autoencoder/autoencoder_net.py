import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from load_dataset import CustomImageDataset, IMAGE_FOLDER, LABEL_FILE
from torchvision.transforms import transforms
from device import DEVICE

class Down(nn.Module):
    def __init__(self, input_channels, ouput_channels):
        super().__init__() #__init__ for nn.Module
        self.relu = nn.ReLU()
        # Define a downsample layer
        self.seq = nn.Sequential(
           # first convolution layer
            nn.Conv2d(in_channels=input_channels, out_channels=ouput_channels,
                      kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(num_features=ouput_channels),
            nn.ReLU(), # activation function
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        )
    
    def forward(self, x):
        return self.seq(x)

class Up(nn.Module):
    def __init__(self, input_channels, ouput_channels):
        super().__init__()
        self.relu = nn.ReLU()
        # Define a upsample layer
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels=input_channels, out_channels=ouput_channels,
                               kernel_size=(3, 3), stride=(2, 2), padding=(1,1), output_padding=(1,1)),
            nn.BatchNorm2d(num_features=ouput_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.seq(x)
    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            Down(3, 8),
            Down(8, 16),
            Down(16, 32),
            Down(32, 64),
            Down(64, 128),
            Down(128, 256),
            #nn.Flatten()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            Up(256, 128),
            Up(128, 64),
            Up(64, 32),
            Up(32, 16),
            Up(16, 8),
            Up(8, 3),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.decoder(x)
        return x
    
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(
            Encoder(),
            Decoder()
        )
    
    def forward(self, x):
        return self.seq(x)

# verify the network
if __name__ == '__main__':
    
    # setup a network    
    net = Network().to(DEVICE)
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"Network has {params} total parameters")

    transform = torch.nn.Sequential(
        transforms.Resize((320, 320), antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    )

    dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, transform=transform,
                                 target_label = 'dennis')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    batch, labels = dataloader.__iter__().__next__()
    print(batch.shape)
    x = net(batch)