import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from device import DEVICE
from load_dataset import CustomImageDataset, IMAGE_FOLDER, LABEL_FILE
import numpy as np
from autoencoder_net import Network
from tqdm import tqdm

# Define a model class
class Model:
    def __init__(self, network, loss_function):
        self.network = network
        self.loss_function = loss_function

        self.optimizer = self.get_optimizer()
    
    def get_optimizer(self):
        # Optimizers specified in the torch.optim package
        return torch.optim.Adam(self.network.parameters(), lr=0.0001)
    
    def train(self, dataloader, num_epochs):
        for epoch in range(num_epochs):
            # Set up tqdm progress bar
            progress = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

            total_loss = 0.0
            num_batches = 0

            for batch, _ in progress:
        

                # Move batch to device if available
                batch = batch.to(DEVICE)
                
                # zero the gradient
                self.optimizer.zero_grad()

                # convert 'Byte' to 'float32' --> MSE only works with float
                batch = batch.to(dtype=torch.float32) #.requires_grad_(True)

                # forward pass
                outputs = self.network(batch).to(dtype=torch.float32)

                # calculate the loss
                loss = self.loss_function(outputs, batch)#.requires_grad_(True)
            

                # backward pass
                loss.backward()
                self.optimizer.step()

                # Accumulate the loss
                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                progress.set_postfix(loss=total_loss / num_batches)

            # Print the average loss for the epoch
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / num_batches:.4f}')

if __name__ == '__main__':
    transform = torch.nn.Sequential(
        transforms.Resize((320, 320), antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    )

    # load the datasets
    dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # init the network and model
    network = Network().to(DEVICE)
    model = Model(network, loss_function=nn.MSELoss()) # use MSE as loss function

    model.train(dataloader=dataloader, num_epochs=10)
