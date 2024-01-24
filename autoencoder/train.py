import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from device import DEVICE
from load_dataset import CustomImageDataset, IMAGE_FOLDER, LABEL_FILE
import numpy as np
from autoencoder_net import Network, Encoder, Decoder
from tqdm import tqdm
import os

from torch.utils.tensorboard import SummaryWriter

# Define a model class
class Model:
    def __init__(self, network, loss_function):
        self.network = network
        self.loss_function = loss_function
        self.writer = SummaryWriter()

        self.optimizer = self.get_optimizer()
    
    def get_optimizer(self):
        # Optimizers specified in the torch.optim package
        return torch.optim.Adam(self.network.parameters(), lr=0.001)
    
    def train(self, dataloader, num_epochs, save_path=None):
        best_avg_loss = float('inf')

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

                # forward pass
                outputs = self.network(batch)

                # calculate the loss
                loss = self.loss_function(outputs, batch)
            

                # backward pass
                loss.backward()
                self.optimizer.step()

                # Accumulate the loss
                total_loss += loss.item()
                num_batches += 1

                # Update progress bar
                progress.set_postfix(loss=total_loss / num_batches)

                # Log input images to TensorBoard
                if num_batches % 10 == 0:  # Adjust this frequency as needed
                    input_images = batch  # Adjust the number of images to log
                    output_images = outputs

                    self.writer.add_images('Input Images', input_images, epoch * len(dataloader) + num_batches)
                    self.writer.add_images('Output Images', output_images, epoch * len(dataloader) + num_batches)

            # Log training loss to TensorBoard
            self.writer.add_scalar('Training Loss', total_loss / num_batches, epoch + 1)

            # Print the average loss for the epoch
            average_loss = total_loss / num_batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / num_batches:.4f}')

            # # save the model after every 10th epoch if save_path is available
            # if save_path is not None and (epoch + 1) % 10 == 0:
            #     self.save_model(save_path, epoch + 1, average_loss) # call the function to save the model

            if average_loss < best_avg_loss:
                # update the best average loss
                best_avg_loss = average_loss
                # save the model if save_path is available
                if save_path is not None:
                    self.save_model(save_path, epoch + 1, best_avg_loss) # call the function to save the model
        
        self.writer.close()

    def save_model(self, save_path, epoch, average_loss):
        # Create a directory to save the model if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save the model state dictionary and other information
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': average_loss,
        }

        torch.save(checkpoint, os.path.join(save_path, f'best_model_checkpoint.pth'))

    def predict(self, dataloader):
        total_loss = 0.0
        num_batches = 0
        # set up tqdm progress bar
        progress = tqdm(dataloader, desc='Predicting')

        for batch, _ in progress:
            # Move batch to device if available
            batch = batch.to(DEVICE)

            # load the batch to the model
            outputs = self.network(batch)

            # calculate the loss
            loss = self.loss_function(outputs, batch)

            # Accumulate the loss
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            progress.set_postfix(loss=total_loss / num_batches)




if __name__ == '__main__':
    # Create a SummaryWriter for TensorBoard logging
    #writer = SummaryWriter()

    transform = torch.nn.Sequential(
        transforms.Resize((320, 320), antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    )

    # load the datasets
    dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, 
                                 transform=transform, target_label='dennis')
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # init the network and model
    network = Network().to(DEVICE)

    # path for the model checkpoints
    save_path = r'autoencoder/model_checkpoints/new_model/dennis'

    model = Model(network, loss_function=nn.MSELoss()) # use MSE as loss function

    model.train(dataloader=dataloader, num_epochs=10, save_path=save_path)
