import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from autoencoder.device import DEVICE
from autoencoder.load_dataset import CustomImageDataset
import numpy as np
from autoencoder.autoencoder_net import Network, Encoder, Decoder
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import argparse

# Commented out for copy-paste purposes during live presentation
# LABEL_FILE = r"D:\clean_dataset\label.csv"
# IMAGE_FOLDER = r"D:\clean_dataset"

parser = argparse.ArgumentParser(description="Autoencoder Training Script")
parser.add_argument("--images_folder", required=True, help="Please specify path of your face images folder.")
args = parser.parse_args()

IMAGE_FOLDER = args.images_folder
LABEL_FILE = IMAGE_FOLDER + "\\label.csv"

# Define a model class
class Model:
    def __init__(self, network, loss_function):
        self.network = network
        self.loss_function = loss_function
        self.writer = SummaryWriter()
        self.optimizer = self.get_optimizer()
    
    def get_optimizer(self):
        return torch.optim.Adam(self.network.parameters(), lr=0.001)
    
    def train(self, train_dataloader, val_dataloader, num_epochs=10, save_path=None):
        '''
        This method is used to start the training loop. The dataloader is required for both training and validation set.
        Num of epochs is default at 10. You can adjusted this values accordingly.
        If you wish to save the model checkpoint, please specify the path for it.
        '''
        best_avg_loss = float('inf') # init highest avg loss value

        for epoch in range(num_epochs):
            # Setup tqdm progress bar
            progress = tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{num_epochs}')

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
                if num_batches % 40 == 0:  # Adjust the frequency as needed
                    input_images = batch
                    output_images = outputs

                    self.writer.add_images('Input Images', input_images, epoch * len(train_dataloader) + num_batches)
                    self.writer.add_images('Output Images', output_images, epoch * len(train_dataloader) + num_batches)

            # Log training loss to TensorBoard
            self.writer.add_scalar('Training Loss', total_loss / num_batches, epoch + 1)

            # Print the average loss for the epoch
            average_loss = total_loss / num_batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {total_loss / num_batches:.4f}')
            
            # Get the biggest validation loss in every epoch
            max_val_loss = 0

            # Switch to evaluation mode
            self.network.eval()
            with torch.no_grad():
                for batch, _ in val_dataloader:
                    # Move batch to device if available
                    batch = batch.to(DEVICE)

                    # forward pass
                    outputs = self.network(batch)

                    # calculate the loss
                    val_loss = self.loss_function(outputs, batch)

                    # update the max validation loss value
                    if val_loss.item() > max_val_loss:
                        max_val_loss = val_loss.item()
            
            print(f'Max. validation loss: {max_val_loss:.4f}')

            # Log average validation loss to TensorBoard
            self.writer.add_scalar('Max. validation loss', max_val_loss, epoch + 1)

            # Switch back to training mode
            self.network.train()

            if average_loss < best_avg_loss:
                # update the best average loss
                best_avg_loss = average_loss
                # save the model if save_path is available
                if save_path is not None:
                    self.save_model(save_path, epoch + 1, best_avg_loss) # save the model
        
        self.writer.close()

    def save_model(self, save_path, epoch, average_loss):
        '''
        This method is used to save the best model checkpoint in the local storage.
        '''
        # Create a directory to save the model if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save the model state dictionary and other information
        checkpoint = {
            'epoch': epoch,
            'net_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': average_loss,
        }
        torch.save(checkpoint, os.path.join(save_path, f'autoencoder_best_model.pth'))

if __name__ == '__main__':   
    # transformation of the images, resize to (320, 320), applied data augmentation and convert to float32 (normalized)
    train_transform = torch.nn.Sequential(
        transforms.Resize((320, 320), antialias=True),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ConvertImageDtype(torch.float32)
    )

    # transformation of the images, resize to (320, 320) and convert to float32 (normalized)
    valid_transform = torch.nn.Sequential(
        transforms.Resize((320, 320), antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    )

    # load the training dataset
    train_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, transform=train_transform, subset='train', balance_class=True)
    print(f'Training dataset size: {len(train_dataset)} images')

    # load the validation dataset
    val_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, transform=valid_transform, subset='val')
    print(f'Validation dataset size: {len(val_dataset)} images')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # init the network and model
    network = Network().to(DEVICE)

    if not os.path.exists("models"):
        os.makedirs("models")

    # path for the model checkpoints
    save_path = "models\\"

    model = Model(network, loss_function=nn.MSELoss()) # use MSE as loss function

    model.train(train_dataloader=train_dataloader, val_dataloader=val_dataloader, num_epochs=20, save_path=save_path)
