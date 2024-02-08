import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from PIL import Image
from torchvision import datasets, transforms
from network.device import DEVICE
from network.load_dataset import CustomImageDataset
import numpy as np
from network.transform import training_transform,validation_transform
from network.network import NeuralNetwork
import torch.optim as optim
from tqdm import tqdm
import json
import datetime
from torch.utils.tensorboard import SummaryWriter
import tensorboard

LABEL_FILE = r"D:\clean_dataset\face_label_encoded.csv"
IMAGE_FOLDER = r"D:\clean_dataset"

train_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='train',balance_class=True,transform=training_transform)
val_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='val',transform=validation_transform)
test_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='test',transform=validation_transform)
dataloader_train = DataLoader(train_dataset, batch_size=64, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=64, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=3, shuffle=True)
num_labels = 12 # for 12 faces/classes
net = NeuralNetwork(num_labels).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) - Commented out because SGD does not have state_dict 
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Only for local training
# Create directory to save models
if not os.path.exists("models"):
    os.makedirs("models")


try:
    chkpts = list(os.listdir("models"))
    chkpt_paths = ["models/"+ i for i in chkpts] 
    last_chkpt = chkpt_paths[-1]
    checkpoint = torch.load(last_chkpt)
    print(f"Checkpoint found.\nStarting from session {last_chkpt}....")
    net.load_state_dict(checkpoint["net_state_dict"])
    optimizer.load_state_dict(checkpoint["optim_state_dict"])
except:
    print("No checkpoint found.\nStarting from scratch...")

def train_epoch(epoch_id,tb_writer):
    # Initialize running loss and last loss for every batch
    running_loss = 0
    last_loss = 0
    running_acc = 0
    last_acc = 0
    labels_count = 0
    top_loss_values = None
    top_loss_samples = None
    # Try with one batch
    for i, data in enumerate(dataloader_train):
        # Get inputs and labels
        inputs, labels = data
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)
        labels = torch.argmax(labels,dim=1)

        # Zero out gradients for every batch
        optimizer.zero_grad()

        # Make predictions
        outputs = net(inputs)

        # Compute the loss and its gradients and backpropagate
        loss = loss_fn(outputs,labels)

        # Collect top loss samples
        if top_loss_values is None:
            top_loss_values = loss.detach()
            top_loss_samples = inputs.detach()
        else:
            top_loss_values = torch.cat([top_loss_values,loss.detach()])
            top_loss_samples = torch.cat([top_loss_samples,inputs.detach()])
            top_loss_values, indices = torch.sort(top_loss_values, descending=True)
            top_loss_values = top_loss_values[:inputs.size(0)]
            top_loss_samples = top_loss_samples[indices,:,:]
        top_loss_samples = top_loss_samples[:64,:,:]
        img = make_grid(top_loss_samples, nrow=8)
        tb_writer.add_image("top_loss_samples",img,global_step=epoch_id)
        loss = torch.mean(loss)
        # Test 
        loss.backward()

        # Adjust weights
        optimizer.step()

         # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        running_acc += torch.sum(preds == labels)

        labels_count += labels.size(0)

        # Report batch performance
        running_loss += loss.item()
        last_loss = running_loss / (labels_count)  # loss per batch
        last_acc = running_acc.float() / (labels_count)
        print(f'     batch {i+1} loss: {last_loss}, acc: {last_acc}')

    return last_loss,last_acc

# Initialize timestamp and summary writer
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter()

# Initialize a large value for first best validation loss and small value for first best val accuracy
best_vloss = 3
best_vacc = 0.1

# Initialize early stopping params
patience = 5
patience_counter = 0

# Initialize starting epoch
epoch = 0

while True:

    print(f'EPOCH {epoch+1}')

    # Set to training mode
    net.train(True)
    avg_loss, avg_acc = train_epoch(epoch,writer)
    # Set to evaluation mode to validate the epoch
    running_vloss = 0
    running_vacc = 0
    vlabels_count = 0
    net.eval()

    # Disable gradient computation
    with torch.no_grad():
        for i, vdata in enumerate(dataloader_val):
            vinputs, vlabels = vdata
            # Assign inputs and labels to device
            vinputs = vinputs.to(DEVICE)
            vlabels = vlabels.to(DEVICE)
            # Get  labels
            vlabels = torch.argmax(vlabels, dim=1)
            # Get outputs
            voutputs = net(vinputs)
            # Calculate validation loss
            vloss = loss_fn(voutputs,vlabels)
            running_vloss += vloss.sum()
            # Calculate validation accuracy
            _, preds = torch.max(voutputs, 1)
            # Possibility for improvement
            # Try to fit this in a for loop. 1. Make an empty tensor of fixed length outside of loop and keep concatenating for the each class.
            # 2. Once tensor is full, add the images to the grid.
            # Output the indices where the prediction is for class 0
            # Return the images from inputs where predictions are class 0
            running_vacc += torch.sum(preds == vlabels)
            vlabels_count += vlabels.size(0)
            # Calculate average running accuracy
            avg_vacc = running_vacc.float() / (vlabels_count)
            # Calculate average running loss
            avg_vloss = running_vloss/(vlabels_count)
            
    print(f'LOSS -- train : {avg_loss}, validation : {avg_vloss} \n ACC -- train : {avg_acc}, validation : {avg_vacc}')

    # Logging
    writer.add_scalars('Loss',
                       { 'Training' : avg_loss, 'Validation': avg_vloss},
                       epoch + 1)
    writer.add_scalars('Accuracy',
                       { 'Training' : avg_acc, 'Validation' : avg_vacc},
                       epoch + 1)
    writer.flush()

    # Track best performance and save model state
    if avg_vloss < best_vloss and avg_vacc > best_vacc:
        # Updates the current best validation loss
        best_vloss = avg_vloss
        best_vacc = avg_vacc
        patience_counter = 0 # Reset patience counter to 0 if there's improvement
        # Employ saving state dicts of network and optimizer as a checkpoint dictionary
        chkpt_path = f'models/model_{timestamp}_{epoch}.pt'
        torch.save({
            "net_state_dict" : net.state_dict(),
            "optim_state_dict" : optimizer.state_dict(),
            }, chkpt_path)
    else:
        patience_counter += 1

    # Check if patience criteria is met
    if patience_counter >= patience:
        print(f"No improvement after {epoch+1} epochs.")
        break

    epoch += 1


