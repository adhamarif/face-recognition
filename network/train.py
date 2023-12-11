import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from device import DEVICE
from load_dataset import CustomImageDataset, LABEL_FILE, IMAGE_FOLDER
import numpy as np
from transform import training_transform,validation_transform, label_to_tensor
from network.network import NeuralNetwork
import torch.optim as optim
from tqdm import tqdm
import json
import datetime
from torch.utils.tensorboard import SummaryWriter
import tensorboard




train_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='train',balance_class=True,transform=training_transform)
val_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='val',transform=validation_transform)
test_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='test',transform=validation_transform)
# Transform after dataloader
dataloader_train = DataLoader(train_dataset, batch_size=32, shuffle=True)
dataloader_val = DataLoader(val_dataset, batch_size=32, shuffle=True)
dataloader_test = DataLoader(test_dataset, batch_size=3, shuffle=True)
num_labels = train_dataset.img_labels.iloc[:, 1].nunique()
net = NeuralNetwork(num_labels).to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train_epoch(epoch_id,tb_writer):
    # Initialize running loss and last loss for every batch
    running_loss : 0
    last_loss = 0

    # Try with one batch
    for i, data in enumerate(dataloader_train):
        # Get inputs and labels
        inputs, labels = data
        # inputs = training_transform(inputs)
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Zero out gradients for every batch
        optimizer.zero_grad()

        # Make predictions
        outputs = net(inputs)

        # Compute the loss and its gradients and backpropagate
        loss = loss_fn(outputs,labels)
        loss.backward()

        # Adjust weights
        optimizer.step()

        # Report batch performance
        running_loss += loss.item()

        # Reports for batch size of 1000
        if i == 1000:
            last_loss = running_loss / 1000 # loss per batch
            print(f'     batch {i+1} loss: {last_loss}')
            # Add counter
            tb_x = epoch_id*len(dataloader_train) + i + 1
            tb_writer.add_scalar('Loss/Train', last_loss, tb_x)
            running_loss = 0.
    return last_loss

timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f'runs/face_detection/{timestamp}'
writer = SummaryWriter(log_dir)

epoch_number = 0

EPOCHS = 5

# Initialize a large value for first best validation loss
best_vloss = 1_000_000.

for epoch in range(EPOCHS):
    print(f'EPOCH {epoch_number+1}')

    # Set to training mode
    net.train(True)
    avg_loss = train_epoch(epoch_number,writer)
    # Set to evaluation mode to validate the epoch
    running_vloss = 0
    net.eval()

    # Disable gradient computation
    with torch.no_grad():
        for i, vdata in enumerate(dataloader_val):
            vinputs, vlabels = vdata
            voutputs = net(vinputs)
            vloss = loss_fn(voutputs,vlabels)
            running_vloss += vloss
    # Calculate average running loss
    avg_vloss = running_vloss/(i+1)
    print(f'LOSS -- train : {avg_loss}, validation : {avg_vloss}')

    # Logging
    writer.add_scalars('Training vs Validation Loss',
                       { 'Training' : avg_loss, 'Validation': avg_vloss},
                       epoch_number + 1)
    writer.flush()

    # Track best performance and save model state
    if avg_loss < best_vloss:
        # Updates the current best validation loss
        best_vloss = avg_loss
        model_path = f'model_{timestamp}_{epoch_number}.h5'
        torch.save(net.state_dict(), model_path)
    
    epoch_number + 1







# test_mode = "train"



# if __name__ == "__main__":

#     if test_mode == "epoch":
#         dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, transform=training_transform)
#         dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#         num_labels = dataset.img_labels.iloc[:, 1].nunique()
#         net = NeuralNetwork(num_labels).to(DEVICE)
#         criterion = torch.nn.CrossEntropyLoss()
#         optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#         # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=5,verbose=True)
#         trainer = Trainer(net,criterion,optimizer,scheduler,device=DEVICE)
#         trainer.epoch(dataloader)
    
#     elif test_mode == "train":
#         dataset_train = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,transform=training_transform)
#         dataset_val = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,transform=validation_transform)
#         dataloader_train = DataLoader(dataset_train, batch_size=32, shuffle=True)
#         dataloader_val = DataLoader(dataset_val, batch_size=32, shuffle=True)
#         num_labels = dataset_train.img_labels.iloc[:, 1].nunique()
#         net = NeuralNetwork(num_labels).to(DEVICE)
#         criterion = torch.nn.CrossEntropyLoss()
#         optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#         # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.1,patience=5,verbose=True)
#         checkpoint_path = "C:\\Users\\ASUS\\Desktop\\Sem5\\AdvIntSys\\face-recognition\\checkpoints\\checkpoint.pth"
#         trainer = Trainer(net,criterion,optimizer,checkpoint_path=checkpoint_path,scheduler=scheduler,device=DEVICE)
#         trainer.train(dataloader_train,dataloader_val)


