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
from PIL import Image


# import the netwrok
net = Network().to(DEVICE)

# checkpoint_path
checkpoint_path = r'autoencoder\model_checkpoints\autoencoder_best_model.pth'

# load the checkpoint
try:
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net_state_dict'])
except:
    print('Could not find model from the local disk')
    exit()

### validate using image folder

# transform = torch.nn.Sequential(
#     transforms.Resize((320, 320), antialias=True),
#     transforms.ConvertImageDtype(torch.float32)
# )

# load the datasets
# dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, 
#                                 transform=transform, subset='val', target_label='liza_brille')
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# import the base model
model = Model(net, loss_function=nn.MSELoss())

#model.predict(dataloader)

### validate with single image

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((320, 320)),  # Resize the image
    transforms.ToTensor()  # Convert PIL Image to PyTorch Tensor
])

# Load a random image
#image_path = r"C:\Users\Asus\Desktop\Project\face-recognition\datasets\test_folder\justin-bieber-test.jpg"
#image_path = r"C:\Users\Asus\Desktop\Project\face-recognition\datasets\image\adham\00000.png"
#image_path = r"C:\Users\Asus\Desktop\Project\face-recognition\datasets\face\adham-test\00004.png"
image_path = r"D:\daisy_dataset\face\dennis\00138.png"
image = Image.open(image_path).convert('RGB')
# Apply the transformation to the image
input_image = transform(image).unsqueeze(0).to(DEVICE)

reconstructed_image = net(input_image)

# Calculate the MSE loss between the original image and the reconstructed image
loss = nn.MSELoss()
mse_loss = loss(input_image, reconstructed_image)

print("MSE Loss:", mse_loss.item())





