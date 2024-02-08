import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
from .load_dataset import CustomImageDataset, LABEL_FILE, IMAGE_FOLDER
import random
from matplotlib import pyplot as plt
from .device import DEVICE
import numpy as np


training_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.CenterCrop((256, 256)),
        # transforms.RandomCrop((128, 128)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ConvertImageDtype(torch.float32)
        ]
    )

validation_transform = transforms.Compose([
        transforms.CenterCrop((256, 256)),
        transforms.ConvertImageDtype(torch.float32)
        ]
    )

image_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((360, 360), antialias=True),
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor()
        ]
    )

def label_to_tensor(label):
    return torch.Tensor([label]).type(torch.LongTensor).to(DEVICE).reshape(-1)

if __name__ == "__main__":

    # Load dataset
    train_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='train',balance_class=True,transform=training_transform)
    print(f'training dataset size: {len(train_dataset)}.')
    val_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='val',transform=validation_transform)
    print(f'validation dataset size: {len(val_dataset)}.')
    test_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='test',transform=validation_transform)
    print(f'test dataset size: {len(test_dataset)}.')
    image, label = test_dataset[int(random.uniform(0, len(test_dataset)))]
    batch = None
    for index in range(32):
        transformed = training_transform(image)
        transformed = transformed.reshape(1, 3, 256, 256)
        if batch is None:
            batch = transformed
        else:
            batch = torch.cat((batch, transformed))

    grid = make_grid(batch, 8, padding=4).permute(1,2,0)

    plt.imshow(grid.cpu())
    plt.show()