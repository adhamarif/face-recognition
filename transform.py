import torch
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from tqdm import tqdm
from load_dataset import CustomImageDataset, LABEL_FILE, IMAGE_FOLDER
import random
from matplotlib import pyplot as plt
from device import DEVICE

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

if __name__ == "__main__":

    dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER)
    image, label = dataset[int(random.uniform(0, len(dataset)))]
    print(image.shape)
    batch = None
    for index in range(32):
        transformed = training_transform(image)
        print(transformed.shape)
        transformed = transformed.reshape(1, 3, 256, 256)
        if batch is None:
            batch = transformed
        else:
            batch = torch.cat((batch, transformed))

    grid = make_grid(batch, 8, padding=4).permute(1,2,0)

    plt.imshow(grid.cpu())
    plt.show()