import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import make_grid

LABEL_FILE = r"C:\Users\Asus\Desktop\Project\face-recognition\datasets\label.csv"
IMAGE_FOLDER = r"C:\Users\Asus\Desktop\Project\face-recognition\datasets\face"

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':
    dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    batch, label = dataloader.__iter__().__next__()
        
    grid = make_grid(batch, 8, padding=4).permute(1,2,0)

    plt.imshow(grid.cpu())
    plt.show()
