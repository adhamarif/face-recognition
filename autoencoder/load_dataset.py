import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .device import DEVICE
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import make_grid
from sklearn.model_selection import train_test_split

# Please replace with your own respective paths for each
LABEL_FILE = r"D:\clean_dataset\label.csv"
IMAGE_FOLDER = r"D:\clean_dataset"

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, subset="train", balance_class=False, target_label=None): #, target_label=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.subset = subset
        self.balance_class = balance_class
        self.target_label = target_label
        if target_label:
            # mask the option based on target
            self.img_labels = self.img_labels[self.img_labels['label'] == self.target_label]

        # Split the data into training, validation, and test sets
        train_data, val_data = train_test_split(self.img_labels, test_size=0.2, random_state=42)

        if self.subset == 'train':
            self.data = train_data
        elif self.subset == 'val':
            self.data = val_data
        else:
            raise ValueError("Subset must be one of 'train', 'val'")
        
        # Optionally balance classes in the training set
        if self.balance_class and self.subset == 'train':
            self.data = self.balance_classes(self.data)
            

    def __len__(self):
        return len(self.data)  # Filter by target_label)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = read_image(img_path).to(DEVICE)
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def balance_classes(self, data):
        # Implement class balancing logic here

        class_counts = data['label'].value_counts()
        max_class_count = class_counts.max()

        balanced_data = pd.DataFrame(columns=data.columns)

        for class_label, count in class_counts.items():
            class_data = data[data['label'] == class_label]
            if count < max_class_count:
                # Oversamples according to the difference between the label count and the highest label count
                oversampled_data = class_data.sample(max_class_count - count, replace=True, random_state=42)
                class_data = pd.concat([class_data, oversampled_data], ignore_index=True)
            balanced_data = pd.concat([balanced_data, class_data], ignore_index=True)

        return balanced_data

if __name__ == '__main__':
    # apply transformation to the images
    transform = torch.nn.Sequential(
        transforms.Resize((480, 480), antialias=True),
        transforms.ConvertImageDtype(torch.float32)
    )

    # load the training dataset
    train_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, transform=transform, subset='train', balance_class=True, target_label='dennis')
    print(f'Training dataset size: {len(train_dataset)} images')

    # load the validtaion dataset
    val_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER, transform=transform, subset='val', target_label='dennis')
    print(f'Validation dataset size: {len(val_dataset)} images')

    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    batch, label = dataloader.__iter__().__next__()

    # plot the images grid    
    grid = make_grid(batch, 8, padding=4).permute(1,2,0)
    plt.imshow(grid.cpu())
    plt.show()
