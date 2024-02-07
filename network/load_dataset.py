import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.io import read_image
from torchvision.utils import make_grid
from .device import DEVICE
import torch
import numpy as np
from sklearn.model_selection import train_test_split

LABEL_FILE = "C:\\Users\\ASUS\\datasets\\face_label_encoded.csv"
IMAGE_FOLDER = "C:\\Users\\ASUS\\datasets\\cleaned_face"

resize_transform = transforms.Compose([
    transforms.Resize((360, 360), antialias=True),
])



class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None,subset='train',balance_class=False): #mode = DataSetMode.Training):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.subset = subset
        self.balance_class = balance_class

        # Split the data into training, validation, and test sets
        train_data, test_data = train_test_split(self.img_labels, test_size=0.1, random_state=42)
        train_data, val_data = train_test_split(train_data, test_size=0.15, random_state=42)

        if self.subset == 'train':
            self.data = train_data
        elif self.subset == 'val':
            self.data = val_data
        elif self.subset == 'test':
            self.data = test_data
        else:
            raise ValueError("Subset must be one of 'train', 'val', or 'test'.")

        # Optionally balance classes in the training set
        if self.balance_class and self.subset == 'train':
            self.data = self.balance_classes(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data.iloc[idx, 1])
        image = read_image(img_path).to(DEVICE)
        label = self.data.iloc[idx, 2:14].values.astype(np.float32) # Convert Pandas Series to NumPy array
        label = torch.tensor(label, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor
        # # Resize to target size, done here because all cropped faces are of different sizes
        image = resize_transform(image)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
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
    # Load dataset
    train_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='train',balance_class=True)
    print(f'training dataset size: {len(train_dataset)}.')
    val_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='val')
    print(f'validation dataset size: {len(val_dataset)}.')
    test_dataset = CustomImageDataset(LABEL_FILE, IMAGE_FOLDER,subset='test')
    print(f'test dataset size: {len(test_dataset)}.')
    dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    batch, label = dataloader.__iter__().__next__()
        
    grid = make_grid(batch, 8, padding=4).permute(1,2,0)

    plt.imshow(grid.cpu())
    plt.show()
