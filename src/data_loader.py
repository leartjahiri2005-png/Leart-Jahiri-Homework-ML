import os
import glob
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import torchvision.transforms.functional as F

IMG_SIZE =128 
class SODDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

        self.img_transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
        ])

        self.mask_transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
        ])
        self.color_jitter = T.ColorJitter(brightness=0.2)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.augment:
            if random.random() < 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)
            img = self.color_jitter(img)

        img = self.img_transform(img)  
        mask = self.mask_transform(mask) 

        mask = (mask > 0.5).float()

        return img, mask

def load_dataset_paths():
    img_dir = "data/raw/ecssd/images"
    mask_dir = "data/raw/ecssd/masks"

    image_paths = sorted(
        glob.glob(os.path.join(img_dir, "*.jpg")) +
        glob.glob(os.path.join(img_dir, "*.jpeg")) +
        glob.glob(os.path.join(img_dir, "*.png"))
    )

    mask_paths = sorted(
        glob.glob(os.path.join(mask_dir, "*.png")) +
        glob.glob(os.path.join(mask_dir, "*.jpg")) +
        glob.glob(os.path.join(mask_dir, "*.jpeg"))
    )

    if len(image_paths) == 0:
        raise RuntimeError("No images found in data/raw/ecssd/images")

    if len(mask_paths) == 0:
        raise RuntimeError("No masks found in data/raw/ecssd/masks")

    if len(image_paths) != len(mask_paths):
        print("The number of images and masks does not match!")

    return image_paths, mask_paths

def get_dataloaders(batch_size=8):
    image_paths, mask_paths = load_dataset_paths()

    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, mask_paths, test_size=0.30, shuffle=True, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, shuffle=True, random_state=42
    )

    train_dataset = SODDataset(X_train, y_train, augment=True)
    val_dataset   = SODDataset(X_val,   y_val,   augment=False)
    test_dataset  = SODDataset(X_test,  y_test,  augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
    print("Train samples:", len(train_loader.dataset))
    print("Val samples:",   len(val_loader.dataset))
    print("Test samples:",  len(test_loader.dataset))