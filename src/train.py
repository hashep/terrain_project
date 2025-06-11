import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from dataset import DroneDataset
from model import get_model
from utils import set_seed, compute_iou
from PIL import Image

def main():
    set_seed(42)

    image_dir = "C:/Anaconda/terrain_project/data/images"
    mask_dir = "C:/Anaconda/terrain_project/data/masks"
    num_classes = 9
    batch_size = 8
    epochs = 50
    lr = 0.0003
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    mask_transforms = T.Compose([
        T.Resize((256, 256), interpolation=Image.NEAREST),
    ])

    dataset = DroneDataset(image_dir, mask_dir, image_transforms=image_transforms, mask_transforms=mask_transforms)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = get_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss / len(train_loader):.4f}")

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
