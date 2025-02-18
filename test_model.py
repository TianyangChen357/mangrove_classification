import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Define a custom dataset class for image segmentation
class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask

# Define the training and validation loops
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in dataloader:
        images = images.to(device, dtype=torch.float32)
        masks = masks.to(device, dtype=torch.float32)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device, dtype=torch.float32)
            masks = masks.to(device, dtype=torch.float32)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main():
    # Paths to your training and validation data
    # Replace these with actual data paths or preprocessing pipelines
    train_images = np.random.rand(100, 6, 256, 256).astype(np.float32)  # Example data
    train_masks = np.random.randint(0, 2, (100, 1, 256, 256)).astype(np.float32)  # Example masks
    val_images = np.random.rand(20, 6, 256, 256).astype(np.float32)
    val_masks = np.random.randint(0, 2, (20, 1, 256, 256)).astype(np.float32)

    # Define transformations (optional)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Create datasets and dataloaders
    train_dataset = SegmentationDataset(train_images, train_masks)
    val_dataset = SegmentationDataset(val_images, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Define the model
    model = smp.Unet(
        encoder_name="resnet34",        # Use ResNet-34 as the encoder
        encoder_weights="imagenet",    # Pretrained on ImageNet
        in_channels=6,                 # Number of input channels (e.g., RGB)
        classes=1                      # Binary segmentation
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer, loss function, and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Training parameters
    num_epochs = 20

    # Training loop
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save the model
    torch.save(model.state_dict(), "unet_resnet34.pth")

if __name__ == "__main__":
    main()
