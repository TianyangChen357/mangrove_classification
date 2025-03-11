import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import rasterio
import random
import sys
from torch.utils.tensorboard import SummaryWriter

"""
Script Name: 4_training.py
Description:
This script trains a U-Net model on Landsat imagery subsets and corresponding mangrove masks.
It logs Loss, IoU, Precision, Recall, and F1-score for both Training and Validation.

Usage:
    python 4_training.py scene_list_display_id.txt <learning_rate> <num_epochs> <batch_size>

Example:
    python 4_training.py scene_list_display_id.txt 0.0001 1000 16

Requirements:
- Pre-processed image subsets in `3_imagery_subsets/`
- Pre-processed mask subsets in `3_mask_subsets/`
- A text file (`scene_list_display_id.txt`) with Landsat scene names (one per line)

Output:
- **Only the best model checkpoint** is saved in `4_model/`
- **Training logs** are stored in `4_runs/` for TensorBoard visualization.
"""

# Set a fixed random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Ensures reproducibility

def load_tif_dataset_and_masks(tif_dir, mask_dir, scene_names):
    """Loads TIFF images and corresponding masks into NumPy arrays."""
    image_list = []
    mask_list = []

    for scene in scene_names:
        for filename in os.listdir(tif_dir):
            if filename.startswith(scene) and filename.endswith(".tif"):
                image_path = os.path.join(tif_dir, filename)
                mask_path = os.path.join(mask_dir, filename)

                with rasterio.open(image_path) as src:
                    img = src.read()  # Shape: (6, 256, 256)

                with rasterio.open(mask_path) as src:
                    mask = src.read(1)  # Shape: (256, 256), single channel

                image_list.append(img)
                mask_list.append(mask[np.newaxis, :, :])  # Add channel dimension

    if image_list and mask_list:
        images = np.stack(image_list, axis=0).astype(np.float32)  # (num_samples, 6, 256, 256)
        masks = np.stack(mask_list, axis=0).astype(np.float32)  # (num_samples, 1, 256, 256)
        print(f"Loaded dataset - Images: {images.shape}, Masks: {masks.shape}")
        return images, masks
    else:
        print("No valid images or masks found.")
        return None, None

class SegmentationDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        return image, mask

def compute_metrics(y_pred, y_true):
    """Computes IoU, Precision, Recall, and F1-score for the Mangrove class."""
    y_pred = (y_pred > 0.5).astype(np.uint8).flatten()
    y_true = y_true.astype(np.uint8).flatten()

    intersection = np.logical_and(y_pred, y_true).sum()
    union = np.logical_or(y_pred, y_true).sum()
    iou = intersection / union if union > 0 else 0.0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return iou, precision, recall, f1

def train_or_validate(model, dataloader, optimizer, criterion, device, train=True):
    """Performs one epoch of training or validation."""
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    iou_scores, precisions, recalls, f1_scores = [], [], [], []

    with torch.set_grad_enabled(train):
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, masks)

            if train:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)

            outputs_np = outputs.cpu().detach().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(outputs_np.shape[0]):
                iou, precision, recall, f1 = compute_metrics(outputs_np[i, 0], masks_np[i, 0])
                iou_scores.append(iou)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

    return (running_loss / len(dataloader.dataset),
            np.mean(iou_scores), np.mean(precisions), np.mean(recalls), np.mean(f1_scores))

def main():
    # Read command-line arguments
    scene_list_file = sys.argv[1]
    learning_rate = float(sys.argv[2])
    num_epochs = int(sys.argv[3])
    batch_size = int(sys.argv[4])

    tif_dir, mask_dir = "./3_imagery_subsets", "./3_mask_subsets"

    with open(scene_list_file, "r") as f:
        scene_names = [line.strip() for line in f.readlines() if line.strip()]

    # Ensure reproducibility in dataset split
    random.shuffle(scene_names)

    split_index = int(len(scene_names) * 0.8)
    train_scenes, val_scenes = scene_names[:split_index], scene_names[split_index:]

    train_images, train_masks = load_tif_dataset_and_masks(tif_dir, mask_dir, train_scenes)
    val_images, val_masks = load_tif_dataset_and_masks(tif_dir, mask_dir, val_scenes)

    if train_images is None or val_images is None:
        print("Error: No training or validation data found. Exiting...")
        return

    train_loader = DataLoader(SegmentationDataset(train_images, train_masks), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SegmentationDataset(val_images, val_masks), batch_size=batch_size, shuffle=False)

    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=6, classes=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs("4_model", exist_ok=True)
    best_val_iou = 0.0
    model_filename = f"unet_res34_lr{learning_rate}_epochs{num_epochs}_bs{batch_size}.pth"
    model_path = os.path.join("4_model", model_filename)

    for epoch in range(num_epochs):
        train_loss, train_iou, _, _, _ = train_or_validate(model, train_loader, optimizer, criterion, device, train=True)
        val_loss, val_iou, _, _, _ = train_or_validate(model, val_loader, optimizer, criterion, device, train=False)

        

        # Save only the best model based on validation IoU
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), model_path)
            # print(f"âœ… New best model saved: {model_path}")
            print(f"new best Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train IoU={train_iou:.4f}, Val IoU={val_iou:.4f}")

    # print(f"Training complete! Best model saved at: {model_path}")

if __name__ == "__main__":
    main()
