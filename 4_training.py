import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
from sklearn.metrics import confusion_matrix
import numpy as np
import rasterio
import time
import tqdm
from torch.utils.tensorboard import SummaryWriter
import random

def load_tif_dataset_and_masks(tif_dir, mask_dir, scene_names):
    """
    Loads TIFF images and corresponding masks into NumPy arrays, 
    excluding images where the entire mask is non-mangrove (all zeros).
    
    Parameters:
        tif_dir (str): Directory containing the TIFF files (training images).
        mask_dir (str): Directory containing the mask files.
        scene_names (list): List of scene name prefixes to filter relevant files.

    Returns:
        tuple: (images, masks) NumPy arrays.
    """
    image_list = []
    mask_list = []
    removed_count = 0

    for scene in scene_names:
        for filename in os.listdir(tif_dir):
            if filename.startswith(scene) and filename.endswith(".tif"):
                image_path = os.path.join(tif_dir, filename)
                mask_path = os.path.join(mask_dir, filename)

                with rasterio.open(image_path) as src:
                    img = src.read()  # Shape: (6, 256, 256)

                with rasterio.open(mask_path) as src:
                    mask = src.read(1)  # Shape: (256, 256), single channel
                # mask[mask == 255] = -1
                    if np.any(mask == 255) or np.all(mask == 0):  
                        removed_count += 1  # Count removed subsets
                        continue  # Skip this image

                    # Otherwise, add the image and mask to the dataset
                    image_list.append(img)
                    mask_list.append(mask[np.newaxis, :, :])  # Add channel dimension

    if image_list and mask_list:
        images = np.stack(image_list, axis=0).astype(np.float32)  # (num_samples, 6, 256, 256)
        masks = np.stack(mask_list, axis=0).astype(np.float32)  # (num_samples, 1, 256, 256)
        print(f"Loaded dataset - Images: {images.shape}, Masks: {masks.shape}")
        print(f"Removed {removed_count} subsets with no mangrove pixels.")
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
def compute_iou(y_pred, y_true):
    """
    Compute IoU for the Mangrove class.
    """
    y_pred = (y_pred > 0.5).astype(np.uint8)  # Convert logits to binary mask
    y_true = y_true.astype(np.uint8)

    intersection = np.logical_and(y_pred, y_true).sum()
    union = np.logical_or(y_pred, y_true).sum()

    iou = intersection / union if union > 0 else 0.0
    return iou

def compute_confusion_matrix(y_pred, y_true):
    """
    Compute the confusion matrix for a single batch.
    """
    y_pred = (y_pred > 0.5).astype(np.uint8).flatten()
    y_true = y_true.astype(np.uint8).flatten()

    return confusion_matrix(y_true, y_pred, labels=[0, 1])

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    iou_scores = []

    for images, masks in tqdm.tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        # Convert outputs and masks to numpy
        outputs_np = outputs.cpu().detach().numpy()
        masks_np = masks.cpu().numpy()

        # Compute IoU for each sample in batch
        for i in range(outputs_np.shape[0]):
            iou_scores.append(compute_iou(outputs_np[i, 0], masks_np[i, 0]))

    avg_iou = np.mean(iou_scores)
    return running_loss / len(dataloader.dataset), avg_iou


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    iou_scores = []
    all_conf_matrix = np.zeros((2, 2))

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            # Convert outputs and masks to numpy
            outputs_np = outputs.cpu().numpy()
            masks_np = masks.cpu().numpy()

            # Compute IoU for Mangrove class
            for i in range(outputs_np.shape[0]):  # Loop over batch
                iou_scores.append(compute_iou(outputs_np[i, 0], masks_np[i, 0]))
                conf_matrix = compute_confusion_matrix(outputs_np[i, 0], masks_np[i, 0])
                all_conf_matrix += conf_matrix

    epoch_loss = running_loss / len(dataloader.dataset)
    avg_iou = np.mean(iou_scores)

    # print("\nConfusion Matrix:")
    print(all_conf_matrix)
    print(f"IoU for Mangrove Class: {avg_iou:.4f}\n")

    return epoch_loss, avg_iou

def main():
    start = time.time()
    tif_dir = "./3_imagery_subsets"  # Update with actual path
    mask_dir = "./3_mask_subsets"  # Update with actual path
    scene_list_file = "scene_list_display_id_test4.txt"
    with open(scene_list_file, "r") as f:
        scene_names = [line.strip() for line in f.readlines() if line.strip()]

    split_ratio = 0.8  # Adjust as needed

    # Shuffle the scenes to ensure randomness (but with a fixed seed for reproducibility)
    random.shuffle(scene_names)

    # Split the data
    split_index = int(len(scene_names) * split_ratio)
    train_scenes = scene_names[:split_index]
    val_scenes = scene_names[split_index:]

    # Print results
    print(f"Total scenes: {len(scene_names)}")
    print(f"Training scenes ({len(train_scenes)}): {train_scenes}")
    print(f"Validation scenes ({len(val_scenes)}): {val_scenes}")
    print(val_scenes)
    train_images, train_masks = load_tif_dataset_and_masks(tif_dir, mask_dir, train_scenes)
    val_images, val_masks = load_tif_dataset_and_masks(tif_dir, mask_dir, val_scenes)

    if train_images is None or train_masks is None or val_images is None or val_masks is None:
        print("Error: No training or validation data found. Exiting...")
        return

    train_dataset = SegmentationDataset(train_images, train_masks)
    val_dataset = SegmentationDataset(val_images, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    loadtime = time.time() - start
    print(f'Loading time: {loadtime:.2f} seconds')
    start = time.time()

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=6,
        classes=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 10

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir="runs/mangrove_segmentation")

    for epoch in range(num_epochs):
        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch + 1)
        writer.add_scalar("IoU/Train", train_iou, epoch + 1)
        writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
        writer.add_scalar("IoU/Validation", val_iou, epoch + 1)

    torch.save(model.state_dict(), "./model/unet_resnet34.pth")
    writer.close()  # Close TensorBoard writer

    loadtime=time.time()-start
    print(f'training time is {loadtime:.2f} seconds')
    start=time.time()


if __name__ == "__main__":
    main()
