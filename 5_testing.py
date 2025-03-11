import os
import torch
import numpy as np
import rasterio
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import segmentation_models_pytorch as smp
from collections import defaultdict
import sys
import random
import csv
from rasterio.windows import Window
from rasterio.merge import merge
from rasterio.io import MemoryFile

"""
Script Name: 5_testing.py
Description:
This script evaluates a trained U-Net model on Landsat test imagery. It handles:
- Image tiling (256x256 subsets)
- Excluding NoData values
- Merging predictions back into full scenes
- Computing evaluation metrics per scene

Outputs:
- Merged scene predictions (GeoTIFFs) stored in `./6_merged_predictions/`
- Evaluation metrics saved in `scene_metrics.csv`
"""

def create_raster_memory(array, transform, meta):
    """Creates an in-memory raster dataset from a NumPy array."""
    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        transform=transform,
        crs=meta["crs"],
    ) as dataset:
        dataset.write(array, 1)
    return memfile.open()

def load_test_data(tif_dir, mask_dir, scene_names, subset_size=256):
    """Loads and tiles Landsat images and masks, ensuring valid inference data."""
    image_dict, mask_dict, meta_dict = {}, {}, {}

    for scene in scene_names:
        image_path = os.path.join(tif_dir, f"{scene}_merged.tif")
        mask_path = os.path.join(mask_dir, f"{scene}_mask.TIF")

        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print(f"⚠️ Skipping {scene}: Missing imagery or mask.")
            continue

        with rasterio.open(image_path) as src_imagery, rasterio.open(mask_path) as src_mask:
            width, height = src_imagery.width, src_imagery.height
            count, nodata = src_imagery.count, src_imagery.nodata
            meta = src_imagery.meta.copy()

            num_rows = (height + subset_size - 1) // subset_size
            num_cols = (width + subset_size - 1) // subset_size

            image_dict[scene] = []
            mask_dict[scene] = []
            meta_dict[scene] = meta

            for row in range(num_rows):
                for col in range(num_cols):
                    row_start, col_start = row * subset_size, col * subset_size
                    row_end, col_end = min((row + 1) * subset_size, height), min((col + 1) * subset_size, width)

                    window = Window(col_start, row_start, col_end - col_start, row_end - row_start)

                    imagery_data = src_imagery.read(window=window, boundless=True, fill_value=0)
                    mask_data = src_mask.read(1, window=window, boundless=True, fill_value=0)

                    # Skip subsets with NoData values
                    if nodata is not None and np.any(imagery_data == nodata):
                        continue

                    # Apply padding
                    padded_imagery = np.zeros((count, subset_size, subset_size), dtype=imagery_data.dtype)
                    padded_imagery[:, :imagery_data.shape[1], :imagery_data.shape[2]] = imagery_data

                    padded_mask = np.zeros((subset_size, subset_size), dtype=np.uint8)
                    padded_mask[:mask_data.shape[0], :mask_data.shape[1]] = mask_data

                    image_dict[scene].append((padded_imagery, padded_mask, row, col))

    return image_dict, mask_dict, meta_dict

class TestDataset(Dataset):
    """PyTorch dataset for test images and masks."""
    def __init__(self, image_list):
        self.image_list = image_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image, mask, row, col = self.image_list[idx]
        return (
            torch.tensor(image, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.float32),
            row,
            col
        )

def run_inference(model, dataloader, device, meta, scene):
    """Runs inference, merges results, and saves full raster."""
    model.eval()
    scene_predictions = []
    scene_conf_matrix = np.zeros((2, 2), dtype=int)

    with torch.no_grad():
        for images, masks, rows, cols in dataloader:
            images = images.to(device)
            outputs = model(images)

            preds = (outputs.cpu().numpy() > 0.5).astype(np.uint8)

            for j in range(preds.shape[0]):
                row, col = rows[j].item(), cols[j].item()
                pred_mask = preds[j, 0]
                true_mask = masks[j].cpu().numpy().squeeze()

                valid_pixels = (true_mask != 255)

                true_mask_valid = true_mask[valid_pixels].flatten()
                pred_mask_valid = pred_mask[valid_pixels].flatten()

                if len(true_mask_valid) > 0 and len(pred_mask_valid) > 0:
                    conf_matrix = confusion_matrix(true_mask_valid, pred_mask_valid, labels=[0, 1])
                    scene_conf_matrix += conf_matrix

                window = Window(col * 256, row * 256, 256, 256)
                transform = rasterio.windows.transform(window, meta["transform"])
                scene_predictions.append((pred_mask, transform))

    # Convert NumPy predictions to raster datasets
    raster_datasets = [create_raster_memory(pred, transform, meta) for pred, transform in scene_predictions]

    # Merge tiled predictions into a full raster
    merged_pred, merged_transform = merge(raster_datasets)

    # Save merged output
    output_dir = "./6_merged_predictions"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{scene}_predicted.tif")

    meta.update({
        "driver": "GTiff",
        "height": merged_pred.shape[1],
        "width": merged_pred.shape[2],
        "transform": merged_transform,
        "count": 1,
        "dtype": "uint8"
    })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(merged_pred[0], 1)

    print(f"✅ Merged prediction saved: {output_path}")

    # Compute scene metrics
    tn, fp, fn, tp = scene_conf_matrix.ravel()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return [scene, round(iou, 4), round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4), tn, fp, fn, tp]

def main():
    model_dir = os.path.join('./4_model', "unet_res34_lr0.0001_epochs500_bs32.pth")
    scene_list_file="scene_list_display_id.txt"
    # model_dir=sys.argv[2]
    # scene_list_file=sys.argv[1]
    test_tif_dir = "./1_imagery_merge"
    test_mask_dir = "./2_Mask"
    
    with open(scene_list_file, "r") as f:
        scene_names = [line.strip() for line in f.readlines() if line.strip()]

    image_dict, mask_dict, meta_dict = load_test_data(test_tif_dir, test_mask_dir, scene_names)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=6, classes=1)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.to(device)

    csv_filename = "scene_metrics.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)

        # ✅ Write correct header
        writer.writerow(["Scene", "IoU", "Accuracy", "Precision", "Recall", "F1-score", "TN", "FP", "FN", "TP"])

        for scene, image_list in image_dict.items():
            test_dataset = TestDataset(image_list)
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

            # ✅ Get metrics from run_inference()
            metrics = run_inference(model, test_loader, device, meta_dict[scene], scene)

            # ✅ Write the metrics for each scene
            writer.writerow(metrics)


if __name__ == "__main__":
    main()
