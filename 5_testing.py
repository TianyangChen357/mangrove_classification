import os
import torch
import numpy as np
import rasterio
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score, accuracy_score, confusion_matrix
import segmentation_models_pytorch as smp
from collections import defaultdict


def load_test_data(tif_dir, mask_dir, scene_names):
    """
    Loads test images and corresponding masks.
    """
    image_list, mask_list, filenames = [], [], []
    
    for scene in scene_names:
        for filename in os.listdir(tif_dir):
            if filename.startswith(scene) and filename.endswith(".tif"):
                image_path = os.path.join(tif_dir, filename)
                mask_path = os.path.join(mask_dir, filename)
                
                with rasterio.open(image_path) as src:
                    img = src.read()  # Shape: (6, 256, 256)
                with rasterio.open(mask_path) as src:
                    mask = src.read(1)  # Shape: (256, 256)
                    meta = src.meta.copy()
                
                image_list.append(img)
                mask_list.append(mask[np.newaxis, :, :])  # Add channel dimension
                filenames.append((scene, filename, meta))
    
    images = np.stack(image_list, axis=0).astype(np.float32)
    masks = np.stack(mask_list, axis=0).astype(np.float32)
    return images, masks, filenames


class TestDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        return image, mask


def run_inference(model, dataloader, device, output_dir, filenames):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    scene_conf_matrices = defaultdict(lambda: np.zeros((2, 2), dtype=int))
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(dataloader):
            images = images.to(device)
            outputs = model(images)
            preds = (outputs.cpu().numpy() > 0.5).astype(np.uint8)
            
            for j in range(preds.shape[0]):
                scene, filename, meta = filenames[i * dataloader.batch_size + j]
                output_path = os.path.join(output_dir, filename)
                meta.update({"count": 1, "dtype": "uint8"})
                
                # Convert binary predictions to TP (3), FP (2), FN (1), TN (0)
                pred_mask = preds[j, 0]
                true_mask = masks[j, 0].cpu().numpy()
                confusion_map = np.zeros_like(pred_mask, dtype=np.uint8)
                confusion_map[(pred_mask == 1) & (true_mask == 1)] = 3  # TP
                confusion_map[(pred_mask == 1) & (true_mask == 0)] = 2  # FP
                confusion_map[(pred_mask == 0) & (true_mask == 1)] = 1  # FN
                confusion_map[(pred_mask == 0) & (true_mask == 0)] = 0  # TN
                
                with rasterio.open(output_path, 'w', **meta) as dst:
                    dst.write(confusion_map, 1)
                
                # Update confusion matrix for the scene
                conf_matrix = confusion_matrix(true_mask.flatten(), pred_mask.flatten(), labels=[0, 1])
                scene_conf_matrices[scene] += conf_matrix
    
    # Compute and print metrics per scene
    for scene, conf_matrix in scene_conf_matrices.items():
        tn, fp, fn, tp = conf_matrix.ravel()
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        print(f"Scene {scene}: IoU={iou:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-score={f1:.4f}")


def main():
    model_dir = './model/initial_train.pth'
    test_tif_dir = "./3_imagery_subsets"  # Update with actual path
    test_mask_dir = "./3_mask_subsets"  # Update with actual path
    test_scenes = [
        "LT05_L2SP_107070_20070629_20200830_02_T1",
    ]  # Update with actual test scenes

    output_dir = "./test_predictions"
    
    images, masks, filenames = load_test_data(test_tif_dir, test_mask_dir, test_scenes)
    test_dataset = TestDataset(images, masks)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=6, classes=1)
    model.load_state_dict(torch.load(model_dir, map_location=device))
    model.to(device)
    
    run_inference(model, test_loader, device, output_dir, filenames)


if __name__ == "__main__":
    main()
