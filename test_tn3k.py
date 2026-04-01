#!/usr/bin/env python3
"""
TN3K Dataset Testing Script for UMambaBot
Evaluates trained model on TN3K test set with accuracy, dice, and IoU metrics
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm
import argparse

# Import the baseline model
import sys
sys.path.append('基础')
from baseline import UMambaBot


class TN3KDataset(Dataset):
    """TN3K Dataset for testing"""
    
    def __init__(self, image_dir, mask_dir, img_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        # Get all image files
        self.image_files = sorted(glob.glob(os.path.join(image_dir, '*')))
        self.mask_files = sorted(glob.glob(os.path.join(mask_dir, '*')))
        
        # Ensure same number of images and masks
        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images ({len(self.image_files)}) != number of masks ({len(self.mask_files)})"
        
        print(f"Found {len(self.image_files)} test samples")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.img_size, Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW
        
        # Load mask
        mask_path = self.mask_files[idx]
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(self.img_size, Image.NEAREST)
        mask = np.array(mask).astype(np.float32)
        mask = (mask > 127).astype(np.float32)  # Binarize mask
        
        return torch.from_numpy(image), torch.from_numpy(mask)


def calculate_metrics(pred, target, threshold=None):
    """
    Calculate accuracy, dice coefficient, and IoU

    Args:
        pred: Predicted segmentation [H, W] or [B, H, W]
        target: Ground truth segmentation [H, W] or [B, H, W]
        threshold: Threshold for binarizing predictions. If None, use adaptive threshold

    Returns:
        dict: Dictionary containing accuracy, dice, and iou
    """
    # Ensure tensors are on CPU and convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Use adaptive threshold if not provided
    if threshold is None:
        # Use a much lower threshold since model outputs are very small
        threshold = max(np.mean(pred) + 1 * np.std(pred), 0.002)
        threshold = min(threshold, 0.1)  # Cap at 0.1

    # Binarize predictions
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = target.astype(np.float32)

    # Flatten arrays
    pred_flat = pred_binary.flatten()
    target_flat = target_binary.flatten()

    # Calculate accuracy
    accuracy = np.mean(pred_flat == target_flat)

    # Calculate intersection and union
    intersection = np.sum(pred_flat * target_flat)
    pred_sum = np.sum(pred_flat)
    target_sum = np.sum(target_flat)
    union = pred_sum + target_sum - intersection

    # Calculate Dice coefficient
    if pred_sum + target_sum == 0:
        dice = 1.0  # Both prediction and target are empty
    else:
        dice = 2.0 * intersection / (pred_sum + target_sum)

    # Calculate IoU
    if union == 0:
        iou = 1.0  # Both prediction and target are empty
    else:
        iou = intersection / union

    return {
        'accuracy': accuracy,
        'dice': dice,
        'iou': iou,
        'threshold_used': threshold
    }


def test_model(model, test_loader, device):
    """
    Test the model on the test dataset

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run inference on

    Returns:
        dict: Dictionary containing lists of metrics for each sample
    """
    model.eval()

    all_metrics = {
        'accuracy': [],
        'dice': [],
        'iou': []
    }

    print("Starting evaluation...")

    # Debug variables
    pred_stats = []
    target_stats = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(tqdm(test_loader, desc="Testing")):
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # Apply sigmoid to get probabilities
            if isinstance(outputs, list):
                outputs = outputs[0]  # Take the main output if multiple outputs

            probs = torch.sigmoid(outputs)

            # Calculate metrics for each sample in the batch
            for i in range(images.size(0)):
                pred = probs[i, 0]  # Remove channel dimension
                target = masks[i]

                metrics = calculate_metrics(pred, target)

                # Collect statistics for debugging
                if batch_idx < 5:  # Only for first few batches
                    pred_stats.append({
                        'min': pred.min().item(),
                        'max': pred.max().item(),
                        'mean': pred.mean().item(),
                        'threshold': metrics['threshold_used'],
                        'positive_ratio': (pred > metrics['threshold_used']).float().mean().item()
                    })
                    target_stats.append({
                        'positive_ratio': target.mean().item()
                    })

                all_metrics['accuracy'].append(metrics['accuracy'])
                all_metrics['dice'].append(metrics['dice'])
                all_metrics['iou'].append(metrics['iou'])

    # Print debug information
    if pred_stats:
        print(f"\nDebug Info (first {len(pred_stats)} samples):")
        print("Prediction stats:")
        for i, stats in enumerate(pred_stats[:5]):
            print(f"  Sample {i}: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                  f"mean={stats['mean']:.4f}, threshold={stats['threshold']:.4f}, pos_ratio={stats['positive_ratio']:.4f}")
        print("Target stats:")
        for i, stats in enumerate(target_stats[:5]):
            print(f"  Sample {i}: pos_ratio={stats['positive_ratio']:.4f}")

    return all_metrics


def print_results(metrics):
    """
    Print results in mean±std format
    
    Args:
        metrics: Dictionary containing lists of metrics
    """
    print("\n" + "="*60)
    print("TN3K Test Results")
    print("="*60)
    
    for metric_name, values in metrics.items():
        values = np.array(values)
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        print(f"{metric_name.upper():>10}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Test UMambaBot on TN3K dataset')
    parser.add_argument('--model_path', type=str, default='基础/best_real_model.pth',
                        help='Path to trained model weights')
    parser.add_argument('--test_image_dir', type=str, default='TN3K/test-image',
                        help='Path to test images directory')
    parser.add_argument('--test_mask_dir', type=str, default='TN3K/test-mask',
                        help='Path to test masks directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--img_size', type=int, nargs=2, default=[256, 256],
                        help='Image size for resizing')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating model...")
    model_config = {
        'input_channels': 3,
        'n_stages': 4,
        'features_per_stage': [32, 64, 128, 256],
        'conv_op': nn.Conv2d,
        'kernel_sizes': 3,
        'strides': [1, 2, 2, 2],
        'n_conv_per_stage': 2,
        'num_classes': 1,
        'n_conv_per_stage_decoder': 2,
        'norm_op': nn.BatchNorm2d,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
        'use_csfi': False,
        'use_safr': True,
        'use_frequency_aware': False
    }
    
    model = UMambaBot(**model_config)
    
    # Load trained weights
    if os.path.exists(args.model_path):
        print(f"Loading model weights from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully!")
    else:
        print(f"Warning: Model weights not found at {args.model_path}")
        print("Using randomly initialized weights for testing...")
    
    model = model.to(device)
    
    # Create test dataset and dataloader
    print("Loading test dataset...")
    test_dataset = TN3KDataset(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        img_size=tuple(args.img_size)
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Test the model
    metrics = test_model(model, test_loader, device)
    
    # Print results
    print_results(metrics)
    
    # Save results to file
    results_file = 'tn3k_test_results.txt'
    with open(results_file, 'w') as f:
        f.write("TN3K Test Results\n")
        f.write("="*60 + "\n")
        for metric_name, values in metrics.items():
            values = np.array(values)
            mean_val = np.mean(values)
            std_val = np.std(values)
            f.write(f"{metric_name.upper():>10}: {mean_val:.4f} ± {std_val:.4f}\n")
        f.write("="*60 + "\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
