import os
import sys
import argparse
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image

# Add networks to path
sys.path.append('networks')
from networks.net_factory import net_factory
from dataset import get_dataloaders
from metrics import MetricsCalculator, evaluate_model


def save_predictions(model, dataloader, device, save_dir, dataset_name):
    """
    Save model predictions as images
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Saving predictions")):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            filenames = batch['filename']
            
            outputs = model(images)
            
            # Convert outputs to predictions
            if outputs.dim() == 4 and outputs.shape[1] > 1:  # Multi-class
                preds = torch.softmax(outputs, dim=1)
                preds = torch.argmax(preds, dim=1)  # (B, H, W)
            else:  # Binary
                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                if preds.dim() == 4:
                    preds = preds.squeeze(1)  # (B, H, W)
            
            # Save each prediction in the batch
            for i in range(len(filenames)):
                filename = filenames[i]
                pred = preds[i].cpu().numpy().astype(np.uint8) * 255
                mask = masks[i].cpu().numpy().astype(np.uint8) * 255
                
                # Save prediction
                pred_path = os.path.join(save_dir, f"pred_{filename}")
                cv2.imwrite(pred_path, pred)
                
                # Save ground truth for comparison
                gt_path = os.path.join(save_dir, f"gt_{filename}")
                cv2.imwrite(gt_path, mask)


def evaluate_single_model(model_path, model_name, dataset_name, data_root, 
                         img_size=512, batch_size=8, num_classes=2, save_predictions_flag=False):
    """
    Evaluate a single model on a dataset
    """
    print(f"\nEvaluating {model_name} on {dataset_name}")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=data_root,
        dataset_name=dataset_name,
        batch_size=batch_size,
        img_size=img_size,
        num_workers=4
    )
    
    # Create model
    in_chns = 3  # RGB images
    model = net_factory(
        net_type=model_name,
        in_chns=in_chns,
        class_num=num_classes,
        device=device
    )
    
    if model is None:
        raise ValueError(f"Model {model_name} not found!")
    
    # Load checkpoint
    print(f"Loading checkpoint: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, val_loader, device, num_classes)
    
    print("Validation Results:")
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Dice: {val_metrics['dice']:.4f}")
    print(f"IoU: {val_metrics['iou']:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, num_classes)
    
    print("Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Dice: {test_metrics['dice']:.4f}")
    print(f"IoU: {test_metrics['iou']:.4f}")
    
    # Save predictions if requested
    if save_predictions_flag:
        save_dir = f"predictions/{model_name}_{dataset_name}"
        print(f"\nSaving predictions to {save_dir}...")
        save_predictions(model, test_loader, device, save_dir, dataset_name)
    
    return {
        'model': model_name,
        'dataset': dataset_name,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }


def compare_models(model_configs, data_root, img_size=512, batch_size=8, 
                  num_classes=2, save_predictions_flag=False):
    """
    Compare multiple models on multiple datasets
    
    Args:
        model_configs: List of dictionaries with 'model_name', 'model_path', 'dataset_name'
    """
    results = []
    
    for config in model_configs:
        try:
            result = evaluate_single_model(
                model_path=config['model_path'],
                model_name=config['model_name'],
                dataset_name=config['dataset_name'],
                data_root=data_root,
                img_size=img_size,
                batch_size=batch_size,
                num_classes=num_classes,
                save_predictions_flag=save_predictions_flag
            )
            results.append(result)
        except Exception as e:
            print(f"Error evaluating {config['model_name']} on {config['dataset_name']}: {e}")
            continue
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<15} {'Dataset':<10} {'Val Dice':<10} {'Test Dice':<10} {'Test Acc':<10} {'Test IoU':<10}")
    print("-"*80)
    
    for result in results:
        val_dice = result['val_metrics']['dice']
        test_dice = result['test_metrics']['dice']
        test_acc = result['test_metrics']['accuracy']
        test_iou = result['test_metrics']['iou']
        
        print(f"{result['model']:<15} {result['dataset']:<10} {val_dice:<10.4f} "
              f"{test_dice:<10.4f} {test_acc:<10.4f} {test_iou:<10.4f}")
    
    # Find best models
    print("\n" + "="*50)
    print("BEST MODELS (by Test Dice Score)")
    print("="*50)
    
    # Group by dataset
    datasets = list(set([r['dataset'] for r in results]))
    for dataset in datasets:
        dataset_results = [r for r in results if r['dataset'] == dataset]
        if dataset_results:
            best_result = max(dataset_results, key=lambda x: x['test_metrics']['dice'])
            print(f"{dataset}: {best_result['model']} (Dice: {best_result['test_metrics']['dice']:.4f})")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Medical Image Segmentation Evaluation')
    parser.add_argument('--model_dir', type=str, required=True,
                       help='Directory containing model checkpoints')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory of the datasets')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['swinumamba', 'lkmunet', 'segmamba', 'emnet'],
                       help='Models to evaluate')
    parser.add_argument('--datasets', type=str, nargs='+',
                       default=['TN3K', 'DDTI'],
                       help='Datasets to evaluate on')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--img_size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save prediction images')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json',
                       help='Output file for results')
    
    args = parser.parse_args()
    
    # Create model configurations
    model_configs = []
    for model_name in args.models:
        for dataset_name in args.datasets:
            # Look for best model checkpoint
            pattern = f"{model_name}_{dataset_name}_*_best.pth"
            import glob
            checkpoint_files = glob.glob(os.path.join(args.model_dir, pattern))
            
            if checkpoint_files:
                # Use the most recent checkpoint
                checkpoint_path = max(checkpoint_files, key=os.path.getctime)
                model_configs.append({
                    'model_name': model_name,
                    'dataset_name': dataset_name,
                    'model_path': checkpoint_path
                })
                print(f"Found checkpoint for {model_name} on {dataset_name}: {checkpoint_path}")
            else:
                print(f"Warning: No checkpoint found for {model_name} on {dataset_name}")
    
    if not model_configs:
        print("No model checkpoints found!")
        return
    
    # Run evaluation
    results = compare_models(
        model_configs=model_configs,
        data_root=args.data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        save_predictions_flag=args.save_predictions
    )
    
    # Save results to file
    output_data = {
        'evaluation_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': vars(args),
        'results': results
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output_file}")


if __name__ == '__main__':
    main()
