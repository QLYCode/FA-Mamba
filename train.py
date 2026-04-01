import os
import sys
import argparse
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
# from torch.utils.tensorboard import SummaryWriter  # 移除tensorboard依赖
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler

# Add networks to path
sys.path.append('networks')
from dataset import get_dataloaders
from metrics import MetricsCalculator, evaluate_model
from losses import DiceLoss, FocalLoss, CombinedLoss, CombinedBinaryLoss

# Setup device
device_id = 3
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device_id)  # 可选，但有些库喜欢这个
print("Using", device)

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, best_dice, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_dice': best_dice,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_dice']


def train_epoch(model, train_loader, criterion, optimizer, device, metrics_calc, scaler=None, use_safr=True):
    """Train for one epoch with AMP, grad clipping, and optional SAFR supervision"""

    metrics_calc.reset()
    total_loss = 0.0
    model = model.to(device)
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        images = batch['image']
        masks = batch['mask']

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if masks.dtype != torch.long:
            masks = masks.long()
        masks = torch.clamp(masks, 0, 1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        optimizer.zero_grad(set_to_none=True)

        # Prepare seg_gt for SAFR (train-time only)
        seg_gt = None
        if use_safr:
            seg_gt = masks.float().unsqueeze(1)

        # Forward + loss with AMP
        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            # outputs = model(images, seg_gt=seg_gt)
            # 数值安全：裁剪/替换 NaN/Inf
            outputs = torch.clamp(outputs, -20, 20)
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)

            # 验证输出形状
            if len(outputs.shape) != 4 or outputs.shape[1] != 2:
                raise ValueError(f"模型输出形状错误: {outputs.shape}, 期望 (B, 2, H, W)")
            if len(masks.shape) != 3:
                raise ValueError(f"掩码形状错误: {masks.shape}, 期望 (B, H, W)")

            loss = criterion(outputs, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                # 跳过异常 batch，避免污染优化器状态
                print(f"Warning: loss is NaN/Inf at batch {batch_idx}, skipping step")
                continue

        if scaler is not None and device.type == 'cuda':
            scaler.scale(loss).backward()
            # Gradient clipping after unscale
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        metrics_calc.update(outputs.detach(), masks)

        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')

    avg_loss = total_loss / len(train_loader)
    metrics = metrics_calc.get_metrics()

    return avg_loss, metrics


def validate_epoch(model, val_loader, criterion, device, metrics_calc, use_safr=False):
    """Validate for one epoch (use BN batch stats to avoid small-batch mismatch)"""
    # 使用train()以便BatchNorm使用batch统计，避免小batch下eval模式过度偏置
    model.train()
    metrics_calc.reset()
    total_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            if masks.dtype != torch.long:
                masks = masks.long()
            masks = torch.clamp(masks, 0, 1)
            if len(masks.shape) == 4 and masks.shape[1] == 1:
                masks = masks.squeeze(1)

            # No seg_gt during validation to avoid leakage; SAFR会在内部回退到特征边界
            outputs = model(images)
            # 数值安全
            outputs = torch.clamp(outputs, -20, 20)
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            metrics_calc.update(outputs, masks)

    avg_loss = total_loss / len(val_loader)
    metrics = metrics_calc.get_metrics()

    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Medical Image Segmentation Training')
    # choices = ['swinumamba', 'lkmunet', 'segmamba', 'emnet']
    parser.add_argument('--model', type=str, default='famamba', help='Model architecture to use')
    # choices = ['TN3K', 'DDTI'],
    # A no csi, no fp
    # B csi, no fp
    # C no csi, fp
    # D csi, fp
    # source ~/anaconda3/etc/profile.d/conda.sh
    parser.add_argument('--exp_name', type=str, default='TN3K_Model_D', help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='TN3K',
                        help='Dataset to use')
    # "../DDTI dataset" "../TN3K 4/TN3K"
    parser.add_argument('--data_root', type=str, default="../TN3K 4/TN3K" ,
                        help='Root directory of the dataset')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Input image size')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classes')
    parser.add_argument('--loss', type=str, default='combined',
                        choices=['ce', 'dice', 'focal', 'combined'],
                        help='Loss function to use')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'{args.exp_name}_{args.dataset}_{timestamp}'

    # Setup logging
    log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)  # 确保日志目录存在
    # writer = SummaryWriter(log_dir)  # 移除tensorboard依赖
    writer = None

    # Save configuration
    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create model
    print(f"Creating model: {args.model}")

    # Determine input channels based on dataset
    in_chns = 3  # RGB images

    model_config = {
        'input_channels': 3,
        'n_stages': 4,
        'features_per_stage': [32, 64, 128, 256],
        'conv_op': nn.Conv2d,
        'kernel_sizes': 3,
        'strides': [1, 2, 2, 2],
        'n_conv_per_stage': 2,
        'num_classes': args.num_classes,
        'n_conv_per_stage_decoder': 2,
        'norm_op': nn.BatchNorm2d,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
        'use_csfi': True,
        'use_safr': False,
        'use_frequency_aware': False
    }

    from networks.net_factory import net_factory
    model = net_factory(net_type="famamba")
    # UMambaBot(**model_config)

    if model is None:
        raise ValueError(f"Model {args.model} not found!")

    print(f"Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss function with class weights for imbalanced data
    # 为类别不平衡添加权重：背景类权重较小，前景类权重较大
    class_weights = torch.tensor([0.7, 0.3]).to(device)  # 提高背景权重，强力抑制误检（FP）

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == 'dice':
        criterion = DiceLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss()
    elif args.loss == 'combined':
        # 若输出为2通道（多类logits）：用CE+前景Dice；若为1通道（二值logits）：用BCE+BinaryDice
        # 我们的 UMambaBot 当前输出通道数=2，仍保留CE+Dice策略；如果后续切为1通道，可自动使用二值组合损失
        criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5, class_weights=class_weights)  # 加大CE抑制FP

    # Create optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP GradScaler
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    # Initialize metrics calculators
    train_metrics = MetricsCalculator(args.num_classes)
    val_metrics = MetricsCalculator(args.num_classes)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_dice = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_dice = load_checkpoint(model, optimizer, scheduler, args.resume)
        print(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")

    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Train (enable SAFR supervision during training)
        train_loss, train_metrics_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics, scaler=scaler, use_safr=True
        )

        # Validate (no SAFR supervision)
        val_loss, val_metrics_dict = validate_epoch(
            model, val_loader, criterion, device, val_metrics, use_safr=False
        )

        # Update scheduler
        scheduler.step()

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train - Acc: {train_metrics_dict['accuracy']:.4f}, "
              f"Dice: {train_metrics_dict['dice']:.4f}, IoU: {train_metrics_dict['iou']:.4f}")
        print(f"Val - Acc: {val_metrics_dict['accuracy']:.4f}, "
              f"Dice: {val_metrics_dict['dice']:.4f}, IoU: {val_metrics_dict['iou']:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s, LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Log to tensorboard (disabled)
        if writer is not None:
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Metrics/Train_Accuracy', train_metrics_dict['accuracy'], epoch)
            writer.add_scalar('Metrics/Train_Dice', train_metrics_dict['dice'], epoch)
            writer.add_scalar('Metrics/Train_IoU', train_metrics_dict['iou'], epoch)
            writer.add_scalar('Metrics/Val_Accuracy', val_metrics_dict['accuracy'], epoch)
            writer.add_scalar('Metrics/Val_Dice', val_metrics_dict['dice'], epoch)
            writer.add_scalar('Metrics/Val_IoU', val_metrics_dict['iou'], epoch)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        # Save best model
        current_dice = val_metrics_dict['dice']
        if current_dice > best_dice:
            best_dice = current_dice
            best_model_path = os.path.join(args.save_dir, f'{exp_name}_best.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_dice, best_model_path)
            print(f"New best model saved! Dice: {best_dice:.4f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f'{exp_name}_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, best_dice, checkpoint_path)

    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, args.num_classes)
    print("Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Dice: {test_metrics['dice']:.4f}")
    print(f"IoU: {test_metrics['iou']:.4f}")

    # Save final results
    results = {
        'best_val_dice': best_dice,
        'test_metrics': test_metrics,
        'config': vars(args)
    }

    results_path = os.path.join(log_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    if writer is not None:
        writer.close()
    print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
