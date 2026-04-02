import os
import sys
import argparse
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import random
from torch.cuda.amp import autocast, GradScaler

sys.path.append('networks')
from dataset import get_dataloaders
from metrics import MetricsCalculator, evaluate_model
from losses import DiceLoss, FocalLoss, CombinedLoss, CombinedBinaryLoss

device_id = 3
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device_id)
print("Using", device)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scheduler, epoch, best_dice, save_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_dice': best_dice,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['best_dice']


def train_epoch(model, train_loader, criterion, optimizer, device, metrics_calc, scaler=None):
    metrics_calc.reset()
    total_loss = 0.0
    model.to(device)
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)

        if masks.dtype != torch.long:
            masks = masks.long()
        masks = torch.clamp(masks, 0, 1)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(device.type == 'cuda')):
            outputs = model(images)
            outputs = torch.clamp(outputs, -20, 20)
            outputs = torch.nan_to_num(outputs, nan=0.0, posinf=20.0, neginf=-20.0)

            if len(outputs.shape) != 4 or outputs.shape[1] != 2:
                raise ValueError(f"模型输出形状错误: {outputs.shape}, 期望 (B, 2, H, W)")
            if len(masks.shape) != 3:
                raise ValueError(f"掩码形状错误: {masks.shape}, 期望 (B, H, W)")

            loss = criterion(outputs, masks)
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: loss is NaN/Inf at batch {batch_idx}, skipping step")
                continue

        if scaler is not None and device.type == 'cuda':
            scaler.scale(loss).backward()
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


def validate_epoch(model, val_loader, criterion, device, metrics_calc):
    """
    Fix 1: 使用 model.eval() 而不是 model.train()。
    BatchNorm 在 eval 模式下使用运行均值/方差，验证结果稳定可信。
    """
    model.eval()
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

            outputs = model(images)
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
    parser.add_argument('--model', type=str, default='famamba')
    parser.add_argument('--exp_name', type=str, default='TN3K_Model_D')
    parser.add_argument('--dataset', type=str, default='TN3K')
    parser.add_argument('--data_root', type=str, default="../TN3K 4/TN3K")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--loss', type=str, default='combined',
                        choices=['ce', 'dice', 'focal', 'combined'])
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_freq', type=int, default=20)

    args = parser.parse_args()
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'{args.exp_name}_{args.dataset}_{timestamp}'

    log_dir = os.path.join(args.log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)

    config_path = os.path.join(log_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("Creating data loaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_root=args.data_root,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    print(f"Creating model: {args.model}")

    from networks.net_factory import net_factory
    model = net_factory(net_type="famamba")
    if model is None:
        raise ValueError(f"Model {args.model} not found!")
    model = model.to(device)

    print(f"Model created successfully")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fix 2: 前景（病灶）是少数类，权重应更大；原代码 [0.7, 0.3] 反了
    # 背景=类别0权重小，前景=类别1权重大
    class_weights = torch.tensor([0.3, 0.7]).to(device)

    if args.loss == 'ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif args.loss == 'dice':
        criterion = DiceLoss()
    elif args.loss == 'focal':
        criterion = FocalLoss()
    elif args.loss == 'combined':
        criterion = CombinedLoss(
            ce_weight=0.5,
            dice_weight=0.5,
            class_weights=class_weights,
        )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Fix 3: 加 warmup：前 10 epoch 线性升温，之后 cosine 衰减
    # 这样避免训练初期 LR 过大导致不稳定，中期也不会因 LR 骤变而跳Loss
    warmup_epochs = 10

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性 warmup
        return 1.0  # warmup 结束后交给 CosineAnnealingLR

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - warmup_epochs,
        eta_min=1e-6,
    )

    scaler = GradScaler(enabled=(device.type == 'cuda'))

    train_metrics = MetricsCalculator(args.num_classes)
    val_metrics = MetricsCalculator(args.num_classes)

    start_epoch = 0
    best_dice = 0.0

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch, best_dice = load_checkpoint(model, optimizer, cosine_scheduler, args.resume)
        print(f"Resumed from epoch {start_epoch}, best dice: {best_dice:.4f}")

    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        train_loss, train_metrics_dict = train_epoch(
            model, train_loader, criterion, optimizer, device, train_metrics, scaler=scaler,
        )

        val_loss, val_metrics_dict = validate_epoch(
            model, val_loader, criterion, device, val_metrics,
        )

        # Fix 3: warmup 阶段用 warmup_scheduler，之后用 cosine_scheduler
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Train - Acc: {train_metrics_dict['accuracy']:.4f}, "
              f"Dice: {train_metrics_dict['dice']:.4f}, IoU: {train_metrics_dict['iou']:.4f}")
        print(f"Val   - Acc: {val_metrics_dict['accuracy']:.4f}, "
              f"Dice: {val_metrics_dict['dice']:.4f}, IoU: {val_metrics_dict['iou']:.4f}")
        print(f"Epoch time: {epoch_time:.2f}s, LR: {current_lr:.2e}")

        current_dice = val_metrics_dict['dice']
        if current_dice > best_dice:
            best_dice = current_dice
            best_model_path = os.path.join(args.save_dir, f'{exp_name}_best.pth')
            save_checkpoint(model, optimizer, cosine_scheduler, epoch, best_dice, best_model_path)
            print(f"New best model saved! Dice: {best_dice:.4f}")

        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.save_dir, f'{exp_name}_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, cosine_scheduler, epoch, best_dice, checkpoint_path)

    print("\nEvaluating on test set...")
    test_metrics = evaluate_model(model, test_loader, device, args.num_classes)
    print("Test Results:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Dice:     {test_metrics['dice']:.4f}")
    print(f"IoU:      {test_metrics['iou']:.4f}")

    results = {
        'best_val_dice': best_dice,
        'test_metrics': test_metrics,
        'config': vars(args),
    }
    results_path = os.path.join(log_dir, 'final_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining completed! Best validation Dice: {best_dice:.4f}")
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()