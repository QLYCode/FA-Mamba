import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


def calculate_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy
    Args:
        pred: Predicted segmentation mask (B, H, W) or (B, C, H, W)
        target: Ground truth mask (B, H, W)
    Returns:
        accuracy: Pixel-wise accuracy
    """
    if pred.dim() == 4:  # (B, C, H, W)
        pred = torch.argmax(pred, dim=1)  # (B, H, W)
    
    pred = pred.flatten()
    target = target.flatten()
    
    correct = (pred == target).sum().item()
    total = target.numel()
    
    return correct / total


def calculate_dice_score(pred, target, smooth=1e-6, threshold=0.25):
    """
    Calculate Dice coefficient for foreground class with improved handling
    Args:
        pred: Predicted segmentation mask (B, H, W) or (B, C, H, W)
        target: Ground truth mask (B, H, W)
        smooth: Smoothing factor to avoid division by zero
        threshold: Threshold for binary prediction (降低阈值以捕获更多前景)
    Returns:
        dice: Dice coefficient
    """
    if pred.dim() == 4:  # (B, C, H, W)
        if pred.shape[1] == 1:  # Binary segmentation with single channel
            pred = torch.sigmoid(pred)
            pred = (pred > threshold).float()
        else:  # Multi-class segmentation - extract foreground class
            pred = torch.softmax(pred, dim=1)
            pred = pred[:, 1, :, :]  # 提取前景类（类别1）的概率
            pred = (pred > threshold).float()  # 使用更低的阈值

    # Convert to binary masks for foreground class
    pred = pred.flatten().float()
    target = (target == 1).flatten().float()  # 只考虑前景类

    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    union = pred_sum + target_sum

    # 改进的Dice计算，处理边界情况
    if target_sum == 0 and pred_sum == 0:
        # 如果真实标签和预测都没有前景，返回1（完美匹配）
        return 1.0
    elif target_sum == 0:
        # 如果真实标签没有前景但预测有，返回0
        return 0.0
    else:
        # 正常情况
        dice = (2.0 * intersection + smooth) / (union + smooth)
        return dice.item()


def calculate_iou(pred, target, smooth=1e-6, threshold=0.25):
    """
    Calculate Intersection over Union (IoU) for foreground class with improved handling
    Args:
        pred: Predicted segmentation mask (B, H, W) or (B, C, H, W)
        target: Ground truth mask (B, H, W)
        smooth: Smoothing factor to avoid division by zero
        threshold: Threshold for binary prediction (降低阈值以捕获更多前景)
    Returns:
        iou: IoU score
    """
    if pred.dim() == 4:  # (B, C, H, W)
        if pred.shape[1] == 1:  # Binary segmentation with single channel
            pred = torch.sigmoid(pred)
            pred = (pred > threshold).float()
        else:  # Multi-class segmentation - extract foreground class
            pred = torch.softmax(pred, dim=1)
            pred = pred[:, 1, :, :]  # 提取前景类（类别1）的概率
            pred = (pred > threshold).float()  # 使用更低的阈值

    # Convert to binary masks for foreground class
    pred = pred.flatten().float()
    target = (target == 1).flatten().float()  # 只考虑前景类

    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    union = pred_sum + target_sum - intersection

    # 改进的IoU计算，处理边界情况
    if target_sum == 0 and pred_sum == 0:
        # 如果真实标签和预测都没有前景，返回1（完美匹配）
        return 1.0
    elif target_sum == 0:
        # 如果真实标签没有前景但预测有，返回0
        return 0.0
    elif union == 0:
        # 防止除零
        return 0.0
    else:
        # 正常情况
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()


def calculate_multiclass_dice(pred, target, num_classes, smooth=1e-6):
    """
    Calculate Dice coefficient for multi-class segmentation
    Args:
        pred: Predicted segmentation logits (B, C, H, W)
        target: Ground truth mask (B, H, W)
        num_classes: Number of classes
        smooth: Smoothing factor
    Returns:
        dice_scores: List of dice scores for each class
        mean_dice: Mean dice score across all classes
    """
    pred = torch.softmax(pred, dim=1)
    dice_scores = []
    
    for class_idx in range(num_classes):
        pred_class = pred[:, class_idx, :, :].flatten()
        target_class = (target == class_idx).float().flatten()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum()
        
        dice = (2.0 * intersection + smooth) / (union + smooth)
        dice_scores.append(dice.item())
    
    mean_dice = np.mean(dice_scores)
    return dice_scores, mean_dice


def calculate_multiclass_iou(pred, target, num_classes, smooth=1e-6):
    """
    Calculate IoU for multi-class segmentation
    Args:
        pred: Predicted segmentation logits (B, C, H, W)
        target: Ground truth mask (B, H, W)
        num_classes: Number of classes
        smooth: Smoothing factor
    Returns:
        iou_scores: List of IoU scores for each class
        mean_iou: Mean IoU score across all classes
    """
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)  # (B, H, W)
    
    iou_scores = []
    
    for class_idx in range(num_classes):
        pred_class = (pred == class_idx).float().flatten()
        target_class = (target == class_idx).float().flatten()
        
        intersection = (pred_class * target_class).sum()
        union = pred_class.sum() + target_class.sum() - intersection
        
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())
    
    mean_iou = np.mean(iou_scores)
    return iou_scores, mean_iou


class MetricsCalculator:
    """
    Utility class to calculate and track metrics during training/evaluation
    """
    def __init__(self, num_classes=2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_accuracy = 0.0
        self.total_dice = 0.0
        self.total_iou = 0.0
        self.count = 0
        self.class_dice_scores = []
        self.class_iou_scores = []
    
    def update(self, pred, target):
        """
        Update metrics with new batch
        Args:
            pred: Predicted segmentation (B, C, H, W) or (B, H, W)
            target: Ground truth mask (B, H, W)
        """
        # Calculate metrics
        accuracy = calculate_accuracy(pred, target)
        
        if self.num_classes == 2:  # Binary segmentation
            # 使用更低的阈值来捕获更多前景预测
            dice = calculate_dice_score(pred, target, threshold=0.3)
            iou = calculate_iou(pred, target, threshold=0.3)

            self.total_dice += dice
            self.total_iou += iou
        else:  # Multi-class segmentation
            dice_scores, mean_dice = calculate_multiclass_dice(pred, target, self.num_classes)
            iou_scores, mean_iou = calculate_multiclass_iou(pred, target, self.num_classes)
            
            self.class_dice_scores.append(dice_scores)
            self.class_iou_scores.append(iou_scores)
            self.total_dice += mean_dice
            self.total_iou += mean_iou
        
        self.total_accuracy += accuracy
        self.count += 1
    
    def get_metrics(self):
        """
        Get average metrics
        Returns:
            dict: Dictionary containing average metrics
        """
        if self.count == 0:
            return {'accuracy': 0.0, 'dice': 0.0, 'iou': 0.0}
        
        metrics = {
            'accuracy': self.total_accuracy / self.count,
            'dice': self.total_dice / self.count,
            'iou': self.total_iou / self.count
        }
        
        if self.num_classes > 2 and self.class_dice_scores:
            # Calculate per-class averages
            class_dice_avg = np.mean(self.class_dice_scores, axis=0)
            class_iou_avg = np.mean(self.class_iou_scores, axis=0)
            
            for i in range(self.num_classes):
                metrics[f'dice_class_{i}'] = class_dice_avg[i]
                metrics[f'iou_class_{i}'] = class_iou_avg[i]
        
        return metrics
    
    def print_metrics(self, prefix=""):
        """Print current metrics"""
        metrics = self.get_metrics()
        print(f"{prefix}Accuracy: {metrics['accuracy']:.4f}")
        print(f"{prefix}Dice: {metrics['dice']:.4f}")
        print(f"{prefix}IoU: {metrics['iou']:.4f}")
        
        if self.num_classes > 2:
            for i in range(self.num_classes):
                if f'dice_class_{i}' in metrics:
                    print(f"{prefix}Class {i} - Dice: {metrics[f'dice_class_{i}']:.4f}, IoU: {metrics[f'iou_class_{i}']:.4f}")


def evaluate_model(model, dataloader, device, num_classes=2):
    """
    Evaluate model on a dataset
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        num_classes: Number of classes
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator(num_classes)
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(images)
            
            metrics_calc.update(outputs, masks)
    
    return metrics_calc.get_metrics()


if __name__ == "__main__":
    # Test metrics calculation
    batch_size, num_classes, height, width = 2, 2, 256, 256
    
    # Create dummy data
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test metrics
    accuracy = calculate_accuracy(pred, target)
    dice = calculate_dice_score(pred, target)
    iou = calculate_iou(pred, target)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"IoU: {iou:.4f}")
    
    # Test MetricsCalculator
    calc = MetricsCalculator(num_classes)
    calc.update(pred, target)
    calc.print_metrics("Test - ")
