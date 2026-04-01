import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    """
    def __init__(self, smooth=1e-6, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        # 确保target是正确的数据类型和形状
        if target.dtype != torch.long:
            target = target.long()

        # 确保target值在正确范围内
        target = torch.clamp(target, 0, pred.shape[1] - 1)

        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)

        # Convert target to one-hot encoding - 更安全的方式
        num_classes = pred.shape[1]
        try:
            target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        except Exception as e:
            print(f"one_hot编码失败: {e}")
            print(f"target形状: {target.shape}, num_classes: {num_classes}")
            print(f"target值范围: [{target.min()}, {target.max()}]")
            # 手动创建one-hot编码
            B, H, W = target.shape
            target_one_hot = torch.zeros(B, num_classes, H, W, device=target.device, dtype=torch.float32)
            for c in range(num_classes):
                target_one_hot[:, c, :, :] = (target == c).float()
            print("使用手动one-hot编码")

        # Flatten tensors
        pred = pred.view(pred.shape[0], pred.shape[1], -1)  # (B, C, H*W)
        target_one_hot = target_one_hot.view(target_one_hot.shape[0], target_one_hot.shape[1], -1)  # (B, C, H*W)

        # Calculate Dice coefficient for each class
        intersection = (pred * target_one_hot).sum(dim=2)  # (B, C)
        union = pred.sum(dim=2) + target_one_hot.sum(dim=2)  # (B, C)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (B, C)

        # Average across classes and batch
        dice_loss = 1.0 - dice.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=1.0, gamma=2.0, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        # Calculate cross entropy - 修复ignore_index问题
        if self.ignore_index is None:
            ce_loss = F.cross_entropy(pred, target, reduction='none')
        else:
            ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        
        # Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # Calculate focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice loss
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha  # Weight for false positives
        self.beta = beta    # Weight for false negatives
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Convert target to one-hot encoding
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Flatten tensors
        pred = pred.view(pred.shape[0], pred.shape[1], -1)
        target_one_hot = target_one_hot.view(target_one_hot.shape[0], target_one_hot.shape[1], -1)
        
        # Calculate Tversky coefficient
        true_pos = (pred * target_one_hot).sum(dim=2)
        false_neg = (target_one_hot * (1 - pred)).sum(dim=2)
        false_pos = ((1 - target_one_hot) * pred).sum(dim=2)
        
        tversky = (true_pos + self.smooth) / (true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth)
        
        return 1.0 - tversky.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function using Cross Entropy + Dice Loss
    """
    def __init__(self, ce_weight=0.5, dice_weight=0.5, ignore_index=None, class_weights=None):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        # 修复ignore_index问题，添加类别权重支持
        if ignore_index is None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=class_weights)
        self.dice_loss = DiceLoss()
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        # 详细的数据验证和修复
        # 1. 检查和修复target的数据类型
        if target.dtype != torch.long:
            target = target.long()

        # 2. 检查和修复target的值范围
        target = torch.clamp(target, 0, pred.shape[1] - 1)

        # 3. 检查形状匹配
        if len(target.shape) == 4 and target.shape[1] == 1:
            target = target.squeeze(1)  # 移除多余的通道维度

        # 4. 确保形状正确
        if len(pred.shape) != 4:
            raise ValueError(f"pred形状错误: {pred.shape}, 期望 (B, C, H, W)")
        if len(target.shape) != 3:
            raise ValueError(f"target形状错误: {target.shape}, 期望 (B, H, W)")
        if pred.shape[0] != target.shape[0] or pred.shape[2] != target.shape[1] or pred.shape[3] != target.shape[2]:
            raise ValueError(f"pred和target形状不匹配: pred={pred.shape}, target={target.shape}")

        # 5. 计算损失
        try:
            # 先单独测试CrossEntropyLoss
            # 数值安全：对 logits 做裁剪/NaN 处理
            pred = torch.clamp(pred, -20, 20)
            pred = torch.nan_to_num(pred, nan=0.0, posinf=20.0, neginf=-20.0)

            ce = self.ce_loss(pred, target)

            # 前景优先的 Dice：只对前景通道计算
            pred_soft = F.softmax(pred, dim=1)
            if pred_soft.shape[1] < 2:
                pred_fg = torch.sigmoid(pred[:, 0])
            else:
                pred_fg = pred_soft[:, 1]
            target_fg = (target == 1).float()
            if pred_fg.dim() == 4 and pred_fg.shape[1] == 1:
                pred_fg = pred_fg.squeeze(1)
            smooth = 1e-6
            intersection = (pred_fg * target_fg).sum(dim=(1, 2))
            pred_sum = pred_fg.sum(dim=(1, 2))
            target_sum = target_fg.sum(dim=(1, 2))
            dice_fg = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)
            dice = 1.0 - dice_fg.mean()

            # 若出现 NaN，回退到仅 CE
            if torch.isnan(ce) or torch.isinf(ce) or torch.isnan(dice) or torch.isinf(dice):
                return self.ce_loss(pred, target)

            return self.ce_weight * ce + self.dice_weight * dice

        except RuntimeError as e:
            if "CUDA" in str(e) or "out of memory" in str(e):
                print(f"CUDA内存错误: {e}")
                print("尝试减少batch_size或使用CPU")
                # 尝试在CPU上计算
                pred_cpu = pred.cpu()
                target_cpu = target.cpu()
                ce = self.ce_loss(pred_cpu, target_cpu)
                dice = self.dice_loss(pred_cpu, target_cpu)
                return (self.ce_weight * ce + self.dice_weight * dice).to(pred.device)
            else:
                raise e
        except Exception as e:
            print(f"损失计算错误:")
            print(f"  pred形状: {pred.shape}, 数据类型: {pred.dtype}")
            print(f"  target形状: {target.shape}, 数据类型: {target.dtype}")
            print(f"  pred值范围: [{pred.min():.3f}, {pred.max():.3f}]")
            print(f"  target值范围: [{target.min():.0f}, {target.max():.0f}]")
            print(f"  target唯一值: {torch.unique(target)}")
            print(f"  错误类型: {type(e).__name__}")
            print(f"  错误信息: {str(e)}")

            # 尝试简单的CrossEntropyLoss
            print("尝试简单的CrossEntropyLoss...")
            try:
                simple_ce = nn.CrossEntropyLoss()
                ce_simple = simple_ce(pred, target)
                print(f"简单CE成功: {ce_simple.item():.4f}")
                return ce_simple  # 如果CE成功，就只返回CE损失
            except Exception as e2:
                print(f"简单CE也失败: {e2}")
                raise e


class WeightedCombinedLoss(nn.Module):
    """
    Weighted combined loss with class weights
    """
    def __init__(self, class_weights=None, ce_weight=0.5, dice_weight=0.5):
        super(WeightedCombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss()

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)

        return self.ce_weight * ce + self.dice_weight * dice


class BinaryDiceLoss(nn.Module):
    """Binary Dice loss for 1-channel logits"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    def forward(self, pred, target):
        # pred: (B,1,H,W) logits, target: (B,1,H,W) float {0,1}
        pred_prob = torch.sigmoid(pred)
        intersection = (pred_prob * target).sum(dim=(1,2,3))
        pred_sum = pred_prob.sum(dim=(1,2,3))
        target_sum = target.sum(dim=(1,2,3))
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        return 1.0 - dice.mean()

class CombinedBinaryLoss(nn.Module):
    """BCEWithLogits + Binary Dice loss for 1-channel output"""
    def __init__(self, bce_weight=0.2, dice_weight=0.8, pos_weight=3.0):
        super().__init__()
        # pos_weight >1 增强前景召回
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        self.dice = BinaryDiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
    def forward(self, pred, target):
        # pred: (B,1,H,W) logits, target: (B,1,H,W) float {0,1}
        pred = torch.clamp(pred, -20, 20)
        pred = torch.nan_to_num(pred, nan=0.0, posinf=20.0, neginf=-20.0)
        bce = self.bce(pred, target)
        dice = self.dice(pred, target)
        if torch.isnan(bce) or torch.isinf(bce) or torch.isnan(dice) or torch.isinf(dice):
            return self.bce(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


class BoundaryLoss(nn.Module):
    """
    Boundary Loss for better edge segmentation
    """
    def __init__(self, theta0=3, theta=5):
        super(BoundaryLoss, self).__init__()
        self.theta0 = theta0
        self.theta = theta
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        # Apply softmax to get probabilities
        pred_soft = F.softmax(pred, dim=1)
        
        # Convert target to one-hot
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Calculate distance transform (simplified version)
        # In practice, you might want to use scipy.ndimage.distance_transform_edt
        boundary_loss = 0
        
        for b in range(pred.shape[0]):
            for c in range(num_classes):
                pred_c = pred_soft[b, c]
                target_c = target_one_hot[b, c]
                
                # Simple boundary approximation using gradients
                grad_x = torch.abs(pred_c[1:, :] - pred_c[:-1, :])
                grad_y = torch.abs(pred_c[:, 1:] - pred_c[:, :-1])
                
                boundary_loss += grad_x.mean() + grad_y.mean()
        
        return boundary_loss / (pred.shape[0] * num_classes)


class LovaszSoftmaxLoss(nn.Module):
    """
    Lovász-Softmax loss for multiclass segmentation
    """
    def __init__(self, ignore_index=None):
        super(LovaszSoftmaxLoss, self).__init__()
        self.ignore_index = ignore_index
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        # Apply softmax
        pred = F.softmax(pred, dim=1)
        
        # Flatten
        pred = pred.permute(0, 2, 3, 1).contiguous().view(-1, pred.shape[1])  # (B*H*W, C)
        target = target.view(-1)  # (B*H*W,)
        
        # Remove ignored pixels
        if self.ignore_index is not None:
            valid = target != self.ignore_index
            pred = pred[valid]
            target = target[valid]
        
        # Calculate Lovász loss (simplified version)
        losses = []
        for c in range(pred.shape[1]):
            target_c = (target == c).float()
            if target_c.sum() == 0:
                continue
            
            pred_c = pred[:, c]
            errors = torch.abs(target_c - pred_c)
            errors_sorted, perm = torch.sort(errors, descending=True)
            target_sorted = target_c[perm]
            
            # Calculate Lovász extension
            intersection = target_sorted.sum() - target_sorted.cumsum(0)
            union = target_sorted.sum() + (1 - target_sorted).cumsum(0)
            jaccard = 1 - intersection / union
            
            if len(jaccard) > 1:
                jaccard[1:] = jaccard[1:] - jaccard[:-1]
            
            losses.append((jaccard * errors_sorted).sum())
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0, device=pred.device)


def get_loss_function(loss_name, num_classes=2, class_weights=None):
    """
    Factory function to get loss function by name
    
    Args:
        loss_name: Name of the loss function
        num_classes: Number of classes
        class_weights: Class weights for weighted losses
    
    Returns:
        Loss function
    """
    if loss_name == 'ce':
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=class_weights)
    
    elif loss_name == 'dice':
        return DiceLoss()
    
    elif loss_name == 'focal':
        return FocalLoss()
    
    elif loss_name == 'tversky':
        return TverskyLoss()
    
    elif loss_name == 'combined':
        return CombinedLoss()
    
    elif loss_name == 'weighted_combined':
        return WeightedCombinedLoss(class_weights=class_weights)
    
    elif loss_name == 'boundary':
        return BoundaryLoss()
    
    elif loss_name == 'lovasz':
        return LovaszSoftmaxLoss()
    
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")


if __name__ == "__main__":
    # Test loss functions
    batch_size, num_classes, height, width = 2, 2, 64, 64
    
    # Create dummy data
    pred = torch.randn(batch_size, num_classes, height, width)
    target = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test different losses
    losses = {
        'CrossEntropy': nn.CrossEntropyLoss(),
        'Dice': DiceLoss(),
        'Focal': FocalLoss(),
        'Tversky': TverskyLoss(),
        'Combined': CombinedLoss(),
        'Boundary': BoundaryLoss(),
    }
    
    print("Testing loss functions:")
    for name, loss_fn in losses.items():
        try:
            loss_value = loss_fn(pred, target)
            print(f"{name}: {loss_value.item():.4f}")
        except Exception as e:
            print(f"{name}: Error - {e}")
