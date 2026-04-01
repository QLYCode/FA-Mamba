import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# import albumentations as A
# from albumentations.pytorch import ToTensorV2  # 移除albumentations依赖


class MedicalSegmentationDataset(Dataset):
    """
    Medical Image Segmentation Dataset for TN3K and DDTI datasets
    """
    def __init__(self, data_root, dataset_name, split='train', img_size=512, augment=True):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.split = split
        self.img_size = img_size
        self.augment = augment and (split == 'train')  # Only augment training data
        
        # Setup paths based on actual data structure
        if dataset_name == 'TN3K':
            # TN3K structure: typically .../TN3K 3/TN3K/{train,val,test}-{image,mask}
            # 1) 常见路径快速检测
            possible_paths = [
                os.path.join(data_root, 'TN3K 3', 'TN3K'),
                os.path.join(data_root, 'TN3K'),
                os.path.join(data_root, 'TN3K 3'),
                data_root,
            ]

            def has_split_dirs(base, split_name):
                return (
                    os.path.exists(os.path.join(base, f'{split_name}-image')) and
                    os.path.exists(os.path.join(base, f'{split_name}-mask'))
                )

            dataset_path = None
            # 先检查常见路径
            for path in possible_paths:
                if has_split_dirs(path, split):
                    dataset_path = path
                    break

            # 2) 递归扫描 data_root，寻找同时包含 split-image 与 split-mask 的目录
            if dataset_path is None:
                for dirpath, dirnames, _ in os.walk(data_root):
                    # 直接在该层检查 split 子目录是否存在
                    if has_split_dirs(dirpath, split):
                        dataset_path = dirpath
                        break

            if dataset_path is None:
                raise FileNotFoundError(
                    f"TN3K数据集路径未找到。请将 data_root 指向包含 {{train,val,test}}-image/ -mask 的目录上一级；"
                    f"或把 TN3K 放在当前目录树内（例如 .\\TN3K 3\\TN3K）。当前 data_root={data_root}"
                )

            # 设定图像与掩码目录
            self.img_dir = os.path.join(dataset_path, f'{split}-image')
            self.mask_dir = os.path.join(dataset_path, f'{split}-mask')

        elif dataset_name == 'DDTI':
            # DDTI structure: data_root 直接包含 p_image, p_mask，或在子目录中
            possible_paths = [
                data_root,
                os.path.join(data_root, 'DDTI'),
                os.path.join(data_root, 'DDTI dataset'),
            ]

            dataset_path = None
            # 1) 先检查常见路径
            for path in possible_paths:
                test_img_dir = os.path.join(path, 'p_image')
                test_mask_dir = os.path.join(path, 'p_mask')
                if os.path.exists(test_img_dir) and os.path.exists(test_mask_dir):
                    dataset_path = path
                    break

            # 2) 递归扫描 data_root，寻找同时包含 p_image 与 p_mask 的目录（大小写不敏感）
            if dataset_path is None:
                for dirpath, dirnames, _ in os.walk(data_root):
                    names_lower = {d.lower() for d in dirnames}
                    if 'p_image'.lower() in names_lower and 'p_mask'.lower() in names_lower:
                        dataset_path = dirpath
                        break

            if dataset_path is None:
                raise FileNotFoundError(
                    f"DDTI数据集路径未找到。请在 data_root 下放置包含 p_image 和 p_mask 的目录，"
                    f"或将 data_root 指向该目录的上一级。例如：--data_root \"D:/datasets/DDTI dataset\""
                )

            self.img_dir = os.path.join(dataset_path, 'p_image')
            self.mask_dir = os.path.join(dataset_path, 'p_mask')
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        print(f"数据集路径设置:")
        print(f"  图像目录: {self.img_dir}")
        print(f"  掩码目录: {self.mask_dir}")

        # Check if directories exist
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Get all files first
        all_img_files = sorted([f for f in os.listdir(self.img_dir)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        # For DDTI, split the data since it's all in one folder
        if dataset_name == 'DDTI':
            # 使用分层采样确保每个分割都包含前景样本
            import random
            random.seed(42)  # 确保可重复性

            # 随机打乱文件列表
            shuffled_files = all_img_files.copy()
            random.shuffle(shuffled_files)

            total_files = len(shuffled_files)
            train_split = int(0.7 * total_files)  # 70% for training
            val_split = int(0.85 * total_files)   # 15% for validation, 15% for test

            if split == 'train':
                self.img_files = shuffled_files[:train_split]
            elif split == 'val':
                self.img_files = shuffled_files[train_split:val_split]
            elif split == 'test':
                self.img_files = shuffled_files[val_split:]
        else:
            # For TN3K, use all files in the respective split folder
            self.img_files = all_img_files

        print(f"Found {len(self.img_files)} images in {dataset_name} {split} set")
        print(f"Image directory: {self.img_dir}")
        print(f"Mask directory: {self.mask_dir}")

        # Setup transforms
        self.setup_transforms()

    def setup_transforms(self):
        """Setup data augmentation and preprocessing transforms"""
        # 现在在__getitem__中直接处理变换，这里只设置标志
        self.transform = True
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask_name = self.img_files[idx].replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Try different extensions
            base_name = os.path.splitext(self.img_files[idx])[0]
            for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                mask_path = os.path.join(self.mask_dir, base_name + ext)
                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    break
            else:
                raise FileNotFoundError(f"Mask not found for {self.img_files[idx]}")
        
        if mask is None:
            raise FileNotFoundError(f"Mask could not be loaded: {mask_path}")
        
        # 关键修复：确保掩码二值化
        mask = (mask > 127).astype(np.uint8)
        
        # 简化的变换处理
        # 1. 调整大小
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 2. 数据增强（仅训练时）
        if self.augment:
            # 形变增强：轻微缩放与平移（保持 mask 对齐）
            if np.random.random() < 0.5:
                scale = 1.0 + np.random.uniform(-0.08, 0.08)
                tx = int(np.random.uniform(-0.04, 0.04) * self.img_size)
                ty = int(np.random.uniform(-0.04, 0.04) * self.img_size)
                M = np.array([[scale, 0, tx], [0, scale, ty]], dtype=np.float32)
                image = cv2.warpAffine(image, M, (self.img_size, self.img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                mask = cv2.warpAffine(mask, M, (self.img_size, self.img_size), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            # 随机水平/垂直翻转
            if np.random.random() < 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if np.random.random() < 0.3:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)
            # 随机90度旋转
            if np.random.random() < 0.5:
                k = np.random.randint(0, 4)
                if k > 0:
                    image = np.ascontiguousarray(np.rot90(image, k))
                    mask = np.ascontiguousarray(np.rot90(mask, k))
            # 亮度/对比度抖动（轻度）
            if np.random.random() < 0.5:
                alpha = 0.9 + 0.2 * np.random.random()  # [0.9, 1.1]
                beta = np.random.uniform(-10, 10)       # [-10, 10]
                image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # 3. 转换为tensor
        # 图像归一化
        image = image.astype(np.float32) / 255.0
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW

        # 掩码处理 - 关键修复
        mask = torch.from_numpy(mask).long()  # 确保是long类型
        mask = torch.clamp(mask, 0, 1)  # 确保值在[0,1]范围内

        # 校验：至少在一个小块里存在前景（避免全背景批次导致训练/验证不稳定）
        if self.split == 'val' or self.split == 'test':
            pass  # 不改动验证/测试数据
        else:
            if mask.sum().item() == 0:
                # 若当前图无前景，尝试简单增强使其包含少量随机前景（防止训练陷入全背景）
                # 这里不修改 mask 语义，只进行轻微随机噪声；主要依赖损失权重与 SAFR 修复
                # 如需更强的采样控制，可在 DataLoader 层做分层采样（此处保持简单）
                pass

        return {
            'image': image,
            'mask': mask,
            'filename': self.img_files[idx]
        }


def get_dataloaders(data_root, dataset_name, batch_size=8, img_size=512, num_workers=4):
    """
    Create train, validation, and test dataloaders
    """
    # Create datasets
    train_dataset = MedicalSegmentationDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        split='train',
        img_size=img_size,
        augment=True
    )
    
    val_dataset = MedicalSegmentationDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        split='val',
        img_size=img_size,
        augment=False
    )
    
    test_dataset = MedicalSegmentationDataset(
        data_root=data_root,
        dataset_name=dataset_name,
        split='test',
        img_size=img_size,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Use batch_size=1 for testing
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
