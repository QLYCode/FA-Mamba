#!/usr/bin/env python3
"""
测试数据加载脚本
验证数据集是否能正确加载
"""

import os
import sys
import torch
from data_config import print_data_summary, verify_data_structure
from dataset import MedicalSegmentationDataset, get_dataloaders

def test_dataset_loading(data_root="."):
    """测试数据集加载"""
    
    print("🔍 开始测试数据加载...")
    print(f"数据根目录: {os.path.abspath(data_root)}")
    
    # 验证数据结构
    results = print_data_summary(data_root)
    
    # 测试每个数据集
    datasets_to_test = []
    
    if results['TN3K']['exists']:
        datasets_to_test.append('TN3K')
    if results['DDTI']['exists']:
        datasets_to_test.append('DDTI')
    
    if not datasets_to_test:
        print("❌ 没有找到可用的数据集!")
        return False
    
    print(f"\n🧪 测试数据集: {datasets_to_test}")
    
    success_count = 0
    total_tests = 0
    
    for dataset_name in datasets_to_test:
        print(f"\n{'='*50}")
        print(f"测试 {dataset_name} 数据集")
        print(f"{'='*50}")
        
        # 确定要测试的分割
        if dataset_name == 'TN3K':
            splits_to_test = ['train', 'val', 'test']
        else:  # DDTI
            splits_to_test = ['train', 'val', 'test']  # DDTI会自动分割
        
        for split in splits_to_test:
            total_tests += 1
            print(f"\n📂 测试 {dataset_name} {split} 集...")
            
            try:
                # 创建数据集
                dataset = MedicalSegmentationDataset(
                    data_root=data_root,
                    dataset_name=dataset_name,
                    split=split,
                    img_size=256,  # 使用较小尺寸加快测试
                    augment=False  # 测试时不使用增强
                )
                
                print(f"  ✅ 数据集创建成功，包含 {len(dataset)} 个样本")
                
                # 测试加载第一个样本
                if len(dataset) > 0:
                    sample = dataset[0]
                    image = sample['image']
                    mask = sample['mask']
                    filename = sample['filename']
                    
                    print(f"  ✅ 样本加载成功:")
                    print(f"     - 文件名: {filename}")
                    print(f"     - 图像形状: {image.shape}")
                    print(f"     - 掩码形状: {mask.shape}")
                    print(f"     - 图像数据类型: {image.dtype}")
                    print(f"     - 掩码数据类型: {mask.dtype}")
                    print(f"     - 图像值范围: [{image.min():.3f}, {image.max():.3f}]")
                    print(f"     - 掩码唯一值: {torch.unique(mask)}")
                    
                    success_count += 1
                else:
                    print(f"  ⚠️  数据集为空")
                
            except Exception as e:
                print(f"  ❌ 数据集加载失败: {str(e)}")
                import traceback
                traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"测试结果: {success_count}/{total_tests} 成功")
    print(f"{'='*60}")
    
    return success_count == total_tests


def test_dataloader(data_root="."):
    """测试数据加载器"""
    
    print(f"\n🔄 测试数据加载器...")
    
    try:
        # 测试TN3K
        if os.path.exists(os.path.join(data_root, 'TN3K 3')):
            print(f"\n📊 测试TN3K数据加载器...")
            train_loader, val_loader, test_loader = get_dataloaders(
                data_root=data_root,
                dataset_name='TN3K',
                batch_size=2,
                img_size=256,
                num_workers=0  # 避免多进程问题
            )
            
            print(f"  ✅ TN3K数据加载器创建成功")
            print(f"     - 训练批次数: {len(train_loader)}")
            print(f"     - 验证批次数: {len(val_loader)}")
            print(f"     - 测试批次数: {len(test_loader)}")
            
            # 测试加载一个批次
            for batch in train_loader:
                images = batch['image']
                masks = batch['mask']
                filenames = batch['filename']
                
                print(f"  ✅ 批次加载成功:")
                print(f"     - 批次图像形状: {images.shape}")
                print(f"     - 批次掩码形状: {masks.shape}")
                print(f"     - 文件名数量: {len(filenames)}")
                break
        
        # 测试DDTI
        if os.path.exists(os.path.join(data_root, 'DDTI dataset')):
            print(f"\n📊 测试DDTI数据加载器...")
            train_loader, val_loader, test_loader = get_dataloaders(
                data_root=data_root,
                dataset_name='DDTI',
                batch_size=2,
                img_size=256,
                num_workers=0
            )
            
            print(f"  ✅ DDTI数据加载器创建成功")
            print(f"     - 训练批次数: {len(train_loader)}")
            print(f"     - 验证批次数: {len(val_loader)}")
            print(f"     - 测试批次数: {len(test_loader)}")
            
            # 测试加载一个批次
            for batch in train_loader:
                images = batch['image']
                masks = batch['mask']
                filenames = batch['filename']
                
                print(f"  ✅ 批次加载成功:")
                print(f"     - 批次图像形状: {images.shape}")
                print(f"     - 批次掩码形状: {masks.shape}")
                print(f"     - 文件名数量: {len(filenames)}")
                break
        
        return True
        
    except Exception as e:
        print(f"  ❌ 数据加载器测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    
    # 获取数据根目录
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "."
    
    print("🚀 数据加载测试开始")
    print("="*60)
    
    # 测试数据集加载
    dataset_success = test_dataset_loading(data_root)
    
    # 测试数据加载器
    dataloader_success = test_dataloader(data_root)
    
    # 总结
    print(f"\n🎯 测试总结:")
    print(f"  数据集加载: {'✅ 成功' if dataset_success else '❌ 失败'}")
    print(f"  数据加载器: {'✅ 成功' if dataloader_success else '❌ 失败'}")
    
    if dataset_success and dataloader_success:
        print(f"\n🎉 所有测试通过! 数据已准备就绪，可以开始训练。")
        print(f"\n下一步:")
        print(f"  1. 运行模型测试: python3 test_models.py")
        print(f"  2. 开始训练: ./start_training.sh")
        return True
    else:
        print(f"\n❌ 测试失败，请检查数据结构和路径配置。")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
