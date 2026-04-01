#!/usr/bin/env python3
"""
数据路径配置文件
根据实际的数据结构配置路径
"""

import os

def get_data_paths(data_root, dataset_name, split):
    """
    根据数据集名称和分割类型返回正确的路径
    
    Args:
        data_root: 数据根目录
        dataset_name: 数据集名称 ('TN3K' 或 'DDTI')
        split: 数据分割 ('train', 'val', 'test')
    
    Returns:
        tuple: (img_dir, mask_dir)
    """
    
    if dataset_name == 'TN3K':
        # TN3K数据结构: TN3K 3/TN3K/train-image, train-mask, etc.
        dataset_path = os.path.join(data_root, 'TN3K 3', 'TN3K')
        
        if split == 'train':
            img_dir = os.path.join(dataset_path, 'train-image')
            mask_dir = os.path.join(dataset_path, 'train-mask')
        elif split == 'val':
            img_dir = os.path.join(dataset_path, 'val-image')
            mask_dir = os.path.join(dataset_path, 'val-mask')
        elif split == 'test':
            img_dir = os.path.join(dataset_path, 'test-image')
            mask_dir = os.path.join(dataset_path, 'test-mask')
        else:
            raise ValueError(f"Unknown split: {split}")
            
    elif dataset_name == 'DDTI':
        # DDTI数据结构: DDTI dataset/DDTI dataset/p_image, p_mask
        dataset_path = os.path.join(data_root, 'DDTI dataset', 'DDTI dataset')
        img_dir = os.path.join(dataset_path, 'p_image')
        mask_dir = os.path.join(dataset_path, 'p_mask')
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return img_dir, mask_dir


def verify_data_structure(data_root):
    """
    验证数据结构是否正确
    
    Args:
        data_root: 数据根目录
    
    Returns:
        dict: 验证结果
    """
    results = {
        'TN3K': {'exists': False, 'splits': {}},
        'DDTI': {'exists': False, 'total_files': 0}
    }
    
    # 检查TN3K
    tn3k_path = os.path.join(data_root, 'TN3K 3', 'TN3K')
    if os.path.exists(tn3k_path):
        results['TN3K']['exists'] = True
        
        for split in ['train', 'val', 'test']:
            img_dir, mask_dir = get_data_paths(data_root, 'TN3K', split)
            
            if os.path.exists(img_dir) and os.path.exists(mask_dir):
                img_count = len([f for f in os.listdir(img_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
                mask_count = len([f for f in os.listdir(mask_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
                
                results['TN3K']['splits'][split] = {
                    'img_count': img_count,
                    'mask_count': mask_count,
                    'matched': img_count == mask_count
                }
    
    # 检查DDTI
    ddti_path = os.path.join(data_root, 'DDTI dataset', 'DDTI dataset')
    if os.path.exists(ddti_path):
        results['DDTI']['exists'] = True
        
        img_dir, mask_dir = get_data_paths(data_root, 'DDTI', 'train')  # split不影响DDTI路径
        
        if os.path.exists(img_dir) and os.path.exists(mask_dir):
            img_count = len([f for f in os.listdir(img_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            mask_count = len([f for f in os.listdir(mask_dir) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
            
            results['DDTI']['total_files'] = img_count
            results['DDTI']['img_count'] = img_count
            results['DDTI']['mask_count'] = mask_count
            results['DDTI']['matched'] = img_count == mask_count
    
    return results


def print_data_summary(data_root):
    """
    打印数据摘要信息
    """
    print("="*60)
    print("数据集结构验证")
    print("="*60)
    
    results = verify_data_structure(data_root)
    
    # TN3K摘要
    print(f"\n📁 TN3K数据集:")
    if results['TN3K']['exists']:
        print("  ✅ 数据集存在")
        for split, info in results['TN3K']['splits'].items():
            status = "✅" if info['matched'] else "❌"
            print(f"  {status} {split}: {info['img_count']} 图像, {info['mask_count']} 掩码")
    else:
        print("  ❌ 数据集不存在")
    
    # DDTI摘要
    print(f"\n📁 DDTI数据集:")
    if results['DDTI']['exists']:
        print("  ✅ 数据集存在")
        if results['DDTI'].get('matched', False):
            total = results['DDTI']['total_files']
            train_count = int(0.7 * total)
            val_count = int(0.15 * total)
            test_count = total - train_count - val_count
            
            print(f"  📊 总文件数: {total}")
            print(f"  📊 训练集: {train_count} (70%)")
            print(f"  📊 验证集: {val_count} (15%)")
            print(f"  📊 测试集: {test_count} (15%)")
        else:
            print(f"  ❌ 图像和掩码数量不匹配")
    else:
        print("  ❌ 数据集不存在")
    
    print("\n" + "="*60)
    
    return results


if __name__ == "__main__":
    # 测试数据路径配置
    import sys
    
    if len(sys.argv) > 1:
        data_root = sys.argv[1]
    else:
        data_root = "."  # 当前目录
    
    print(f"检查数据根目录: {data_root}")
    results = print_data_summary(data_root)
    
    # 测试路径获取
    print("\n🔍 路径测试:")
    try:
        for dataset in ['TN3K', 'DDTI']:
            for split in ['train', 'val', 'test']:
                img_dir, mask_dir = get_data_paths(data_root, dataset, split)
                exists = os.path.exists(img_dir) and os.path.exists(mask_dir)
                status = "✅" if exists else "❌"
                print(f"  {status} {dataset} {split}: {img_dir}")
    except Exception as e:
        print(f"  ❌ 路径测试失败: {e}")
