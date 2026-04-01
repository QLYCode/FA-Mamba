#!/usr/bin/env python3
"""
Test script to verify all models can be loaded and run forward pass
"""

import sys
import torch
import traceback
import warnings
warnings.filterwarnings('ignore')

# Add networks to path
sys.path.append('networks')
from networks.net_factory import net_factory


def test_model(model_name, input_shape=(1, 3, 512, 512), num_classes=2):
    """
    Test if a model can be loaded and perform forward pass
    
    Args:
        model_name: Name of the model to test
        input_shape: Input tensor shape (B, C, H, W)
        num_classes: Number of output classes
    
    Returns:
        dict: Test results
    """
    print(f"\nTesting {model_name}...")
    print("-" * 40)
    
    try:
        # Create model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        model = net_factory(
            net_type=model_name,
            in_chns=input_shape[1],
            class_num=num_classes,
            device=device
        )
        
        if model is None:
            return {
                'status': 'failed',
                'error': f'Model {model_name} not found in net_factory',
                'parameters': 0,
                'output_shape': None
            }
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            dummy_input = torch.randn(input_shape).to(device)
            print(f"Input shape: {dummy_input.shape}")
            
            output = model(dummy_input)
            
            if isinstance(output, (list, tuple)):
                output_shape = [o.shape for o in output]
                print(f"Output shapes: {output_shape}")
            else:
                output_shape = output.shape
                print(f"Output shape: {output_shape}")
            
            # Check if output has correct number of classes
            if isinstance(output, torch.Tensor):
                if len(output.shape) == 4:  # (B, C, H, W)
                    if output.shape[1] != num_classes:
                        print(f"Warning: Expected {num_classes} classes, got {output.shape[1]}")
                elif len(output.shape) == 3:  # (B, H, W) for binary segmentation
                    if num_classes == 2:
                        print("Binary segmentation output detected")
                    else:
                        print(f"Warning: Expected {num_classes} classes, got binary output")
        
        print(f"✅ {model_name} test passed!")
        
        return {
            'status': 'success',
            'error': None,
            'parameters': total_params,
            'trainable_parameters': trainable_params,
            'output_shape': output_shape
        }
        
    except Exception as e:
        print(f"❌ {model_name} test failed!")
        print(f"Error: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        
        return {
            'status': 'failed',
            'error': str(e),
            'parameters': 0,
            'trainable_parameters': 0,
            'output_shape': None
        }


def main():
    """Test all target models"""
    print("="*60)
    print("MODEL TESTING SCRIPT")
    print("="*60)
    
    # Models to test
    target_models = ['swinumamba', 'lkmunet', 'segmamba', 'emnet']
    
    # Test configuration
    input_shape = (2, 3, 512, 512)  # Batch size 2, RGB, 512x512
    num_classes = 2  # Binary segmentation
    
    print(f"Test configuration:")
    print(f"- Input shape: {input_shape}")
    print(f"- Number of classes: {num_classes}")
    print(f"- Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    if torch.cuda.is_available():
        print(f"- GPU: {torch.cuda.get_device_name()}")
        print(f"- GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test each model
    results = {}
    for model_name in target_models:
        results[model_name] = test_model(model_name, input_shape, num_classes)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    successful_models = []
    failed_models = []
    
    for model_name, result in results.items():
        status = "✅ PASS" if result['status'] == 'success' else "❌ FAIL"
        params = f"{result['parameters']:,}" if result['parameters'] > 0 else "N/A"
        
        print(f"{model_name:<15} {status:<8} Parameters: {params}")
        
        if result['status'] == 'success':
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
            print(f"  Error: {result['error']}")
    
    print(f"\nResults: {len(successful_models)}/{len(target_models)} models passed")
    
    if successful_models:
        print(f"✅ Working models: {', '.join(successful_models)}")
    
    if failed_models:
        print(f"❌ Failed models: {', '.join(failed_models)}")
        print("\nTo fix failed models:")
        print("1. Check if required dependencies are installed")
        print("2. Install missing packages: pip install -r requirements.txt")
        print("3. For Mamba models: pip install mamba-ssm causal-conv1d")
        print("4. For MONAI models: pip install monai")
        print("5. Check model implementation files in networks/")
    
    # Additional tests
    print(f"\n" + "="*60)
    print("ADDITIONAL TESTS")
    print("="*60)
    
    # Test different input sizes
    if successful_models:
        test_model_name = successful_models[0]
        print(f"\nTesting {test_model_name} with different input sizes:")
        
        test_sizes = [(1, 3, 256, 256), (1, 3, 768, 768)]
        for test_shape in test_sizes:
            try:
                result = test_model(test_model_name, test_shape, num_classes)
                status = "✅" if result['status'] == 'success' else "❌"
                print(f"  {test_shape}: {status}")
            except Exception as e:
                print(f"  {test_shape}: ❌ (Error: {str(e)})")
    
    print(f"\nTesting complete!")
    
    return len(failed_models) == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
