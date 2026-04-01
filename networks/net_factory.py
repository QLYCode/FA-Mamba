import argparse
from random import random
import numpy as np
from torch.backends import cudnn
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def net_factory(net_type="unet", in_chns=3, class_num=2, device=None, use_channel_converter=False, conv_type='3x3', reverse_converter=False):
    """
    Create network with optional channel converter

    Args:
        net_type: Type of network to create
        in_chns: Input channels (will be used for channel converter if enabled)
        class_num: Number of output classes
        device: Device to place the network on
        use_channel_converter: If True, adds a channel converter before the network
        conv_type: Type of convolution for channel converter ('1x1' or '3x3')
        reverse_converter: If True, converts 1->3 channels (for models expecting RGB)

    Returns:
        Network model (optionally wrapped with channel converter)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if net_type == "famamba":
        from networks.famamba import UMambaBot
        net = UMambaBot(input_channels=3,
                        n_stages=4,  # 包含stem的4个编码阶段
                        features_per_stage=[32, 64, 128, 256],  # 各阶段特征通道数
                        conv_op=nn.Conv2d,
                        kernel_sizes=[[3,3], [3,3], [3,3], [3,3]],  # 每个阶段的卷积核尺寸
                        strides=[1, 2, 2, 2],  # 下采样策略（第一次保持分辨率）
                        n_conv_per_stage=[2, 2, 2, 2],  # 编码器各阶段卷积层数
                        num_classes=2,
                        n_conv_per_stage_decoder=[2, 2, 2],  # 解码器各阶段卷积层数（比编码器少1阶）
                        conv_bias=False,
                        norm_op=nn.BatchNorm2d,
                        norm_op_kwargs={"eps": 1e-5, "momentum": 0.1},
                        nonlin=nn.LeakyReLU,
                        nonlin_kwargs={"inplace": True},
                        deep_supervision=False
                        ).to(device)
    elif net_type == "lkmunet":
        from networks.lkmunet import LKMUNet
        net = LKMUNet(in_channels=actual_in_chns, out_channels=class_num, kernel_sizes=[21, 15, 9, 3]).to(device)


    else:
        net = None

    # Apply channel converter if requested
    if use_channel_converter and net is not None:
        if reverse_converter:
            # For models expecting RGB input (like SegMamba)
            from networks.channel_converter import GrayscaleToRGBConverter

            class ReversedNetworkWrapper(nn.Module):
                def __init__(self, base_network):
                    super().__init__()
                    self.converter = GrayscaleToRGBConverter(method='replicate')
                    self.base_network = base_network

                def forward(self, x):
                    print(f"[DEBUG] 包装器输入形状: {x.shape}")
                    x_rgb = self.converter(x)  # 1 -> 3 channels
                    print(f"[DEBUG] 转换后形状: {x_rgb.shape}")
                    output = self.base_network(x_rgb)
                    print(f"[DEBUG] 网络输出形状: {output.shape}")
                    return output

            net = ReversedNetworkWrapper(net).to(device)
            print(f"Applied reverse channel converter: {in_chns} -> 3 channels (grayscale to RGB)")
        else:
            # Normal converter: RGB to grayscale
            from networks.channel_converter import NetworkWithChannelConverter
            net = NetworkWithChannelConverter(
                base_network=net,
                conv_type=conv_type,
                input_channels=in_chns,
                output_channels=1
            ).to(device)
            print(f"Applied channel converter: {in_chns} -> 1 channels using {conv_type} convolution")

    return net

