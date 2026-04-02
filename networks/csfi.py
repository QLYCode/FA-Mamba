import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Setup device
device_id = 0
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device_id)


def get_freq_indices(method):
    assert method in [
        'top1',  'top2',  'top4',  'top8',  'top16',  'top32',
        'bot1',  'bot2',  'bot4',  'bot8',  'bot16',  'bot32',
        'low1',  'low2',  'low4',  'low8',  'low16',  'low32',
    ]
    num_freq = int(method[3:])

    if 'top' in method:
        all_top_indices_x = [
            0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2,
            4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1,
        ]
        all_top_indices_y = [
            0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2,
            6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3,
        ]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]

    elif 'low' in method:
        all_low_indices_x = [
            0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0,
            1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4,
        ]
        all_low_indices_y = [
            0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5,
            4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3,
        ]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]

    elif 'bot' in method:
        all_bot_indices_x = [
            6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5,
            6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6,
        ]
        all_bot_indices_y = [
            6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1,
            4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3,
        ]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]

    else:
        raise NotImplementedError

    return mapper_x, mapper_y


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.register_buffer('total_ops', torch.zeros(1, dtype=torch.float64))
        self.channel = channel
        self.register_buffer(
            'weight',
            self.get_dct_filter(height, width, mapper_x, mapper_y, channel),
        )

    def forward(self, x):
        n, c, h, w = x.shape
        assert c == self.channel, f"Input C={c} != DCT C={self.channel}"
        x = x * self.weight.unsqueeze(0).to(x.device)
        return torch.sum(x, dim=(2, 3))

    def build_filter(self, pos, freq, POS):
        val = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        return val if freq == 0 else val * math.sqrt(2)

    def get_dct_filter(self, H, W, mapper_x, mapper_y, C):
        dct_filter = torch.zeros(C, H, W)
        c_part = C // len(mapper_x)
        for i, (ux, vy) in enumerate(zip(mapper_x, mapper_y)):
            for x in range(H):
                for y in range(W):
                    dct_filter[i * c_part:(i + 1) * c_part, x, y] = (
                        self.build_filter(x, ux, H) * self.build_filter(y, vy, W)
                    )
        return dct_filter


class FrequencyChannelAttention(nn.Module):
    """
    通道频率注意力。
    Fix: 在 __init__ 中立即构建 dct_layer 和 fc，确保它们被注册为正式子模块，
    state_dict 保存/加载和多卡训练都不会出现权重重置问题。
    返回 x * channel_attention。
    """
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super().__init__()
        self.dct_h = dct_h
        self.dct_w = dct_w
        self.reduction = reduction
        self.freq_sel_method = freq_sel_method

        # 立即构建，不延迟——保证子模块在 state_dict 中
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        mapper_x = [ux * (dct_h // 7) for ux in mapper_x]
        mapper_y = [vy * (dct_w // 7) for vy in mapper_y]

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)

        mid = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = (
            x if (h == self.dct_h and w == self.dct_w)
            else F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        )
        y = self.dct_layer(x_pooled)  # [N, C]
        y = self.fc(y)                # [N, C]，(0, 1)
        y = y.view(n, c, 1, 1)
        return x * y


class MultiSpectralPatchDCT2D(nn.Module):
    """
    以 patch 为单位的 2D-DCT 投影层。
    输入:  x [B, C, H, W]
    输出:  Freq_hw [B, M, H, W]
    """

    def __init__(self, patch_h: int, patch_w: int, mapper_x, mapper_y):
        super().__init__()
        assert len(mapper_x) == len(mapper_y)
        self.ph = int(patch_h)
        self.pw = int(patch_w)
        self.num_freq = len(mapper_x)

        bases = []
        for u, v in zip(mapper_x, mapper_y):
            Bu = torch.cos(torch.pi * (torch.arange(self.ph) + 0.5) * u / self.ph) / math.sqrt(self.ph)
            if u != 0:
                Bu = Bu * math.sqrt(2)
            Bv = torch.cos(torch.pi * (torch.arange(self.pw) + 0.5) * v / self.pw) / math.sqrt(self.pw)
            if v != 0:
                Bv = Bv * math.sqrt(2)
            bases.append(torch.ger(Bu, Bv))
        self.register_buffer('dct_bases', torch.stack(bases, dim=0))  # [M, ph, pw]

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, ph: int, pw: int):
        H, W = x.shape[-2:]
        pad_h = (ph - H % ph) % ph
        pad_w = (pw - W % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        return x, (pad_h, pad_w)

    @staticmethod
    def _crop(x: torch.Tensor, pad_hw):
        pad_h, pad_w = pad_hw
        if pad_h == 0 and pad_w == 0:
            return x
        return x[..., :-pad_h if pad_h else None, :-pad_w if pad_w else None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ph, pw = self.ph, self.pw
        M = self.num_freq

        x_mean = x.mean(dim=1, keepdim=True)
        x_mean, pad_hw = self._pad_to_multiple(x_mean, ph, pw)
        Hpad, Wpad = x_mean.shape[-2], x_mean.shape[-1]
        Hp, Wp = Hpad // ph, Wpad // pw

        patches = F.unfold(x_mean, kernel_size=(ph, pw), stride=(ph, pw))

        bases = self.dct_bases.to(dtype=patches.dtype, device=patches.device)
        bases_flat = bases.view(M, ph * pw)
        coeff = torch.einsum('b n l, m n -> b m l', patches, bases_flat)

        coeff_grid = coeff.view(B, M, Hp, Wp)
        freq_map = F.interpolate(coeff_grid, scale_factor=(ph, pw), mode='nearest')
        freq_map = self._crop(freq_map, pad_hw)
        return freq_map


class FrequencySpatialAttention(nn.Module):
    """
    空间频率注意力。返回 x * spatial_gate。
    """

    def __init__(self, channel=32, patch_h=8, patch_w=8, freq_sel_method='low8'):
        super().__init__()
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        mapper_x = [ux * (patch_h // 8) for ux in mapper_x]
        mapper_y = [vy * (patch_w // 8) for vy in mapper_y]

        self.dct_patch = MultiSpectralPatchDCT2D(patch_h, patch_w, mapper_x, mapper_y)
        M = len(mapper_x)

        self.proj = nn.Conv2d(M, channel, kernel_size=1, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_hw = self.dct_patch(x)
        gate = self.act(self.proj(freq_hw))  # [B, C, H, W]，(0, 1)
        return x * gate


class CSFI(nn.Module):
    """
    Channel-Spatial Frequency Injection。

    output = fca(x) + fsa(x) + x
           = x*attn_c + x*attn_s + x
           = x * (attn_c + attn_s + 1)

    attn_c, attn_s ∈ (0,1)，整体增益在 (1,3)，数值稳定。

    调用方 BasicResBlock.forward 中：
        y = self.csfi(y)     ← 正确，csfi 内部已含残差
        y = self.csfi(y) + x ← 错误，会重复加残差
    """

    def __init__(self, channels, dct_h=7, dct_w=7, reduction=16, freq_sel_method='top16'):
        super(CSFI, self).__init__()
        self.fca = FrequencyChannelAttention(
            channel=channels,
            dct_h=dct_h,
            dct_w=dct_w,
            reduction=reduction,
            freq_sel_method=freq_sel_method,
        )
        self.fsa = FrequencySpatialAttention(
            channel=channels,
            patch_h=8,
            patch_w=8,
            freq_sel_method=freq_sel_method,
        )

    def forward(self, x):
        return self.fca(x) + self.fsa(x) + x