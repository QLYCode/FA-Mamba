import math
import torch.nn as nn
import torch.nn.functional as F
import torch


# Setup device
device_id = 3
device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(device_id)  #

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
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
            self.get_dct_filter(height, width, mapper_x, mapper_y, channel)  # [C,H,W]
        )

    def forward(self, x):
        n, c, h, w = x.shape
        assert c == self.channel, f"Input C={c} != DCT C={self.channel}"
        x = x * self.weight.unsqueeze(0).to(x.device)   # [N,C,H,W]
        return torch.sum(x, dim=(2, 3))                 # [N,C]

    def build_filter(self, pos, freq, POS):
        val = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        return val if freq == 0 else val * math.sqrt(2)

    def get_dct_filter(self, H, W, mapper_x, mapper_y, C):
        dct_filter = torch.zeros(C, H, W)
        c_part = C // len(mapper_x)
        for i, (ux, vy) in enumerate(zip(mapper_x, mapper_y)):
            for x in range(H):
                for y in range(W):
                    dct_filter[i*c_part:(i+1)*c_part, x, y] = \
                        self.build_filter(x, ux, H) * self.build_filter(y, vy, W)
        return dct_filter


class FrequencyChannelAttention(nn.Module):
    def __init__(self, channel_hint, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        """
        channel_hint: 初始通道提示（不强制），真实通道以首次forward的输入为准
        """
        super().__init__()
        self.dct_h, self.dct_w = dct_h, dct_w
        self.reduction = reduction
        self.freq_sel_method = freq_sel_method

        self.channel = None
        self.dct_layer = None
        self.fc = None

    def _get_mappers(self):
        mapper_x, mapper_y = get_freq_indices(self.freq_sel_method)
        mapper_x = [ux * (self.dct_h // 7) for ux in mapper_x]
        mapper_y = [vy * (self.dct_w // 7) for vy in mapper_y]
        return mapper_x, mapper_y

    def _build_if_needed(self, C):
        if self.channel == C and self.dct_layer is not None and self.fc is not None:
            return
        mapper_x, mapper_y = self._get_mappers()
        self.dct_layer = MultiSpectralDCTLayer(self.dct_h, self.dct_w, mapper_x, mapper_y, C)

        mid = max(1, C // self.reduction)
        self.fc = nn.Sequential(
            nn.Linear(C, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, C, bias=False),
            nn.Sigmoid()
        )
        self.channel = C

    def forward(self, x):
        n, c, h, w = x.shape
        self._build_if_needed(c)
        x_device = x.device

        if next(self.fc.parameters()).device != x_device:
            self.fc.to(x_device)
        if any(p.device != x_device for p in self.dct_layer.parameters()):
            self.dct_layer.to(x_device)

        x_pooled = x if (h == self.dct_h and w == self.dct_w) \
                   else F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))

        y = self.dct_layer(x_pooled)  # -> x_device
        y = self.fc(y)  # -> x_device
        y = y.view(n, c, 1, 1)

        return x * y


# -------------------------------------------------------------------------
# Fix: 按论文 Eq.(5-8) 重写 Spatial Frequency Transform
#
# 论文流程：
#   1. 1×1 Conv 将 C 通道压缩为单通道 x' ∈ R^{1×H×W}
#   2. 分为 h×w (8×8) 的 patch，每个 patch 做 2D DCT
#   3. zigzag scan 保留 Is 中的 16 个系数，其余位置置零，重组为
#      sparse freq map F_req_hw ∈ R^{1×H×W}
#   4. Conv1×1(1→1) 可学习缩放 → sigmoid → 空间门控 → 逐点乘原特征
#
# 原代码问题：
#   1. 用 x.mean(dim=1) 代替 1×1 Conv 做通道压缩          ← 改
#   2. 输出形状为 [B, M, H, W]（M 个频率基独立保留），
#      而非论文的单通道 sparse map（选中系数保留原位，其余置零）← 改
#   3. proj: Conv2d(M→C) 而非 Conv2d(1→1)               ← 改
# -------------------------------------------------------------------------

def _build_zigzag_indices(h: int, w: int, num_freq: int):
    """
    按 zigzag 顺序生成 (h, w) 块内的坐标列表，取前 num_freq 个。
    返回 (rows, cols) 两个列表，长度均为 num_freq。
    """
    coords = []
    for s in range(h + w - 1):
        if s % 2 == 0:
            r_start = min(s, h - 1)
            c_start = s - r_start
            while r_start >= 0 and c_start < w:
                coords.append((r_start, c_start))
                r_start -= 1
                c_start += 1
        else:
            c_start = min(s, w - 1)
            r_start = s - c_start
            while c_start >= 0 and r_start < h:
                coords.append((r_start, c_start))
                r_start += 1
                c_start -= 1
    rows = [c[0] for c in coords[:num_freq]]
    cols = [c[1] for c in coords[:num_freq]]
    return rows, cols


class MultiSpectralPatchDCT2D(nn.Module):
    """
    按论文 Eq.(5-8) 实现的 Spatial Frequency Transform 核心层。

    输入:  x_single [B, 1, H, W]  ← 已由 1×1 Conv 压缩为单通道
    输出:  F_req_hw [B, 1, H, W]  ← sparse freq map（选中系数保留原位，其余置零）

    步骤：
      ① 将 x_single 按 (ph, pw) 分块（unfold）
      ② 每块做 2D DCT
      ③ 按 zigzag 顺序保留前 num_freq 个系数，其余置零
      ④ 将处理后的块重组回 (H, W)（fold）
    """

    def __init__(self, patch_h: int, patch_w: int, num_freq: int = 16):
        super().__init__()
        self.ph = int(patch_h)
        self.pw = int(patch_w)
        self.num_freq = num_freq

        # 预计算 2D DCT 基矩阵 [ph*pw, ph*pw]，用于整块 DCT
        # dct_mat[k, n] = cos(pi*(n+0.5)*k/N) * norm
        ph, pw = self.ph, self.pw
        mat_h = self._dct1d_matrix(ph)  # [ph, ph]
        mat_w = self._dct1d_matrix(pw)  # [pw, pw]
        # 2D DCT: D = mat_h @ X @ mat_w.T，向量化为 kron 积
        dct2d = torch.kron(mat_h, mat_w)  # [ph*pw, ph*pw]
        self.register_buffer('dct2d', dct2d)

        # zigzag mask：在 ph×pw 块中标记保留位置
        rows, cols = _build_zigzag_indices(ph, pw, num_freq)
        flat_indices = [r * pw + c for r, c in zip(rows, cols)]
        mask = torch.zeros(ph * pw, dtype=torch.bool)
        mask[flat_indices] = True
        self.register_buffer('zigzag_mask', mask)  # [ph*pw]

    @staticmethod
    def _dct1d_matrix(N: int) -> torch.Tensor:
        """正交 DCT-II 基矩阵 [N, N]"""
        k = torch.arange(N).float()
        n = torch.arange(N).float()
        mat = torch.cos(math.pi / N * (n.unsqueeze(0) + 0.5) * k.unsqueeze(1))  # [N, N]
        mat[0] /= math.sqrt(N)
        mat[1:] /= math.sqrt(N / 2)
        return mat

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
        """
        x: [B, 1, H, W]
        返回 F_req_hw: [B, 1, H, W]，sparse freq map
        """
        B, C, H, W = x.shape
        assert C == 1, "MultiSpectralPatchDCT2D 期望单通道输入"
        ph, pw = self.ph, self.pw

        # 1) pad
        x_pad, pad_hw = self._pad_to_multiple(x, ph, pw)
        Hpad, Wpad = x_pad.shape[-2], x_pad.shape[-1]
        Hp, Wp = Hpad // ph, Wpad // pw
        L = Hp * Wp  # 总 patch 数

        # 2) unfold: [B, ph*pw, L]
        patches = F.unfold(x_pad, kernel_size=(ph, pw), stride=(ph, pw))

        # 3) 2D DCT（矩阵乘法形式）: [B, ph*pw, L]
        dct2d = self.dct2d.to(dtype=patches.dtype)       # [ph*pw, ph*pw]
        dct_patches = torch.einsum('mn, bnl -> bml', dct2d, patches)  # [B, ph*pw, L]

        # 4) zigzag 掩码：非选中系数置零
        mask = self.zigzag_mask.to(dct_patches.device)   # [ph*pw]
        dct_patches = dct_patches * mask.unsqueeze(0).unsqueeze(-1)  # [B, ph*pw, L]

        # 5) fold 重组回空间图
        freq_map_pad = F.fold(
            dct_patches,
            output_size=(Hpad, Wpad),
            kernel_size=(ph, pw),
            stride=(ph, pw)
        )  # [B, ph*pw, Hpad, Wpad] ← fold 要求 C_out=ph*pw，不符合

        # fold 输出 C = ph*pw（每个像素只被一个 patch 覆盖，无重叠），
        # 我们需要把 ph*pw 维"展开"回 (ph, pw) 并还原为单通道。
        # 更直接：reshape 后直接拼接
        dct_patches_r = dct_patches.view(B, ph, pw, L)          # [B, ph, pw, L]
        dct_patches_r = dct_patches_r.permute(0, 3, 1, 2)        # [B, L, ph, pw]
        dct_patches_r = dct_patches_r.contiguous().view(B * L, 1, ph, pw)

        # 还原到 patch 网格
        freq_grid = dct_patches_r.view(B, Hp, Wp, ph, pw)        # [B, Hp, Wp, ph, pw]
        freq_map_pad = freq_grid.permute(0, 1, 3, 2, 4).contiguous()  # [B, Hp, ph, Wp, pw]
        freq_map_pad = freq_map_pad.view(B, 1, Hpad, Wpad)        # [B, 1, Hpad, Wpad]

        # 6) 裁掉 padding
        freq_map = self._crop(freq_map_pad, pad_hw)               # [B, 1, H, W]
        return freq_map


class FrequencySpatialAttention(nn.Module):
    """
    Spatial Frequency Transform (SFT)，严格按论文 Eq.(5-8) 实现：

    论文：
      ① 1×1 Conv: C → 1
      ② 8×8 patch DCT + zigzag 保留 num_freq 个系数 → sparse map [1×H×W]
      ③ 1×1 Conv(1→1): 可学习缩放
      ④ sigmoid → 空间门控

    原代码偏差（已修正）：
      - 通道压缩改为 nn.Conv2d(channel, 1, 1)（原为 x.mean）
      - DCT 核心改为单通道 sparse map（原为 M 通道独立内积）
      - 最终投影改为 Conv2d(1→1)（原为 Conv2d(M→C)）
    """
    def __init__(self, channel=32, patch_h=8, patch_w=8, num_freq=16):
        super().__init__()
        # ① 1×1 Conv 压缩通道：C → 1
        self.compress = nn.Conv2d(channel, 1, kernel_size=1, bias=False)

        # ② patch DCT + zigzag sparse map
        self.dct_patch = MultiSpectralPatchDCT2D(patch_h, patch_w, num_freq)

        # ③ 1×1 Conv 可学习缩放：1 → 1
        self.proj = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        # ④ sigmoid
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        返回: 与 x 同形，做空间频域门控后的结果
        """
        # ① 压缩为单通道
        x_single = self.compress(x)          # [B, 1, H, W]

        # ② sparse freq map
        freq_hw = self.dct_patch(x_single)   # [B, 1, H, W]

        # ③④ 可学习缩放 + sigmoid → 空间门控
        gate = self.act(self.proj(freq_hw))  # [B, 1, H, W]

        return x * gate                      # 广播到 [B, C, H, W]


class CSFI(nn.Module):

    def __init__(self, channels, dct_h=7, dct_w=7, reduction=16, freq_sel_method='top16'):
        super(CSFI, self).__init__()
        self.fca = FrequencyChannelAttention(
            channel_hint=channels,
            dct_h=dct_h,
            dct_w=dct_w,
            reduction=reduction,
            freq_sel_method=freq_sel_method
        ).to(device)
        # Fix: FrequencySpatialAttention 不再需要 freq_sel_method，
        # num_freq 直接从 freq_sel_method 中解析
        num_freq = int(freq_sel_method[3:])
        self.fsa = FrequencySpatialAttention(
            channel=channels, patch_h=8, patch_w=8, num_freq=num_freq
        )

    def forward(self, x):
        # 论文 Eq.(9): FT(x) = CFT(x) + SFT(x)
        output =  self.fca(x) + self.fsa(x)
        return output