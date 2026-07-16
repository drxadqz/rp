from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

try:
    from timm.layers import DropPath
except Exception:  # pragma: no cover - timm is present in the paper env.
    class DropPath(nn.Identity):
        pass


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> None:
    try:
        nn.init.trunc_normal_(tensor, std=std)
    except AttributeError:  # pragma: no cover
        nn.init.normal_(tensor, std=std)


def create_haar_2d_wavelet_filter(in_size: int, out_size: int, dtype: torch.dtype = torch.float) -> tuple[torch.Tensor, torch.Tensor]:
    """Create fixed Haar analysis/synthesis filters used by RSPNet WTConv."""

    inv_sqrt2 = 2.0 ** -0.5
    dec_lo = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=dtype)
    dec_hi = torch.tensor([inv_sqrt2, -inv_sqrt2], dtype=dtype)
    rec_lo = torch.tensor([inv_sqrt2, inv_sqrt2], dtype=dtype)
    rec_hi = torch.tensor([inv_sqrt2, -inv_sqrt2], dtype=dtype)
    dec_filters = torch.stack(
        [
            dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
            dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    rec_filters = torch.stack(
        [
            rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
            rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
        ],
        dim=0,
    )
    return dec_filters[:, None].repeat(in_size, 1, 1, 1), rec_filters[:, None].repeat(out_size, 1, 1, 1)


def wavelet_2d_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    out = F.conv2d(x, filters, stride=2, groups=channels, padding=pad)
    return out.reshape(batch, channels, 4, height // 2, width // 2)


def inverse_2d_wavelet_transform(x: torch.Tensor, filters: torch.Tensor) -> torch.Tensor:
    batch, channels, _, height_half, width_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    out = x.reshape(batch, channels * 4, height_half, width_half)
    return F.conv_transpose2d(out, filters, stride=2, groups=channels, padding=pad)


class ScaleModule(nn.Module):
    def __init__(self, dims: list[int], init_scale: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(*dims) * float(init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x


class WTConv2d(nn.Module):
    """Depthwise convolution in Haar wavelet sub-bands.

    This keeps RSPNet's frequency-texture inductive bias without requiring
    PyWavelets in the experiment environment.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        bias: bool = True,
        wt_levels: int = 1,
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("WTConv2d requires in_channels == out_channels.")
        self.in_channels = int(in_channels)
        self.wt_levels = int(wt_levels)
        wt_filter, iwt_filter = create_haar_2d_wavelet_filter(self.in_channels, self.in_channels, torch.float)
        self.wt_filter = nn.Parameter(wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(iwt_filter, requires_grad=False)
        padding = int(kernel_size) // 2
        self.base_conv = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size,
            padding=padding,
            stride=1,
            dilation=1,
            groups=self.in_channels,
            bias=bias,
        )
        self.base_scale = ScaleModule([1, self.in_channels, 1, 1])
        self.wavelet_convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.in_channels * 4,
                    self.in_channels * 4,
                    kernel_size,
                    padding=padding,
                    stride=1,
                    dilation=1,
                    groups=self.in_channels * 4,
                    bias=False,
                )
                for _ in range(self.wt_levels)
            ]
        )
        self.wavelet_scale = nn.ModuleList(
            [ScaleModule([1, self.in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )
        self.do_stride = nn.AvgPool2d(kernel_size=1, stride=int(stride)) if int(stride) > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []
        curr_x_ll = x
        for level in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if curr_shape[2] % 2 > 0 or curr_shape[3] % 2 > 0:
                curr_x_ll = F.pad(curr_x_ll, (0, curr_shape[3] % 2, 0, curr_shape[2] % 2))
            curr_x = wavelet_2d_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]
            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[level](self.wavelet_convs[level](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)
            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll: torch.Tensor | int = 0
        for _ in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()
            curr_x_ll = curr_x_ll + next_x_ll
            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_2d_wavelet_transform(curr_x, self.iwt_filter)
            next_x_ll = next_x_ll[:, :, : curr_shape[2], : curr_shape[3]]
        x_tag = next_x_ll
        x_base = self.base_scale(self.base_conv(x))
        out = x_base + x_tag
        if self.do_stride is not None:
            out = self.do_stride(out)
        return out


class ConvBN(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        with_bn: bool = True,
    ) -> None:
        super().__init__()
        self.add_module("conv", nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups))
        if with_bn:
            self.add_module("bn", nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 0.0)


class MultiScaleConv(nn.Module):
    def __init__(self, in_channels: int, n: int = 2, activation: nn.Module | None = None) -> None:
        super().__init__()
        self.n = int(n)
        if in_channels < self.n:
            raise ValueError("in_channels must be >= number of scales.")
        activation = activation if activation is not None else nn.ReLU6(inplace=True)
        base_channels = in_channels // self.n
        remainder = in_channels % self.n
        self.channel_allocations = [base_channels + (1 if idx < remainder else 0) for idx in range(self.n)]
        self.branches = nn.ModuleList()
        for idx, channels in enumerate(self.channel_allocations):
            kernel_size = 2 * (idx + 1) + 1
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False, groups=channels),
                    nn.BatchNorm2d(channels),
                    activation,
                )
            )
        self.fusion = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, 1), nn.BatchNorm2d(in_channels), activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        chunks = []
        start = 0
        for channels in self.channel_allocations:
            end = start + channels
            chunks.append(x[:, start:end])
            start = end
        out = torch.cat([branch(chunk) for branch, chunk in zip(self.branches, chunks, strict=True)], dim=1)
        return self.fusion(out) + identity


class FFN(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            ConvBN(dim, dim, 1, 1, 0, 1, 1, True),
            nn.ReLU6(),
            ConvBN(dim, dim, 1, 1, 0, 1, 1, True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class Block(nn.Module):
    def __init__(self, dim: int, levels: int, mlp_ratio: int = 3, drop_path: float = 0.0, ratios: tuple[float, float, float] = (0.6, 0.3, 0.1)) -> None:
        super().__init__()
        if abs(sum(ratios) - 1.0) >= 1e-6:
            raise ValueError("branch ratios must sum to 1.")
        self.channels = [int(dim * ratio) for ratio in ratios]
        self.channels[-1] = int(dim) - sum(self.channels[:-1])
        self.has_branch1 = self.channels[0] > 0
        self.has_branch2 = self.channels[1] > 0
        self.has_branch3 = self.channels[2] > 0
        if self.has_branch1:
            channels = self.channels[0]
            self.WTConv = WTConv2d(channels, channels, kernel_size=3, stride=1, bias=True, wt_levels=levels)
            self.dwconv = ConvBN(channels, channels, 7, 1, 3, groups=channels, with_bn=True)
            self.f1 = ConvBN(channels, mlp_ratio * channels, 1, with_bn=False)
            self.f2 = ConvBN(channels, mlp_ratio * channels, 1, with_bn=False)
            self.g = ConvBN(mlp_ratio * channels, channels, 1, with_bn=True)
            self.dwconv2 = ConvBN(channels, channels, 7, 1, 3, groups=channels, with_bn=False)
        if self.has_branch2:
            self.MKC = MultiScaleConv(self.channels[1], 2)
        self.ffn = FFN(dim)
        self.act = nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y1, y2, y3 = torch.split(z, self.channels, dim=1)
        outputs = []
        if self.has_branch1:
            y1_wtc = self.WTConv(y1)
            x = self.dwconv(y1)
            x1, x2 = self.f1(x), self.f2(x)
            x = self.act(x1) * x2
            x = self.dwconv2(self.g(x))
            outputs.append(y1_wtc + y1 + self.drop_path(x))
        if self.has_branch2:
            outputs.append(self.MKC(y2))
        if self.has_branch3:
            outputs.append(y3)
        return self.ffn(torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0])


class RSPNet(nn.Module):
    def __init__(
        self,
        mlp_ratio: int,
        base_dim: int = 32,
        depths: list[int] | tuple[int, ...] = (1, 3, 6, 3),
        drop_path_rate: float = 0.0,
        num_classes: int = 27,
        wt_levels: list[int] | tuple[int, ...] = (1, 1, 1, 1),
        branch_ratios: list[tuple[float, float, float]] | tuple[float, float, float] | None = None,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.in_channel = 32
        if len(wt_levels) != len(depths):
            raise ValueError("wt_levels must have the same length as depths.")
        if branch_ratios is None:
            branch_ratios = [(0.6, 0.3, 0.1)] * len(depths)
        elif isinstance(branch_ratios, tuple) and len(branch_ratios) == 3 and isinstance(branch_ratios[0], float):
            branch_ratios = [branch_ratios] * len(depths)
        elif len(branch_ratios) != len(depths):
            raise ValueError("branch_ratios must have the same length as depths.")
        self.stem = nn.Sequential(ConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1), nn.ReLU6())
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i_layer, depth in enumerate(depths):
            embed_dim = int(base_dim) * 2**i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [
                Block(
                    dim=self.in_channel,
                    levels=int(wt_levels[i_layer]),
                    mlp_ratio=int(mlp_ratio),
                    drop_path=dpr[cur + block_idx],
                    ratios=branch_ratios[i_layer],
                )
                for block_idx in range(int(depth))
            ]
            cur += int(depth)
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        self.norm = nn.BatchNorm2d(self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.in_channel, self.num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            _trunc_normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


def rspnet_s(num_classes: int = 27, branch_ratios: list[tuple[float, float, float]] | None = None) -> RSPNet:
    if branch_ratios is None:
        branch_ratios = [(0.5, 0.5, 0.0), (0.5, 0.5, 0.0), (0.5, 0.4, 0.1), (0.5, 0.4, 0.1)]
    return RSPNet(
        mlp_ratio=2,
        base_dim=32,
        depths=[1, 2, 6, 2],
        num_classes=num_classes,
        wt_levels=[2, 1, 1, 1],
        branch_ratios=branch_ratios,
    )


def rspnet_m(num_classes: int = 27, branch_ratios: list[tuple[float, float, float]] | None = None) -> RSPNet:
    if branch_ratios is None:
        branch_ratios = [(0.5, 0.5, 0.0), (0.5, 0.5, 0.0), (0.5, 0.4, 0.1), (0.5, 0.4, 0.1)]
    return RSPNet(
        mlp_ratio=2,
        base_dim=40,
        depths=[1, 2, 6, 2],
        num_classes=num_classes,
        wt_levels=[3, 2, 1, 1],
        branch_ratios=branch_ratios,
    )


def rspnet_l(num_classes: int = 27, branch_ratios: list[tuple[float, float, float]] | None = None) -> RSPNet:
    if branch_ratios is None:
        branch_ratios = [(0.5, 0.5, 0.0), (0.5, 0.5, 0.0), (0.5, 0.4, 0.1), (0.5, 0.4, 0.1)]
    return RSPNet(
        mlp_ratio=2,
        base_dim=48,
        depths=[1, 3, 6, 3],
        num_classes=num_classes,
        wt_levels=[3, 2, 1, 1],
        branch_ratios=branch_ratios,
    )


class RSPNetFeatureBackbone(nn.Module):
    def __init__(self, variant: str, out_dim: int) -> None:
        super().__init__()
        builders = {"s": rspnet_s, "m": rspnet_m, "l": rspnet_l}
        if variant not in builders:
            raise ValueError(f"Unknown RSPNet variant: {variant}")
        self.model = builders[variant](num_classes=27)
        self.proj = nn.Identity() if int(out_dim) == int(self.model.in_channel) else nn.Linear(self.model.in_channel, out_dim)
        self.out_dim = int(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.model.forward_features(x))
