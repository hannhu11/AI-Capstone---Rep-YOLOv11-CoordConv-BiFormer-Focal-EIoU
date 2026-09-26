"""
Research modules for Stage 2 custom ablations.

These modules are intentionally self-contained. They are not wired into the
baseline notebook because Stage 1 should remain stable and comparable. Use them
after selecting the top-2 backbones from `benchmark_results.csv`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k: int, p: int | None = None, d: int = 1) -> int:
    """Return padding that preserves spatial shape for odd kernels."""
    if p is None:
        p = d * (k - 1) // 2
    return p


class ConvBNAct(nn.Module):
    """Small Conv-BN-SiLU block compatible with YOLO-style modules."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        d: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class CoordConv(nn.Module):
    """
    CoordConv layer: append normalized x/y coordinate channels before a conv.

    Use sparingly in the stem or selected neck layers. Replacing every conv can
    overfit fixed camera geometry.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, with_r: bool = False) -> None:
        super().__init__()
        self.with_r = with_r
        extra = 3 if with_r else 2
        self.conv = ConvBNAct(c1 + extra, c2, k=k, s=s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        yy = torch.linspace(-1.0, 1.0, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
        coords = [xx, yy]
        if self.with_r:
            rr = torch.sqrt(torch.clamp(xx.square() + yy.square(), min=0.0))
            coords.append(rr)
        return self.conv(torch.cat([x, *coords], dim=1))


class RepConv(nn.Module):
    """
    RepConv block with 3x3, 1x1, and optional identity branches during training.

    Call `switch_to_deploy()` before ONNX/TensorRT export. The deploy path is a
    single 3x3 Conv2d plus activation.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, deploy: bool = False, act: bool = True) -> None:
        super().__init__()
        assert k == 3, "This RepConv implementation fuses to a 3x3 kernel."
        self.deploy = deploy
        self.in_channels = c1
        self.out_channels = c2
        self.stride = s
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, 3, s, 1, bias=True)
        else:
            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, 3, s, 1, bias=False),
                nn.BatchNorm2d(c2),
            )
            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d(c1, c2, 1, s, 0, bias=False),
                nn.BatchNorm2d(c2),
            )
            self.rbr_identity = nn.BatchNorm2d(c1) if c1 == c2 and s == 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.deploy:
            return self.act(self.rbr_reparam(x))
        out = self.rbr_dense(x) + self.rbr_1x1(x)
        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)
        return self.act(out)

    @staticmethod
    def _fuse_conv_bn(branch: nn.Sequential | nn.BatchNorm2d, channels: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(branch, nn.Sequential):
            conv = branch[0]
            bn = branch[1]
            kernel = conv.weight
        else:
            bn = branch
            input_dim = channels
            kernel = torch.zeros((input_dim, input_dim, 3, 3), device=bn.weight.device, dtype=bn.weight.dtype)
            for i in range(input_dim):
                kernel[i, i, 1, 1] = 1.0

        std = torch.sqrt(bn.running_var + bn.eps)
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        fused_kernel = kernel * t
        fused_bias = bn.bias - bn.running_mean * bn.weight / std
        return fused_kernel, fused_bias

    @staticmethod
    def _pad_1x1_to_3x3(kernel: torch.Tensor) -> torch.Tensor:
        if kernel.size(2) == 3:
            return kernel
        return F.pad(kernel, [1, 1, 1, 1])

    def get_equivalent_kernel_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        k3, b3 = self._fuse_conv_bn(self.rbr_dense, self.out_channels)
        k1, b1 = self._fuse_conv_bn(self.rbr_1x1, self.out_channels)
        if self.rbr_identity is None:
            kid = torch.zeros_like(k3)
            bid = torch.zeros_like(b3)
        else:
            kid, bid = self._fuse_conv_bn(self.rbr_identity, self.out_channels)
        return k3 + self._pad_1x1_to_3x3(k1) + kid.to(k3.device), b3 + b1 + bid.to(b3.device)

    def switch_to_deploy(self) -> None:
        if self.deploy:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            3,
            self.stride,
            1,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel.detach().clone()
        self.rbr_reparam.bias.data = bias.detach().clone()
        del self.rbr_dense
        del self.rbr_1x1
        if hasattr(self, "rbr_identity"):
            del self.rbr_identity
        self.deploy = True


class BiFormerBlockLite(nn.Module):
    """
    Lightweight BiFormer-inspired block for ablation.

    This is not a drop-in copy of the official BiFormer implementation. It keeps
    the core idea for experiments: region-level routing followed by attention on
    a small set of routed regions. Use it as an ablation candidate, then compare
    latency against ECA/CBAM/Triplet attention.
    """

    def __init__(self, channels: int, num_heads: int = 4, region_size: int = 8, topk: int = 4) -> None:
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.region_size = region_size
        self.topk = topk
        self.qkv = nn.Conv2d(channels, channels * 3, 1, bias=False)
        self.proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        rs = self.region_size
        pad_h = (rs - h % rs) % rs
        pad_w = (rs - w % rs) % rs
        x_pad = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x_pad.shape[-2:]
        gh, gw = hp // rs, wp // rs

        q, k, v = self.qkv(x_pad).chunk(3, dim=1)
        q_regions = q.unfold(2, rs, rs).unfold(3, rs, rs).contiguous()
        k_regions = k.unfold(2, rs, rs).unfold(3, rs, rs).contiguous()
        v_regions = v.unfold(2, rs, rs).unfold(3, rs, rs).contiguous()

        # [B, C, GH, GW, RS, RS] -> [B, R, T, C]
        q_tokens = q_regions.permute(0, 2, 3, 4, 5, 1).reshape(b, gh * gw, rs * rs, c)
        k_tokens = k_regions.permute(0, 2, 3, 4, 5, 1).reshape(b, gh * gw, rs * rs, c)
        v_tokens = v_regions.permute(0, 2, 3, 4, 5, 1).reshape(b, gh * gw, rs * rs, c)

        q_region = q_tokens.mean(dim=2)
        k_region = k_tokens.mean(dim=2)
        route_logits = torch.matmul(q_region, k_region.transpose(-1, -2)) / (c ** 0.5)
        topk = min(self.topk, gh * gw)
        route_idx = route_logits.topk(topk, dim=-1).indices

        out_regions = []
        head_dim = c // self.num_heads
        for region_idx in range(gh * gw):
            selected = route_idx[:, region_idx]
            k_sel = torch.stack([k_tokens[bi, selected[bi]].reshape(topk * rs * rs, c) for bi in range(b)], dim=0)
            v_sel = torch.stack([v_tokens[bi, selected[bi]].reshape(topk * rs * rs, c) for bi in range(b)], dim=0)
            q_cur = q_tokens[:, region_idx]

            qh = q_cur.reshape(b, rs * rs, self.num_heads, head_dim).transpose(1, 2)
            kh = k_sel.reshape(b, topk * rs * rs, self.num_heads, head_dim).transpose(1, 2)
            vh = v_sel.reshape(b, topk * rs * rs, self.num_heads, head_dim).transpose(1, 2)
            attn = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) / (head_dim ** 0.5), dim=-1)
            out = torch.matmul(attn, vh).transpose(1, 2).reshape(b, rs * rs, c)
            out_regions.append(out)

        y = torch.stack(out_regions, dim=1).reshape(b, gh, gw, rs, rs, c)
        y = y.permute(0, 5, 1, 3, 2, 4).reshape(b, c, hp, wp)
        y = y[:, :, :h, :w]
        return x + self.norm(self.proj(y))


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    half_w = w / 2
    half_h = h / 2
    return torch.stack((x - half_w, y - half_h, x + half_w, y + half_h), dim=-1)


def focal_eiou_loss(
    pred_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    xywh: bool = True,
    gamma: float = 0.5,
    eps: float = 1e-7,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal-EIoU box regression loss.

    Args:
        pred_boxes: [N, 4] predicted boxes.
        target_boxes: [N, 4] target boxes.
        xywh: True when boxes are center-x, center-y, width, height.
        gamma: focal exponent from the Focal-EIoU paper family.
    """
    if xywh:
        pred = xywh_to_xyxy(pred_boxes)
        target = xywh_to_xyxy(target_boxes)
    else:
        pred = pred_boxes
        target = target_boxes

    px1, py1, px2, py2 = pred.unbind(-1)
    tx1, ty1, tx2, ty2 = target.unbind(-1)
    pw = (px2 - px1).clamp(min=eps)
    ph = (py2 - py1).clamp(min=eps)
    tw = (tx2 - tx1).clamp(min=eps)
    th = (ty2 - ty1).clamp(min=eps)

    inter_x1 = torch.maximum(px1, tx1)
    inter_y1 = torch.maximum(py1, ty1)
    inter_x2 = torch.minimum(px2, tx2)
    inter_y2 = torch.minimum(py2, ty2)
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    union = pw * ph + tw * th - inter + eps
    iou = (inter / union).clamp(min=eps, max=1.0)

    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2
    center_dist = (pcx - tcx).square() + (pcy - tcy).square()

    cw = (torch.maximum(px2, tx2) - torch.minimum(px1, tx1)).clamp(min=eps)
    ch = (torch.maximum(py2, ty2) - torch.minimum(py1, ty1)).clamp(min=eps)
    c2 = cw.square() + ch.square() + eps

    eiou = 1.0 - iou + center_dist / c2 + (pw - tw).square() / (cw.square() + eps) + (ph - th).square() / (ch.square() + eps)
    loss = iou.pow(gamma) * eiou

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def alpha_balanced_focal_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor | float = 0.75,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Focal BCE for class imbalance.

    For two class logits in this project, pass an alpha tensor such as
    `torch.tensor([0.85, 0.15], device=logits.device)` to weight `hat` higher
    than `person`.
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)

    if not torch.is_tensor(alpha):
        alpha_t = torch.tensor(alpha, device=logits.device, dtype=logits.dtype)
    else:
        alpha_t = alpha.to(device=logits.device, dtype=logits.dtype)
    while alpha_t.ndim < logits.ndim:
        alpha_t = alpha_t.view(*([1] * (logits.ndim - 1)), -1)

    weight = alpha_t * targets + (1.0 - alpha_t) * (1.0 - targets)
    loss = weight * (1.0 - p_t).pow(gamma) * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Unsupported reduction: {reduction}")


def smoke_test() -> None:
    x = torch.randn(2, 16, 32, 32)
    coord = CoordConv(16, 24)
    rep = RepConv(24, 24)
    attn = BiFormerBlockLite(24, num_heads=4, region_size=8, topk=2)
    with torch.no_grad():
        y = attn(rep(coord(x)))
    assert y.shape == (2, 24, 32, 32)

    rep.eval()
    with torch.no_grad():
        before = rep(coord(x))
        rep.switch_to_deploy()
        after = rep(coord(x))
    max_diff = (before - after).abs().max().item()
    print({"shape": tuple(y.shape), "repconv_fusion_max_diff": max_diff})

    pred = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.1, 0.2]])
    target = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.45, 0.4, 0.1, 0.2]])
    print({"focal_eiou": float(focal_eiou_loss(pred, target))})


if __name__ == "__main__":
    smoke_test()
