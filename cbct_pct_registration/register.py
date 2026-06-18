"""配准优化主流程：阶段一（多分辨率）刚体，阶段二形变。"""

import numpy as np
import torch
import torch.nn.functional as F

from .losses import gradient_loss, local_ncc, masked_mse
from .model import RegistrationModel


def _to_tensor(arr, device):
    return torch.from_numpy(arr).float()[None, None].to(device)


def _downsample(t, factor):
    """三线性下采样 (N,1,D,H,W)，factor=1 时原样返回。"""
    if factor == 1:
        return t
    size = [max(1, s // factor) for s in t.shape[2:]]
    return F.interpolate(t, size=size, mode="trilinear", align_corners=True)


def register(moving, fixed, mask=None, device="cpu",
             rigid_iters=300, deform_iters=300,
             rigid_lr=0.01, deform_lr=0.05,
             ncc_win=9, reg_weight=1.0, allow_scale=True,
             flow_downsample=4, pyramid=(4, 2, 1), verbose=True):
    """对公共网格上的 moving/fixed 体数据做刚体 + 形变配准。

    moving, fixed, mask: numpy (D, H, W)，已归一化到 [0, 1]。
    pyramid: 刚体阶段的多分辨率下采样倍数（由粗到细），用于扩大捕获范围。
    返回 (model, warped_numpy, history)。
    """
    device = torch.device(device)
    mv = _to_tensor(moving, device)
    fx = _to_tensor(fixed, device)
    mk = _to_tensor(mask, device) if mask is not None else None

    size = moving.shape  # (D, H, W)
    model = RegistrationModel(size, flow_downsample=flow_downsample,
                              allow_scale=allow_scale).to(device)

    history = {"rigid": [], "deform": []}

    # ---------- 阶段一：多分辨率刚体（+各向同性缩放） ----------
    params = [model.rot, model.trans] + ([model.log_scale] if allow_scale else [])
    for level in pyramid:
        mv_l, fx_l = _downsample(mv, level), _downsample(fx, level)
        mk_l = _downsample(mk, level) if mk is not None else None
        model.size = tuple(mv_l.shape[2:])
        win = max(3, ncc_win // level | 1)  # 保持窗口为奇数
        opt = torch.optim.Adam(params, lr=rigid_lr)
        for it in range(rigid_iters):
            opt.zero_grad()
            warped = model.warp(mv_l, use_flow=False)
            loss = local_ncc(warped, fx_l, win=win, mask=mk_l)
            loss.backward()
            opt.step()
            history["rigid"].append(float(loss.detach()))
            if verbose and (it % 50 == 0 or it == rigid_iters - 1):
                print("  [刚体 x%d] iter %4d/%d  LNCC loss=%.5f"
                      % (level, it, rigid_iters, float(loss.detach())))
    model.size = size  # 恢复全分辨率

    # ---------- 阶段二：形变（冻结刚体，优化位移场） ----------
    model.rot.requires_grad_(False)
    model.trans.requires_grad_(False)
    model.log_scale.requires_grad_(False)
    opt = torch.optim.Adam([model.flow_lowres], lr=deform_lr)
    for it in range(deform_iters):
        opt.zero_grad()
        warped = model.warp(mv, use_flow=True)
        sim = local_ncc(warped, fx, win=ncc_win, mask=mk)
        reg = gradient_loss(model.flow_lowres_tensor())
        loss = sim + reg_weight * reg
        loss.backward()
        opt.step()
        history["deform"].append((float(sim.detach()), float(reg.detach())))
        if verbose and (it % 50 == 0 or it == deform_iters - 1):
            print("  [形变] iter %4d/%d  LNCC=%.5f  reg=%.5f"
                  % (it, deform_iters, float(sim.detach()), float(reg.detach())))

    with torch.no_grad():
        warped = model.warp(mv, use_flow=True).cpu().numpy()[0, 0]
    return model, warped, history


def evaluate(moving, fixed, warped, mask=None):
    """返回配准前后的 LNCC 与 MSE 指标（字典）。"""
    def t(a):
        return torch.from_numpy(a).float()[None, None]
    mk = t(mask) if mask is not None else None
    out = {
        "ncc_before": 1 - float(local_ncc(t(moving), t(fixed), mask=mk)),
        "ncc_after": 1 - float(local_ncc(t(warped), t(fixed), mask=mk)),
        "mse_before": float(masked_mse(t(moving), t(fixed), mk)),
        "mse_after": float(masked_mse(t(warped), t(fixed), mk)),
    }
    return out
