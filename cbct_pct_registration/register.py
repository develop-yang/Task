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
             rigid_lr=0.01, deform_lr=0.01,
             ncc_win=9, reg_weight=20.0, allow_scale=False,
             flow_downsample=8, smooth_sigma=0.8, int_steps=7,
             pyramid=(4, 2, 1), verbose=True):
    """对公共网格上的 moving/fixed 体数据做刚体 + 形变配准。

    moving, fixed, mask: numpy (D, H, W)，已归一化到 [0, 1]。
    pyramid: 刚体阶段的多分辨率下采样倍数（由粗到细），用于扩大捕获范围。
    形变默认强正则 + 粗控制网格 + 速度场高斯平滑，做“小幅平滑细化”而非剧烈揉变，
    保证 Jacobian 处处为正、无折叠。
    返回 (model, warped_deform, warped_rigid, history)。
    """
    device = torch.device(device)
    mv = _to_tensor(moving, device)
    fx = _to_tensor(fixed, device)
    mk = _to_tensor(mask, device) if mask is not None else None

    size = moving.shape  # (D, H, W)
    model = RegistrationModel(size, flow_downsample=flow_downsample,
                              allow_scale=allow_scale, int_steps=int_steps,
                              smooth_sigma=smooth_sigma).to(device)

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

    # 记录“仅刚体”结果（用于和 +形变 对照）
    with torch.no_grad():
        warped_rigid = model.warp(mv, use_flow=False).cpu().numpy()[0, 0]

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
    return model, warped, warped_rigid, history


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


# ----------------------------------------------------------------------------
# 与优化目标(LNCC)无关的独立验证指标 + 物理刚体参数导出
# ----------------------------------------------------------------------------

def _dice(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2.0 * inter / denom) if denom > 0 else 0.0


def bone_dice(moving_hu, fixed_hu, warped_hu, mask=None, thresh=200.0):
    """骨结构 Dice（独立于 LNCC 目标的结构一致性指标）。

    在 HU 空间按 >thresh 取骨掩膜，只在 CBCT 视野(mask)内统计配准前/后 Dice。
    """
    if mask is not None:
        m = mask.astype(bool)
    else:
        m = np.ones_like(fixed_hu, bool)
    bone_fixed = (fixed_hu > thresh) & m
    bone_moving = (moving_hu > thresh) & m
    bone_warped = (warped_hu > thresh) & m
    return {
        "bone_dice_before": _dice(bone_moving, bone_fixed),
        "bone_dice_after": _dice(bone_warped, bone_fixed),
    }


def decompose_euler(R):
    """把旋转矩阵分解为 (rx, ry, rz)（弧度），约定 R = Rz @ Ry @ Rx。"""
    sy = -R[2, 0]
    sy = max(-1.0, min(1.0, float(sy)))
    ry = np.arcsin(sy)
    if abs(sy) < 0.99999:
        rx = np.arctan2(R[2, 1], R[2, 2])
        rz = np.arctan2(R[1, 0], R[0, 0])
    else:  # 万向锁
        rx = np.arctan2(-R[1, 2], R[1, 1])
        rz = 0.0
    return np.array([rx, ry, rz])


def rigid_to_physical(model, ref_origin, ref_spacing, ref_shape):
    """把归一化坐标下的刚体参数换算成物理单位(mm/度)。

    立方体各向同性网格下，affine_grid 的归一化旋转就是物理旋转。
    采样网格变换 theta 把 fixed(pCT) 坐标映射到 moving(CBCT) 坐标：
        world_moving = R · world_fixed + T
    其中 T = c - R·c + s·t_norm，c 为网格中心世界坐标，s 为半物理边长。
    临床摆位关心的是 CBCT->pCT 的修正量，即其逆变换。
    """
    R = model.affine_theta().detach().cpu().numpy()[0, :, :3]
    scale = float(np.cbrt(abs(np.linalg.det(R))))
    Rrot = R / scale if scale > 1e-6 else R  # 去掉缩放得到纯旋转
    t_norm = model.trans.detach().cpu().numpy()

    n = ref_shape[0]
    s = (n - 1) * float(ref_spacing[0]) / 2.0  # 半物理边长(mm)，三轴相同
    c = np.asarray(ref_origin) + (n - 1) / 2.0 * np.asarray(ref_spacing)

    # fixed(pCT) -> moving(CBCT)
    T = c - Rrot @ c + s * t_norm
    # moving(CBCT) -> fixed(pCT)，即摆位修正
    R_inv = Rrot.T
    T_inv = -R_inv @ T

    return {
        "scale": scale,
        "rotation_deg_cbct_to_pct": np.degrees(decompose_euler(R_inv)),
        "translation_mm_cbct_to_pct": T_inv,
        "rotation_deg_pct_to_cbct": np.degrees(decompose_euler(Rrot)),
        "translation_mm_pct_to_cbct": T,
        "matrix_cbct_to_pct": np.vstack([np.hstack([R_inv, T_inv[:, None]]),
                                         [0, 0, 0, 1]]),
    }


def jacobian_stats(model, mask=None):
    """统计**形变场**(微分同胚位移)的雅可比行列式。

    刚体部分是旋转(det≈1)，这里只评估形变映射 id+位移 的折叠情况。
    负值=折叠=非法形变。返回 min / mean / 负值占比。
    """
    with torch.no_grad():
        grid = (model._identity_grid() + model.full_flow()
                ).detach().cpu().numpy()[0]
    D, H, W = model.size
    # 归一化坐标(x,y,z) -> 体素坐标
    vx = (grid[..., 0] + 1) / 2 * (W - 1)
    vy = (grid[..., 1] + 1) / 2 * (H - 1)
    vz = (grid[..., 2] + 1) / 2 * (D - 1)
    # phi 分量按 (z,y,x) 排列，对输入轴 (0=D=z,1=H=y,2=W=x) 求梯度
    phi = [vz, vy, vx]
    J = np.empty((3, 3) + (D, H, W), np.float64)
    for a in range(3):
        g = np.gradient(phi[a])
        for b in range(3):
            J[a, b] = g[b]
    Jm = np.moveaxis(J, [0, 1], [-2, -1])  # (D,H,W,3,3)
    det = np.linalg.det(Jm)

    if mask is not None:
        m = mask.astype(bool)
        det = det[m]
    return {
        "jacobian_min": float(det.min()),
        "jacobian_mean": float(det.mean()),
        "jacobian_neg_fraction": float((det <= 0).mean()),
    }
