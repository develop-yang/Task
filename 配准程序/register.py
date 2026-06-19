import numpy as np
import torch
import torch.nn.functional as F

from losses import gradient_loss, local_ncc, masked_mse
from model import RegistrationModel


def _t(arr, device):
    return torch.from_numpy(arr).float()[None, None].to(device)


def _down(t, f):
    if f == 1:
        return t
    return F.interpolate(t, size=[max(1, s // f) for s in t.shape[2:]],
                         mode="trilinear", align_corners=True)


def register(moving, fixed, mask=None, device="cpu",
             rigid_iters=150, deform_iters=60, rigid_lr=0.01, deform_lr=0.01,
             ncc_win=9, reg_weight=20.0, allow_scale=False,
             flow_downsample=8, smooth_sigma=0.8, int_steps=7,
             pyramid=(4, 2, 1), verbose=True):
    device = torch.device(device)
    mv, fx = _t(moving, device), _t(fixed, device)
    mk = _t(mask, device) if mask is not None else None
    size = moving.shape

    model = RegistrationModel(size, flow_downsample=flow_downsample, allow_scale=allow_scale,
                              int_steps=int_steps, smooth_sigma=smooth_sigma).to(device)

    # 阶段一：多分辨率刚体（金字塔由粗到细扩大捕获范围）
    params = [model.rot, model.trans] + ([model.log_scale] if allow_scale else [])
    for lv in pyramid:
        mvl, fxl = _down(mv, lv), _down(fx, lv)
        mkl = _down(mk, lv) if mk is not None else None
        model.size = tuple(mvl.shape[2:])
        win = max(3, ncc_win // lv | 1)
        opt = torch.optim.Adam(params, lr=rigid_lr)
        for it in range(rigid_iters):
            opt.zero_grad()
            loss = local_ncc(model.warp(mvl, use_flow=False), fxl, win=win, mask=mkl)
            loss.backward(); opt.step()
            if verbose and it % 50 == 0:
                print("  [rigid x%d] %3d/%d  loss=%.4f" % (lv, it, rigid_iters, float(loss.detach())))
    model.size = size

    with torch.no_grad():
        warped_rigid = model.warp(mv, use_flow=False).cpu().numpy()[0, 0]

    # 阶段二：冻结刚体，优化受控的微分同胚形变（强正则 + 平滑 + 粗网格 -> 小幅、无折叠）
    for p in (model.rot, model.trans, model.log_scale):
        p.requires_grad_(False)
    opt = torch.optim.Adam([model.flow_lowres], lr=deform_lr)
    for it in range(deform_iters):
        opt.zero_grad()
        sim = local_ncc(model.warp(mv, use_flow=True), fx, win=ncc_win, mask=mk)
        reg = gradient_loss(model.flow_lowres)
        (sim + reg_weight * reg).backward(); opt.step()
        if verbose and it % 30 == 0:
            print("  [deform]  %3d/%d  ncc=%.4f reg=%.4f"
                  % (it, deform_iters, float(sim.detach()), float(reg.detach())))

    with torch.no_grad():
        warped = model.warp(mv, use_flow=True).cpu().numpy()[0, 0]
    return model, warped, warped_rigid


def evaluate(moving, fixed, warped, mask=None):
    f = lambda a: torch.from_numpy(a).float()[None, None]
    mk = f(mask) if mask is not None else None
    return {"ncc_before": 1 - float(local_ncc(f(moving), f(fixed), mask=mk)),
            "ncc_after": 1 - float(local_ncc(f(warped), f(fixed), mask=mk)),
            "mse_before": float(masked_mse(f(moving), f(fixed), mk)),
            "mse_after": float(masked_mse(f(warped), f(fixed), mk))}


def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return float(2 * np.logical_and(a, b).sum() / s) if s else 0.0


def bone_dice(moving_hu, fixed_hu, warped_hu, mask=None, thresh=200.0):
    # 独立于优化目标(LNCC)的结构一致性指标：HU>thresh 的骨掩膜在 CBCT 视野内的 Dice
    m = mask.astype(bool) if mask is not None else np.ones_like(fixed_hu, bool)
    bf, bm, bw = (fixed_hu > thresh) & m, (moving_hu > thresh) & m, (warped_hu > thresh) & m
    return {"bone_dice_before": _dice(bm, bf), "bone_dice_after": _dice(bw, bf)}


def _decompose_euler(R):
    sy = max(-1.0, min(1.0, float(-R[2, 0])))
    ry = np.arcsin(sy)
    if abs(sy) < 0.99999:
        rx, rz = np.arctan2(R[2, 1], R[2, 2]), np.arctan2(R[1, 0], R[0, 0])
    else:
        rx, rz = np.arctan2(-R[1, 2], R[1, 1]), 0.0
    return np.array([rx, ry, rz])


def rigid_to_physical(model, ref_origin, ref_spacing, ref_shape):
    # 归一化坐标下的刚体参数 -> 物理 mm/度（立方体各向同性网格下归一化旋转即物理旋转）
    R = model.affine_theta().detach().cpu().numpy()[0, :, :3]
    scale = float(np.cbrt(abs(np.linalg.det(R))))
    Rr = R / scale if scale > 1e-6 else R
    t = model.trans.detach().cpu().numpy()
    n = ref_shape[0]
    s = (n - 1) * float(ref_spacing[0]) / 2.0
    c = np.asarray(ref_origin) + (n - 1) / 2.0 * np.asarray(ref_spacing)
    T = c - Rr @ c + s * t                       # fixed(pCT) -> moving(CBCT)
    Rinv, Tinv = Rr.T, -Rr.T @ (c - Rr @ c + s * t)  # CBCT -> pCT（摆位修正）
    return {"scale": scale,
            "rot_deg": np.degrees(_decompose_euler(Rinv)),
            "trans_mm": Tinv,
            "matrix": np.vstack([np.hstack([Rinv, Tinv[:, None]]), [0, 0, 0, 1]])}


def jacobian_stats(model, mask=None):
    # 形变场 Jacobian 行列式：负值=折叠。统计 min/mean/负值占比
    with torch.no_grad():
        grid = (model._identity_grid() + model.full_flow()).cpu().numpy()[0]
    D, H, W = model.size
    vz = (grid[..., 2] + 1) / 2 * (D - 1)
    vy = (grid[..., 1] + 1) / 2 * (H - 1)
    vx = (grid[..., 0] + 1) / 2 * (W - 1)
    J = np.empty((3, 3, D, H, W))
    for a, comp in enumerate((vz, vy, vx)):
        g = np.gradient(comp)
        for b in range(3):
            J[a, b] = g[b]
    det = np.linalg.det(np.moveaxis(J, [0, 1], [-2, -1]))
    if mask is not None:
        det = det[mask.astype(bool)]
    return {"min": float(det.min()), "mean": float(det.mean()),
            "neg_fraction": float((det <= 0).mean())}
