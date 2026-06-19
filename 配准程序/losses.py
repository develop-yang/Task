import torch
import torch.nn.functional as F


def local_ncc(pred, target, win=9, mask=None, eps=1e-5):
    # 局部归一化互相关，对 CBCT/pCT 之间的线性灰度差异鲁棒；返回 1-LNCC
    if mask is not None:
        pred, target = pred * mask, target * mask
    pad = win // 2
    filt = torch.ones(1, 1, win, win, win, device=pred.device, dtype=pred.dtype)
    conv = lambda x: F.conv3d(x, filt, padding=pad)

    I, J = pred, target
    Is, Js = conv(I), conv(J)
    I2s, J2s, IJs = conv(I * I), conv(J * J), conv(I * J)
    npix = win ** 3
    uI, uJ = Is / npix, Js / npix
    cross = IJs - uJ * Is - uI * Js + uI * uJ * npix
    Iv = I2s - 2 * uI * Is + uI * uI * npix
    Jv = J2s - 2 * uJ * Js + uJ * uJ * npix
    cc = cross * cross / (Iv * Jv + eps)

    if mask is not None:
        m = (conv(mask) > npix * 0.5).float()
        return 1.0 - (cc * m).sum() / (m.sum() + eps)
    return 1.0 - cc.mean()


def gradient_loss(flow):
    # 位移/速度场的扩散正则（一阶差分 L2），保证场平滑
    dz = flow[:, :, 1:] - flow[:, :, :-1]
    dy = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()) / 3.0


def masked_mse(pred, target, mask=None, eps=1e-5):
    if mask is None:
        return F.mse_loss(pred, target)
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() + eps)
