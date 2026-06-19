"""配准用的相似性度量与正则项（PyTorch 实现）。

CBCT 与 pCT 的灰度并不严格一致（CBCT 有杯状伪影、HU 标定不同），因此采用
对线性灰度差异更鲁棒的“局部归一化互相关”(LNCC)。形变场使用扩散正则
(diffusion / L2 梯度) 约束其平滑性。
"""

import torch
import torch.nn.functional as F


def local_ncc(y_pred, y_true, win=9, mask=None, eps=1e-5):
    """局部归一化互相关损失，返回 ``1 - LNCC``（越小越好）。

    y_pred, y_true: (N, 1, D, H, W)
    win: 局部窗口边长（体素）。
    mask: 可选 (N, 1, D, H, W)，只在掩膜内统计。
    """
    if mask is not None:
        y_pred = y_pred * mask
        y_true = y_true * mask

    pad = win // 2
    sum_filt = torch.ones(1, 1, win, win, win, device=y_pred.device,
                          dtype=y_pred.dtype)

    def _conv(x):
        return F.conv3d(x, sum_filt, padding=pad)

    I, J = y_pred, y_true
    I2, J2, IJ = I * I, J * J, I * J

    I_sum, J_sum = _conv(I), _conv(J)
    I2_sum, J2_sum, IJ_sum = _conv(I2), _conv(J2), _conv(IJ)

    win_size = win ** 3
    u_I = I_sum / win_size
    u_J = J_sum / win_size

    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

    cc = (cross * cross) / (I_var * J_var + eps)

    if mask is not None:
        m = (_conv(mask) > (win_size * 0.5)).float()
        return 1.0 - (cc * m).sum() / (m.sum() + eps)
    return 1.0 - cc.mean()


def gradient_loss(flow):
    """形变场的扩散正则（各方向一阶差分的 L2）。

    flow: (N, 3, D, H, W)，单位为归一化坐标位移。
    """
    dz = flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :]
    dy = flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :]
    dx = flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]
    return (dz.pow(2).mean() + dy.pow(2).mean() + dx.pow(2).mean()) / 3.0


def masked_mse(y_pred, y_true, mask=None, eps=1e-5):
    """掩膜内的均方误差，仅用于评估指标。"""
    if mask is None:
        return F.mse_loss(y_pred, y_true)
    diff = (y_pred - y_true) ** 2 * mask
    return diff.sum() / (mask.sum() + eps)
