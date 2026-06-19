"""刚体 + 形变配准模型（纯 PyTorch，单对图像的实例优化）。

思路：在公共网格上，把 moving(CBCT) 通过 ``grid_sample`` 形变去匹配 fixed(pCT)。
采样网格 = 刚体仿射网格 + 上采样后的稠密位移场。先优化刚体参数，再优化位移场。
所有坐标都在 PyTorch 的归一化坐标系 [-1, 1] 下进行（公共网格各向同性，
归一化旋转与物理旋转一致）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def euler_to_matrix(angles):
    """由 (rx, ry, rz) 欧拉角（弧度）构造 3x3 旋转矩阵。"""
    rx, ry, rz = angles[0], angles[1], angles[2]
    zero = torch.zeros((), device=angles.device, dtype=angles.dtype)
    one = torch.ones((), device=angles.device, dtype=angles.dtype)

    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    Rx = torch.stack([torch.stack([one, zero, zero]),
                      torch.stack([zero, cx, -sx]),
                      torch.stack([zero, sx, cx])])
    Ry = torch.stack([torch.stack([cy, zero, sy]),
                      torch.stack([zero, one, zero]),
                      torch.stack([-sy, zero, cy])])
    Rz = torch.stack([torch.stack([cz, -sz, zero]),
                      torch.stack([sz, cz, zero]),
                      torch.stack([zero, zero, one])])
    return Rz @ Ry @ Rx


class RegistrationModel(nn.Module):
    """刚体（可选各向同性缩放）+ 稠密位移场。

    size: 公共网格形状 (D, H, W)。
    flow_downsample: 位移场相对全分辨率的下采样倍数（控制平滑度与显存）。
    """

    def __init__(self, size, flow_downsample=4, allow_scale=False,
                 diffeomorphic=True, int_steps=7):
        super().__init__()
        self.size = size
        self.allow_scale = allow_scale
        self.diffeomorphic = diffeomorphic
        self.int_steps = int_steps

        # 刚体参数：3 旋转 + 3 平移（归一化坐标）
        self.rot = nn.Parameter(torch.zeros(3))
        self.trans = nn.Parameter(torch.zeros(3))
        self.log_scale = nn.Parameter(torch.zeros(1))  # 各向同性缩放 exp(log_scale)

        # 低分辨率位移场参数 (1, 3, d, h, w)
        d = max(1, size[0] // flow_downsample)
        h = max(1, size[1] // flow_downsample)
        w = max(1, size[2] // flow_downsample)
        self.flow_lowres = nn.Parameter(torch.zeros(1, 3, d, h, w))

    # ---- 刚体部分 ----
    def affine_theta(self):
        R = euler_to_matrix(self.rot)
        if self.allow_scale:
            R = R * torch.exp(self.log_scale)
        theta = torch.zeros(1, 3, 4, device=self.rot.device, dtype=self.rot.dtype)
        theta[0, :, :3] = R
        theta[0, :, 3] = self.trans
        return theta

    def affine_grid(self):
        theta = self.affine_theta()
        return F.affine_grid(theta, (1, 1) + tuple(self.size),
                             align_corners=True)  # (1, D, H, W, 3)

    # ---- 形变部分 ----
    def _identity_grid(self):
        theta = torch.eye(3, 4, device=self.rot.device, dtype=self.rot.dtype)[None]
        return F.affine_grid(theta, (1, 1) + tuple(self.size), align_corners=True)

    def _integrate(self, vel):
        """对速度场做 scaling-and-squaring 积分，得到微分同胚位移场。

        vel: (1, 3, D, H, W)，归一化坐标下的速度。返回同形位移场。
        保证（速度足够小、步数足够）变换可逆、Jacobian 处处为正，几乎无折叠。
        """
        disp = vel / (2 ** self.int_steps)
        grid = self._identity_grid()  # (1, D, H, W, 3)
        for _ in range(self.int_steps):
            d_perm = disp.permute(0, 2, 3, 4, 1)  # (1, D, H, W, 3)
            sampled = F.grid_sample(disp, grid + d_perm, mode="bilinear",
                                    padding_mode="border", align_corners=True)
            disp = disp + sampled
        return disp

    def full_flow(self):
        """把低分辨率速度/位移场上采样到全分辨率，返回 (1, D, H, W, 3)。"""
        flow = F.interpolate(self.flow_lowres, size=tuple(self.size),
                             mode="trilinear", align_corners=True)
        if self.diffeomorphic:
            flow = self._integrate(flow)  # 速度场 -> 微分同胚位移场
        return flow.permute(0, 2, 3, 4, 1)  # -> (1, D, H, W, 3)

    def sampling_grid(self, use_flow=True):
        grid = self.affine_grid()
        if use_flow:
            grid = grid + self.full_flow()
        return grid

    def warp(self, moving, use_flow=True, mode="bilinear"):
        grid = self.sampling_grid(use_flow=use_flow)
        return F.grid_sample(moving, grid, mode=mode, padding_mode="border",
                             align_corners=True)

    def flow_lowres_tensor(self):
        """返回 (1, 3, d, h, w) 形式的低分辨率位移场，供正则项使用。"""
        return self.flow_lowres
