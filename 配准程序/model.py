import torch
import torch.nn as nn
import torch.nn.functional as F


def euler_to_matrix(a):
    rx, ry, rz = a[0], a[1], a[2]
    z = torch.zeros((), device=a.device, dtype=a.dtype)
    o = torch.ones((), device=a.device, dtype=a.dtype)
    cx, sx, cy, sy, cz, sz = (torch.cos(rx), torch.sin(rx), torch.cos(ry),
                              torch.sin(ry), torch.cos(rz), torch.sin(rz))
    Rx = torch.stack([torch.stack([o, z, z]), torch.stack([z, cx, -sx]), torch.stack([z, sx, cx])])
    Ry = torch.stack([torch.stack([cy, z, sy]), torch.stack([z, o, z]), torch.stack([-sy, z, cy])])
    Rz = torch.stack([torch.stack([cz, -sz, z]), torch.stack([sz, cz, z]), torch.stack([z, z, o])])
    return Rz @ Ry @ Rx


class RegistrationModel(nn.Module):
    """刚体(6 DOF，可选缩放) + 速度场积分得到的微分同胚形变。"""

    def __init__(self, size, flow_downsample=8, allow_scale=False,
                 int_steps=7, smooth_sigma=0.8):
        super().__init__()
        self.size = size
        self.allow_scale = allow_scale
        self.int_steps = int_steps
        self.smooth_sigma = smooth_sigma

        self.rot = nn.Parameter(torch.zeros(3))
        self.trans = nn.Parameter(torch.zeros(3))
        self.log_scale = nn.Parameter(torch.zeros(1))

        d = max(1, size[0] // flow_downsample)
        h = max(1, size[1] // flow_downsample)
        w = max(1, size[2] // flow_downsample)
        self.flow_lowres = nn.Parameter(torch.zeros(1, 3, d, h, w))   # 低分辨率速度场

    # ----- 刚体 -----
    def affine_theta(self):
        R = euler_to_matrix(self.rot)
        if self.allow_scale:
            R = R * torch.exp(self.log_scale)
        theta = torch.zeros(1, 3, 4, device=self.rot.device, dtype=self.rot.dtype)
        theta[0, :, :3] = R
        theta[0, :, 3] = self.trans
        return theta

    def affine_grid(self):
        return F.affine_grid(self.affine_theta(), (1, 1) + tuple(self.size), align_corners=True)

    # ----- 形变 -----
    @staticmethod
    def _gauss1d(sigma, device, dtype):
        r = max(1, int(round(3 * sigma)))
        x = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (x / sigma) ** 2)
        return k / k.sum()

    def _smooth(self, v):
        if not self.smooth_sigma or self.smooth_sigma <= 0:
            return v
        k = self._gauss1d(self.smooth_sigma, v.device, v.dtype)
        n, pad = k.numel(), k.numel() // 2
        for ax in (2, 3, 4):
            shape = [1, 1, 1, 1, 1]; shape[ax] = n
            ker = k.view(shape).repeat(3, 1, 1, 1, 1)
            p = [0, 0, 0]; p[ax - 2] = pad
            v = F.conv3d(v, ker, padding=tuple(p), groups=3)
        return v

    def _identity_grid(self):
        theta = torch.eye(3, 4, device=self.rot.device, dtype=self.rot.dtype)[None]
        return F.affine_grid(theta, (1, 1) + tuple(self.size), align_corners=True)

    def _integrate(self, vel):
        # scaling-and-squaring：速度场积分成微分同胚位移场，保证可逆、无折叠
        disp = vel / (2 ** self.int_steps)
        grid = self._identity_grid()
        for _ in range(self.int_steps):
            dperm = disp.permute(0, 2, 3, 4, 1)
            disp = disp + F.grid_sample(disp, grid + dperm, mode="bilinear",
                                        padding_mode="border", align_corners=True)
        return disp

    def full_flow(self):
        vel = self._smooth(self.flow_lowres)
        flow = F.interpolate(vel, size=tuple(self.size), mode="trilinear", align_corners=True)
        return self._integrate(flow).permute(0, 2, 3, 4, 1)

    def sampling_grid(self, use_flow=True):
        g = self.affine_grid()
        return g + self.full_flow() if use_flow else g

    def warp(self, moving, use_flow=True, mode="bilinear"):
        return F.grid_sample(moving, self.sampling_grid(use_flow), mode=mode,
                             padding_mode="border", align_corners=True)
