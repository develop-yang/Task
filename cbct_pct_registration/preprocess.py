"""把两个体数据重采样到同一参考网格，并做强度归一化。

由于两套 DICOM 都带有正确的世界坐标（LPS）信息，先按物理坐标重采样到一个
公共的、各向同性的网格上，就已经完成了“基于 DICOM 头的初始对齐”。之后再用
PyTorch 做刚体 + 形变细化。

约定：以 CBCT 的世界包围盒作为公共参考网格（CBCT 视野较小），
fixed = pCT 重采样结果，moving = CBCT 重采样结果，二者形状相同。
"""

import numpy as np
from scipy.ndimage import map_coordinates

from .dicom_io import Volume

AIR_HU = -1000.0  # 视野外填充值（空气）


def make_reference_grid(vol_for_extent, spacing_mm):
    """根据给定体数据的世界包围盒，构造**立方体**各向同性参考网格。

    立方体（D=H=W）+ 各向同性体素，保证 ``F.affine_grid`` 归一化坐标
    [-1,1] 在三个轴上对应相同的物理长度，于是归一化旋转 == 物理刚体旋转，
    不会沿 z 引入剪切（修复“归一化坐标下非真刚体”的问题）。

    返回 (ref_origin(x,y,z), ref_spacing(x,y,z), shape(z,y,x))。
    """
    mn, mx = vol_for_extent.world_extent()
    center = (mn + mx) / 2.0
    spacing = np.array([spacing_mm, spacing_mm, spacing_mm], np.float64)
    size_xyz = np.ceil((mx - mn) / spacing).astype(int) + 1
    n = int(size_xyz.max())  # 取最大边长，做成立方体
    ref_origin = center - (n - 1) / 2.0 * spacing
    return ref_origin, spacing, (n, n, n)


def resample_to_grid(vol, ref_origin, ref_spacing, ref_shape, order=1,
                     fill=AIR_HU):
    """把 ``vol`` 重采样到参考网格（轴位、方向为单位阵）。

    对参考网格的每个体素，计算其世界坐标，再换算成源体数据的连续索引，
    用三线性插值采样。
    """
    nz, ny, nx = ref_shape
    # 参考网格各体素的世界坐标（方向为单位阵，可直接线性映射）
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    wx = ref_origin[0] + xx * ref_spacing[0]
    wy = ref_origin[1] + yy * ref_spacing[1]
    wz = ref_origin[2] + zz * ref_spacing[2]

    # 世界坐标 -> 源体数据连续索引（i=z, j=y, k=x）
    src_k = (wx - vol.origin[0]) / vol.spacing[0]
    src_j = (wy - vol.origin[1]) / vol.spacing[1]
    src_i = (wz - vol.origin[2]) / vol.spacing[2]

    coords = np.stack([src_i.ravel(), src_j.ravel(), src_k.ravel()])
    out = map_coordinates(vol.data, coords, order=order, mode="constant",
                          cval=fill, prefilter=False)
    return out.reshape(ref_shape).astype(np.float32)


def normalize_hu(arr, win_min=-1000.0, win_max=1000.0):
    """把 HU 裁剪到给定窗宽并归一化到 [0, 1]。"""
    arr = np.clip(arr, win_min, win_max)
    return (arr - win_min) / (win_max - win_min)


def body_centroid_world(vol, thresh=-400.0):
    """估计“身体”体素（HU > thresh）的世界坐标质心 (x, y, z)。"""
    body = vol.data > thresh
    if body.sum() < 10:
        nz, ny, nx = vol.data.shape
        c_zyx = np.array([nz, ny, nx]) / 2.0
    else:
        c_zyx = np.argwhere(body).mean(0)  # (z, y, x)
    c_xyz = c_zyx[::-1]
    return vol.origin + c_xyz * vol.spacing


def axial_area_profile(vol, thresh=-300.0):
    """每层身体横截面积（HU>thresh 的体素数）随物理 z 的剖面。

    返回 (z_world(每层世界 z 坐标), area(每层面积))。
    """
    area = (vol.data > thresh).sum(axis=(1, 2)).astype(np.float64)  # (z,)
    z = vol.origin[2] + np.arange(vol.data.shape[0]) * vol.spacing[2]
    return z, area


def estimate_z_offset(cbct, pct, dz=1.0, thresh=-300.0):
    """用身体横截面积剖面的 1D 归一化互相关，估计 z 方向初始偏移(mm)。

    pCT 的 z 覆盖远大于 CBCT，二者“质心 z”落在不同解剖范围上并不可比，
    直接用质心做 z 对齐会偏。这里把 CBCT 的面积剖面在 pCT 的面积剖面上滑动，
    取归一化互相关最大的位置，得到稳健的 z 初值。

    返回 offset_z，使得参考网格（CBCT 坐标系）中世界 z 处应采样 pCT 的
    (z + offset_z) 位置。
    """
    zc, ac = axial_area_profile(cbct, thresh)
    zp, ap = axial_area_profile(pct, thresh)

    # 重采样到统一的细 z 网格
    zc_fine = np.arange(zc.min(), zc.max() + dz, dz)
    zp_fine = np.arange(zp.min(), zp.max() + dz, dz)
    ac_f = np.interp(zc_fine, zc, ac)
    ap_f = np.interp(zp_fine, zp, ap)

    if len(ac_f) >= len(ap_f):  # CBCT 比 pCT 还长，退化为质心 z
        return body_centroid_world(pct)[2] - body_centroid_world(cbct)[2]

    ac_n = ac_f - ac_f.mean()
    ac_norm = np.linalg.norm(ac_n) + 1e-6
    n = len(ac_n)

    best_score, best_s = -np.inf, 0
    for s in range(len(ap_f) - n + 1):
        w = ap_f[s:s + n]
        w_n = w - w.mean()
        score = float(np.dot(w_n, ac_n) / (np.linalg.norm(w_n) * ac_norm + 1e-6))
        if score > best_score:
            best_score, best_s = score, s

    # CBCT 的起始 z (zc_fine[0]) 对应 pCT 世界 z 为 zp_fine[best_s]
    return float(zp_fine[best_s] - zc_fine[0])


def prepare_pair(cbct, pct, spacing_mm=1.5, align_centroid=True):
    """生成公共网格上的 (moving=CBCT, fixed=pCT) 及掩膜。

    CBCT（治疗机等中心坐标系）与 pCT（CT 床坐标系）通常不在同一世界坐标系，
    因此默认先做一次初始平移对齐：x、y 用身体质心，z 用横截面积剖面的 1D
    互相关（更稳健），把两者拉到大致重叠，再交给 PyTorch 做刚体 + 形变细化。

    返回字典，含归一化体数据、原始 HU 体数据、参考几何、CBCT 视野掩膜。
    """
    ref_origin, ref_spacing, ref_shape = make_reference_grid(cbct, spacing_mm)

    # 以 CBCT 立方体网格为公共网格；采样 pCT 时加上初始偏移，使其落入该网格
    offset = np.zeros(3)
    if align_centroid:
        c = body_centroid_world(pct) - body_centroid_world(cbct)
        offset[0], offset[1] = c[0], c[1]          # x, y 用质心
        offset[2] = estimate_z_offset(cbct, pct)    # z 用 1D 互相关

    moving_hu = resample_to_grid(cbct, ref_origin, ref_spacing, ref_shape)
    fixed_hu = resample_to_grid(pct, ref_origin + offset, ref_spacing, ref_shape)

    # CBCT 视野掩膜：重采样时视野外被填成 AIR_HU，这里据此标出有效区域
    mask = resample_to_grid(
        Volume(np.ones_like(cbct.data), cbct.spacing, cbct.origin, cbct.direction),
        ref_origin, ref_spacing, ref_shape, order=0, fill=0.0) > 0.5

    return {
        "moving": normalize_hu(moving_hu),
        "fixed": normalize_hu(fixed_hu),
        "moving_hu": moving_hu,
        "fixed_hu": fixed_hu,
        "mask": mask.astype(np.float32),
        "ref_origin": ref_origin,
        "ref_spacing": ref_spacing,
        "ref_shape": ref_shape,
        "centroid_offset": offset,
    }


def to_nifti_affine(ref_origin, ref_spacing):
    """构造 NIfTI 仿射矩阵（LPS->RAS：翻转 x、y 符号）。"""
    aff = np.diag([-ref_spacing[0], -ref_spacing[1], ref_spacing[2], 1.0])
    aff[0, 3] = -ref_origin[0]
    aff[1, 3] = -ref_origin[1]
    aff[2, 3] = ref_origin[2]
    return aff
