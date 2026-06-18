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
    """根据给定体数据的世界包围盒，构造各向同性参考网格。

    返回 (ref_origin(x,y,z), ref_spacing(x,y,z), shape(z,y,x))。
    """
    mn, mx = vol_for_extent.world_extent()
    spacing = np.array([spacing_mm, spacing_mm, spacing_mm], np.float64)
    size_xyz = np.ceil((mx - mn) / spacing).astype(int) + 1
    shape_zyx = (int(size_xyz[2]), int(size_xyz[1]), int(size_xyz[0]))
    return mn.copy(), spacing, shape_zyx


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


def prepare_pair(cbct, pct, spacing_mm=1.5, align_centroid=True):
    """生成公共网格上的 (moving=CBCT, fixed=pCT) 及掩膜。

    CBCT（治疗机等中心坐标系）与 pCT（CT 床坐标系）通常不在同一世界坐标系，
    因此默认先按“身体质心”做一次初始平移对齐，把两者拉到大致重叠，
    再交给 PyTorch 做刚体 + 形变细化。

    返回字典，含归一化体数据、原始 HU 体数据、参考几何、CBCT 视野掩膜。
    """
    ref_origin, ref_spacing, ref_shape = make_reference_grid(cbct, spacing_mm)

    # 以 CBCT 包围盒为公共网格；采样 pCT 时加上质心偏移，使其落入该网格
    offset = np.zeros(3)
    if align_centroid:
        offset = body_centroid_world(pct) - body_centroid_world(cbct)

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
