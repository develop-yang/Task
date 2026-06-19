import numpy as np
from scipy.ndimage import map_coordinates

from dicom_io import Volume

AIR_HU = -1000.0


def make_reference_grid(vol, spacing_mm):
    # 立方体 + 各向同性网格：保证后面 affine_grid 的归一化旋转等于物理刚体旋转
    mn, mx = vol.world_extent()
    center = (mn + mx) / 2.0
    spacing = np.array([spacing_mm] * 3, np.float64)
    n = int(np.ceil((mx - mn) / spacing).max()) + 1
    origin = center - (n - 1) / 2.0 * spacing
    return origin, spacing, (n, n, n)


def resample_to_grid(vol, ref_origin, ref_spacing, ref_shape, order=1, fill=AIR_HU):
    nz, ny, nx = ref_shape
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    wx = ref_origin[0] + xx * ref_spacing[0]
    wy = ref_origin[1] + yy * ref_spacing[1]
    wz = ref_origin[2] + zz * ref_spacing[2]
    src = np.stack([(wz - vol.origin[2]) / vol.spacing[2],
                    (wy - vol.origin[1]) / vol.spacing[1],
                    (wx - vol.origin[0]) / vol.spacing[0]])
    out = map_coordinates(vol.data, src.reshape(3, -1), order=order,
                          mode="constant", cval=fill, prefilter=False)
    return out.reshape(ref_shape).astype(np.float32)


def normalize_hu(arr, lo=-1000.0, hi=1000.0):
    return (np.clip(arr, lo, hi) - lo) / (hi - lo)


def body_centroid(vol, thresh=-400.0):
    body = vol.data > thresh
    c = np.argwhere(body).mean(0) if body.sum() > 10 else np.array(vol.data.shape) / 2.0
    return vol.origin + c[::-1] * vol.spacing


def axial_area_profile(vol, thresh=-300.0):
    area = (vol.data > thresh).sum(axis=(1, 2)).astype(np.float64)
    z = vol.origin[2] + np.arange(vol.data.shape[0]) * vol.spacing[2]
    return z, area


def _ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 1e-6 else -1.0


def estimate_z_offset(cbct, pct, search_spacing=5.0, step=5.0):
    # CBCT(等中心系)与 pCT(床系)不共坐标系，z 相差几百 mm。
    # 直接在一组候选 z 偏移上重采样 pCT，与 CBCT 比 masked-NCC 取最优，
    # 比单纯靠横截面积剖面互相关稳健（不会因头颈面积曲线多峰而挑错）。
    ro, rs, sh = make_reference_grid(cbct, search_spacing)
    moving = normalize_hu(resample_to_grid(cbct, ro, rs, sh))
    mask = resample_to_grid(Volume(np.ones_like(cbct.data), cbct.spacing, cbct.origin),
                            ro, rs, sh, order=0, fill=0.0) > 0.5
    cxy = body_centroid(pct) - body_centroid(cbct)
    mv = moving[mask]

    czmin, czmax = cbct.world_extent()[0][2], cbct.world_extent()[1][2]
    pzmin, pzmax = pct.world_extent()[0][2], pct.world_extent()[1][2]
    lo, hi = pzmin - czmin, pzmax - czmax            # 让 CBCT 的 z 落进 pCT 范围
    if lo > hi:
        lo, hi = hi, lo
    best, best_off = -2.0, (lo + hi) / 2.0
    for off in np.arange(lo, hi + 1e-6, step):
        fx = normalize_hu(resample_to_grid(pct, ro + np.array([cxy[0], cxy[1], off]), rs, sh))
        s = _ncc(mv, fx[mask])
        if s > best:
            best, best_off = s, float(off)
    return best_off


def prepare_pair(cbct, pct, spacing_mm=3.0):
    ref_origin, ref_spacing, ref_shape = make_reference_grid(cbct, spacing_mm)

    # CBCT(等中心系) 与 pCT(床系) 不在同一世界坐标系，先做初始平移把两者拉到重叠
    off = np.zeros(3)
    c = body_centroid(pct) - body_centroid(cbct)
    off[0], off[1] = c[0], c[1]                       # x,y 用质心
    off[2] = estimate_z_offset(cbct, pct)             # z 用面积剖面互相关

    moving_hu = resample_to_grid(cbct, ref_origin, ref_spacing, ref_shape)
    fixed_hu = resample_to_grid(pct, ref_origin + off, ref_spacing, ref_shape)
    mask = resample_to_grid(Volume(np.ones_like(cbct.data), cbct.spacing, cbct.origin),
                            ref_origin, ref_spacing, ref_shape, order=0, fill=0.0) > 0.5

    return {
        "moving": normalize_hu(moving_hu), "fixed": normalize_hu(fixed_hu),
        "moving_hu": moving_hu, "fixed_hu": fixed_hu, "mask": mask.astype(np.float32),
        "ref_origin": ref_origin, "ref_spacing": ref_spacing, "ref_shape": ref_shape,
        "init_offset": off,
    }


def nifti_affine(ref_origin, ref_spacing):
    # LPS -> RAS：x,y 取负
    aff = np.diag([-ref_spacing[0], -ref_spacing[1], ref_spacing[2], 1.0])
    aff[0, 3], aff[1, 3], aff[2, 3] = -ref_origin[0], -ref_origin[1], ref_origin[2]
    return aff
