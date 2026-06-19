"""DICOM 读写工具。

这些 DICOM 文件（Elekta CBCT / Philips pCT）缺少标准的 File-Meta 头，
因此需要用 ``force=True`` 读取，并手动补上传输语法（Implicit VR Little Endian）。
本模块负责把一个序列目录读成 (z, y, x) 的 HU 体数据，并返回其几何信息
（体素间距、世界坐标原点、方向余弦）。
"""

import glob
import os

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian


class Volume:
    """带几何信息的 3D 体数据。

    约定：数组维度顺序为 (z, y, x)。
    spacing/origin 均按 (x, y, z) 给出，单位 mm，坐标系为 DICOM 的 LPS。
    """

    def __init__(self, data, spacing, origin, direction=None):
        self.data = data.astype(np.float32)          # (z, y, x), HU
        self.spacing = np.asarray(spacing, np.float64)  # (sx, sy, sz)
        self.origin = np.asarray(origin, np.float64)    # (ox, oy, oz)，第一层第一像素的世界坐标
        # 方向余弦（行/列方向），本数据均为轴位，方向为单位阵
        self.direction = np.eye(3) if direction is None else np.asarray(direction, np.float64)

    @property
    def shape(self):
        return self.data.shape

    def world_extent(self):
        """返回体数据在世界坐标下的包围盒 (min_xyz, max_xyz)。"""
        nz, ny, nx = self.data.shape
        # 八个角点
        idx = np.array([[0, 0, 0], [nx - 1, 0, 0], [0, ny - 1, 0], [0, 0, nz - 1],
                        [nx - 1, ny - 1, 0], [nx - 1, 0, nz - 1], [0, ny - 1, nz - 1],
                        [nx - 1, ny - 1, nz - 1]], np.float64)
        world = self.origin + idx * self.spacing  # 方向为单位阵，可直接相乘
        return world.min(0), world.max(0)


def _decode(ds):
    """补上缺失的传输语法，使 pydicom 能解码像素。"""
    if not hasattr(ds, "file_meta") or ds.file_meta is None or \
            "TransferSyntaxUID" not in ds.file_meta:
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = (
            ImplicitVRLittleEndian if ds.is_implicit_VR else ExplicitVRLittleEndian
        )
    return ds


def load_series(folder):
    """读取一个 DICOM 序列目录，返回 :class:`Volume`（HU 体数据）。"""
    # 用 os.listdir 只枚举一次目录，按扩展名(忽略大小写)过滤；
    # 不要用 glob("*.DCM")+glob("*.dcm")：Windows 大小写不敏感会让同一文件取两遍，
    # 切片翻倍后层间距会被算成 0
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith(".dcm")]
    if not files:
        raise FileNotFoundError("目录下没有找到 DICOM 文件: %s" % folder)

    slices = [_decode(pydicom.dcmread(f, force=True)) for f in files]
    # 按层位置 z 升序排列，保证 z 间距为正
    slices.sort(key=lambda d: float(d.ImagePositionPatient[2]))

    # 按 z 去重，防止重复层把层间距污染成 0
    uniq, zset = [], set()
    for s in slices:
        zr = round(float(s.ImagePositionPatient[2]), 3)
        if zr not in zset:
            zset.add(zr)
            uniq.append(s)
    slices = uniq

    vol = np.stack([
        s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1.0))
        + float(getattr(s, "RescaleIntercept", 0.0))
        for s in slices
    ])  # (z, y, x), HU

    ps = [float(v) for v in slices[0].PixelSpacing]      # (row=y, col=x)
    spacing_x, spacing_y = ps[1], ps[0]
    zs = np.array([float(s.ImagePositionPatient[2]) for s in slices])
    spacing_z = float(np.median(np.diff(zs))) if len(zs) > 1 else 1.0
    if spacing_z == 0:
        spacing_z = float(getattr(slices[0], "SliceThickness", 1.0)) or 1.0

    origin = [float(v) for v in slices[0].ImagePositionPatient]  # (x, y, z)
    iop = [float(v) for v in slices[0].ImageOrientationPatient]
    direction = np.array([
        [iop[0], iop[3], 0.0],
        [iop[1], iop[4], 0.0],
        [iop[2], iop[5], 0.0],
    ])
    # 第三列（层方向）取行列方向的叉乘
    direction[:, 2] = np.cross(direction[:, 0], direction[:, 1])

    return Volume(vol, (spacing_x, spacing_y, spacing_z), origin, direction)
