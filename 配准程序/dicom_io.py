import os

import numpy as np
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, ImplicitVRLittleEndian


class Volume:
    """带几何信息的体数据，数组维度 (z, y, x)，HU。"""

    def __init__(self, data, spacing, origin, direction=None):
        self.data = data.astype(np.float32)
        self.spacing = np.asarray(spacing, np.float64)   # x, y, z (mm)
        self.origin = np.asarray(origin, np.float64)     # 首层首像素世界坐标 (x,y,z)
        self.direction = np.eye(3) if direction is None else np.asarray(direction, np.float64)

    @property
    def shape(self):
        return self.data.shape

    def world_extent(self):
        nz, ny, nx = self.data.shape
        idx = np.array([[0, 0, 0], [nx - 1, 0, 0], [0, ny - 1, 0], [0, 0, nz - 1],
                        [nx - 1, ny - 1, 0], [nx - 1, 0, nz - 1], [0, ny - 1, nz - 1],
                        [nx - 1, ny - 1, nz - 1]], np.float64)
        w = self.origin + idx * self.spacing
        return w.min(0), w.max(0)


def _fix_meta(ds):
    # 这两组 DICOM 缺标准 file meta，手动补传输语法后才能解码像素
    if not getattr(ds, "file_meta", None) or "TransferSyntaxUID" not in ds.file_meta:
        ds.file_meta = pydicom.dataset.FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = (
            ImplicitVRLittleEndian if ds.is_implicit_VR else ExplicitVRLittleEndian)
    return ds


def load_series(folder):
    # 用 os.listdir 只枚举一次目录，按扩展名(忽略大小写)过滤。
    # 不要用 glob("*.DCM")+glob("*.dcm")：Windows 大小写不敏感会让同一文件被取两遍，
    # 切片翻倍后 np.diff(z) 半数为 0 -> 层间距算成 0 -> 重采样除零 -> 全部失效。
    files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
             if f.lower().endswith(".dcm")]
    if not files:
        raise FileNotFoundError("没找到 DICOM: " + folder)

    sl = [_fix_meta(pydicom.dcmread(f, force=True)) for f in files]
    sl.sort(key=lambda d: float(d.ImagePositionPatient[2]))   # 按层位置 z 升序

    # 按 z 去重，防止重复层把层间距污染成 0
    uniq, zset = [], set()
    for s in sl:
        zr = round(float(s.ImagePositionPatient[2]), 3)
        if zr not in zset:
            zset.add(zr)
            uniq.append(s)
    sl = uniq

    print("  %s: %d 文件 -> %d 层" % (os.path.basename(folder.rstrip("/\\")),
                                     len(files), len(sl)))

    vol = np.stack([s.pixel_array.astype(np.float32) * float(getattr(s, "RescaleSlope", 1.0))
                    + float(getattr(s, "RescaleIntercept", 0.0)) for s in sl])

    ps = [float(v) for v in sl[0].PixelSpacing]               # [row(y), col(x)]
    zs = np.array([float(s.ImagePositionPatient[2]) for s in sl])
    diffs = np.diff(zs)
    diffs = diffs[diffs > 0]                                  # 只取正间距，避免重复层把中位数拖成 0
    dz = float(np.median(diffs)) if diffs.size else float(getattr(sl[0], "SliceThickness", 1.0)) or 1.0
    origin = [float(v) for v in sl[0].ImagePositionPatient]

    iop = [float(v) for v in sl[0].ImageOrientationPatient]
    d = np.array([[iop[0], iop[3], 0.0], [iop[1], iop[4], 0.0], [iop[2], iop[5], 0.0]])
    d[:, 2] = np.cross(d[:, 0], d[:, 1])
    return Volume(vol, (ps[1], ps[0], dz), origin, d)
