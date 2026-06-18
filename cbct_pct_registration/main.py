"""端到端运行 CBCT -> pCT 配准。

用法示例：
    python -m cbct_pct_registration.main \
        --cbct data/M24557-CBCT --pct data/M24557-pCT --out results

默认参数即可直接运行；有 GPU 时自动使用 CUDA。
"""

import argparse
import json
import os

import numpy as np
import torch

from .dicom_io import load_series
from .preprocess import prepare_pair, to_nifti_affine
from .register import evaluate, register
from .visualize import history_figure, overlay_figure


def _save_volume(path, arr_zyx, affine):
    """保存为 NIfTI（需要 nibabel），否则退化为 .npy。"""
    try:
        import nibabel as nib
        # NIfTI 习惯 (x, y, z)，这里把 (z, y, x) 转回 (x, y, z)
        img = nib.Nifti1Image(np.transpose(arr_zyx, (2, 1, 0)), affine)
        nib.save(img, path)
        return path
    except Exception as e:  # pragma: no cover
        np.save(path + ".npy", arr_zyx)
        print("  (未能保存 NIfTI: %s，已改存 %s.npy)" % (e, path))
        return path + ".npy"


def main():
    p = argparse.ArgumentParser(description="CBCT->pCT PyTorch 刚体+形变配准")
    p.add_argument("--cbct", default="data/M24557-CBCT", help="CBCT 序列目录")
    p.add_argument("--pct", default="data/M24557-pCT", help="pCT 序列目录")
    p.add_argument("--out", default="results", help="输出目录")
    p.add_argument("--spacing", type=float, default=1.5, help="公共网格各向同性间距(mm)")
    p.add_argument("--rigid-iters", type=int, default=300)
    p.add_argument("--deform-iters", type=int, default=300)
    p.add_argument("--ncc-win", type=int, default=9, help="LNCC 窗口边长")
    p.add_argument("--reg-weight", type=float, default=1.0, help="形变场正则权重")
    p.add_argument("--flow-downsample", type=int, default=4)
    p.add_argument("--no-scale", action="store_true", help="刚体阶段禁用各向同性缩放")
    p.add_argument("--device", default=None, help="cpu / cuda；默认自动选择")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备:", device)

    print("[1/5] 读取 DICOM 序列 ...")
    cbct = load_series(args.cbct)
    pct = load_series(args.pct)
    print("  CBCT:", cbct.shape, "spacing", cbct.spacing)
    print("  pCT :", pct.shape, "spacing", pct.spacing)

    print("[2/5] 重采样到公共网格 (%.1f mm) ..." % args.spacing)
    data = prepare_pair(cbct, pct, spacing_mm=args.spacing)
    print("  公共网格形状 (z,y,x):", data["ref_shape"])
    print("  质心初始平移 (x,y,z) mm:", np.round(data["centroid_offset"], 1))

    print("[3/5] PyTorch 刚体 + 形变优化 ...")
    model, warped, history = register(
        data["moving"], data["fixed"], mask=data["mask"], device=device,
        rigid_iters=args.rigid_iters, deform_iters=args.deform_iters,
        ncc_win=args.ncc_win, reg_weight=args.reg_weight,
        allow_scale=not args.no_scale, flow_downsample=args.flow_downsample,
    )

    print("[4/5] 评估指标 ...")
    metrics = evaluate(data["moving"], data["fixed"], warped, mask=data["mask"])
    for k, v in metrics.items():
        print("  %-12s %.5f" % (k, v))

    print("[5/5] 保存结果到 %s ..." % args.out)
    affine = to_nifti_affine(data["ref_origin"], data["ref_spacing"])
    # 把归一化的配准结果还原到 HU 再保存
    warped_hu = warped * 2000.0 - 1000.0
    _save_volume(os.path.join(args.out, "fixed_pct.nii.gz"),
                 data["fixed_hu"], affine)
    _save_volume(os.path.join(args.out, "moving_cbct_init.nii.gz"),
                 data["moving_hu"], affine)
    _save_volume(os.path.join(args.out, "warped_cbct.nii.gz"),
                 warped_hu, affine)

    # 保存变换参数与位移场
    torch.save({
        "rot": model.rot.detach().cpu(),
        "trans": model.trans.detach().cpu(),
        "log_scale": model.log_scale.detach().cpu(),
        "flow_lowres": model.flow_lowres.detach().cpu(),
        "ref_origin": data["ref_origin"],
        "ref_spacing": data["ref_spacing"],
        "ref_shape": data["ref_shape"],
    }, os.path.join(args.out, "transform.pt"))

    overlay_figure(data["moving"], data["fixed"], warped,
                   os.path.join(args.out, "overlay.png"))
    history_figure(history, os.path.join(args.out, "loss_curve.png"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("完成。可用 ITK-SNAP / 3D Slicer 打开 results/*.nii.gz，"
          "或查看 results/overlay.png 对比配准前后。")


if __name__ == "__main__":
    main()
