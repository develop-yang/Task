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
from .register import (bone_dice, evaluate, jacobian_stats, register,
                       rigid_to_physical)
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
    p.add_argument("--reg-weight", type=float, default=3.0, help="形变场正则权重")
    p.add_argument("--deform-lr", type=float, default=0.02, help="形变阶段学习率")
    p.add_argument("--flow-downsample", type=int, default=4)
    p.add_argument("--scale", action="store_true",
                   help="刚体阶段额外开启各向同性缩放(默认关闭,纯6自由度刚体)")
    p.add_argument("--bone-thresh", type=float, default=200.0,
                   help="骨结构 Dice 的 HU 阈值")
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
    print("  初始平移 (x,y质心 / z互相关) mm:",
          np.round(data["centroid_offset"], 1))

    print("[3/5] PyTorch 刚体 + 形变优化 (%s) ..."
          % ("刚体+缩放+形变" if args.scale else "纯刚体6DOF+形变"))
    model, warped, history = register(
        data["moving"], data["fixed"], mask=data["mask"], device=device,
        rigid_iters=args.rigid_iters, deform_iters=args.deform_iters,
        ncc_win=args.ncc_win, reg_weight=args.reg_weight, deform_lr=args.deform_lr,
        allow_scale=args.scale, flow_downsample=args.flow_downsample,
    )

    print("[4/5] 评估指标 (含独立验证: 骨 Dice / Jacobian) ...")
    warped_hu = warped * 2000.0 - 1000.0  # 归一化 -> HU
    metrics = evaluate(data["moving"], data["fixed"], warped, mask=data["mask"])
    metrics.update(bone_dice(data["moving_hu"], data["fixed_hu"], warped_hu,
                             mask=data["mask"], thresh=args.bone_thresh))
    metrics.update(jacobian_stats(model, mask=data["mask"]))
    for k, v in metrics.items():
        print("  %-22s %.5f" % (k, v))

    # 物理单位刚体变换（配准的核心“答案”）
    phys = rigid_to_physical(model, data["ref_origin"], data["ref_spacing"],
                             data["ref_shape"])

    print("[5/5] 保存结果到 %s ..." % args.out)
    affine = to_nifti_affine(data["ref_origin"], data["ref_spacing"])
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

    _write_rigid_txt(os.path.join(args.out, "rigid_transform.txt"), phys,
                     data, metrics, args)

    print("\n===== 刚体配准结果 (CBCT -> pCT 摆位修正) =====")
    print("  平移 (mm)  [x:L+, y:P+, z:S+] :",
          np.round(phys["translation_mm_cbct_to_pct"], 2))
    print("  旋转 (度)  [rx, ry, rz]       :",
          np.round(phys["rotation_deg_cbct_to_pct"], 2))
    print("  骨 Dice: %.3f -> %.3f   Jacobian 负值占比: %.4f%%"
          % (metrics["bone_dice_before"], metrics["bone_dice_after"],
             100 * metrics["jacobian_neg_fraction"]))
    print("完成。结果在 %s/：overlay.png / metrics.json / rigid_transform.txt"
          % args.out)


def _write_rigid_txt(path, phys, data, metrics, args):
    """把物理单位刚体变换与关键指标写入文本文件。"""
    tc = phys["translation_mm_cbct_to_pct"]
    rc = phys["rotation_deg_cbct_to_pct"]
    lines = [
        "CBCT -> pCT 刚体配准结果 (病例 M24557)",
        "=" * 56,
        "坐标系: DICOM 患者坐标 (x: 右->左 L+, y: 前->后 P+, z: 下->上 S+)",
        "缩放: %s (scale=%.4f)" % (
            "开启" if args.scale else "关闭(纯6DOF刚体)", phys["scale"]),
        "公共网格(立方体): %s  各向同性间距 %.2f mm"
        % (str(data["ref_shape"]), float(data["ref_spacing"][0])),
        "",
        "== 摆位修正量 (把 CBCT 对到 pCT 所需的刚体变换) ==",
        "平移 tx, ty, tz (mm): %8.3f  %8.3f  %8.3f" % (tc[0], tc[1], tc[2]),
        "旋转 rx, ry, rz (度): %8.3f  %8.3f  %8.3f" % (rc[0], rc[1], rc[2]),
        "",
        "== 反方向 (pCT -> CBCT) 供参考 ==",
        "平移 (mm): %8.3f  %8.3f  %8.3f"
        % tuple(phys["translation_mm_pct_to_cbct"]),
        "旋转 (度): %8.3f  %8.3f  %8.3f"
        % tuple(phys["rotation_deg_pct_to_cbct"]),
        "",
        "== 4x4 齐次变换矩阵 (CBCT->pCT, 世界坐标 mm) ==",
    ]
    for row in phys["matrix_cbct_to_pct"]:
        lines.append("  " + "  ".join("%10.4f" % v for v in row))
    lines += [
        "",
        "== 验证指标 ==",
        "LNCC      : %.4f -> %.4f" % (metrics["ncc_before"], metrics["ncc_after"]),
        "骨 Dice   : %.4f -> %.4f (HU>%.0f, 独立于优化目标)"
        % (metrics["bone_dice_before"], metrics["bone_dice_after"], args.bone_thresh),
        "形变 Jacobian: min=%.4f mean=%.4f 负值占比=%.4f%%"
        % (metrics["jacobian_min"], metrics["jacobian_mean"],
           100 * metrics["jacobian_neg_fraction"]),
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
