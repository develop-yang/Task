import argparse
import json
import os

import numpy as np
import torch

from dicom_io import load_series
from preprocess import nifti_affine, prepare_pair
from register import bone_dice, evaluate, jacobian_stats, register, rigid_to_physical


def save_nifti(path, vol_zyx, affine):
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.transpose(vol_zyx, (2, 1, 0)), affine), path)


def write_rigid_txt(path, phys, rg, df, jac, gain):
    t, r = phys["trans_mm"], phys["rot_deg"]
    L = ["CBCT -> pCT 刚体配准结果 (M24557)",
         "坐标系 DICOM 患者系: x 右->左(L+), y 前->后(P+), z 下->上(S+)",
         "",
         "摆位修正 (把 CBCT 对到 pCT):",
         "  平移 tx,ty,tz (mm): %8.3f %8.3f %8.3f" % (t[0], t[1], t[2]),
         "  旋转 rx,ry,rz (度): %8.3f %8.3f %8.3f" % (r[0], r[1], r[2]),
         "",
         "4x4 变换矩阵 (CBCT->pCT, mm):"]
    for row in phys["matrix"]:
        L.append("  " + "  ".join("%9.4f" % v for v in row))
    L += ["",
          "验证指标 (仅刚体 / +形变):",
          "  LNCC   : %.4f -> %.4f / %.4f" % (rg["ncc_before"], rg["ncc_after"], df["ncc_after"]),
          "  骨 Dice: %.4f -> %.4f / %.4f  (HU>200, 独立于优化目标)"
          % (rg["bone_dice_before"], rg["bone_dice_after"], df["bone_dice_after"]),
          "  形变 Jacobian: min=%.4f mean=%.4f 负值占比=%.4f%%"
          % (jac["min"], jac["mean"], 100 * jac["neg_fraction"]),
          "  结论: 刚体为主结果, 形变为受控小幅细化 (Dice 增益 %.4f)" % gain]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="CBCT 到 pCT 配准")
    ap.add_argument("--cbct", default="data/M24557-CBCT")
    ap.add_argument("--pct", default="data/M24557-pCT")
    ap.add_argument("--out", default="results")
    ap.add_argument("--spacing", type=float, default=3.0)     # 公共网格各向同性间距(mm)
    ap.add_argument("--rigid-iters", type=int, default=150)
    ap.add_argument("--deform-iters", type=int, default=60)
    ap.add_argument("--scale", action="store_true")           # 开缩放(默认纯刚体)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    cbct, pct = load_series(args.cbct), load_series(args.pct)
    print("CBCT", cbct.shape, "spacing", cbct.spacing)
    print("pCT ", pct.shape, "spacing", pct.spacing)

    data = prepare_pair(cbct, pct, spacing_mm=args.spacing)
    print("公共网格", data["ref_shape"], "初始平移(x,y,z)mm", np.round(data["init_offset"], 1))

    model, warped, warped_rigid = register(
        data["moving"], data["fixed"], mask=data["mask"], device=device,
        rigid_iters=args.rigid_iters, deform_iters=args.deform_iters, allow_scale=args.scale)

    to_hu = lambda x: x * 2000.0 - 1000.0
    rg = {**evaluate(data["moving"], data["fixed"], warped_rigid, data["mask"]),
          **bone_dice(data["moving_hu"], data["fixed_hu"], to_hu(warped_rigid), data["mask"])}
    df = {**evaluate(data["moving"], data["fixed"], warped, data["mask"]),
          **bone_dice(data["moving_hu"], data["fixed_hu"], to_hu(warped), data["mask"])}
    jac = jacobian_stats(model, data["mask"])
    gain = df["bone_dice_after"] - rg["bone_dice_after"]
    phys = rigid_to_physical(model, data["ref_origin"], data["ref_spacing"], data["ref_shape"])

    print("仅刚体  骨Dice %.3f->%.3f" % (rg["bone_dice_before"], rg["bone_dice_after"]))
    print("+形变   骨Dice ->%.3f (增益 %.3f)" % (df["bone_dice_after"], gain))
    print("Jacobian min=%.3f 负值占比=%.4f%%" % (jac["min"], 100 * jac["neg_fraction"]))
    print("摆位修正 平移(mm)", np.round(phys["trans_mm"], 2), "旋转(度)", np.round(phys["rot_deg"], 2))

    aff = nifti_affine(data["ref_origin"], data["ref_spacing"])
    save_nifti(os.path.join(args.out, "fixed_pct.nii.gz"), data["fixed_hu"], aff)
    save_nifti(os.path.join(args.out, "moving_cbct.nii.gz"), data["moving_hu"], aff)
    save_nifti(os.path.join(args.out, "warped_cbct.nii.gz"), to_hu(warped), aff)
    torch.save({"rot": model.rot.detach().cpu(), "trans": model.trans.detach().cpu(),
                "flow_lowres": model.flow_lowres.detach().cpu()},
               os.path.join(args.out, "transform.pt"))
    json.dump({"rigid_only": rg, "deform": df, "jacobian": jac, "deform_gain": gain},
              open(os.path.join(args.out, "metrics.json"), "w"), indent=2, ensure_ascii=False)
    write_rigid_txt(os.path.join(args.out, "rigid_transform.txt"), phys, rg, df, jac, gain)
    print("结果已保存到", args.out)


if __name__ == "__main__":
    main()
