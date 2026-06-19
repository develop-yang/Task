"""为汇报 slides 生成高分辨率图（中文标注），输出到 results_sample/figures/。

依赖已完成的配准产物 results_sample/transform.pt 与原始数据 data/。
脚本不重新优化，只复用已保存的变换重建 warped / Jacobian 等。

用法:
    python -m cbct_pct_registration.make_slide_figures \
        --cbct data/M24557-CBCT --pct data/M24557-pCT --out results_sample
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .dicom_io import load_series
from .model import RegistrationModel
from .preprocess import (axial_area_profile, estimate_z_offset, normalize_hu,
                         prepare_pair)

# ---- 统一风格：白底 + 单一强调色（深蓝），对照色用深灰 ----
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"
ACCENT = "#1F3B6E"   # 深蓝（强调色）
SECOND = "#6B7280"   # 次级灰
BEFORE = "#9CA3AF"   # 配准前统一灰
DPI = 170


def _checker(a, b, n=8):
    out = a.copy()
    sh, sw = a.shape
    bh, bw = max(1, sh // n), max(1, sw // n)
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                out[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = \
                    b[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
    return out


def _show(ax, img, title="", cmap="gray", aspect=1.0):
    ax.imshow(np.rot90(img), cmap=cmap, aspect=aspect)
    if title:
        ax.set_title(title, fontsize=13)
    ax.axis("off")


def _jacobian_det(model):
    """返回形变场(id+位移)的 Jacobian 行列式体数据 (D,H,W)。"""
    with torch.no_grad():
        grid = (model._identity_grid() + model.full_flow()).cpu().numpy()[0]
    D, H, W = model.size
    vx = (grid[..., 0] + 1) / 2 * (W - 1)
    vy = (grid[..., 1] + 1) / 2 * (H - 1)
    vz = (grid[..., 2] + 1) / 2 * (D - 1)
    phi = [vz, vy, vx]
    J = np.empty((3, 3) + (D, H, W))
    for a in range(3):
        g = np.gradient(phi[a])
        for b in range(3):
            J[a, b] = g[b]
    return np.linalg.det(np.moveaxis(J, [0, 1], [-2, -1]))


# ---------------------------------------------------------------------------
# 各张图
# ---------------------------------------------------------------------------

def fig_data_overview(data, path):
    """CBCT / pCT 各取 轴/冠/矢 三视图（公共各向同性网格，等大、统一窗宽窗位）。"""
    mv, fx = data["moving_hu"], data["fixed_hu"]
    names = ["轴位", "冠状位", "矢状位"]
    rows = [("CBCT（Elekta XVI · 1.0³mm）", mv, ACCENT),
            ("pCT（Philips · 1.16×1.16×3.0mm）", fx, SECOND)]
    vmin, vmax = -500, 1000  # 统一窗宽窗位
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.6))
    for r, (label, vol, color) in enumerate(rows):
        views = _views(vol)
        for c in range(3):
            ax = axes[r, c]
            ax.imshow(np.rot90(views[c]), cmap="gray", vmin=vmin, vmax=vmax,
                      aspect=1.0)
            ax.axis("off")
            if r == 0:
                ax.set_title(names[c], fontsize=14)
        axes[r, 0].text(-0.08, 0.5, label, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=12,
                        color=color)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_coord_offset(cbct, pct, path):
    cz0, cz1 = cbct.world_extent()[0][2], cbct.world_extent()[1][2]
    pz0, pz1 = pct.world_extent()[0][2], pct.world_extent()[1][2]
    gap = ((cz0 + cz1) / 2) - ((pz0 + pz1) / 2)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.barh(1.0, cz1 - cz0, left=cz0, height=0.45, color=ACCENT)
    ax.barh(0.3, pz1 - pz0, left=pz0, height=0.45, color=SECOND)
    ax.text(cz1 + 15, 1.0, "CBCT z∈[%.0f, %.0f]" % (cz0, cz1),
            va="center", color=ACCENT, fontsize=12)
    ax.text(pz0 - 15, 0.3, "pCT z∈[%.0f, %.0f]" % (pz0, pz1),
            va="center", ha="right", color=SECOND, fontsize=12)
    ax.annotate("", xy=((cz0 + cz1) / 2, 0.66), xytext=((pz0 + pz1) / 2, 0.66),
                arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
    ax.text(((cz0 + cz1) / 2 + (pz0 + pz1) / 2) / 2, 0.74,
            "中心相差 ≈ %.0f mm" % abs(gap), ha="center", fontsize=13)
    ax.set_xlabel("世界坐标 z (mm)", fontsize=12)
    ax.set_yticks([])
    ax.set_ylim(-0.1, 1.5)
    ax.set_title("两套数据的 z 世界坐标完全不重叠（不在同一 Frame of Reference）",
                 fontsize=13)
    for s in ["top", "left", "right"]:
        ax.spines[s].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_intensity_cupping(data, path):
    """初始对齐后同一中心层，沿水平线比较 CBCT 与 pCT 的 HU 剖面。"""
    mv, fx, mask = data["moving_hu"], data["fixed_hu"], data["mask"]
    z = mv.shape[0] // 2
    # 取该层身体最“宽”的行，保证穿过身体
    body = (fx[z] > -400) & (mask[z] > 0.5)
    row = int(np.argmax(body.sum(axis=1)))
    cols = np.where(mask[z, row] > 0.5)[0]
    c0, c1 = cols.min(), cols.max()
    xs = np.arange(c0, c1)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(xs, fx[z, row, c0:c1], color=SECOND, lw=2, label="pCT（计划CT）")
    ax.plot(xs, mv[z, row, c0:c1], color=ACCENT, lw=2, label="CBCT（锥束CT）")
    ax.set_xlabel("沿水平线的体素位置", fontsize=12)
    ax.set_ylabel("HU", fontsize=12)
    ax.set_title("同层水平 HU 剖面：CBCT 中心偏低/起伏 → 散射 cupping、HU 标定差异",
                 fontsize=12)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_zinit_xcorr(cbct, pct, path):
    zc, ac = axial_area_profile(cbct)
    zp, ap = axial_area_profile(pct)
    off = estimate_z_offset(cbct, pct)  # pCT 采样位置 = cbct_z + off

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(zp, ap, color=SECOND, lw=2, label="pCT 各层横截面积")
    # 把 CBCT 的 z 映射到 pCT 坐标系：cbct_z + off
    ax.plot(zc + off, ac, color=ACCENT, lw=2,
            label="CBCT 各层横截面积（按互相关偏移对齐）")
    ax.set_xlabel("pCT 世界坐标 z (mm)", fontsize=12)
    ax.set_ylabel("身体横截面积（体素数）", fontsize=12)
    ax.set_title("z 初始化：横截面积剖面 1D 互相关定位（偏移 ≈ %.0f mm）" % off,
                 fontsize=12)
    ax.legend(fontsize=11, frameon=False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _views(vol):
    d, h, w = vol.shape
    return [vol[d // 2], vol[:, h // 2], vol[:, :, w // 2]]


def fig_checkerboard(data, warped, path):
    mv, fx = data["moving"], data["fixed"]
    names = ["轴位", "冠状位", "矢状位"]
    mvv, fxv, wpv = _views(mv), _views(fx), _views(warped)
    fig, axes = plt.subplots(3, 2, figsize=(8, 11))
    for r in range(3):
        _show(axes[r, 0], _checker(fxv[r], mvv[r]), "%s · 配准前" % names[r])
        _show(axes[r, 1], _checker(fxv[r], wpv[r]), "%s · 配准后" % names[r])
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_hero(data, warped, path):
    """封面用的干净单图：配准后轴位，pCT 灰底 + 配准后 CBCT 骨结构(强调色)叠加。"""
    fx_hu, mask = data["fixed_hu"], data["mask"]
    warped_hu = warped * 2000.0 - 1000.0
    z = fx_hu.shape[0] // 2
    base = np.clip(fx_hu[z], -500, 1000)
    bone = (warped_hu[z] > 200) & (mask[z] > 0.5)
    overlay = np.ma.masked_where(~bone, np.ones_like(base))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.rot90(base), cmap="gray")
    from matplotlib.colors import ListedColormap
    ax.imshow(np.rot90(overlay), cmap=ListedColormap([ACCENT]), alpha=0.55,
              vmin=0, vmax=1)
    ax.axis("off")
    plt.subplots_adjust(0, 0, 1, 1)
    plt.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def fig_difference(data, warped, path):
    mv, fx = data["moving"], data["fixed"]
    names = ["轴位", "冠状位"]
    mvv, fxv, wpv = _views(mv), _views(fx), _views(warped)
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    for r in range(2):
        before = np.abs(fxv[r] - mvv[r]) * 2000
        after = np.abs(fxv[r] - wpv[r]) * 2000
        _show(axes[r, 0], before, "%s · 配准前 |差异|" % names[r], cmap="hot")
        _show(axes[r, 1], after, "%s · 配准后 |差异|" % names[r], cmap="hot")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_metrics_bar(metrics, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, (key_b, key_a, title) in zip(axes, [
        ("ncc_before", "ncc_after", "LNCC（优化目标）"),
        ("bone_dice_before", "bone_dice_after", "骨结构 Dice（独立验证）")]):
        vals = [metrics[key_b], metrics[key_a]]
        bars = ax.bar(["配准前", "配准后"], vals, color=[BEFORE, ACCENT], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, "%.3f" % v,
                    ha="center", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=12)
        ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def fig_jacobian_map(det, mask, path, jac=None):
    z = det.shape[0] // 2
    sl = det[z].astype(float)
    sl[mask[z] <= 0.5] = np.nan
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(np.rot90(sl), cmap="RdBu_r", vmin=0.5, vmax=1.5)
    if jac and jac["neg_fraction"] <= 1e-9 and jac["min"] > 0:
        sub = "处处为正 (min=%.2f) → 微分同胚、无折叠" % jac["min"]
    elif jac:
        sub = "min=%.2f，负值占比 %.2f%%" % (
            jac["min"], 100 * jac["neg_fraction"])
    else:
        sub = ""
    ax.set_title("形变场 Jacobian 行列式（中心轴位层）\n" + sub, fontsize=11)
    ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("det(J)", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cbct", default="data/M24557-CBCT")
    p.add_argument("--pct", default="data/M24557-pCT")
    p.add_argument("--out", default="results_sample")
    args = p.parse_args()

    figdir = os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)

    ckpt_path = os.path.join(args.out, "transform.pt")
    try:  # torch>=2.6 默认 weights_only=True，需关掉以载入 numpy 几何信息
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:  # torch 1.10 没有该参数
        ckpt = torch.load(ckpt_path, map_location="cpu")
    with open(os.path.join(args.out, "metrics.json")) as f:
        metrics = json.load(f)

    spacing = float(ckpt["ref_spacing"][0])
    print("加载数据并按 %.1f mm 重建公共网格 ..." % spacing)
    cbct = load_series(args.cbct)
    pct = load_series(args.pct)
    data = prepare_pair(cbct, pct, spacing_mm=spacing)

    size = data["ref_shape"]
    model = RegistrationModel(size, diffeomorphic=True,
                             smooth_sigma=ckpt.get("smooth_sigma", 0.0),
                             int_steps=ckpt.get("int_steps", 7))
    # 直接用已保存的变换覆盖参数（位移场分辨率由 ckpt 决定，上采样时自适应）
    model.rot.data = ckpt["rot"]
    model.trans.data = ckpt["trans"]
    model.flow_lowres = torch.nn.Parameter(ckpt["flow_lowres"])
    with torch.no_grad():
        mv = torch.from_numpy(data["moving"]).float()[None, None]
        warped_rigid = model.warp(mv, use_flow=False).numpy()[0, 0]
        warped_deform = model.warp(mv, use_flow=True).numpy()[0, 0]
    det = _jacobian_det(model)
    # 主结果为刚体：定性图用刚体结果，形变仅在 Jacobian 图体现其合法性
    warped_primary = warped_rigid

    print("生成图 ...")
    fig_data_overview(data, os.path.join(figdir, "data_overview.png"))
    fig_coord_offset(cbct, pct, os.path.join(figdir, "coord_offset.png"))
    fig_intensity_cupping(data, os.path.join(figdir, "intensity_cupping.png"))
    fig_zinit_xcorr(cbct, pct, os.path.join(figdir, "zinit_xcorr.png"))
    fig_checkerboard(data, warped_primary, os.path.join(figdir, "checkerboard.png"))
    fig_difference(data, warped_primary, os.path.join(figdir, "difference.png"))
    fig_hero(data, warped_primary, os.path.join(figdir, "hero.png"))
    fig_metrics_bar(metrics["rigid_only"], os.path.join(figdir, "metrics_bar.png"))
    fig_jacobian_map(det, data["mask"], os.path.join(figdir, "jacobian_map.png"),
                     jac=metrics.get("jacobian"))
    print("完成，图已写入", figdir)


if __name__ == "__main__":
    main()
