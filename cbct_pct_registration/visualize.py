"""配准结果可视化：三视图叠加、棋盘格、差异图、loss 曲线。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _slices(vol):
    """取体数据三个正交方向的中间切片 (axial, coronal, sagittal)。"""
    d, h, w = vol.shape
    return vol[d // 2], vol[:, h // 2], vol[:, :, w // 2]


def _checker(a, b, n=8):
    """生成 a、b 的棋盘格拼接。"""
    out = a.copy()
    sh, sw = a.shape
    bh, bw = sh // n, sw // n
    for i in range(n):
        for j in range(n):
            if (i + j) % 2 == 0:
                out[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw] = \
                    b[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
    return out


def overlay_figure(moving, fixed, warped, out_path):
    """三行（axial/coronal/sagittal）× 四列（fixed、配准前棋盘、配准后棋盘、差异）。"""
    names = ["Axial", "Coronal", "Sagittal"]
    mv_s, fx_s, wp_s = _slices(moving), _slices(fixed), _slices(warped)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for r in range(3):
        before = _checker(fx_s[r], mv_s[r])
        after = _checker(fx_s[r], wp_s[r])
        diff = np.abs(fx_s[r] - wp_s[r])
        for c, (img, title, cmap) in enumerate([
            (fx_s[r], "%s: fixed(pCT)" % names[r], "gray"),
            (before, "before (checker)", "gray"),
            (after, "after (checker)", "gray"),
            (diff, "|fixed - warped|", "hot"),
        ]):
            axes[r, c].imshow(np.rot90(img), cmap=cmap)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def history_figure(history, out_path):
    """绘制刚体/形变两阶段的 loss 曲线。"""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history["rigid"])
    ax[0].set_title("Stage 1: rigid (LNCC loss)")
    ax[0].set_xlabel("iter"); ax[0].set_ylabel("1 - LNCC")
    if history["deform"]:
        sim = [s for s, _ in history["deform"]]
        reg = [r for _, r in history["deform"]]
        ax[1].plot(sim, label="1 - LNCC")
        ax[1].plot(reg, label="reg (diffusion)")
        ax[1].legend()
    ax[1].set_title("Stage 2: deformable")
    ax[1].set_xlabel("iter")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
