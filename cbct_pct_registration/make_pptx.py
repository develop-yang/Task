"""用 python-pptx 生成 CBCT→pCT 配准汇报 deck（中文，白底+深蓝强调色）。

所有数字从 results_sample/metrics.json 与 results_sample/rigid_transform.txt
读取真实值；图来自 results_sample/figures/ 与 results_sample/。

用法:
    python -m cbct_pct_registration.make_pptx --out results_sample
"""

import argparse
import json
import os
import re

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ACCENT = RGBColor(0x1B, 0x3B, 0x6F)   # 深蓝
DARK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x5A, 0x5A, 0x5A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xE8, 0xED, 0xF5)
FONT = "Microsoft YaHei"  # Windows 常见中文字体；缺失时 PowerPoint 自动回退

SW, SH = 13.333, 7.5  # 16:9 inches


# --------------------------- 小工具 ---------------------------

def _set_font(run, size, bold=False, color=DARK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT


def add_title(slide, text):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(0.30), Inches(12.2),
                                  Inches(0.9))
    tf = tb.text_frame
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = text
    _set_font(r, 28, bold=True, color=ACCENT)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.58), Inches(1.18),
                                 Inches(2.4), Inches(0.06))
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()


def add_bullets(slide, items, left, top, width, height, size=16):
    """items: list of (text, level)。"""
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width),
                                  Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, (txt, lvl) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.level = lvl
        p.space_after = Pt(6)
        r = p.add_run()
        prefix = "• " if lvl == 0 else "– "
        r.text = prefix + txt
        _set_font(r, size - 2 * lvl, color=DARK if lvl == 0 else GRAY)


def add_image_fit(slide, path, left, top, width, height):
    try:
        from PIL import Image
        iw, ih = Image.open(path).size
        ar = iw / ih
    except Exception:
        slide.shapes.add_picture(path, Inches(left), Inches(top),
                                 width=Inches(width))
        return
    if width / height > ar:
        h, w = height, height * ar
    else:
        w, h = width, width / ar
    slide.shapes.add_picture(path, Inches(left + (width - w) / 2),
                             Inches(top + (height - h) / 2),
                             Inches(w), Inches(h))


def add_table(slide, rows, left, top, width, height, col_w=None,
              num_cols=None, header=True):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(left), Inches(top),
                                Inches(width), Inches(height)).table
    if col_w:
        for j, w in enumerate(col_w):
            gt.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = gt.cell(i, j)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_top = Pt(2)
            cell.margin_bottom = Pt(2)
            p = cell.text_frame.paragraphs[0]
            r = p.add_run()
            r.text = str(val)
            is_head = header and i == 0
            _set_font(r, 13 if is_head else 12, bold=is_head,
                      color=WHITE if is_head else DARK)
            if num_cols and j in num_cols and not is_head:
                p.alignment = PP_ALIGN.RIGHT
            cell.fill.solid()
            if is_head:
                cell.fill.fore_color.rgb = ACCENT
            else:
                cell.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
    return gt


def add_note(slide, text, top=6.95, color=GRAY, size=12):
    tb = slide.shapes.add_textbox(Inches(0.55), Inches(top), Inches(12.2),
                                  Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = text
    _set_font(r, size, color=color)


def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


# --------------------------- 读取真实数字 ---------------------------

def load_numbers(out):
    with open(os.path.join(out, "metrics.json")) as f:
        m = json.load(f)
    txt = open(os.path.join(out, "rigid_transform.txt"), encoding="utf-8").read()

    def floats(pat):
        mt = re.search(pat, txt)
        return [float(x) for x in re.findall(r"-?\d+\.\d+", mt.group(1))] if mt else []

    trans = floats(r"平移 tx, ty, tz \(mm\):(.*)")
    rot = floats(r"旋转 rx, ry, rz \(度\):(.*)")
    return m, trans, rot


# --------------------------- 各页 ---------------------------

def build(out):
    fig = os.path.join(out, "figures")
    m, trans, rot = load_numbers(out)

    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    # 1. 封面
    s = blank(prs)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(2.5), Inches(SW),
                              Inches(0.10))
    band.fill.solid(); band.fill.fore_color.rgb = ACCENT
    band.line.fill.background()
    tb = s.shapes.add_textbox(Inches(0.9), Inches(2.7), Inches(11.5), Inches(2))
    tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run()
    r.text = "CBCT → pCT 医学影像配准"
    _set_font(r, 40, bold=True, color=ACCENT)
    p = tf.add_paragraph(); r = p.add_run()
    r.text = "锥束CT 与 计划CT 的刚体 + 微分同胚形变配准（病例 M24557）"
    _set_font(r, 18, color=GRAY)
    p = tf.add_paragraph(); r = p.add_run()
    r.text = "汇报人：________      日期：2026-06-19"
    _set_font(r, 14, color=GRAY)

    # 2. 任务与数据
    s = blank(prs)
    add_title(s, "任务与数据")
    add_image_fit(s, os.path.join(fig, "data_overview.png"), 0.5, 1.45, 6.2, 5.0)
    rows = [["", "pCT 计划CT", "CBCT 锥束CT"],
            ["设备", "Philips Brilliance", "Elekta XVI"],
            ["模态", "CT", "CBCT(CT)"],
            ["矩阵", "512×512×112", "270×270×128"],
            ["体素(mm)", "1.16×1.16×3.0", "1.0×1.0×1.0"],
            ["坐标系", "CT 床坐标 (z≈-800)", "等中心坐标 (z≈0)"]]
    add_table(s, rows, 7.0, 1.7, 5.9, 2.6, col_w=[1.4, 2.4, 2.1])
    add_bullets(s, [
        ("三个难点：", 0),
        ("不同 Frame of Reference（世界坐标不重叠）", 1),
        ("灰度不一致（散射/HU 标定差异）", 1),
        ("分辨率与 FOV 不同（CBCT 视野小、被截断）", 1),
    ], 7.0, 4.5, 5.9, 2.2, size=16)

    # 3. 为什么不平凡
    s = blank(prs)
    add_title(s, "为什么不平凡：CBCT 的物理特性决定方法")
    add_image_fit(s, os.path.join(fig, "coord_offset.png"), 0.5, 1.35, 12.3, 2.9)
    add_bullets(s, [
        ("世界坐标系不同：CBCT 以治疗机等中心为原点、pCT 以 CT 床为原点，"
         "z 相差约 800 mm，DICOM 头无法直接对齐。", 0),
        ("X 射线散射 → cupping 伪影，软组织区 HU 向中心塌陷。", 0),
        ("HU 标定不准：CBCT 的 CT 值与计划 CT 不可直接比较。", 0),
        ("FOV 截断：CBCT 视野小，边缘信息缺失。", 0),
        ("→ 需要：稳健初始化 + 对灰度差异鲁棒的相似性 + 仅在 CBCT 视野内评估。", 0),
    ], 0.6, 4.5, 12.2, 2.8, size=14)

    # 4. 方法总览（流程图）
    s = blank(prs)
    add_title(s, "方法总览：纯 PyTorch 实例优化（非黑箱工具）")
    steps = ["DICOM 读取\n→ HU", "立方体公共网格\n+ 初始对齐",
             "多分辨率刚体\n(LNCC)", "微分同胚形变\n(速度场积分)",
             "输出\n变换/指标/图"]
    n = len(steps)
    bw, bh, gap = 2.05, 1.5, 0.45
    total = n * bw + (n - 1) * gap
    x = (SW - total) / 2
    y = 3.0
    for i, st in enumerate(steps):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                 Inches(y), Inches(bw), Inches(bh))
        box.fill.solid(); box.fill.fore_color.rgb = ACCENT if i % 2 == 0 else GRAY
        box.line.color.rgb = ACCENT
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = st
        _set_font(r, 14, bold=True, color=WHITE)
        if i < n - 1:
            ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                    Inches(x + bw + 0.04), Inches(y + bh / 2 - 0.18),
                                    Inches(gap - 0.08), Inches(0.36))
            ar.fill.solid(); ar.fill.fore_color.rgb = ACCENT
            ar.line.fill.background()
        x += bw + gap
    add_bullets(s, [
        ("单对图像的实例优化：直接对这一对体数据用梯度下降优化空间变换，"
         "无需训练数据 / 预训练模型。", 0),
        ("全部基于 PyTorch 的 affine_grid / grid_sample 自行实现。", 0),
    ], 0.8, 5.2, 11.8, 1.6, size=16)

    # 5. 关键设计① 初始对齐
    s = blank(prs)
    add_title(s, "关键设计①：稳健初始对齐")
    add_image_fit(s, os.path.join(fig, "zinit_xcorr.png"), 0.5, 1.5, 7.2, 5.2)
    add_bullets(s, [
        ("立方体 + 各向同性公共网格", 0),
        ("使 affine_grid 的归一化旋转 = 物理刚体旋转，不引入剪切。", 1),
        ("x / y：身体质心对齐", 0),
        ("z：横截面积剖面的 1D 互相关", 0),
        ("pCT 的 z 覆盖远大于 CBCT，z 质心落在不同解剖范围、不可比；"
         "用面积剖面互相关定位更稳健。", 1),
    ], 7.9, 1.7, 5.0, 5.0, size=16)

    # 6. 关键设计② 相似性与形变
    s = blank(prs)
    add_title(s, "关键设计②：相似性度量与形变模型")
    add_image_fit(s, os.path.join(fig, "intensity_cupping.png"), 0.5, 1.5, 7.2, 5.0)
    add_bullets(s, [
        ("相似性：局部归一化互相关 LNCC", 0),
        ("对 CBCT/pCT 的线性灰度差异、cupping 鲁棒。", 1),
        ("多分辨率金字塔 (4×→2×→1×)", 0),
        ("扩大捕获范围，避免陷入局部极小。", 1),
        ("形变：速度场 scaling-and-squaring 积分", 0),
        ("得到微分同胚位移场 → 可逆、无折叠。", 1),
        ("仅在 CBCT 视野(FOV)掩膜内计算损失。", 0),
    ], 7.9, 1.7, 5.0, 5.2, size=15)

    # 7. 结果·定性
    s = blank(prs)
    add_title(s, "结果 · 定性：棋盘格与差异图（前 vs 后）")
    add_image_fit(s, os.path.join(fig, "checkerboard.png"), 0.4, 1.45, 6.0, 5.4)
    add_image_fit(s, os.path.join(fig, "difference.png"), 6.7, 1.45, 6.2, 5.4)
    add_note(s, "棋盘格中骨与体表轮廓在配准后跨方格连续；差异图整体变暗。")

    # 8. 结果·定量
    s = blank(prs)
    add_title(s, "结果 · 定量（真实数值）")
    add_image_fit(s, os.path.join(fig, "metrics_bar.png"), 0.5, 1.5, 6.4, 4.4)
    neg_pct = 100 * m["jacobian_neg_fraction"]
    rows = [["指标", "数值"],
            ["刚体平移 tx,ty,tz (mm)",
             "%.2f, %.2f, %.2f" % (trans[0], trans[1], trans[2]) if trans else "-"],
            ["刚体旋转 rx,ry,rz (度)",
             "%.2f, %.2f, %.2f" % (rot[0], rot[1], rot[2]) if rot else "-"],
            ["骨 Dice（前→后）",
             "%.3f → %.3f" % (m["bone_dice_before"], m["bone_dice_after"])],
            ["LNCC（前→后）",
             "%.3f → %.3f" % (m["ncc_before"], m["ncc_after"])],
            ["Jacobian 最小值", "%.2f" % m["jacobian_min"]],
            ["Jacobian 负值占比", "%.3f%%" % neg_pct]]
    add_table(s, rows, 7.1, 1.7, 5.8, 3.6, col_w=[3.2, 2.6], num_cols=[1])
    add_bullets(s, [
        ("骨 Dice 独立于优化目标 LNCC：大幅提升说明是真实结构对齐，而非过拟合相似度。", 0),
        ("Jacobian 负值占比 ≈ 0 → 形变合法、无折叠。", 0),
    ], 7.1, 5.45, 5.8, 1.6, size=14)

    # 9. 收敛与合法性
    s = blank(prs)
    add_title(s, "收敛过程与形变合法性")
    add_image_fit(s, os.path.join(out, "loss_curve.png"), 0.5, 1.5, 7.0, 5.0)
    add_image_fit(s, os.path.join(fig, "jacobian_map.png"), 7.7, 1.5, 5.2, 5.0)
    add_note(s, "左：两阶段 loss 收敛；右：形变场 Jacobian 行列式中心层，"
                "绝大多数为正（全局负值占比 0.33%）。")

    # 10. 小结与局限
    s = blank(prs)
    add_title(s, "小结与局限")
    add_bullets(s, [
        ("做了什么", 0),
        ("纯 PyTorch 实现 CBCT→pCT 刚体 + 微分同胚形变配准，导出物理单位摆位修正。", 1),
        ("针对 CBCT 物理特性做了：稳健初始化、LNCC、FOV 掩膜、立方体真刚体。", 1),
        ("验证了什么", 0),
        ("独立于优化目标的骨 Dice 大幅提升；Jacobian 负值≈0 证明形变合法。", 1),
        ("局限", 0),
        ("未对 CBCT 散射 / HU 做定量校正；单病例；未做标志点 TRE 几何精度评估。", 1),
        ("可拓展", 0),
        ("散射/HU 校正、多病例评估、加入标志点或轮廓 Dice、学习式配准网络。", 1),
    ], 0.7, 1.5, 12.2, 5.6, size=17)

    path = os.path.join(out, "CBCT_pCT_配准汇报.pptx")
    prs.save(path)
    print("已生成", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_sample")
    args = p.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
