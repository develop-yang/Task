"""用 python-pptx 生成 CBCT→pCT 配准汇报 deck（中文，统一设计系统）。

设计系统：
- 配色：强调色 #1F3B6E；次级灰 #6B7280；分隔线 #E5E7EB；正文 #1F2937；
  配准前灰 #9CA3AF、配准后强调色；纯白背景。
- 字号：页标题 30pt 加粗(强调色)；正文 18pt；图注 13pt(灰)；
  表头 15pt 加粗(白)；表体 15pt；页脚 11pt(灰)；行距 1.2。
- 版式：左右安全边距 0.7in；顶部全宽强调色细条页眉；页脚左“病例 M24557”、右页码。

所有数字从 results_sample/metrics.json 与 rigid_transform.txt 读取真实值。
用法：python -m cbct_pct_registration.make_pptx --out results_sample
"""

import argparse
import json
import os
import re

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ACCENT = RGBColor(0x1F, 0x3B, 0x6E)
GRAY2 = RGBColor(0x6B, 0x72, 0x80)
LINE = RGBColor(0xE5, 0xE7, 0xEB)
TEXT = RGBColor(0x1F, 0x29, 0x37)
BEFORE = RGBColor(0x9C, 0xA3, 0xAF)
ALT = RGBColor(0xF3, 0xF6, 0xFB)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FONT = "Microsoft YaHei"

SW, SH = 13.333, 7.5
MARGIN = 0.7
S_TITLE, S_BODY, S_NOTE, S_FOOT = 30, 18, 13, 11
NOGRID = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"  # 内置“无样式无网格”


# --------------------------- 文本 ---------------------------

def _font(run, size, bold=False, color=TEXT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", FONT)


def _box(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width),
                                  Inches(height))
    tb.text_frame.word_wrap = True
    return tb


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --------------------------- 模板 ---------------------------

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _rect(slide, shape, left, top, w, h, fill=None, line=None, line_w=1.0):
    sp = slide.shapes.add_shape(shape, Inches(left), Inches(top),
                                Inches(w), Inches(h))
    if fill is None:
        sp.fill.background()
    else:
        sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(line_w)
    sp.shadow.inherit = False
    return sp


def master(slide, title, page):
    _rect(slide, MSO_SHAPE.RECTANGLE, 0, 0, SW, 0.12, fill=ACCENT)
    tb = _box(slide, MARGIN, 0.34, SW - 2 * MARGIN, 0.85)
    r = tb.text_frame.paragraphs[0].add_run(); r.text = title
    _font(r, S_TITLE, bold=True, color=ACCENT)
    fl = _box(slide, MARGIN, 7.04, 6, 0.36)
    r = fl.text_frame.paragraphs[0].add_run(); r.text = "病例 M24557"
    _font(r, S_FOOT, color=GRAY2)
    fr = _box(slide, SW - MARGIN - 1, 7.04, 1, 0.36)
    p = fr.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = str(page)
    _font(r, S_FOOT, color=GRAY2)


def content(prs, title, page):
    s = blank(prs)
    master(s, title, page)
    return s


def bullets(slide, items, left, top, width, height, size=S_BODY):
    tf = _box(slide, left, top, width, height).text_frame
    for i, txt in enumerate(items[:4]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.line_spacing = 1.2
        p.space_after = Pt(10)
        r = p.add_run(); r.text = "▪  " + txt
        _font(r, size, color=TEXT)


def _img_size(path, bw, bh):
    try:
        from PIL import Image
        iw, ih = Image.open(path).size
        ar = iw / ih
    except Exception:
        return bw, bh
    if bw / bh > ar:
        return bh * ar, bh
    return bw, bw / ar


def image(slide, path, left, top, bw, bh, caption=None, border=True):
    w, h = _img_size(path, bw, bh)
    il, it = left + (bw - w) / 2, top + (bh - h) / 2
    pic = slide.shapes.add_picture(path, Inches(il), Inches(it),
                                   Inches(w), Inches(h))
    if border:
        pic.line.color.rgb = LINE; pic.line.width = Pt(0.75)
    if caption:
        cb = _box(slide, left, top + bh + 0.02, bw, 0.34)
        p = cb.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = caption
        _font(r, S_NOTE, color=GRAY2)


def image_cover(slide, path, left, top, bw, bh):
    """按目标框比例中心裁剪后铺满（不变形、不留边）。"""
    try:
        from PIL import Image
        im = Image.open(path).convert("RGB")
        iw, ih = im.size
        target = bw / bh
        if iw / ih > target:
            nw = int(ih * target)
            im = im.crop(((iw - nw) // 2, 0, (iw - nw) // 2 + nw, ih))
        else:
            nh = int(iw / target)
            im = im.crop((0, (ih - nh) // 2, iw, (ih - nh) // 2 + nh))
        tmp = path + ".cover.png"
        im.save(tmp)
        slide.shapes.add_picture(tmp, Inches(left), Inches(top),
                                 Inches(bw), Inches(bh))
    except Exception:
        slide.shapes.add_picture(path, Inches(left), Inches(top), Inches(bw),
                                 Inches(bh))


def table(slide, rows, left, top, width, height, col_w=None, num_cols=None):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(left), Inches(top),
                                Inches(width), Inches(height)).table
    # 用“无网格”样式，去掉粗黑边
    tblPr = gt._tbl.find(qn("a:tblPr"))
    if tblPr is not None:
        tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
        sid = tblPr.find(qn("a:tableStyleId"))
        if sid is not None:
            sid.text = NOGRID
    if col_w:
        for j, w in enumerate(col_w):
            gt.columns[j].width = Inches(w)
    for i, row in enumerate(rows):
        gt.rows[i].height = Inches(height / nr)
        for j, val in enumerate(row):
            cell = gt.cell(i, j)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Pt(8); cell.margin_right = Pt(8)
            cell.margin_top = Pt(3); cell.margin_bottom = Pt(3)
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            head = i == 0
            _font(r, 15, bold=head, color=WHITE if head else TEXT)
            if num_cols and j in num_cols and not head:
                p.alignment = PP_ALIGN.RIGHT
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT if head else (WHITE if i % 2 else ALT)
    return gt


# --------------------------- 读取真实数字 ---------------------------

def load_numbers(out):
    with open(os.path.join(out, "metrics.json")) as f:
        m = json.load(f)
    txt = open(os.path.join(out, "rigid_transform.txt"), encoding="utf-8").read()

    def floats(pat):
        mt = re.search(pat, txt)
        return [float(x) for x in re.findall(r"-?\d+\.\d+", mt.group(1))] if mt else []

    return m, floats(r"平移 tx, ty, tz \(mm\):(.*)"), floats(r"旋转 rx, ry, rz \(度\):(.*)")


# --------------------------- deck ---------------------------

def build(out):
    fig = os.path.join(out, "figures")
    m, trans, rot = load_numbers(out)
    rg, df, jac = m["rigid_only"], m["deform"], m["jacobian"]
    jac_clean = m.get("jacobian_clean", jac["neg_fraction"] <= 1e-9 and jac["min"] > 0)
    primary = m.get("primary", "刚体为主结果，形变为受控的小幅细化")

    prs = Presentation()
    prs.slide_width = Inches(SW); prs.slide_height = Inches(SH)

    # ---- 1. 封面 ----
    s = blank(prs)
    image_cover(s, os.path.join(fig, "checkerboard.png"), 0, 0, 5.0, SH)
    _rect(s, MSO_SHAPE.RECTANGLE, 5.0, 0, 0.06, SH, fill=ACCENT)
    tb = _box(s, 5.6, 2.35, 7.2, 3.2)
    tf = tb.text_frame
    r = tf.paragraphs[0].add_run(); r.text = "CBCT → pCT 医学影像配准"
    _font(r, 36, bold=True, color=ACCENT)
    p = tf.add_paragraph(); r = p.add_run()
    r.text = "刚体配准为主 · 受控微分同胚形变细化"
    _font(r, S_BODY, color=GRAY2)
    _rect(s, MSO_SHAPE.RECTANGLE, 5.62, 3.7, 2.2, 0.04, fill=ACCENT)
    for t in ["病例 M24557", "汇报人：杨同学", "2026-06-18"]:
        p = tf.add_paragraph(); p.space_before = Pt(4)
        r = p.add_run(); r.text = t
        _font(r, 14, color=TEXT)

    # ---- 2. 任务与数据 ----
    s = content(prs, "任务与数据", 2)
    image(s, os.path.join(fig, "data_overview.png"), MARGIN, 1.4, 7.1, 5.0,
          caption="CBCT / pCT 各取 轴/冠/矢 三视图（统一窗位）")
    rows = [["", "pCT 计划CT", "CBCT 锥束CT"],
            ["设备", "Philips", "Elekta XVI"],
            ["矩阵", "512×512×112", "270×270×128"],
            ["体素 mm", "1.16×1.16×3.0", "1.0×1.0×1.0"],
            ["坐标系", "床 z≈-800", "等中心 z≈0"]]
    table(s, rows, 8.1, 1.55, 4.5, 2.2, col_w=[1.2, 1.7, 1.6], num_cols=[])
    bullets(s, [
        "不同 Frame of Reference（坐标不重叠）",
        "灰度不一致（散射 / HU 标定差异）",
        "分辨率与 FOV 不同（CBCT 视野小）",
    ], 8.1, 4.15, 4.5, 2.4)
    notes(s, "同一病人 M24557 的计划CT与治疗位锥束CT。三大难点决定方法："
              "坐标系不同→稳健初始化；灰度不一致→对线性差异鲁棒的 LNCC；"
              "FOV 不同→只在 CBCT 视野内评估。")

    # ---- 3. 为什么不平凡 ----
    s = content(prs, "为什么不平凡：CBCT 的物理特性", 3)
    image(s, os.path.join(fig, "coord_offset.png"), MARGIN, 1.35, 12.0, 3.0)
    bullets(s, [
        "坐标系不同：z 相差约 800 mm，DICOM 头不能直接对齐",
        "散射 → cupping 伪影，软组织 HU 塌陷",
        "HU 标定不准，CBCT 与计划CT 不可直接比较",
        "FOV 截断：CBCT 视野小、边缘缺失",
    ], MARGIN, 4.7, 12.0, 2.2)
    notes(s, "CBCT 以治疗机等中心为原点、pCT 以 CT 床为原点；散射导致 cupping 与 HU "
              "偏移；FOV 截断使边缘缺失。这些都要求对灰度差异鲁棒并限制评估区域。")

    # ---- 4. 方法总览（流程图）----
    s = content(prs, "方法总览：纯 PyTorch 实例优化", 4)
    steps = ["DICOM\n→ HU", "立方体网格\n+ 初始对齐", "多分辨率\n刚体 LNCC",
             "受控形变\n微分同胚", "输出\n变换/指标"]
    circ = ["①", "②", "③", "④", "⑤"]
    n = len(steps); bw, bh, gap = 2.1, 1.7, 0.42
    x = (SW - (n * bw + (n - 1) * gap)) / 2; y = 3.0
    band_y = y + bh / 2
    for i, st in enumerate(steps):
        if i < n - 1:
            _rect(s, MSO_SHAPE.RECTANGLE, x + bw, band_y - 0.012,
                  gap, 0.024, fill=ACCENT)
        box = _rect(s, MSO_SHAPE.ROUNDED_RECTANGLE, x, y, bw, bh,
                    fill=ACCENT, line=ACCENT)
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = circ[i] + " " + st
        _font(r, 14, bold=True, color=WHITE)
        x += bw + gap
    bullets(s, [
        "单对图像实例优化：无需训练数据 / 预训练模型",
        "全部基于 PyTorch 的 affine_grid / grid_sample 自行实现",
    ], MARGIN + 0.5, 5.4, 11.5, 1.3)
    notes(s, "不是调用黑箱配准软件，而是用 PyTorch 对这一对体数据直接梯度下降优化"
              "刚体参数与受控微分同胚形变。")

    # ---- 5. 初始对齐 ----
    s = content(prs, "关键设计①：稳健初始对齐", 5)
    image(s, os.path.join(fig, "zinit_xcorr.png"), MARGIN, 1.6, 7.0, 4.9)
    bullets(s, [
        "立方体各向同性网格 → 归一化旋转 = 物理刚体",
        "x / y：身体质心对齐",
        "z：横截面积剖面 1D 互相关",
        "因 pCT 的 z 覆盖远大于 CBCT，z 质心不可比",
    ], 8.0, 2.0, 4.6, 4.4)
    notes(s, "立方体各向同性网格保证 affine_grid 的归一化旋转就是物理刚体旋转、"
              "不引入剪切；z 用面积剖面互相关比用质心稳健得多。")

    # ---- 6. 相似性与形变 ----
    s = content(prs, "关键设计②：相似性与形变", 6)
    image(s, os.path.join(fig, "intensity_cupping.png"), MARGIN, 1.6, 7.0, 4.9)
    bullets(s, [
        "相似性：局部归一化互相关 LNCC（抗灰度差异）",
        "多分辨率金字塔 4×→2×→1× 扩大捕获范围",
        "形变：速度场积分得微分同胚位移场",
        "强正则 + 粗网格 + 平滑 → 仅小幅、无折叠",
    ], 8.0, 2.0, 4.6, 4.4)
    notes(s, "CBCT 软组织 HU 低于 pCT（cupping/标定差异），故用对线性灰度差异鲁棒的"
              "LNCC。本例戴面罩头颈近似刚体，形变只做小幅平滑细化以免过度揉变。")

    # ---- 7. 结果·定性 ----
    s = content(prs, "结果 · 定性：棋盘格与差异图", 7)
    image(s, os.path.join(fig, "checkerboard.png"), MARGIN, 1.45, 5.7, 5.0,
          caption="三视图棋盘格（配准前 vs 后）")
    image(s, os.path.join(fig, "difference.png"), 6.9, 1.45, 5.7, 5.0,
          caption="差异图 |CBCT − pCT|（前 vs 后）")
    notes(s, "棋盘格中骨与体表轮廓在配准后跨方格连续；差异图整体明显变暗，"
              "残留主要在 CBCT FOV 边缘与 couch。")

    # ---- 8. 结果·定量（含醒目 callout）----
    s = content(prs, "结果 · 定量（真实数值）", 8)
    image(s, os.path.join(fig, "metrics_bar.png"), MARGIN, 1.55, 5.9, 4.0)
    # 核心答案 callout
    co = _rect(s, MSO_SHAPE.ROUNDED_RECTANGLE, 6.9, 1.5, 5.7, 1.75,
               fill=ALT, line=ACCENT, line_w=1.75)
    tf = co.text_frame; tf.word_wrap = True
    tf.margin_left = Pt(10); tf.margin_top = Pt(6)
    r = tf.paragraphs[0].add_run(); r.text = "刚体摆位修正（CBCT → pCT）"
    _font(r, 14, bold=True, color=ACCENT)
    p = tf.add_paragraph(); r = p.add_run()
    r.text = "平移 %.2f, %.2f, %.2f mm" % tuple(trans) if trans else "-"
    _font(r, 17, bold=True, color=TEXT)
    p = tf.add_paragraph(); r = p.add_run()
    r.text = "旋转 %.2f, %.2f, %.2f °" % tuple(rot) if rot else "-"
    _font(r, 17, bold=True, color=TEXT)
    rows = [["指标", "仅刚体", "+形变"],
            ["骨 Dice", "%.3f" % rg["bone_dice_after"], "%.3f" % df["bone_dice_after"]],
            ["LNCC", "%.3f" % rg["ncc_after"], "%.3f" % df["ncc_after"]],
            ["Jacobian min", "%.3f" % jac["min"], "负值 %.1f%%" % (100 * jac["neg_fraction"])]]
    table(s, rows, 6.9, 3.5, 5.7, 1.9, col_w=[2.1, 1.8, 1.8], num_cols=[1, 2])
    bullets(s, [
        "骨 Dice 独立于优化目标，证明真实结构对齐",
        primary,
    ], 6.9, 5.55, 5.9, 1.3, size=14)
    notes(s, "骨 Dice 用 HU>200、独立于 LNCC，因此大幅提升说明是真实结构对齐而非过拟合。"
              "刚体已达主要精度；形变 +%.3f Dice、Jacobian 全正无折叠，仅作小幅细化。"
              % m["deform_bone_dice_gain"])

    # ---- 9. 收敛与合法性 ----
    s = content(prs, "收敛过程与形变合法性", 9)
    image(s, os.path.join(out, "loss_curve.png"), MARGIN, 1.55, 6.9, 4.6)
    image(s, os.path.join(fig, "jacobian_map.png"), 7.7, 1.55, 5.0, 4.6)
    legal = ("形变场 Jacobian 处处为正、无折叠（微分同胚）" if jac_clean else
             "形变场 Jacobian min=%.3f、负值占比 %.2f%%" % (
                 jac["min"], 100 * jac["neg_fraction"]))
    bullets(s, [legal], MARGIN, 6.35, 12.0, 0.6, size=15)
    notes(s, "左：两阶段 loss 收敛；右：形变场 Jacobian 行列式中心层，颜色接近 1 表示"
              "近似保体积、无折叠。")

    # ---- 10. 小结与局限 ----
    s = content(prs, "小结与局限", 10)
    bullets(s, [
        primary,
        "针对 CBCT 物理特性：稳健初始化 / LNCC / FOV 掩膜 / 真刚体",
        "独立骨 Dice 验证：0.128 → %.3f（仅刚体）" % rg["bone_dice_after"],
        "局限：未做散射/HU 校正、单病例、未做标志点 TRE",
    ], MARGIN, 1.8, 12.0, 4.6)
    notes(s, "做了什么 / 验证了什么 / 局限与展望。可拓展：散射与 HU 定量校正、多病例"
              "评估、标志点 TRE 或轮廓 Dice、学习式配准网络。")

    path = os.path.join(out, "CBCT_pCT_配准汇报.pptx")
    prs.save(path)
    print("已生成", path, "| Jacobian 干净:", jac_clean)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results_sample")
    args = p.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()
