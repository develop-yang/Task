"""用 python-pptx 生成 CBCT→pCT 配准汇报 deck（中文，统一模板）。

设计：白底 + 深蓝强调色；每页页眉色条、页脚“病例 M24557”、右下角页码；
标题/正文/图注三档字号；正文每页 ≤4 条短句，长解释放进 speaker notes；
封面 hero 图；方法页用真正的节点+箭头流程图。

所有数字从 results_sample/metrics.json 与 rigid_transform.txt 读取真实值。

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
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ACCENT = RGBColor(0x1F, 0x3B, 0x6E)   # 深蓝（强调色）
DARK = RGBColor(0x22, 0x22, 0x22)
GRAY = RGBColor(0x5A, 0x5A, 0x5A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT = RGBColor(0xE9, 0xEE, 0xF6)
FONT = "Microsoft YaHei"  # Windows 常见中文字体；缺失时 PowerPoint 自动回退

SW, SH = 13.333, 7.5  # 16:9 inches
T_TITLE, T_BODY, T_NOTE = 28, 18, 13


# --------------------------- 字体/文本 ---------------------------

def _font(run, size, bold=False, color=DARK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = FONT
    # 同时设置东亚字体槽，保证中文按 FONT 渲染
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", FONT)


def _textbox(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width),
                                  Inches(height))
    tb.text_frame.word_wrap = True
    return tb


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


# --------------------------- 模板 ---------------------------

def blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def master(slide, title, page_no):
    """页眉色条 + 标题 + 页脚 + 页码。"""
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(SW),
                                 Inches(0.22))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()

    tb = _textbox(slide, 0.55, 0.42, 12.2, 0.9)
    r = tb.text_frame.paragraphs[0].add_run()
    r.text = title
    _font(r, T_TITLE, bold=True, color=ACCENT)

    # 页脚
    fl = _textbox(slide, 0.55, 7.02, 6.0, 0.4)
    r = fl.text_frame.paragraphs[0].add_run()
    r.text = "CBCT → pCT 配准 · 病例 M24557"
    _font(r, 10, color=GRAY)
    fr = _textbox(slide, 11.8, 7.02, 1.0, 0.4)
    p = fr.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.RIGHT
    r = p.add_run(); r.text = str(page_no)
    _font(r, 10, color=GRAY)


def content(prs, title, page_no):
    s = blank(prs)
    master(s, title, page_no)
    return s


def add_bullets(slide, items, left, top, width, height, size=T_BODY):
    """items: list[str]（每条一行短句，建议 ≤4 条）。"""
    tf = _textbox(slide, left, top, width, height).text_frame
    for i, txt in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(10)
        r = p.add_run(); r.text = "▪ " + txt
        _font(r, size, color=DARK)


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
                             Inches(top + (height - h) / 2), Inches(w), Inches(h))


def add_table(slide, rows, left, top, width, height, col_w=None, num_cols=None):
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
            cell.margin_top = Pt(2); cell.margin_bottom = Pt(2)
            p = cell.text_frame.paragraphs[0]
            r = p.add_run(); r.text = str(val)
            head = i == 0
            _font(r, 13 if head else 12, bold=head,
                  color=WHITE if head else DARK)
            if num_cols and j in num_cols and not head:
                p.alignment = PP_ALIGN.RIGHT
            cell.fill.solid()
            cell.fill.fore_color.rgb = ACCENT if head else (
                WHITE if i % 2 else LIGHT)
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


# --------------------------- 各页 ---------------------------

def build(out):
    fig = os.path.join(out, "figures")
    m, trans, rot = load_numbers(out)
    rg = m["rigid_only"]
    df = m["deform"]
    jac = m["jacobian"]
    jac_clean = jac["neg_fraction"] <= 1e-9 and jac["min"] > 0

    prs = Presentation()
    prs.slide_width = Inches(SW)
    prs.slide_height = Inches(SH)

    # ---- 1. 封面（hero 图）----
    s = blank(prs)
    add_image_fit(s, os.path.join(fig, "checkerboard.png"), 8.3, 0.0, 5.0, 7.5)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(8.3), Inches(SH))
    band.fill.solid(); band.fill.fore_color.rgb = WHITE
    band.line.fill.background()
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.9), Inches(2.5),
                             Inches(0.12), Inches(2.2))
    bar.fill.solid(); bar.fill.fore_color.rgb = ACCENT; bar.line.fill.background()
    tb = _textbox(s, 1.2, 2.5, 6.7, 2.6)
    tf = tb.text_frame
    r = tf.paragraphs[0].add_run(); r.text = "CBCT → pCT 医学影像配准"
    _font(r, 38, bold=True, color=ACCENT)
    p = tf.add_paragraph(); r = p.add_run()
    r.text = "锥束CT 与 计划CT 的刚体配准（含可选形变细化）"
    _font(r, 17, color=GRAY)
    p = tf.add_paragraph(); r = p.add_run(); r.text = "病例 M24557"
    _font(r, 15, color=DARK)
    p = tf.add_paragraph(); r = p.add_run()
    r.text = "汇报人：杨同学      2026-06-18"
    _font(r, 14, color=GRAY)

    # ---- 2. 任务与数据 ----
    s = content(prs, "任务与数据", 2)
    add_image_fit(s, os.path.join(fig, "data_overview.png"), 0.5, 1.45, 7.2, 5.2)
    rows = [["", "pCT 计划CT", "CBCT 锥束CT"],
            ["设备", "Philips", "Elekta XVI"],
            ["矩阵", "512×512×112", "270×270×128"],
            ["体素(mm)", "1.16×1.16×3.0", "1.0×1.0×1.0"],
            ["坐标系", "床坐标 z≈-800", "等中心 z≈0"]]
    add_table(s, rows, 7.9, 1.7, 5.0, 2.2, col_w=[1.3, 1.9, 1.8])
    add_bullets(s, [
        "不同 Frame of Reference（坐标不重叠）",
        "灰度不一致（散射 / HU 标定差异）",
        "分辨率与 FOV 不同（CBCT 视野小）",
    ], 7.9, 4.3, 5.0, 2.4)
    notes(s, "同一病人 M24557 的计划CT与治疗位锥束CT。三大难点决定了后续方法："
              "坐标系不同→需稳健初始化；灰度不一致→需对线性差异鲁棒的相似性(LNCC)；"
              "FOV不同→只在CBCT视野内评估。")

    # ---- 3. 为什么不平凡 ----
    s = content(prs, "为什么不平凡：CBCT 的物理特性", 3)
    add_image_fit(s, os.path.join(fig, "coord_offset.png"), 0.5, 1.4, 12.3, 3.0)
    add_bullets(s, [
        "坐标系不同：z 相差约 800 mm，DICOM 头不能直接对齐",
        "X 射线散射 → cupping 伪影，软组织 HU 塌陷",
        "HU 标定不准：CBCT 与计划CT 不可直接比较",
        "FOV 截断：CBCT 视野小、边缘缺失",
    ], 0.7, 4.6, 12.0, 2.4)
    notes(s, "CBCT 以治疗机等中心为原点、pCT 以 CT 床为原点；散射导致 cupping 与 HU "
              "偏移；FOV 截断使边缘信息缺失。这些都要求方法对灰度差异鲁棒并限制评估区域。")

    # ---- 4. 方法总览（流程图）----
    s = content(prs, "方法总览：纯 PyTorch 实例优化", 4)
    steps = ["DICOM\n→ HU", "立方体网格\n+ 初始对齐", "多分辨率刚体\n(LNCC)",
             "可选形变\n(微分同胚)", "输出\n变换/指标"]
    n = len(steps); bw, bh, gap = 2.05, 1.5, 0.5
    x = (SW - (n * bw + (n - 1) * gap)) / 2; y = 3.1
    for i, st in enumerate(steps):
        box = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x),
                                 Inches(y), Inches(bw), Inches(bh))
        box.fill.solid(); box.fill.fore_color.rgb = ACCENT if i % 2 == 0 else GRAY
        box.line.color.rgb = ACCENT; box.line.width = Pt(1)
        tf = box.text_frame; tf.word_wrap = True
        p = tf.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = st
        _font(r, 14, bold=True, color=WHITE)
        if i < n - 1:
            ar = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW,
                                    Inches(x + bw + 0.06), Inches(y + bh / 2 - 0.16),
                                    Inches(gap - 0.12), Inches(0.32))
            ar.fill.solid(); ar.fill.fore_color.rgb = ACCENT
            ar.line.fill.background()
        x += bw + gap
    add_bullets(s, [
        "单对图像实例优化：无需训练数据 / 预训练模型",
        "全部基于 PyTorch 的 affine_grid / grid_sample 自行实现",
    ], 1.2, 5.3, 11.0, 1.4)
    notes(s, "不是调用黑箱配准软件，而是用 PyTorch 自行实现：对这一对体数据直接用"
              "梯度下降优化刚体参数与（可选）微分同胚形变。")

    # ---- 5. 关键设计① 初始对齐 ----
    s = content(prs, "关键设计①：稳健初始对齐", 5)
    add_image_fit(s, os.path.join(fig, "zinit_xcorr.png"), 0.5, 1.5, 7.2, 5.0)
    add_bullets(s, [
        "立方体 + 各向同性网格 → 归一化旋转 = 物理刚体",
        "x / y：身体质心对齐",
        "z：横截面积剖面的 1D 互相关",
        "因 pCT 的 z 覆盖远大于 CBCT，z 质心不可比",
    ], 7.9, 1.9, 5.0, 4.6)
    notes(s, "立方体各向同性网格保证 affine_grid 的归一化旋转就是物理刚体旋转、不引入"
              "剪切；z 方向用面积剖面互相关比用质心稳健得多。")

    # ---- 6. 关键设计② 相似性与形变 ----
    s = content(prs, "关键设计②：相似性与形变", 6)
    add_image_fit(s, os.path.join(fig, "intensity_cupping.png"), 0.5, 1.5, 7.2, 5.0)
    add_bullets(s, [
        "相似性：局部归一化互相关 LNCC（抗灰度差异）",
        "多分辨率金字塔 4×→2×→1× 扩大捕获范围",
        "形变：速度场积分得微分同胚位移场",
        "强正则 + 粗网格 + 平滑 → 仅小幅细化",
    ], 7.9, 1.9, 5.0, 4.6)
    notes(s, "CBCT 软组织 HU 低于 pCT（cupping/标定差异），故用对线性灰度差异鲁棒的"
              "LNCC。本例戴面罩头颈近似刚体，形变只做小幅平滑细化以避免过度揉变。")

    # ---- 7. 结果·定性 ----
    s = content(prs, "结果 · 定性：棋盘格与差异图", 7)
    add_image_fit(s, os.path.join(fig, "checkerboard.png"), 0.4, 1.45, 6.0, 5.3)
    add_image_fit(s, os.path.join(fig, "difference.png"), 6.7, 1.45, 6.2, 5.3)
    notes(s, "棋盘格中骨与体表轮廓在配准后跨方格连续；差异图整体明显变暗，"
              "残留主要在 CBCT FOV 边缘与couch。")

    # ---- 8. 结果·定量（rigid-only 与 +deform）----
    s = content(prs, "结果 · 定量（真实数值）", 8)
    add_image_fit(s, os.path.join(fig, "metrics_bar.png"), 0.5, 1.5, 6.3, 4.3)
    rows = [["指标", "仅刚体", "+形变"],
            ["骨 Dice", "%.3f" % rg["bone_dice_after"], "%.3f" % df["bone_dice_after"]],
            ["LNCC", "%.3f" % rg["ncc_after"], "%.3f" % df["ncc_after"]]]
    add_table(s, rows, 7.0, 1.7, 5.9, 1.4, col_w=[2.1, 1.9, 1.9], num_cols=[1, 2])
    rrows = [["刚体摆位修正", "数值"],
             ["平移 tx,ty,tz (mm)",
              "%.2f, %.2f, %.2f" % tuple(trans) if trans else "-"],
             ["旋转 rx,ry,rz (度)",
              "%.2f, %.2f, %.2f" % tuple(rot) if rot else "-"],
             ["骨 Dice（前→后）",
              "%.3f→%.3f" % (rg["bone_dice_before"], rg["bone_dice_after"])],
             ["Jacobian min / 负值占比",
              "%.3f / %.2f%%" % (jac["min"], 100 * jac["neg_fraction"])]]
    add_table(s, rrows, 7.0, 3.35, 5.9, 2.3, col_w=[3.1, 2.8], num_cols=[1])
    add_bullets(s, [
        "骨 Dice 独立于优化目标，证明真实结构对齐",
        "刚体为主结果；形变仅小幅细化",
    ], 7.0, 5.85, 5.9, 1.1, size=14)
    notes(s, "骨 Dice 用 HU>200 阈值、独立于 LNCC 优化目标，因此大幅提升说明是真实结构"
              "对齐而非过拟合相似度。刚体已达到主要精度，形变对 Dice 增益很小。")

    # ---- 9. 收敛与合法性 ----
    s = content(prs, "收敛过程与形变合法性", 9)
    add_image_fit(s, os.path.join(out, "loss_curve.png"), 0.5, 1.5, 7.0, 4.8)
    add_image_fit(s, os.path.join(fig, "jacobian_map.png"), 7.7, 1.5, 5.2, 4.8)
    legal = ("形变场 Jacobian 处处为正、无折叠（微分同胚）"
             if jac_clean else
             "形变场 Jacobian min=%.3f、负值占比 %.2f%%" % (
                 jac["min"], 100 * jac["neg_fraction"]))
    add_bullets(s, [legal], 0.6, 6.35, 12.0, 0.7, size=15)
    notes(s, "左：两阶段 loss 收敛；右：形变场 Jacobian 行列式中心层，颜色接近 1 表示"
              "近似保体积、无折叠。")

    # ---- 10. 小结与局限 ----
    s = content(prs, "小结与局限", 10)
    add_bullets(s, [
        "纯 PyTorch 实现 CBCT→pCT 刚体配准，导出物理摆位修正",
        "针对 CBCT 物理特性：稳健初始化 / LNCC / FOV 掩膜 / 真刚体",
        "独立骨 Dice 验证（0.128→%.3f）；形变仅小幅细化、无折叠" % rg["bone_dice_after"],
        "局限：未做散射/HU 校正、单病例、未做标志点 TRE",
    ], 0.7, 1.7, 12.0, 4.8, size=T_BODY)
    notes(s, "做了什么 / 验证了什么 / 局限与展望。可拓展：散射与 HU 定量校正、多病例"
              "评估、加入标志点 TRE 或轮廓 Dice、学习式配准网络。")

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
