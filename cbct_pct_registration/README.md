# CBCT → pCT 配准（PyTorch 刚体 + 形变）

用 PyTorch 对一对医学影像做配准：把 **CBCT**（Elekta 锥形束 CT）配准到
**pCT**（Philips 计划 CT）。整套流程是“单对图像的实例优化”——不需要训练数据、
不需要预训练模型，直接对这一对体数据用梯度下降优化空间变换。

## 数据

| | pCT（计划 CT） | CBCT（锥束 CT） |
|---|---|---|
| 设备 | Philips | Elekta |
| 矩阵 | 512×512×112 | 270×270×128 |
| 体素间距 (mm) | 1.16×1.16×3.0 | 1.0×1.0×1.0 |
| 坐标系 | CT 床坐标系 | 治疗机等中心坐标系 |

> 关键点：两套数据**不在同一世界坐标系**（CBCT 以等中心为原点，z≈0；
> pCT 以床坐标为原点，z≈−800）。因此程序先用“身体质心”做初始平移，
> 再用多分辨率刚体把两者拉到重叠，最后做形变细化。

## 环境

与你的环境（PyTorch 1.10 / Python 3.8 / CUDA 11.3）兼容。安装依赖：

```bash
pip install -r requirements.txt
```

## 用法

把两个 zip 解压到 `data/` 下（得到 `data/M24557-CBCT/` 与 `data/M24557-pCT/`），
然后在仓库根目录运行：

```bash
python -m cbct_pct_registration.main \
    --cbct data/M24557-CBCT \
    --pct  data/M24557-pCT \
    --out  results
```

有 GPU 时会自动用 CUDA。默认参数即可，常用可调项：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--spacing` | 1.5 | 公共网格各向同性间距 (mm)，越小越精细越慢 |
| `--rigid-iters` | 300 | 刚体阶段每个分辨率层的迭代数 |
| `--deform-iters` | 300 | 形变阶段迭代数 |
| `--ncc-win` | 9 | 局部归一化互相关 (LNCC) 窗口边长 |
| `--reg-weight` | 1.0 | 形变场平滑正则权重，越大形变越平滑 |
| `--no-scale` | 关 | 刚体阶段禁用各向同性缩放（纯 6 自由度刚体） |
| `--device` | 自动 | `cpu` 或 `cuda` |

## 方法流程

1. **读取**（`dicom_io.py`）：DICOM 缺标准文件头，用 `force=True` + 手动补
   Implicit-VR 传输语法读取，按层位置排序、按 slope/intercept 转 HU。
2. **预处理**（`preprocess.py`）：按身体质心做初始平移，把两套数据重采样到
   以 CBCT 视野为基准的各向同性公共网格，HU 裁剪到 [−1000,1000] 后归一化。
3. **刚体配准**（`model.py` / `register.py`）：用 `F.affine_grid` +
   `F.grid_sample` 优化旋转/平移（可选缩放），采用 **多分辨率金字塔**
   (4×→2×→1×) 扩大捕获范围，相似性度量为 LNCC（对 CBCT/pCT 灰度差异鲁棒）。
4. **形变配准**：冻结刚体参数，优化一个低分辨率稠密位移场（三线性上采样到
   全分辨率），相似性用 LNCC，叠加扩散正则保证位移场平滑。
5. **输出**（`results/`）：
   - `warped_cbct.nii.gz` —— 配准后的 CBCT（已对齐到 pCT 网格）
   - `fixed_pct.nii.gz` / `moving_cbct_init.nii.gz` —— 公共网格上的参考图
   - `transform.pt` —— 刚体参数 + 位移场 + 网格几何
   - `overlay.png` —— 三视图棋盘格对比（配准前/后）+ 差异图
   - `loss_curve.png` —— 两阶段 loss 曲线
   - `metrics.json` —— 配准前后的 LNCC / MSE 指标

用 ITK-SNAP 或 3D Slicer 打开 `results/*.nii.gz` 可叠加查看对齐效果。

## 文件结构

```
cbct_pct_registration/
├── dicom_io.py     # DICOM 序列读取 → HU 体数据 + 几何
├── preprocess.py   # 质心初始对齐、公共网格重采样、归一化
├── losses.py       # LNCC 相似性、形变场正则
├── model.py        # 刚体仿射 + 稠密位移场（grid_sample）
├── register.py     # 多分辨率优化主流程、指标评估
├── visualize.py    # 棋盘格/差异图/loss 曲线
├── main.py         # 端到端命令行入口
└── requirements.txt
```
