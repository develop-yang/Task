# CBCT → pCT 配准（PyTorch 刚体 + 形变）

用 PyTorch 对一对医学影像做配准：把 **CBCT**（Elekta 锥形束 CT）配准到
**pCT**（Philips 计划 CT）。整套流程是“单对图像的实例优化”——不需要训练数据、
不需要预训练模型，直接对这一对体数据用梯度下降优化空间变换。

应用背景是放疗 IGRT，所以**主交付物是刚体配准**（可解读为摆位校正：平移 mm + 旋转
度，见 `results/rigid_transform.txt`），形变配准作为可选的二级细化。

## 正确性要点（这版做了哪些保证）

- **真刚体**：公共网格构造成**立方体 + 各向同性**，于是 `F.affine_grid` 的归一化
  旋转严格等于物理旋转，不会沿 z 引入剪切。
- **默认纯 6-DOF 刚体**：同一病人 + 已标定扫描仪，scale 应≈1，默认关闭缩放
  （需要时 `--scale` 作为对照）。
- **稳健 z 初始化**：pCT 的 z 覆盖远大于 CBCT，z 质心不可比；改用两者“横截面积
  随 z 的剖面”做 1D 归一化互相关定 z 偏移。
- **无折叠形变**：形变用**微分同胚**（速度场 scaling-and-squaring 积分）+ 扩散正则，
  保证位移场 Jacobian 几乎处处为正（负值占比≈0）。
- **独立验证指标**（不拿优化目标 LNCC 当评价，避免循环论证）：
  - **骨结构 Dice**（HU>200 取骨掩膜，CBCT 视野内，配准前/后）；
  - **形变场 Jacobian** 行列式的 min / 均值 / 负值占比。

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
| `--reg-weight` | 3.0 | 形变场平滑正则权重，越大形变越平滑、折叠越少 |
| `--deform-lr` | 0.02 | 形变阶段学习率 |
| `--scale` | 关 | 额外开启各向同性缩放（默认纯 6 自由度刚体） |
| `--bone-thresh` | 200 | 骨 Dice 的 HU 阈值 |
| `--device` | 自动 | `cpu` 或 `cuda` |

> 已提交一份样例结果在 `results_sample/`（overlay.png + metrics.json +
> rigid_transform.txt），clone 下来不用跑就能看到配准效果。

## 方法流程

1. **读取**（`dicom_io.py`）：DICOM 缺标准文件头，用 `force=True` + 手动补
   Implicit-VR 传输语法读取，按层位置排序、按 slope/intercept 转 HU。
2. **预处理**（`preprocess.py`）：x/y 用身体质心、z 用面积剖面 1D 互相关做初始
   平移，把两套数据重采样到以 CBCT 视野为基准的**立方体各向同性**公共网格，
   HU 裁剪到 [−1000,1000] 后归一化。
3. **刚体配准**（`model.py` / `register.py`）：用 `F.affine_grid` +
   `F.grid_sample` 优化旋转/平移（默认 6-DOF，不含缩放），采用 **多分辨率金字塔**
   (4×→2×→1×) 扩大捕获范围，相似性度量为 LNCC（对 CBCT/pCT 灰度差异鲁棒）。
4. **形变配准**：冻结刚体，优化低分辨率**速度场**，经 scaling-and-squaring 积分得
   微分同胚位移场（三线性上采样到全分辨率），LNCC + 扩散正则。
5. **输出**（`results/`）：
   - `rigid_transform.txt` —— **刚体配准的核心答案**：平移(mm)+旋转(度)+4×4 矩阵
   - `metrics.json` —— LNCC / MSE / 骨 Dice / 形变 Jacobian 统计
   - `overlay.png` —— 三视图棋盘格对比（配准前/后）+ 差异图
   - `loss_curve.png` —— 两阶段 loss 曲线
   - `warped_cbct.nii.gz` —— 配准后的 CBCT（已对齐到 pCT 网格）
   - `fixed_pct.nii.gz` / `moving_cbct_init.nii.gz` —— 公共网格上的参考图
   - `transform.pt` —— 刚体参数 + 速度场 + 网格几何

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
