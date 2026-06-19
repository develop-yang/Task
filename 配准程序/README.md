# CBCT 到 pCT 配准

把同一病人(M24557)的锥束CT(CBCT, Elekta)配准到计划CT(pCT, Philips)，
用 PyTorch 对这一对图像直接做优化，不需要训练数据。主要给出刚体配准
(摆位修正的平移 mm / 旋转 度)，形变作为小幅细化。

## 环境
PyTorch 1.10 / Python 3.8 测试通过。

```
pip install -r requirements.txt
```

## 运行
把两个 zip 解压到 data/ 下，得到 data/M24557-CBCT 和 data/M24557-pCT，然后：

```
python main.py --cbct data/M24557-CBCT --pct data/M24557-pCT --out results
```

有 GPU 会自动用 GPU。常用参数：`--spacing` 公共网格间距(mm，越小越精细越慢)、
`--rigid-iters`、`--deform-iters`、`--scale`(开缩放，默认纯刚体)。

## 输出 (results/)
- `rigid_transform.txt` —— 刚体配准结果：平移(mm)、旋转(度)、4x4 矩阵、验证指标
- `metrics.json` —— 仅刚体 / +形变 的 LNCC、骨 Dice，以及形变场 Jacobian 统计
- `warped_cbct.nii.gz` —— 配准后的 CBCT；`fixed_pct.nii.gz`、`moving_cbct.nii.gz` 为参考
- `transform.pt` —— 变换参数

NIfTI 可用 ITK-SNAP / 3D Slicer 打开叠加查看。

## 文件
- `dicom_io.py` 读 DICOM 序列转 HU
- `preprocess.py` 初始对齐 + 重采样到公共网格
- `model.py` 刚体 + 微分同胚形变
- `losses.py` LNCC、正则
- `register.py` 优化流程、指标、刚体参数换算
- `main.py` 入口
