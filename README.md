# DnCNN / U-Net Image Denoising with Jittor

> 课程作业项目：将经典 DnCNN 图像去噪实验迁移到 Jittor，并在盲去噪设置下引入噪声水平图与轻量 U-Net 结构进行实验。

## Highlights

- **任务**：灰度图像加性高斯噪声去除，覆盖固定噪声与盲去噪两种模式。
- **框架**：Jittor，保留 PyTorch 版 DnCNN 的残差学习思想。
- **模型**：输入为 `noisy image + noise level map`，网络预测噪声残差，输出通过 `clean = noisy - predicted_noise` 得到。
- **评估**：在 Set12、Set68 上统计 PSNR / SSIM。

## Repository Layout

```text
.
├── data/              # 训练图像与 Set12 / Set68 测试集
├── logs/DnCNN-B/      # 训练输出目录，默认保存 net.pkl
├── dataset.py         # HDF5 数据预处理与 Jittor Dataset
├── models.py          # U-Net 去噪网络
├── train.py           # 训练入口
├── test.py            # 测试入口
├── utils.py           # 初始化、指标、数据增强工具
└── README.md
```

## Environment

建议使用 Python 3.9+。如果本机有 CUDA，Jittor 会自动编译并调用 GPU 后端。

```bash
python -m pip install -r requirements.txt
```

如果需要手动安装 Jittor：

```bash
python -m pip install jittor
```

## Quick Start

首次训练前需要把 `data/train` 和 `data/Set12` 预处理成 `train.h5`、`val.h5`：

```bash
python train.py --preprocess True --mode B --epochs 10 --outf logs/DnCNN-B
```

已经生成 HDF5 后，可以跳过预处理直接训练：

```bash
python train.py --mode B --batch-size 128 --epochs 10 --val_noiseL 25 --outf logs/DnCNN-B
```

后台训练示例：

```bash
nohup python train.py --preprocess True --mode B --val_noiseL 25 --outf logs/DnCNN-B > logs/DnCNN-B.log 2>&1 &
```

## 评估

### 评估设置

我们在两个标准基准数据集上进行评估，跨越多个噪声水平：

- **Set68 (BSD68)**：68张标准灰度图像，广泛应用于图像去噪算法评估
- **Set12**：12张经典图像复原基准测试集

### 评估指标

采用以下两个量化指标：

- **PSNR (Peak Signal-to-Noise Ratio)**：单位dB，数值越大越好
- **SSIM (Structural Similarity Index)**：范围[0,1]，数值越大表示感知质量越好

### 噪声水平

评估覆盖5个噪声水平：σ ∈ {15, 25, 50, 75, 100}，包括中等噪声和高噪声场景。其中σ ≥ 50时的评估重点为盲去噪性能，即网络需在未知噪声水平下进行去噪。

### 运行评估

使用以下命令格式进行评估：

```bash
python test.py --logdir logs/DnCNN-B --test_data {数据集} --test_noiseL {噪声水平}
```

其中：
- `{数据集}` ∈ {`Set68`, `Set12`}
- `{噪声水平}` ∈ {15, 25, 50, 75, 100}

例如，在Set68上评估σ=25的结果：

```bash
python test.py --logdir logs/DnCNN-B --test_data Set68 --test_noiseL 25
```

对所有数据集和噪声水平重复上述命令可完整复现全部结果。

## 实验结果

### 标准基准上的定量结果

#### Set68 / BSD68 (中等噪声)

| 噪声水平 σ | 基线DnCNN-B PSNR | 本方法 PSNR | 基线DnCNN-B SSIM | 本方法 SSIM | 改进 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 15 | 31.60 | 31.62 | 0.891 | 0.891 | +0.02 dB |
| 25 | 29.15 | 29.16 | 0.827 | 0.829 | +0.01 dB |
| 50 | 26.20 | 26.20 | 0.714 | 0.715 | ≈ |

#### Set68 / BSD68 (高噪声盲去噪)

*在盲去噪设置下（噪声水平未知），本方法的噪声水平图+U-Net架构在高噪声场景表现出显著优势：*

| 噪声水平 σ | DnCNN-B PSNR | 本方法 PSNR | DnCNN-B SSIM | 本方法 SSIM | 改进 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 26.20 | 26.11 | 0.715 | 0.708 | -0.09 dB |
| 75 | 17.89 | **24.47** | 0.294 | **0.621** | **+6.58 dB** ↑ |
| 100 | 13.65 | **22.95** | 0.160 | **0.508** | **+9.30 dB** ↑ |

#### Set12 (中等噪声)

| 噪声水平 σ | 基线DnCNN-B PSNR | 本方法 PSNR | 基线DnCNN-B SSIM | 本方法 SSIM | 改进 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 15 | 32.725 | 32.731 | 0.902 | 0.903 | +0.006 dB |
| 25 | 30.344 | 30.376 | 0.859 | 0.862 | +0.032 dB |
| 50 | 27.138 | 27.132 | 0.773 | 0.777 | -0.006 dB |

#### Set12 (高噪声盲去噪)

| 噪声水平 σ | DnCNN-B PSNR | 本方法 PSNR | DnCNN-B SSIM | 本方法 SSIM | 改进 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 50 | 27.132 | 26.846 | 0.777 | 0.765 | -0.286 dB |
| 75 | 18.113 | **24.749** | 0.290 | **0.668** | **+6.636 dB** ↑ |
| 100 | 13.895 | **22.996** | 0.163 | **0.549** | **+9.101 dB** ↑ |

### 主要观察

1. **中等噪声 (σ ≤ 50)**：在Set68和Set12上的性能与基线DnCNN-B相当，证实了所提架构改进在标准设置下保持了有效性。

2. **高噪声盲去噪 (σ ≥ 75)**：所提的噪声水平图+U-Net架构在高噪声场景取得显著提升，PSNR提升**6-9 dB**，SSIM大幅改进（0.6+相对于0.2-0.3），充分展现了该方法在处理未知噪声水平时的优势。

3. **架构优势**：相比原始残差网络设计，带有跳接的轻量级U-Net能更好地捕捉多尺度噪声特征，特别是在高噪声盲去噪场景中表现突出。

## Notes

- `--mode S` 表示固定噪声水平训练，`--mode B` 表示盲去噪训练。
- `--num_of_layers` 仅为兼容早期 DnCNN 命令保留；当前模型结构定义在 `models.py` 的 `UNet` 中。
- 测试脚本会自动把输入图像 padding 到 8 的倍数，以匹配 3 层下采样结构，再裁剪回原尺寸计算指标。
- 曾尝试加入注意力模块与对抗训练。注意力模块在当前 AWGN 设置下收益有限；对抗训练增加了训练成本且未带来稳定提升，因此最终版本保留更稳定的噪声水平图 + U-Net 方案。

## Reference

- Zhang et al., [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://ieeexplore.ieee.org/document/7839189/)
- Original MATLAB implementation: [cszn/DnCNN](https://github.com/cszn/DnCNN)
