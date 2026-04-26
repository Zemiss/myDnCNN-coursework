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

## Evaluation

Set68:

```bash
python test.py --logdir logs/DnCNN-B --test_data Set68 --test_noiseL 15
python test.py --logdir logs/DnCNN-B --test_data Set68 --test_noiseL 25
python test.py --logdir logs/DnCNN-B --test_data Set68 --test_noiseL 50
python test.py --logdir logs/DnCNN-B --test_data Set68 --test_noiseL 75
python test.py --logdir logs/DnCNN-B --test_data Set68 --test_noiseL 100
```

Set12:

```bash
python test.py --logdir logs/DnCNN-B --test_data Set12 --test_noiseL 15
python test.py --logdir logs/DnCNN-B --test_data Set12 --test_noiseL 25
python test.py --logdir logs/DnCNN-B --test_data Set12 --test_noiseL 50
python test.py --logdir logs/DnCNN-B --test_data Set12 --test_noiseL 75
python test.py --logdir logs/DnCNN-B --test_data Set12 --test_noiseL 100
```

## Results

### Set68 / BSD68

| Noise Level | Baseline DnCNN-B PSNR | Ours PSNR | Baseline DnCNN-B SSIM | Ours SSIM |
|:-----------:|:---------------------:|:---------:|:---------------------:|:---------:|
| 15 | 31.60 | 31.62 | 0.891 | 0.891 |
| 25 | 29.15 | 29.16 | 0.827 | 0.829 |
| 50 | 26.20 | 26.20 | 0.714 | 0.715 |

高噪声盲去噪补充实验：

| Noise Level | DnCNN-B PSNR | Ours PSNR | DnCNN-B SSIM | Ours SSIM |
|:-----------:|:------------:|:---------:|:------------:|:---------:|
| 50 | 26.20 | 26.11 | 0.715 | 0.708 |
| 75 | 17.89 | 24.47 | 0.294 | 0.621 |
| 100 | 13.65 | 22.95 | 0.160 | 0.508 |

### Set12

| Noise Level | Baseline DnCNN-B PSNR | Ours PSNR | Baseline DnCNN-B SSIM | Ours SSIM |
|:-----------:|:---------------------:|:---------:|:---------------------:|:---------:|
| 15 | 32.725 | 32.731 | 0.902 | 0.903 |
| 25 | 30.344 | 30.376 | 0.859 | 0.862 |
| 50 | 27.138 | 27.132 | 0.773 | 0.777 |

高噪声盲去噪补充实验：

| Noise Level | DnCNN-B PSNR | Ours PSNR | DnCNN-B SSIM | Ours SSIM |
|:-----------:|:------------:|:---------:|:------------:|:---------:|
| 50 | 27.132 | 26.846 | 0.777 | 0.765 |
| 75 | 18.113 | 24.749 | 0.290 | 0.668 |
| 100 | 13.895 | 22.996 | 0.163 | 0.549 |

## Notes

- `--mode S` 表示固定噪声水平训练，`--mode B` 表示盲去噪训练。
- `--num_of_layers` 仅为兼容早期 DnCNN 命令保留；当前模型结构定义在 `models.py` 的 `UNet` 中。
- 测试脚本会自动把输入图像 padding 到 8 的倍数，以匹配 3 层下采样结构，再裁剪回原尺寸计算指标。
- 曾尝试加入注意力模块与对抗训练。注意力模块在当前 AWGN 设置下收益有限；对抗训练增加了训练成本且未带来稳定提升，因此最终版本保留更稳定的噪声水平图 + U-Net 方案。

## Reference

- Zhang et al., [Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising](https://ieeexplore.ieee.org/document/7839189/)
- Original MATLAB implementation: [cszn/DnCNN](https://github.com/cszn/DnCNN)
