# P3：PyTorch CPU 训练重建协议

**状态：** 已批准实施；训练结果须由 Ubuntu 和 Intel Mac 实机运行产生。  
**实验类型：** reconstructed experiment，不是历史训练的精确复现。

## 1. 固定数据

- 训练集：train，880 张。
- 验证集：validation，220 张。
- test，200 张公开固定测试图片，不用于训练、模型选择或调参。
- 标签始终作为四位 ASCII 数字字符串读取，前导零有意义。
- 图片从原始 JPEG 字节解码，BILINEAR Resize 到 64×24 RGB。

训练入口会先运行公开数据集审计。模型训练数据集类只允许 train 和 validation 两个 split；传入 test 会直接报错。完整数据审计可能验证公开 Test 文件的完整性，但训练过程不加载 Test 样本，不产生 Test 预测或指标。

## 2. 候选模型

三个模型都使用 3×3、stride 2、padding 1 的卷积，Bias 开启，卷积后 ReLU；FC1 后 ReLU；FC2 输出 40 个 Logits 并 reshape 为 4×10。无 BatchNorm、Dropout 或额外正则化，使用 PyTorch 层默认初始化，不加载历史冻结权重。

| 模型 | 卷积通道 | FC1 | 参数量 |
|---|---|---:|---:|
| A | 3→8→16→24 | 128 | 83,888 |
| B | 3→6→12→16 | 64 | 29,812 |
| C | 3→8→16→32 | 96 | 83,736 |

## 3. 训练配置

- 最大 100 Epoch；连续 15 Epoch 未提升 Validation 整图准确率则停止。
- Batch Size 64；drop_last=False；num_workers=0。
- AdamW，学习率 0.001，Weight Decay 0.0001；不使用 Scheduler 或 Label Smoothing。
- Loss 为四个数字位置的 Cross Entropy 平均值。
- CPU 单线程：PyTorch intra-op / inter-op 均为 1；启用 deterministic algorithms。
- 随机种子：20261127、20261128、20261129。
- 每个 Epoch 的 Shuffle 顺序按 seed 和 epoch 生成；相同 seed/epoch 在 A/B/C 之间一致。
- 增强随机数按 seed、epoch、image_id 派生，不依赖模型初始化消耗的全局随机数；相同图片在三个模型中得到相同增强。

## 4. 数据增强

仅训练集执行，顺序如下：

1. RGB BILINEAR Resize 至 64×24。
2. Brightness：概率 50%，倍率均匀采样 0.9～1.1。
3. Contrast：独立概率 50%，倍率均匀采样 0.9～1.1。
4. 反色归一化：(255 - pixel) / 255.0。
5. Gaussian Noise：概率 30%，σ 均匀采样 0～0.02，RGB 通道独立采样并裁剪到 [0,1]。
6. Translation：概率 50%，水平和垂直偏移独立从 -1、0、1 采样；归一化空间越界填 0。

Validation 仅执行确定性 Resize 和归一化，不做随机增强。

## 5. 运行环境

使用 conda-forge 的 Python 3.12、PyTorch 2.14.1 CPU Generic、NumPy 2.3.3、Pillow 13.0.0。批准平台为 Ubuntu x86_64 和 Intel macOS x86_64；两个平台均须使用 CPU Generic 构建，不使用 CUDA、MPS 或 MKL。

已核对的包构建约束：
- conda-forge PyTorch 2.14.1 cpu_generic 存在 linux-64 与 osx-64、Python 3.12 构建。
- Intel macOS PyTorch 构建要求 macOS 12.0 或更高；macOS 12.7.6 满足该要求。
- Pillow 13.0.0 存在 Linux x86_64 与 macOS x86_64 Python 3.12 构建。

创建环境：

```bash
conda env create -f environment.yml
conda activate emtl-ocr-p3
```

如需生成双平台的完整解析锁文件，在具备 conda-forge 网络访问的环境执行：

```bash
conda-lock lock --file environment.yml --virtual-package-spec virtual-packages.yml --lockfile conda-lock.yml
```

随后可用 conda-lock install --name emtl-ocr-p3 conda-lock.yml --platform linux-64 或 --platform osx-64 安装。virtual-packages.yml 将 linux-64 的最低 glibc 设为 2.17，并将 osx-64 的最低 macOS 版本设为 12.0；锁定命令显式使用该约束文件。CI 的环境锁定 Job 会使用 conda-lock 对两个平台实际求解，并上传 conda-lock.yml Artifact。锁文件由实际 solver 生成，不手工伪造包 URL、build 或 hash；本次 PR 不把未经 solver 验证的锁文件提交到仓库。

## 6. 运行训练

运行完整九次实验：

```bash
python tools/train_models.py
```

单次调试运行：

```bash
python tools/train_models.py --model A --seed 20261127
```

输出路径：

```text
outputs/p3/<platform>/model-<a|b|c>/seed-<seed>/
├── best.pt
├── metrics.csv
├── run_config.json
├── run_status.json
└── run_summary.json
```

已存在的 run 目录不会被覆盖。每次运行保存最佳 Checkpoint、逐 Epoch 指标、实际配置、软件/硬件环境和 Checkpoint SHA-256。失败运行会保留 run_status.json，不得删除不利结果。

## 7. 审计与选择

完成一个或两个平台的九次训练后执行：

```bash
python tools/audit_training_runs.py --root outputs/p3 --output outputs/p3/audit.json
```

每个平台必须有 A/B/C × 三个 Seed 的九个完整结果。报告每个模型三次 Validation 整图准确率均值和样本标准差。若较小模型与最高平均准确率相差不超过 1 个百分点，选择参数量更少者；随后在胜出模型的三个 Seed 中选择 Validation 准确率中位数对应的 Checkpoint，Loss 作为并列时的次级排序依据。

两个平台均存在时，审计脚本按模型和 Seed 配对比较 Validation 整图准确率，任一差异超过 1 个百分点则返回失败状态。该脚本不读取测试集或测试标签。

## 8. 实验产物与限制

outputs/ 已在 .gitignore 中排除。训练 Checkpoint 和运行结果保留在各自机器；需要跨平台审计时，将两个平台的 outputs/p3/<platform>/ 目录合并到同一输出根目录。不得将 Test 指标用于 P3 架构选择。训练记录必须保留失败运行，不得因结果不利而删除。
