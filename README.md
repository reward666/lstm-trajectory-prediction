# LSTM Trajectory Prediction

本仓库默认**不上传原始数据、处理中间文件、训练样本和模型权重**。这些文件通常体积较大，也可能受数据集许可限制，适合像 MiniMind 一样：代码进 Git，数据在服务器上通过命令下载或从本地数据盘拷贝，然后再运行处理脚本生成训练文件。

## 数据目录约定

```text
data/
  raw/                         # 原始数据（不提交）
    trajectories-0750am-0805am.txt
  processed/                   # 处理后的 pkl/npy（不提交）
    ngsim_us101_0750_0805_processed.pkl
    X.npy
    Y.npy
    X_social.npy
    social_masks.npy
outputs/                       # checkpoint、日志、图片（不提交）
```

`.gitignore` 已忽略 `data/`、`outputs/`、`*.npy`、`*.pkl`、`*.pt` 等大文件/产物。

## 1. 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 下载或放置原始数据

### 方式 A：服务器上通过 URL 下载

> NGSIM 等交通数据集可能需要你在官方平台申请/登录后获得下载链接。拿到链接后，把 `<DATA_URL>` 换成真实地址即可。

```bash
python src/download_data.py \
  --url "<DATA_URL>" \
  --output_dir data/raw \
  --filename trajectories-0750am-0805am.txt
```

如果下载的是压缩包，可以加 `--extract`：

```bash
python src/download_data.py \
  --url "<ARCHIVE_URL>" \
  --output_dir data/raw \
  --filename ngsim_us101.zip \
  --extract
```

### 方式 B：从服务器已有路径复制

```bash
python src/download_data.py \
  --source /path/to/trajectories-0750am-0805am.txt \
  --output_dir data/raw \
  --filename trajectories-0750am-0805am.txt
```

## 3. 预处理原始轨迹文件

默认读取 `data/raw/trajectories-0750am-0805am.txt`，输出 `data/processed/ngsim_us101_0750_0805_processed.pkl`：

```bash
python src/preprocess.py
```

也可以显式指定路径：

```bash
python src/preprocess.py \
  --raw_path data/raw/trajectories-0750am-0805am.txt \
  --output_path data/processed/ngsim_us101_0750_0805_processed.pkl
```

## 4. 构造训练样本

### Social LSTM / Attention 训练样本（推荐）

生成 `X.npy`、`Y.npy`、`X_social.npy`、`social_masks.npy`：

```bash
python src/buildsample.py \
  --processed_path data/processed/ngsim_us101_0750_0805_processed.pkl \
  --output_dir data/processed \
  --num_workers 8
```

### 简单 LSTM/Transformer 二维轨迹样本

只生成 `X.npy`、`Y.npy`：

```bash
python src/dataset.py \
  --processed_path data/processed/ngsim_us101_0750_0805_processed.pkl \
  --output_dir data/processed
```

## 5. 训练

Social LSTM with Attention：

```bash
python src/train.py \
  --data_dir data/processed \
  --output_dir outputs \
  --wandb_mode offline
```

Transformer baseline：

```bash
python src/train_transformer.py \
  --data_dir data/processed \
  --output_dir outputs
```

## 常用服务器流水线

```bash
pip install -r requirements.txt
python src/download_data.py --url "<DATA_URL>" --output_dir data/raw --filename trajectories-0750am-0805am.txt
python src/preprocess.py --raw_path data/raw/trajectories-0750am-0805am.txt
python src/buildsample.py --num_workers 8
python src/train.py --wandb_mode offline
```
