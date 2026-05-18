# LSTM Trajectory Prediction

基于 NGSIM 轨迹数据训练 Social LSTM with Attention，用历史轨迹和周边车辆信息预测未来轨迹。

## 目录约定

```text
data/raw/        # 原始数据，不提交
data/processed/  # 预处理结果和训练样本，不提交
outputs/         # checkpoint、日志、图片，不提交
```

## 主链路

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备原始数据

下载公开 CSV：

```bash
python src/download_data.py \
  --url "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD" \
  --output_dir data/raw \
  --filename ngsim_vehicle_trajectories.csv
```

如果服务器上已经有数据文件，也可以复制到项目目录：

```bash
python src/download_data.py \
  --source /path/to/trajectories-0750am-0805am.txt \
  --output_dir data/raw \
  --filename trajectories-0750am-0805am.txt
```

### 3. 预处理

```bash
python src/preprocess.py \
  --raw_path data/raw/ngsim_vehicle_trajectories.csv \
  --output_path data/processed/ngsim_us101_0750_0805_processed.pkl
```

### 4. 构造训练样本

```bash
python src/buildsample.py \
  --processed_path data/processed/ngsim_us101_0750_0805_processed.pkl \
  --output_dir data/processed \
  --num_workers 8
```

该步骤会生成：

```text
data/processed/X.npy
data/processed/Y.npy
data/processed/X_social.npy
data/processed/social_masks.npy
```

### 5. 训练

```bash
python src/train.py \
  --data_dir data/processed \
  --output_dir outputs \
  --wandb_mode offline
```

## 一条命令跑完整流程

```bash
pip install -r requirements.txt
python src/download_data.py --url "https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD" --output_dir data/raw --filename ngsim_vehicle_trajectories.csv
python src/preprocess.py --raw_path data/raw/ngsim_vehicle_trajectories.csv --output_path data/processed/ngsim_us101_0750_0805_processed.pkl
python src/buildsample.py --processed_path data/processed/ngsim_us101_0750_0805_processed.pkl --output_dir data/processed --num_workers 8
python src/train.py --data_dir data/processed --output_dir outputs --wandb_mode offline
```
