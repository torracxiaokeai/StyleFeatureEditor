# FSE Inverter 推理使用指南

本文档介绍如何使用 `FSEInverterInferenceRunner` 进行图像反演和编辑。

## 概述

`FSEInverterInferenceRunner` 是一个专门用于 FSE (Feature Style Editor) Inverter 模型的推理运行器，它可以：

1. **图像反演**: 将真实图像反演到潜在空间
2. **图像编辑**: 使用预定义的编辑方向对图像进行修改（如年龄、表情、发型等）

## 文件结构

```
StyleFeatureEditor/
├── run_fse_inverter_inference.py                      # 完整版推理脚本
├── run_fse_inverter_inversion_only.py                 # 仅反演脚本
├── configs/my_fse_inverter_inference.yaml             # 完整推理配置文件
├── configs/my_fse_inverter_inference_inversion_only.yaml # 仅反演配置文件
└── README_FSE_INVERTER.md                             # 本文档
```

## 使用方法

### 仅反演模式（推荐用于快速测试）

如果您只想进行图像反演而不进行任何编辑，可以使用以下方法：

#### 方法1：使用专用仅反演脚本（命令行）

```bash
cd StyleFeatureEditor
python run_fse_inverter_inversion_only.py \
    --input_dir "path/to/your/images" \
    --output_dir "results/inversion_only" \
    --checkpoint "experiments/fse_inverter_train_005/iteration_47000.pt" \
    --device "0" \
    --batch_size 4
```

#### 方法2：使用仅反演配置文件

```bash
cd StyleFeatureEditor
python run_fse_inverter_inversion_only.py --config my_fse_inverter_inference_inversion_only.yaml
```

#### 方法3：修改现有配置文件

编辑 `configs/my_fse_inverter_inference.yaml`，将 `editings_data` 设置为空：

```yaml
inference:
  inference_runner: fse_inverter_inference_runner
  editings_data: {}  # 空的编辑数据，表示不进行任何编辑
```

然后运行：

```bash
cd StyleFeatureEditor
python run_fse_inverter_inference.py exp.config=my_fse_inverter_inference.yaml
```

### 完整推理模式（反演+编辑）

#### 方法1：使用完整版脚本（推荐）

1. **准备配置文件**

   编辑 `configs/my_fse_inverter_inference.yaml` 文件：

   ```yaml
   # 修改输入图像目录
   data:
     inference_dir: "path/to/your/images"  # 替换为您的图像目录
   
   # 修改模型检查点路径（如果需要）
   model:
     checkpoint_path: "experiments/fse_inverter_train_005/iteration_47000.pt"
   ```

2. **运行推理**

   ```bash
   cd StyleFeatureEditor
   python run_fse_inverter_inference.py exp.config=my_fse_inverter_inference.yaml
   ```

#### 方法2：使用原始推理脚本

```bash
cd StyleFeatureEditor
python scripts/inference.py exp.config=my_fse_inverter_inference.yaml
```

## 参数说明

### 主要配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `data.inference_dir` | 输入图像目录 | `"data/test_images"` |
| `exp.output_dir` | 输出结果目录 | `"results/fse_inverter_inference"` |
| `model.checkpoint_path` | 模型检查点路径 | `"experiments/fse_inverter_train_005/iteration_47000.pt"` |
| `model.device` | GPU设备ID | `"0"` |
| `model.batch_size` | 批处理大小 | `4` |

### 编辑类型

配置文件中的 `inference.editings_data` 定义了可用的编辑类型：

- **年龄编辑**: `age: [-7, -5, -3, 3, 5, 7, 10]`
- **表情编辑**: `fs_smiling: [-9, -6, -3, 3, 6, 9]`
- **发型编辑**: `afro`, `bobcut`, `bowlcut`, `mohawk`, `blond hair`
- **其他编辑**: `glasses`, `face_roundness`, `rotation`, `purple_hair`, `angry`

**仅反演模式**: 设置 `editings_data: {}` 即可跳过所有编辑

## 输出结果

### 仅反演模式输出

```
output_dir/
└── inversion/          # 反演结果（重建图像）
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 完整推理模式输出

```
output_dir/
├── inversion/          # 反演结果（重建图像）
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── edited/            # 编辑结果
    ├── age/           # 年龄编辑结果
    ├── fs_smiling/    # 微笑编辑结果
    ├── glasses/       # 眼镜编辑结果
    └── ...
```

## 环境要求

### 必需的预训练模型

确保以下预训练模型文件存在于 `pretrained_models/` 目录中：

- `psp_ffhq_encode.pt`
- `e4e_ffhq_encode.pt`
- `stylegan2-ffhq-config-f.pt`
- `stylegan2-ffhq-config-f.pkl`
- 其他相关模型文件...

### 训练好的检查点

需要一个训练好的 FSE Inverter 检查点文件，例如：
- `experiments/fse_inverter_train_005/iteration_47000.pt`

## 故障排除

### 常见错误及解决方案

1. **"输入图像目录不存在"**
   - 检查 `data.inference_dir` 路径是否正确
   - 确保目录中包含图像文件

2. **"模型检查点不存在"**
   - 检查 `model.checkpoint_path` 路径是否正确
   - 确保已完成模型训练或下载了预训练检查点

3. **CUDA内存不足**
   - 减小 `model.batch_size` 值
   - 使用更小的图像分辨率

4. **预训练模型缺失**
   - 确保所有必需的预训练模型都已下载到 `pretrained_models/` 目录

### 调试模式

如果遇到问题，可以在脚本中添加调试信息：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 高级用法

### 自定义编辑方向

您可以在配置文件中添加新的编辑方向：

```yaml
inference:
  editings_data:
    # 添加自定义编辑
    "my_custom_edit": [0.1, 0.2, 0.3]
```

### 批量处理

对于大量图像，建议：
1. 增加 `model.batch_size`（如果GPU内存允许）
2. 增加 `model.workers` 数量
3. 使用多GPU并行处理

## 示例

### 仅反演示例

```bash
# 方法1: 使用命令行参数
cd StyleFeatureEditor
python run_fse_inverter_inversion_only.py \
    -i "data/my_test_images" \
    -o "results/inversion_only" \
    -c "experiments/fse_inverter_train_005/iteration_47000.pt" \
    -d "0" \
    -b 4

# 方法2: 使用配置文件
python run_fse_inverter_inversion_only.py --config my_fse_inverter_inference_inversion_only.yaml

# 方法3: 修改现有配置文件中的 editings_data: {}
python run_fse_inverter_inference.py exp.config=my_fse_inverter_inference.yaml
```

### 完整推理示例

```bash
# 1. 准备图像目录
mkdir -p data/my_test_images
# 将您的图像复制到该目录

# 2. 修改配置文件
# 编辑 configs/my_fse_inverter_inference.yaml
# 设置 data.inference_dir: "data/my_test_images"

# 3. 运行推理
cd StyleFeatureEditor
python run_fse_inverter_inference.py exp.config=my_fse_inverter_inference.yaml

# 4. 查看结果
ls results/fse_inverter_inference/
```

## 性能优化建议

1. **GPU使用**: 确保使用GPU进行推理，设置正确的 `device` 参数
2. **批大小**: 根据GPU内存调整 `batch_size`
3. **工作进程**: 适当增加 `workers` 数量以加速数据加载
4. **图像预处理**: 确保输入图像已正确对齐和裁剪
5. **仅反演模式**: 如果只需要重建图像，使用仅反演模式可以显著节省时间

## 联系支持

如果遇到问题，请检查：
1. 环境配置是否正确
2. 所有依赖是否已安装
3. 预训练模型是否完整
4. 检查点文件是否有效 