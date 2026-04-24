```table-of-contents
```

# MyNet 背景说明与实施路线

## 1. 项目定位

`MyNet` 是面向 **DJI Action 4 单类目标** 的 8 角点检测子项目。

项目目标不是恢复 6D 位姿，而是建立一条完整的 8 角点检测路线：

1. 使用 HCCEPose / BlenderProc 渲染生成合成训练数据
2. 将 BOP 数据离线转换为 8 个包围盒角点的训练标签
3. 训练一个以单目 RGB 图像为输入、输出为 8 个角点的网络
4. 在真实 DJI Action 4 图像上验证角点预测效果

其中：

- **HCCEPose** 负责生成 BOP/PBR 合成数据
- **MyNet** 负责把 BOP 数据转成 8 角点训练集，并训练 / 推理 8 角点 heatmap 网络

## 2. 与 HCCEPose、BoxDreamer 的关系

### 2.1 与 HCCEPose 的关系

当前仓库中的 HCCEPose 主网络并不是角点 heatmap 网络。

它已经解决的是：

- 合成数据渲染
- BOP 格式数据组织
- 物体模型与真值信息管理

它没有直接提供：

- 8 个角点的 2D 标签
- 8 通道 heatmap
- 单独的 8 角点检测网络

因此，`MyNet` 的定位很明确：

> 在 HCCEPose 渲染数据之上，构建一个新的 8 角点监督网络分支。

### 2.2 与 BoxDreamer 的关系

BoxDreamer 提供了一个重要参考点：使用 3D bounding box corners 作为监督目标。

本项目借鉴的是：

- 以包围盒角点作为明确监督对象
- 采用 heatmap 形式表达角点位置

本项目不直接照搬的部分是：

- BoxDreamer 原始的数据组织方式
- BoxDreamer 完整训练框架

原因是当前目标很明确：

1. 对象类别只有一个：`DJI Action4`
2. 当前只需要获得 8 个角点，而不是继续做姿态恢复

## 3. 总体技术路线

### 3.1 系统级接口

从系统角度，`MyNet` 的目标接口定义为：

```text
输入：单目 RGB 图像
输出：8 个角点 heatmap -> 8 个角点 2D 坐标
```

### 3.2 工程级主线实现

虽然系统级接口是整张 RGB 图像输入，但工程实现上，主线推荐采用两阶段 ROI 方案：

```text
Stage A: detector 找到 DJI Action4 的 2D bbox
Stage B: 对每个实例 crop，预测该实例的 8 个角点 heatmap
Stage C: heatmap 解码出 8 个角点 2D 坐标
```

推荐原因：

1. 一张图里可能有多个同类 DJI Action 4
2. 如果直接整图输出 8 通道 heatmap，每个通道上可能出现多个峰值
3. 对单类多实例来说，检测后裁剪再预测角点更稳

### 3.3 备选研究路线

文档中保留一个研究备选：

```text
单阶段整图 heatmap 网络
```

即：

```text
RGB 整图 -> backbone -> 8 通道 heatmap
```

这条路线只作为备选，不作为第一版主线。

## 4. 当前仓库已经具备的基础

### 4.1 现有数据来源

当前仓库已经能为 DJI Action 4 生成标准 BOP 训练数据，主要文件包括：

```text
dji-action4/
  models/
    obj_000001.ply
    models_info.json
  train_pbr/
    000000/
      rgb/
      depth/
      mask/
      mask_visib/
      scene_camera.json
      scene_gt.json
      scene_gt_info.json
```

这些文件分别提供了：

- `models_info.json`
  - 物体 3D 包围盒范围
  - `min_x/min_y/min_z`
  - `size_x/size_y/size_z`
- `scene_gt.json`
  - 每帧每个实例的真值位姿
  - `cam_R_m2c`
  - `cam_t_m2c`
  - `obj_id`
- `scene_camera.json`
  - 每帧相机内参
  - `cam_K`
- `scene_gt_info.json`
  - 每个实例的 `bbox_obj`
  - `bbox_visib`
  - `visib_fract`
- `mask_visib`
  - 每个实例的可见区域 mask

### 4.2 8 个 3D 角点已经可以从现有数据唯一确定

`models_info.json` 中已经包含包围盒最小点和尺寸，因此可以直接构造 8 个 3D 角点。

当前仓库中 `HccePose/tester.py` 采用的角点顺序如下：

```text
0: [min_x, min_y, min_z]
1: [max_x, min_y, min_z]
2: [max_x, max_y, min_z]
3: [min_x, max_y, min_z]
4: [min_x, min_y, max_z]
5: [max_x, min_y, max_z]
6: [max_x, max_y, max_z]
7: [min_x, max_y, max_z]
```

这个顺序必须作为 MyNet 的固定标准顺序，训练和推理都不能改。

### 4.3 HCCEPose 现有 crop 方式可以直接借鉴

当前仓库中 `HccePose/bop_loader.py` 已经有 square crop 的基础逻辑，特点是：

- 基于 `bbox`
- 转成正方形
- Resize 到固定分辨率

因此 `MyNet` 的实例 crop 规则可以沿用这个思路。

## 5. 数据集规划

### 5.1 主线数据集定义

`MyNet` 的主线训练数据集不是直接使用原始 `train_pbr`，而是：

> 由 BOP `train_pbr` 离线转换得到的实例级 8 角点数据集

推荐输出目录：

```text
MyNet/datasets/dji_action4_corners/
  images/
  heatmaps/
  annotations.jsonl
  splits/
    train.txt
    val.txt
    test.txt
```

其中：

- `images/`：单实例 RGB crop
- `heatmaps/`：每个样本对应的 `8 x H x W` heatmap 标签
- `annotations.jsonl`：几何真值和元数据
- `splits/`：数据划分

### 5.2 每条样本至少包含的字段

推荐 `annotations.jsonl` 每条样本保存如下字段：

```json
{
  "sample_id": "000000_000123_000002",
  "scene_id": 0,
  "frame_id": 123,
  "inst_id": 2,
  "obj_id": 1,
  "rgb_full_path": "dji-action4/train_pbr/000000/rgb/000123.jpg",
  "rgb_crop_path": "MyNet/datasets/dji_action4_corners/images/000000_000123_000002.png",
  "heatmap_path": "MyNet/datasets/dji_action4_corners/heatmaps/000000_000123_000002.npz",
  "bbox_full_xyxy": [x1, y1, x2, y2],
  "crop_xyxy_full": [x1, y1, x2, y2],
  "resize_hw": [256, 256],
  "cam_K_full": [...9 values...],
  "cam_K_crop": [...9 values...],
  "cam_R_m2c": [...9 values...],
  "cam_t_m2c": [...3 values...],
  "corners_3d": [[...], ... x8],
  "corners_2d_full": [[u, v], ... x8],
  "corners_2d_crop": [[u, v], ... x8],
  "corner_visible": [0, 1, ...],
  "corner_in_image": [0, 1, ...]
}
```

### 5.3 数据划分原则

第一版推荐：

- `train`: 80%
- `val`: 10%
- `test`: 10%

如果使用的是按材质分批渲染的新脚本，优先按 `material index` 做切分。  
如果当前数据来自旧版 `train_pbr`，先按 `scene_id + frame_id` 划分即可。

## 6. 从 BOP `train_pbr` 到 8 角点数据集的转换路线

### 6.1 输入

输入是已经存在的 BOP 数据：

```text
dji-action4/
  models/models_info.json
  train_pbr/*/scene_gt.json
  train_pbr/*/scene_camera.json
  train_pbr/*/scene_gt_info.json
  train_pbr/*/rgb/*
  train_pbr/*/mask_visib/*
```

### 6.2 推荐新增的转换脚本

推荐未来在 `MyNet/datasets/` 下实现：

```text
MyNet/datasets/convert_bop_to_mynet.py
```

建议命令接口：

```bash
python MyNet/datasets/convert_bop_to_mynet.py \
  --bop-root dji-action4 \
  --folder train_pbr \
  --output MyNet/datasets/dji_action4_corners \
  --crop-size 256 \
  --heatmap-size 128 \
  --padding-ratio 1.5 \
  --min-visib-fract 0.05
```

### 6.3 详细转换步骤

1. 读取 `models_info.json`
2. 构造固定顺序的 8 个 3D 角点
3. 遍历 `train_pbr/*/scene_gt.json`
4. 读取对应帧的 `scene_camera.json`
5. 将 8 个 3D 角点投影到图像平面，得到 `corners_2d_full`
6. 读取实例 bbox 与可见性信息
7. 生成 square crop
8. 更新 `cam_K_crop`
9. 计算 `corners_2d_crop`
10. 生成 `8 x H x W` heatmap 标签

## 7. 网络结构规划

### 7.1 主线网络

主线方案采用：

```text
ResNet-FPN / U-Net 风格 encoder-decoder
```

输入：

- 单实例 RGB crop

输出：

- `8 x H x W` heatmap

每个输出通道对应一个固定顺序的角点。

### 7.2 备选网络

保留一个备选方向：

```text
ViT / DINO-style encoder + light decoder
```

这只作为后续升级路线，不作为第一版主线。

## 8. 训练与测试路线

### 8.1 训练阶段

训练阶段主推：

- 仅使用 HCCEPose 渲染得到的合成数据
- 监督形式以 heatmap 为主
- 不做真实数据微调

### 8.2 测试阶段

测试阶段主推：

- 在真实 DJI Action 4 图像上测试
- 重点关注 8 个角点是否稳定、准确

## 9. 代码结构规划

推荐未来的 `MyNet` 结构如下：

```text
MyNet/
  README.md
  BACKGROUND.md
  TRAIN_PBR_DATA.md
  datasets/
    convert_bop_to_mynet.py
    mynet_dataset.py
  models/
    backbones/
    necks/
    heads/
    mynet.py
  tools/
    train.py
    infer.py
    visualize.py
```

## 10. 实施路线图

### 阶段 1：跑通 DJI Action4 BOP/PBR 数据

输入：

- DJI Action4 三维模型
- HCCEPose 渲染脚本

输出：

- `train_pbr`
- `models_info.json`

成功判据：

- 能稳定得到 RGB、mask、真值信息

### 阶段 2：实现 BOP -> 8 角点实例数据转换

输入：

- `train_pbr`

输出：

- `images/`
- `heatmaps/`
- `annotations.jsonl`

成功判据：

- 能得到正确的 8 个角点 2D 标签

### 阶段 3：实现 ResNet 版单实例 heatmap 网络

输入：

- 实例级角点数据集

输出：

- 第一版 8 角点模型

成功判据：

- heatmap 输出形状正确
- 训练过程正常收敛

### 阶段 4：跑通训练与 heatmap 可视化

输入：

- 已训练模型

输出：

- heatmap 可视化结果
- 角点预测结果

成功判据：

- 可视化图中 8 个角点位置合理

### 阶段 5：在真实图像上测试

输入：

- 真实 DJI Action 4 图像

输出：

- 角点预测结果

成功判据：

- 角点位置稳定
- 不同视角下结果一致

## 11. 当前明确结论

当前 `MyNet` 的主线设定为：

- 单类目标：DJI Action4
- 合成数据训练，真实图像测试
- ResNet 主线，ViT 备选
- 系统级输入为单目 RGB
- 工程级主线为 detector + ROI heatmap network
- 最终输出目标是 **8 个角点**
