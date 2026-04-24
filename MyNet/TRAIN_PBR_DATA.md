# DJI Action4 `train_pbr` 数据说明

本文档只说明当前 `dji-action4/train_pbr` 中已经存储了哪些数据。

## 1. 当前数据位置

当前数据主目录为：

```text
dji-action4/
  camera.json
  models/
    models_info.json
    obj_000001.ply
  train_pbr/
    000000/
      rgb/
      depth/
      mask/
      mask_visib/
      scene_camera.json
      scene_gt.json
      scene_gt_coco.json
      scene_gt_info.json
```

其中：

- `models/` 存放物体模型和物体尺寸信息
- `train_pbr/000000/` 是一个 BOP 格式的渲染场景文件夹

## 2. 当前 `000000` 中已有的数据

我当前检查到：

- `rgb/`：65 张图
- `depth/`：65 张图
- `mask/`：1260 张图
- `mask_visib/`：1260 张图

这说明：

- 当前 `000000` 中一共已经渲染了 65 帧
- 每一帧中实例数不固定
- `mask` 和 `mask_visib` 是按“帧内实例”分别保存的

## 3. 各目录中存放的内容

### 3.1 `rgb/`

示例：

```text
rgb/000000.jpg
rgb/000001.jpg
rgb/000002.jpg
```

含义：

- 每个文件对应一帧 RGB 图像
- 当前图像分辨率为 `640 x 480`
- 文件格式为 `.jpg`

### 3.2 `depth/`

示例：

```text
depth/000000.png
depth/000001.png
depth/000002.png
```

含义：

- 每个文件对应一帧深度图
- 文件格式为 `.png`
- 需要结合 `scene_camera.json` 中的 `depth_scale` 一起解释

### 3.3 `mask/`

示例：

```text
mask/000000_000000.png
mask/000000_000001.png
mask/000000_000002.png
```

含义：

- 这是实例级二值掩膜
- 命名规则为：

```text
<frame_id>_<instance_id>.png
```

例如：

- `000025_000003.png` 表示第 25 帧中的第 3 个实例

### 3.4 `mask_visib/`

示例：

```text
mask_visib/000000_000000.png
mask_visib/000000_000001.png
```

含义：

- 这是实例级“可见区域”二值掩膜
- 命名规则与 `mask/` 相同

## 4. 各 JSON 文件中存放的内容

### 4.1 `scene_camera.json`

这是按帧保存的相机信息。

结构示意：

```json
{
  "0": {
    "cam_K": [...],
    "cam_R_w2c": [...],
    "cam_t_w2c": [...],
    "depth_scale": 0.1
  }
}
```

包含字段：

- `cam_K`：相机内参矩阵，长度 9，按行展开
- `cam_R_w2c`：世界坐标系到相机坐标系的旋转矩阵
- `cam_t_w2c`：世界坐标系到相机坐标系的平移向量
- `depth_scale`：深度缩放系数

### 4.2 `scene_gt.json`

这是按帧、按实例保存的物体位姿真值。

结构示意：

```json
{
  "0": [
    {
      "cam_R_m2c": [...],
      "cam_t_m2c": [...],
      "obj_id": 1
    }
  ]
}
```

包含字段：

- `cam_R_m2c`：物体坐标系到相机坐标系的旋转矩阵
- `cam_t_m2c`：物体坐标系到相机坐标系的平移向量
- `obj_id`：物体类别编号，当前为 `1`

### 4.3 `scene_gt_info.json`

这是按帧、按实例保存的 2D 信息和像素统计信息。

结构示意：

```json
{
  "0": [
    {
      "bbox_obj": [x, y, w, h],
      "bbox_visib": [x, y, w, h],
      "px_count_all": 1434,
      "px_count_valid": 1434,
      "px_count_visib": 1434,
      "visib_fract": 1.0
    }
  ]
}
```

包含字段：

- `bbox_obj`：实例整体区域的 2D 框
- `bbox_visib`：实例可见区域的 2D 框
- `px_count_all`：实例总像素数
- `px_count_valid`：有效像素数
- `px_count_visib`：可见像素数
- `visib_fract`：可见比例

### 4.4 `scene_gt_coco.json`

这是 COCO 风格的导出文件。

包含主要部分：

- `images`
- `annotations`
- `categories`

其中：

- `images` 对应每一帧 RGB 图像
- `annotations` 对应每个实例的 COCO 格式标注
- `categories` 记录类别信息

`annotations` 中可见的主要字段有：

- `image_id`
- `category_id`
- `area`
- `bbox`
- `segmentation`
- `ignore`

## 5. 其他相关文件

### 5.1 `dji-action4/camera.json`

这是当前数据集使用的默认相机参数文件。

当前内容包含：

- `fx`
- `fy`
- `cx`
- `cy`
- `width`
- `height`
- `depth_scale`

### 5.2 `dji-action4/models/models_info.json`

这是物体的三维尺寸信息文件。

当前 `obj_id = 1` 下包含：

- `diameter`
- `min_x`
- `min_y`
- `min_z`
- `max_x`
- `max_y`
- `max_z`
- `size_x`
- `size_y`
- `size_z`

这个文件描述了 DJI Action4 的三维包围盒范围。

## 6. 数据对应关系

### 6.1 帧级对应关系

例如帧 `25`：

- `rgb/000025.jpg`
- `depth/000025.png`
- `scene_camera.json["25"]`
- `scene_gt.json["25"]`
- `scene_gt_info.json["25"]`

### 6.2 实例级对应关系

例如第 25 帧中的第 3 个实例：

- `scene_gt.json["25"][3]`
- `scene_gt_info.json["25"][3]`
- `mask/000025_000003.png`
- `mask_visib/000025_000003.png`

也就是说，帧内实例索引是一一对应的。

## 7. 当前数据内容总结

当前 `dji-action4/train_pbr` 中已经包含：

1. 每帧 RGB 图像
2. 每帧深度图
3. 每个实例的完整掩膜
4. 每个实例的可见掩膜
5. 每帧相机内参和相机位姿
6. 每个实例的物体位姿真值
7. 每个实例的 2D bbox 和像素统计
8. COCO 格式的检测/分割导出
9. 物体三维尺寸和三维包围盒信息

这些就是当前已经实际渲染并保存下来的数据内容。
