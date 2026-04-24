# scripts 使用说明

这个目录下的脚本现在主要分成两类：

1. `DJI Action4 / HCCEPose` 数据准备与渲染脚本
2. Hugging Face 演示资源下载脚本

如果你当前关注的是 DJI Action4 的训练数据生成，建议重点看下面这 4 个脚本：

- `prepare_dji_action4_bop.py`
- `render_dji_action4_pbr.sh`
- `render_dji_action4_material_scene.py`
- `render_dji_action4_material_batch.sh`

---

## 1. 推荐使用顺序

### 1.1 标准单物体数据准备流程

适用于把 `dji-action4/Action4_muti.glb` 接入 HCCEPose 原始 PBR 渲染流程。

1. 先把 `.glb` 转成 BOP 所需的 `.ply`
2. 生成 `models_info.json`
3. 用 `render_dji_action4_pbr.sh` 调现有 `s2_p1_gen_pbr_data.py`

示例：

```bash
python scripts/prepare_dji_action4_bop.py
python s1_p3_obj_infos.py --dataset-path dji-action4
bash scripts/render_dji_action4_pbr.sh
```

### 1.2 按材质、每场景 2 个 DJI Action4 的渲染流程

适用于你当前的真实应用分布：每个场景只有 2 个 DJI Action4，并且希望按材质分批跑，避免一次性加载太多材质导致内存爆掉。

推荐顺序：

1. 准备好源数据集 `dji-action4/`
2. 先列出材质索引
3. 单独测试 1 个材质
4. 再按 50 个材质为一批批量跑

示例：

```bash
python scripts/render_dji_action4_material_scene.py --list-materials

python scripts/render_dji_action4_material_scene.py \
  --material-index 0 \
  --skip-done

MATERIAL_START=0 MATERIAL_STOP=50 \
bash scripts/render_dji_action4_material_batch.sh
```

---

## 2. 脚本说明

## 2.1 `prepare_dji_action4_bop.py`

### 作用

把 `dji-action4/Action4_muti.glb` 转成 HCCEPose / BOP 需要的：

```text
dji-action4/models/obj_000001.ply
```

它会做这些事情：

- 导入 `.glb`
- 合并多个 mesh
- 应用变换
- 按包围盒中心把模型平移到原点
- 在需要时把单位从米放大到毫米
- 导出 ASCII `.ply`

### 默认用法

```bash
python scripts/prepare_dji_action4_bop.py
```

### 可用参数

#### `--input-glb`

默认值：

```text
dji-action4/Action4_muti.glb
```

作用：指定输入的 `.glb` 模型路径。  
什么时候改：如果你的模型文件名或位置变了，就改这个参数。

示例：

```bash
python scripts/prepare_dji_action4_bop.py --input-glb my_models/action4.glb
```

#### `--output-ply`

默认值：

```text
dji-action4/models/obj_000001.ply
```

作用：指定导出的 BOP 模型路径。  
什么时候改：如果你想输出到别的目录，或者想先导出到临时位置检查尺寸。

#### `--unit-scale`

可选值：

```text
auto | 1 | 1000
```

默认值：`auto`

作用：

- `auto`：如果模型包围盒最大尺寸小于 `2`，脚本认为单位是米，自动放大 `1000` 倍到毫米
- `1`：不缩放
- `1000`：强制放大 1000 倍

什么时候改：

- 如果导出的尺寸明显偏小，尝试 `--unit-scale 1000`
- 如果导出的尺寸明显偏大，尝试 `--unit-scale 1`

#### `--overwrite`

默认不覆盖已有文件。  
作用：允许覆盖已经存在的 `obj_000001.ply`。

示例：

```bash
python scripts/prepare_dji_action4_bop.py --overwrite
```

### 运行后的检查点

脚本会打印：

- `applied_unit_scale`
- `bbox_mm`

你需要重点看 `bbox_mm size=(...)` 是否接近 DJI Action4 的毫米量级。  
如果这里尺寸不对，后面的渲染结果会非常离谱。

---

## 2.2 `render_dji_action4_pbr.sh`

### 作用

这是一个轻量包装脚本，用来调用原始的：

```text
s2_p1_gen_pbr_data.sh
```

它适合：

- 你已经准备好了 `dji-action4/models/obj_000001.ply`
- 你已经生成了 `dji-action4/models/models_info.json`
- 你想沿用 HCCEPose 原始 PBR 渲染流程

### 默认用法

```bash
bash scripts/render_dji_action4_pbr.sh
```

### 可修改环境变量

#### `GPU_ID`

默认值：`0`

作用：指定使用哪张 GPU。

示例：

```bash
GPU_ID=1 bash scripts/render_dji_action4_pbr.sh
```

#### `SCENE_NUM`

默认值：`1`

作用：指定跑多少个 scene。  
原始 `s2_p1_gen_pbr_data.py` 每次会继续往 `train_pbr` 里追加数据。

示例：

```bash
SCENE_NUM=42 bash scripts/render_dji_action4_pbr.sh
```

#### `DATASET_PATH`

默认值：

```text
$REPO_ROOT/dji-action4
```

作用：指定目标数据集根目录。

#### `TEXTURES_PATH`

默认值：

```text
$REPO_ROOT/cc0textures-512
```

作用：指定材质库路径。

#### `GEN_SCRIPT`

默认值：

```text
$REPO_ROOT/s2_p1_gen_pbr_data.py
```

作用：指定实际被调用的渲染 Python 脚本。  
一般不用改，除非你远端有另外一份定制版 `s2_p1_gen_pbr_data.py`。

### 适用场景

这个脚本适合“继续走原项目默认逻辑”。  
如果你要的是“每个材质只加载一次、每个场景只有两个 DJI Action4、按材质批量分批跑”，请改用下面的 `render_dji_action4_material_scene.py` 和 `render_dji_action4_material_batch.sh`。

---

## 2.3 `render_dji_action4_material_scene.py`

### 作用

这个脚本是给“按材质逐个渲染、避免内存累积”准备的。  
它一次只处理 **1 个材质**，并且固定：

- 只加载 **2 个** `obj_id=1` 的 DJI Action4
- 只渲染 **20 张**图
- 结果追加写入 BOP 的 `train_pbr`
- 用 `material_render_manifest.jsonl` 记录进度

### 这个脚本的关键设计

1. 一次只加载一个材质，降低内存占用
2. 一个材质对应 20 帧
3. `frames_per_chunk=1000`
4. 所以默认 **50 个材质 = 1000 帧 = 1 个 `train_pbr/000xyz/` 文件夹**

### 先查看材质索引

```bash
python scripts/render_dji_action4_material_scene.py --list-materials
```

这个命令会输出：

```text
total_materials=...
0   ...
1   ...
2   ...
```

这里的索引是后面 `--material-index` 要用的零基索引。

### 单独跑一个材质

```bash
python scripts/render_dji_action4_material_scene.py \
  --material-index 0 \
  --skip-done
```

### 常用参数

#### `--gpu-id`

默认值：`0`

作用：指定 GPU 编号。

#### `--textures-path`

默认值：

```text
<repo>/cc0textures-512
```

作用：指定材质库目录。

支持两类材质库：

- `cc0textures-512`
- 完整版 `cc0textures`

不同材质库的材质枚举方式不同，但脚本会稳定排序后再编号。

#### `--source-dataset-path`

默认值：

```text
<repo>/dji-action4
```

作用：指定源数据集目录。  
这个目录里至少要有：

```text
models/
models/models_info.json
camera.json   (可选，没有就自动写默认值)
```

脚本会从这里复用 `models/` 和 `camera.json`。

#### `--output-dataset-path`

默认值：

```text
<repo>/dji-action4-twoobj-materials
```

作用：指定新的输出数据集目录。

建议保持和原来的 `dji-action4/` 分开，避免把“原始多物体分布”和“每场景 2 个 DJI Action4”的数据混在一起。

#### `--material-index`

默认值：无，必须显式指定，除非使用 `--list-materials`

作用：指定当前要跑第几个材质。

示例：

```bash
python scripts/render_dji_action4_material_scene.py --material-index 357
```

#### `--object-id`

默认值：`1`

作用：指定渲染哪个物体类别。  
你现在的 DJI Action4 数据集只有一个物体，通常保持 `1` 即可。

#### `--object-count`

默认值：`2`

作用：指定每个小场景中放几个该物体实例。  
你当前场景固定只有 2 个 DJI Action4，所以默认值就是推荐值。

如果你改成别的数：

- 会改变每帧场景内的实例数量
- 但不会改变“每个材质生成多少帧”

#### `--views-per-material`

默认值：`20`

作用：指定每个材质渲染多少张视角图。

这是一个非常关键的参数。  
当前脚本的默认逻辑是：

```text
20 帧 / 材质
1000 帧 / chunk
=> 50 材质 / chunk
```

如果你把它改成别的数，比如 `10` 或 `25`：

- 每个材质对应的帧数会变
- 每个 chunk 里能装多少个材质也会跟着变

并且脚本要求：

```text
frames_per_chunk % views_per_material == 0
```

也就是 `frames_per_chunk` 必须能被 `views_per_material` 整除。

#### `--frames-per-chunk`

默认值：`1000`

作用：指定 BOP 每个 `train_pbr/000xyz/` 文件夹最多放多少帧。

通常建议保持 `1000`，因为这是 BOP 默认习惯，也和原项目一致。

如果你修改它：

- 目录切换频率会变化
- 每个 chunk 能装多少个材质会变化
- manifest 里的 `chunk_id/frame_start/frame_end` 映射也会变化

#### `--skip-done`

默认值：关闭

作用：如果 manifest 里已经记录某个材质 `status=done`，则跳过，不重复渲染。

这个参数很适合：

- 中断后重跑同一段区间
- 已经跑过一部分，想安全补跑

#### `--list-materials`

默认值：关闭

作用：仅列出材质索引与材质名，不执行渲染。

### manifest 文件说明

输出目录下会有：

```text
material_render_manifest.jsonl
```

每处理一个材质，会写记录。  
常见字段有：

- `material_index`
- `material_name`
- `status`
- `chunk_id`
- `frame_start`
- `frame_end`
- `views_written`
- `timestamp`

其中 `status` 主要有两种：

- `started`
- `done`

### 关于“为什么不能乱序跳着跑”

这个脚本默认要求按连续顺序追加，比如：

```text
0 -> 1 -> 2 -> 3 -> ...
```

原因很简单：它要保证

```text
0-49   -> train_pbr/000000
50-99  -> train_pbr/000001
100-149 -> train_pbr/000002
```

如果你跳着跑，比如先跑 `0`，再跑 `20`，再跑 `5`，那 chunk 内部帧位就会乱掉。  
所以现在的脚本会主动检查“下一个该跑的材质索引”，不允许打乱这个顺序。

---

## 2.4 `render_dji_action4_material_batch.sh`

### 作用

这是 `render_dji_action4_material_scene.py` 的批处理包装。  
它按区间循环材质，每个材质都单独启动一次 Python 进程。

这样做的好处是：

- 一个材质跑完，进程退出
- Blender / 材质 / 物理缓存一起释放
- 更适合你现在远端会 OOM 的情况

### 默认用法

```bash
bash scripts/render_dji_action4_material_batch.sh
```

默认等价于：

```bash
MATERIAL_START=0 MATERIAL_STOP=50 bash scripts/render_dji_action4_material_batch.sh
```

也就是跑第 `0-49` 个材质，正好 50 个材质。

### 常用环境变量

#### `GPU_ID`

默认值：`0`

作用：指定 GPU。

#### `TEXTURES_PATH`

默认值：

```text
$REPO_ROOT/cc0textures-512
```

作用：指定材质库路径。

#### `SOURCE_DATASET_PATH`

默认值：

```text
$REPO_ROOT/dji-action4
```

作用：指定源数据集路径。

#### `OUTPUT_DATASET_PATH`

默认值：

```text
$REPO_ROOT/dji-action4-twoobj-materials
```

作用：指定输出数据集路径。

#### `PY_SCRIPT`

默认值：

```text
$REPO_ROOT/scripts/render_dji_action4_material_scene.py
```

作用：指定被批处理调用的 Python 脚本路径。

一般不需要改，除非你维护了另一份定制版。

#### `MATERIAL_START`

默认值：`0`

作用：指定材质区间起点，包含该值。

#### `MATERIAL_STOP`

默认值：`50`

作用：指定材质区间终点，不包含该值。

所以：

```bash
MATERIAL_START=0 MATERIAL_STOP=50
```

表示跑：

```text
0, 1, 2, ..., 49
```

### 常见用法

#### 跑第一批 50 个材质

```bash
MATERIAL_START=0 MATERIAL_STOP=50 \
bash scripts/render_dji_action4_material_batch.sh
```

#### 跑第二批 50 个材质

```bash
MATERIAL_START=50 MATERIAL_STOP=100 \
bash scripts/render_dji_action4_material_batch.sh
```

#### 只测试 10 个材质

```bash
MATERIAL_START=0 MATERIAL_STOP=10 \
bash scripts/render_dji_action4_material_batch.sh
```

#### 只补跑一个材质

```bash
MATERIAL_START=357 MATERIAL_STOP=358 \
bash scripts/render_dji_action4_material_batch.sh
```

注意：补跑时也要满足顺序约束。  
如果 manifest 和 `train_pbr` 当前状态显示“下一个该跑的是 120”，你直接去跑 `357`，脚本会拒绝执行。

---

## 2.5 `download_hf_assets.py`

### 作用

从 Hugging Face 数据集 `SEU-WYL/HccePose` 下载测试图、示例数据集、权重等资源。

### 默认思路

这个脚本按 `--preset` 选择下载内容，而不是一次把整套都拉下来。

### 常用示例

```bash
python scripts/download_hf_assets.py --preset test
python scripts/download_hf_assets.py --preset dataset
python scripts/download_hf_assets.py --preset test --preset weights
python scripts/download_hf_assets.py --preset all
```

### 参数说明

#### `--preset`

可重复传入。当前支持：

- `test`
- `dataset`
- `tex`
- `weights`
- `all`

作用：指定下载哪一类资源。

#### `--endpoint`

可选：

- `auto`
- `hf`
- `mirror`

默认：`auto`

作用：指定 Hugging Face 官方站还是镜像站。

#### `--dest`

作用：指定下载目标目录。  
默认写到仓库根目录。

#### `--foundationpose`

作用：额外下载 FoundationPose 相关权重。

---

## 2.6 `wget_hf_demo_assets.py`

### 作用

这个脚本也是下载演示资源，但走的是 `wget` 路线。  
适合 `huggingface_hub snapshot_download` 不稳定、容易断开的环境。

### 常用示例

```bash
python scripts/wget_hf_demo_assets.py
python scripts/wget_hf_demo_assets.py --endpoint hf --jobs 8
python scripts/wget_hf_demo_assets.py --verify-only
```

### 常见参数

#### `--endpoint`

作用：指定官方站或镜像站。

#### `--jobs`

作用：指定并发下载线程数。  
数值越大，下载速度可能越快，但网络不稳定时也可能更容易失败。

#### `--verify-only`

作用：只校验本地文件是否完整，不执行下载。

---

## 3. 参数修改时最值得注意的几个点

### 3.1 `views-per-material` 和 `frames-per-chunk` 要匹配

在 `render_dji_action4_material_scene.py` 里：

```text
frames_per_chunk 必须能被 views_per_material 整除
```

当前默认：

```text
20 / 材质
1000 / chunk
=> 50 材质 / chunk
```

如果你改成：

```text
views_per_material = 25
```

那么：

```text
1000 / 25 = 40 材质 / chunk
```

### 3.2 `output_dataset_path` 建议单独目录

推荐把按材质生成的新数据写到：

```text
dji-action4-twoobj-materials
```

不要和原来 `dji-action4/train_pbr` 混着写。  
否则后面很难区分：

- 哪些数据是原始默认多物体分布
- 哪些数据是“每场景只有两个 DJI Action4”的目标分布

### 3.3 `skip-done` 适合重跑，不适合忽略顺序

`--skip-done` 的作用是：

- 已完成的材质直接跳过

但它**不会**让脚本支持乱序插入。  
当前逻辑依然要求 chunk 写入顺序连续。

### 3.4 `object-count` 只影响每帧实例数，不影响每材质帧数

比如：

```bash
--object-count 3
```

表示每张图会尝试放 3 个 DJI Action4，  
但仍然是“每个材质渲染 `views-per-material` 张图”。

---

## 4. 推荐命令模板

### 4.1 查看材质索引

```bash
python scripts/render_dji_action4_material_scene.py --list-materials
```

### 4.2 单材质测试

```bash
python scripts/render_dji_action4_material_scene.py \
  --textures-path /path/to/cc0textures-512 \
  --source-dataset-path /path/to/dji-action4 \
  --output-dataset-path /path/to/dji-action4-twoobj-materials \
  --material-index 0 \
  --skip-done
```

### 4.3 第一批 50 个材质

```bash
GPU_ID=0 \
TEXTURES_PATH=/path/to/cc0textures-512 \
SOURCE_DATASET_PATH=/path/to/dji-action4 \
OUTPUT_DATASET_PATH=/path/to/dji-action4-twoobj-materials \
MATERIAL_START=0 \
MATERIAL_STOP=50 \
bash scripts/render_dji_action4_material_batch.sh
```

### 4.4 下一批 50 个材质

```bash
GPU_ID=0 \
TEXTURES_PATH=/path/to/cc0textures-512 \
SOURCE_DATASET_PATH=/path/to/dji-action4 \
OUTPUT_DATASET_PATH=/path/to/dji-action4-twoobj-materials \
MATERIAL_START=50 \
MATERIAL_STOP=100 \
bash scripts/render_dji_action4_material_batch.sh
```

### 4.5 只补跑一个材质

```bash
GPU_ID=0 \
TEXTURES_PATH=/path/to/cc0textures-512 \
SOURCE_DATASET_PATH=/path/to/dji-action4 \
OUTPUT_DATASET_PATH=/path/to/dji-action4-twoobj-materials \
MATERIAL_START=120 \
MATERIAL_STOP=121 \
bash scripts/render_dji_action4_material_batch.sh
```

---

## 5. 你当前场景下的建议

结合你现在的目标，我建议远端实际使用时优先走这条线：

1. `prepare_dji_action4_bop.py`
2. `s1_p3_obj_infos.py --dataset-path dji-action4`
3. `render_dji_action4_material_scene.py --list-materials`
4. 先单测 `material-index 0`
5. 再按 `0:50`、`50:100`、`100:150` 这样分批跑

这样最稳，内存压力也最可控。
