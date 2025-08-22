# MAGIC + Defect Spectrum：用自然语言在指定部位生成指定缺陷（超级完整实现文档）

> 目标：基于你现有仓库（MAGIC-Anomaly-generation），在 **Defect Spectrum（DS-MVTec）** 数据上训练，让模型“听懂”自然语言指令（中文/英文均可），并能**自动定位**（如“螺丝顶部”）只在该区域**生成指定类型的缺陷**（如“细划痕/腐蚀/磨损/裂纹”，带厚度/强弱等属性），保持背景不变、掩码对齐、位置合理。

---

## 1. 算法总览（通俗版）

MAGIC 的生成底座是 **Stable Diffusion v2 Inpainting**。为满足工业缺陷的三大要求——**不动背景**、**掩码严格重合**、**语义位置合法**——它在训练/推理阶段分别引入三个关键“外挂”：

* **GPP（Gaussian Prompt Perturbation）**：对文本编码（CLIP 文本向量）加高斯噪声，训练和推理都加，用于拓宽“缺陷外观”的多样性，同时保持文本语义不崩。
* **MGNI（Mask-Guided Noise Injection）**：只在**掩码区域**注入额外噪声（按时间步衰减），增强局部纹理的多样性与真实性，不污染背景区域。
* **CAMA（Context-Aware Mask Alignment）**：把掩码与物体语义位置对齐（例如把“划痕”掩码吸附到“螺丝头/螺纹上缘”等合理处），防止“越界”。

**本项目新增**：

* **逐样本文本监督训练**：使用 Defect Spectrum 中的 **captions**（描述性文本）作为每张图的 prompt，使模型学会“类型/形态/强弱/位置”的语言→视觉对齐。
* **文本→区域掩码（可选自动）**：推理时如未提供掩码，我们根据“位置词”（如“顶部/头部/top/head”）从**前景掩码**自动裁出“区域条带”作为 inpaint 掩码；也可选对接文本分割器（如短语分割）进一步智能定位。

---

## 2. 数据与目录

**推荐数据源**：Defect Spectrum 中的 **DS-MVTec** 分支（MIT 许可）。
基本结构（以 `screw` 为例）：

```
Defect_Spectrum/
└─ DS-MVTec/
   ├─ screw/
   │  ├─ image/
   │  │  ├─ scratch_head/        # 缺陷子类（同时暗含位置语义，如 head/top/thread_top/...）
   │  │  ├─ thread_top/
   │  │  └─ ...
   │  ├─ mask/
   │  │  ├─ scratch_head/
   │  │  ├─ thread_top/
   │  │  └─ ...
   │  └─ ...
   ├─ captions.xlsx               # 每张图像的描述性文本（filename → caption）
   └─ ...
```

**外部前景掩码（必需）**：仓库 README 已提供用 U²-Net 离线生成前景掩码的说明（正常图 → 物体前景）。推理阶段我们会用它来\*\*裁剪“位置条带”\*\*与 CAMA 对齐。

---

## 3. 环境与依赖

1. 基础环境（与你仓库一致）：

```bash
conda create -n magic python=3.9
conda activate magic
pip install -r requirements.txt
pip install opencv-python
```

2. 数据表与中文处理：

```bash
pip install pandas openpyxl
```

3. 可选（文本分割器，用于 V2 自动掩码，后文有说明）：

```bash
# 可按需安装，如使用短语分割/CLIPSeg等
pip install transformers>=4.41.0
```

---

## 4. 训练改造：逐样本 caption 监督（最小改动）

目标：让训练不再用统一的 `"a photo of sks defect"`，而是**每张图用自己的文本描述**（来自 `captions.xlsx`），从而学会“类型/厚度/强弱/位置”等语义 → 视觉模式。

### 4.1 修改 `train_dreambooth_noise.py`

**新增参数**（argparse 顶部附近加入）：

```python
parser.add_argument("--use_ds_captions", action="store_true")
parser.add_argument("--ds_caption_xlsx", type=str, default=None)
```

**新增工具函数**（文件开头加）：

```python
def load_caption_map(xlsx_path):
    import pandas as pd
    cap = pd.read_excel(xlsx_path)
    # 兼容不同表头
    cols = {c.lower(): c for c in cap.columns}
    fn_col = cols.get("filename") or list(cap.columns)[0]
    cp_col = cols.get("caption")  or list(cap.columns)[1]
    mapping = {}
    for _, row in cap.iterrows():
        fn = str(row[fn_col]).strip()
        cp = str(row[cp_col]).strip()
        mapping[fn] = cp
    return mapping
```

**在 `main()` 中加载映射（构造数据集前）**：

```python
caption_map = None
if args.use_ds_captions and args.ds_caption_xlsx:
    caption_map = load_caption_map(args.ds_caption_xlsx)
```

**让数据集类支持逐样本 prompt**
给 `MVTecDataset` / `DreamBoothDataset` 的 `__init__` 增加：

```python
def __init__(..., caption_map=None, use_ds_captions=False, ...):
    self.caption_map = caption_map
    self.use_ds_captions = use_ds_captions
```

在 `__getitem__` 中将：

```python
prompt_text = self.instance_prompt
```

替换为：

```python
prompt_text = self.instance_prompt
if self.use_ds_captions and self.caption_map is not None:
    from pathlib import Path
    fname = Path(self.instance_images_path[index % self.num_instance_images]).name
    prompt_text = self.caption_map.get(fname, prompt_text)
```

然后将 `prompt_text` 送入 tokenizer：

```python
example["instance_prompt_ids"] = self.tokenizer(
    prompt_text,
    padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length
).input_ids
```

> 训练阶段的 GPP（`text_noise_scale`）在你代码里已实现：对 `encA` 加噪，增强文本→视觉的多样性而不破坏语义。

### 4.2 修改 `run_train.py`：路径适配 DS-MVTec

把原来指向 MVTec AD 的路径：

```python
instance_data_dir = f"{base_dir}/{category}/test/{defect}"
mask_data_dir     = f"{base_dir}/{category}/ground_truth/{defect}"
```

改成 DS-MVTec 的：

```python
instance_data_dir = f"{base_dir}/{category}/image/{defect}"
mask_data_dir     = f"{base_dir}/{category}/mask/{defect}"
```

并在拼接 `train_dreambooth_noise.py` 的命令时，将新增参数透传：

```
--use_ds_captions \
--ds_caption_xlsx "/path/to/Defect_Spectrum/DS-MVTec/captions.xlsx"
```

### 4.3 训练数据的 prompt 清洗（建议）

* **数值 + 离散并列**：保留“厚度 0.1 mm”原文，同时追加“thin/细薄”等离散词（如：`"thin (0.1mm)"`）。
* **位置词多样化**：为同一张图扩写少数同义版本（如 `"on the screw head"`, `"at the top of the screw"`, `"on the thread top"`）。
* **双语增强**：可以把中文和英文并列在同一 prompt（如 `"在螺丝顶部的细划痕 (thin scratch on the screw head)"`），增强跨语言鲁棒性。

> 这些清洗在“caption 读取环节”统一处理即可，不必改网络。

### 4.4 训练命令（例）

```bash
python run_train.py \
  --base_dir /data/Defect_Spectrum/DS-MVTec \
  --output_name ds_magic \
  --text_noise_scale 1.0 \
  --category screw
```

确保 `run_train.py` 内部调用 `train_dreambooth_noise.py` 的命令里附带：

```
--use_ds_captions \
--ds_caption_xlsx "/data/Defect_Spectrum/DS-MVTec/captions.xlsx"
```

---

## 5. 推理改造：自然语言指令 +（可选）自动掩码

### 5.1 修改 `inference.py`：开放 prompt 与自动区域开关

在 argparse 里新增：

```python
parser.add_argument("--prompt", type=str, default="a photo of a defect")
parser.add_argument("--auto_mask_text", type=str, default=None)  # 如 "top of the screw" / "螺丝顶部"
```

在实际调用 inpaint 时，将硬编码 prompt 改为：

```python
imgs = inpaint(
    pipe, normal_img,
    prompt=args.prompt,             # ← 改这里
    mask=final_mask_pil, n_samples=1, device=device,
    blur_factor=args.blur_factor,
    anomaly_strength=a_strength,
    anomaly_stop_step=args.anomaly_stop_step
)
```

### 5.2 文本→区域掩码（V1：几何条带，稳定、无需新模型）

**思路**：当用户没提供掩码、但传入了 `--auto_mask_text "top of the screw"`，我们用**前景掩码**（物体区域）裁出一块**条带**作为 inpaint 掩码：

* `top/head/顶部/上部` → 取前景掩码上部 25%
* `bottom/底部/下部` → 下部 25%
* `left/右` → 左/右 25%
* 否则 → 中部 30%

> 先验条带与前景掩码相与；如开启 `--CAMA`，CAMA 再做语义对齐与裁边，确保落在物体上。

**函数示例**（放在 `inference.py` 顶部或同文件中）：

```python
def region2mask_from_obj(obj_mask_np, text, band=0.25):
    import numpy as np
    H, W = obj_mask_np.shape
    band = max(0.1, min(0.5, band))
    t = (text or "").lower()

    if any(k in t for k in ["top","head","顶部","上部","thread_top"]):
        rr = slice(0, int(H*band));            cc = slice(0, W)
    elif any(k in t for k in ["bottom","底部","下部"]):
        rr = slice(int(H*(1-band)), H);       cc = slice(0, W)
    elif any(k in t for k in ["left","左侧"]):
        rr = slice(0, H);                     cc = slice(0, int(W*band))
    elif any(k in t for k in ["right","右侧"]):
        rr = slice(0, H);                     cc = slice(int(W*(1-band)), W)
    else:  # middle
        rr = slice(int(H*(0.5-band/2)), int(H*(0.5+band/2)))
        cc = slice(int(W*(0.5-band/2)), int(W*(0.5+band/2)))

    tmp = np.zeros_like(obj_mask_np, np.uint8); tmp[rr, cc] = 1
    return (tmp & (obj_mask_np > 0).astype(np.uint8))
```

**集成位置**：在你当前生成 `final_mask` 之前插入（当用户没给真实缺陷掩码时）：

```python
if args.auto_mask_text and obj_mask is not None:
    auto_m = region2mask_from_obj(obj_mask, args.auto_mask_text, band=0.25)
    if auto_m.sum() > 0:
        final_mask = auto_m
    else:
        final_mask = obj_mask  # 兜底
```

> 然后照常走 CAMA + inpaint。MGNI 会在掩码区域注入额外噪声，增加局部纹理的“真·随机”。

### 5.3 文本→区域掩码（V2：文本分割/短语分割，可选增强）

如需更“智能”的定位，可接入任意文本分割模型（例如短语分割器），将“top of the screw / screw head / near thread top”这类短语直接转为像素掩码。建议：

* 先对整图做文本分割得到概率图；
* 与**前景掩码**相与，去除背景误检；
* 若置信偏低/面积太小，回退到 V1 几何条带先验。

> V2 完全可插拔，不影响主干；**先用 V1 跑通上线**，再考虑 V2。

### 5.4 推理命令示例（中文）

**“在螺丝的顶部创建细划痕，轻微”**：

```bash
python inference.py \
  --model_ckpt_root /checkpoints/ds_magic_noise_1.0 \
  --ddim_scheduler_root ~/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-inpainting/snapshots/<hash> \
  --categories screw --dataset_type mvtec \
  --defect_json /path/to/defect_classification.json \
  --match_json  /path/to/matching_result.json \
  --normal_masks ./obj_foreground_mask \
  --base_dir /data/Defect_Spectrum/DS-MVTec \
  --output_name ./gen_out \
  --auto_mask_text "top of the screw" \
  --prompt "在螺丝顶部生成一处细划痕（thin scratch on the screw head, slight severity）" \
  --text_noise_scale 0.3 \
  --anomaly_strength_min 0.2 --anomaly_strength_max 0.5 \
  --anomaly_stop_step 20 \
  --CAMA
```

> 语义（**画什么**）靠 prompt；力度（**画多狠**）优先用 `anomaly_strength_*`；`text_noise_scale` 控多样性；`anomaly_stop_step` 控采样步数上限。

---

## 6. 评价指标与自测

* **定位准确（自动掩码）**：自动掩码 vs. 人工约定条带（IoU）或 vs. 矢量部位标注（若有）。
* **掩码内一致性**：生成缺陷是否严格落在掩码内（inpaint + DDIM + CAMA 应保证）。
* **画质与多样性**：KID / IC-LPIPS（仓库有评测脚本骨架），补充 FID 亦可。
* **下游任务收益**：用合成缺陷扩充训练集，测试分割/检测/分类在 MVTec/DS-MVTec 上的 AUROC/mIoU/F1 提升。

**快检 checklist**：

1. 同一张“正常图”+ 不同自然语言（薄/厚、轻微/严重）→ 局部风格变化明显且合逻辑；
2. 同一自然语言 + 不同 `anomaly_strength` → 严重程度按幅度递增；
3. “顶部/底部/左/右/中部”定位正确，且不越界（受前景掩码/CAMA 限制）。

---

## 7. 常见问题与调参建议

* **“0.1mm 这种硬数值”难精确**：训练时把数值+离散词并列（`0.1mm (thin)`）；推理也写“thin/slight”。
* **背景被污染**：降低 `anomaly_strength_max`，或适当减小条带带宽；确保前景掩码质量；开启 `--CAMA`。
* **纹理模式单一**：提高 `text_noise_scale`（训练 0.5–1.0；推理 0–0.5），或打开 `use_random_mask`（MGNI 的随机子掩码）。
* **位置偏差**：V1 条带与前景相与 + `--CAMA`；必要时上 V2 文本分割增强；或在 `captions` 中更多样地写位置同义词。
* **收敛/过拟合**：仅训 UNet，学习率小一点（5e-6 左右），步数卡在 3k–5k；多类别/子类混训时注意 batch 均衡。

---

## 8. 最小补丁（可直接对照修改）

> 下面是“最小可用改动”，你按块修改即可。若你的文件已有部分变动，请按逻辑合并。

### 8.1 `train_dreambooth_noise.py`（核心片段）

**新增参数与读取：**

```python
# argparse
parser.add_argument("--use_ds_captions", action="store_true")
parser.add_argument("--ds_caption_xlsx", type=str, default=None)

def load_caption_map(xlsx_path):
    import pandas as pd
    cap = pd.read_excel(xlsx_path)
    cols = {c.lower(): c for c in cap.columns}
    fn_col = cols.get("filename") or list(cap.columns)[0]
    cp_col = cols.get("caption")  or list(cap.columns)[1]
    return {str(r[fn_col]).strip(): str(r[cp_col]).strip() for _, r in cap.iterrows()}

# main() 内
caption_map = None
if args.use_ds_captions and args.ds_caption_xlsx:
    caption_map = load_caption_map(args.ds_caption_xlsx)
```

**数据集类接入**（以 `MVTecDataset` 为例，另一个 `DreamBoothDataset` 做同样改动）：

```python
class MVTecDataset(Dataset):
    def __init__(..., caption_map=None, use_ds_captions=False, ...):
        ...
        self.caption_map = caption_map
        self.use_ds_captions = use_ds_captions
        ...

    def __getitem__(self, index):
        ...
        # 逐样本 prompt
        prompt_text = self.instance_prompt
        if self.use_ds_captions and self.caption_map is not None:
            from pathlib import Path
            fname = Path(self.instance_images_path[index % self.num_instance_images]).name
            prompt_text = self.caption_map.get(fname, prompt_text)

        example["instance_prompt_ids"] = self.tokenizer(
            prompt_text, padding="do_not_pad", truncation=True,
            max_length=self.tokenizer.model_max_length
        ).input_ids
        ...
```

**在构造数据集时传参**（`main()` 创建 `train_dataset` 的地方）：

```python
train_dataset = MVTecDataset(
    instance_data_root=args.instance_data_dir,
    mask_data_root=args.mask_data_dir,
    instance_prompt=args.instance_prompt,
    class_data_root=args.class_data_dir if args.with_prior_preservation else None,
    class_prompt=args.class_prompt,
    tokenizer=tokenizer,
    size=args.resolution,
    center_crop=args.center_crop,
    # 新增
    caption_map=caption_map,
    use_ds_captions=args.use_ds_captions,
)
```

### 8.2 `run_train.py`（路径适配 + 透传参数）

**替换路径：**

```python
instance_data_dir = f"{base_dir}/{category}/image/{defect}"
class_data_dir    = instance_data_dir
mask_data_dir     = f"{base_dir}/{category}/mask/{defect}"
```

**在命令字符串中追加：**

```
--use_ds_captions \
--ds_caption_xlsx="/data/Defect_Spectrum/DS-MVTec/captions.xlsx"
```

### 8.3 `inference.py`（prompt 开放 + 自动区域）

**新增 argparse 参数：**

```python
parser.add_argument("--prompt", type=str, default="a photo of a defect")
parser.add_argument("--auto_mask_text", type=str, default=None)
```

**新增工具函数（放文件顶部或合适位置）：**

```python
def region2mask_from_obj(obj_mask_np, text, band=0.25):
    import numpy as np
    H, W = obj_mask_np.shape
    band = max(0.1, min(0.5, band))
    t = (text or "").lower()

    if any(k in t for k in ["top","head","顶部","上部","thread_top"]):
        rr = slice(0, int(H*band));            cc = slice(0, W)
    elif any(k in t for k in ["bottom","底部","下部"]):
        rr = slice(int(H*(1-band)), H);       cc = slice(0, W)
    elif any(k in t for k in ["left","左侧"]):
        rr = slice(0, H);                     cc = slice(0, int(W*band))
    elif any(k in t for k in ["right","右侧"]):
        rr = slice(0, H);                     cc = slice(int(W*(1-band)), W)
    else:  # middle
        rr = slice(int(H*(0.5-band/2)), int(H*(0.5+band/2)))
        cc = slice(int(W*(0.5-band/2)), int(W*(0.5+band/2)))

    tmp = np.zeros_like(obj_mask_np, np.uint8); tmp[rr, cc] = 1
    return (tmp & (obj_mask_np > 0).astype(np.uint8))
```

**在生成 `final_mask` 之前插入（当未显式提供缺陷掩码时）**：

```python
if args.auto_mask_text and obj_mask is not None:
    auto_m = region2mask_from_obj(obj_mask, args.auto_mask_text, band=0.25)
    final_mask = auto_m if auto_m.sum() > 0 else obj_mask
```

**替换 inpaint 调用的 prompt**：

```python
imgs = inpaint(pipe, normal_img, prompt=args.prompt, ... )
```

---

## 9. 提示词写作（实践指南）

* **结构模板**：`[位置] + [缺陷类型] + [形态/纹理] + [强度/厚度] (+可选中英并列)`

  * 例：`"请在螺丝的顶部生成一处细划痕（thin scratch on the screw head, slight severity, thickness 0.1mm）"`
* **位置词**：`top/head/顶部/上部`，`thread_top/螺纹上缘`，`bottom/底部`，`left/right`，`center/middle`
* **类型**：`scratch`（划痕）, `crack`（裂纹）, `corrosion/rust`（腐蚀/锈）, `dent`（凹痕）, `wear/abrasion`（磨损）, `stain`（污渍）…
* **形态/纹理**：`thin/fine/line-like`, `wide`, `rough`, `granular`, `irregular`, `radial`, `concentric` …
* **强弱/厚度**：`slight/moderate/severe`, `thin/medium/thick`, 数值配合离散词（`0.1mm (thin)`）。

---

## 10. 进阶与扩展

* **V2 自动掩码**：接入文本分割器（短语分割），phrase→mask；与前景掩码相与，失败则回退 V1。
* **多类别联合训练**：`screw + metal_nut + transistor ...`，用 caption 的类别词与部位词增强泛化。
* **半自动标注**：生成缺陷 + CAMA 对齐的 mask 作为伪标注，反馈到缺陷检测/分割器训练（闭环提升）。

---

## 11. 收尾：从零到一的执行清单

1. **准备数据**：下载 Defect\_Spectrum/DS-MVTec，确认 `image/`、`mask/` 与 `captions.xlsx` 完整；用 U²-Net 预生成**前景掩码**。
2. **应用代码补丁**：完成 8.1/8.2/8.3 三处改动。
3. **训练**：跑 `run_train.py`（`--use_ds_captions --ds_caption_xlsx`），产出 `model/ds_magic_noise_1.0/<category>/<defect>/`。
4. **推理**：使用 `--prompt` 输入自然语言、`--auto_mask_text` 指定位置词，开启 `--CAMA`。
5. **验收**：按 6 节指标自测；根据 7 节建议微调超参与 prompt。

完成以上，你就拥有了\*\*“一句话点名缺陷 + 指定部位自动定位 + 只在该区域生成”\*\*的工业缺陷生成系统。
