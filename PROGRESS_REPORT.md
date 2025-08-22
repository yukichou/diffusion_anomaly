# 项目调试与状态报告

本文档记录了对 `MAGIC-Anomaly-generation` 项目进行的一系列调试、功能增强的步骤，以及截至目前的最终状态。

## 1. 初始问题与修复

项目在启动初期遇到了多个环境和代码层面的错误，导致无法运行。

### 1.1. 模块导入错误

*   **问题**: `inference.py` 脚本中存在对已重命名或路径不正确的内部模块的引用。
*   **解决方案**:
    1.  修复 `my_ddim_dynamic_random_mask` 导入：将 `from my_ddim_dynamic_random_mask import DDIMScheduler` 修正为 `from magic_ddim import DDIMScheduler`。
    2.  修复 `pipeline_stable_diffusion_inpaint_dynamic_anomaly_strength` 导入：将其修正为 `from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_magic import StableDiffusionInpaintPipeline_magic as StableDiffusionInpaintPipeline_dynamic`，并使用别名以保证兼容性。

### 1.2. 配置文件加载错误

*   **问题**: `FileNotFoundError`，在加载 DDIM 调度器时找不到 `scheduler_config.json`。
*   **解决方案**: 修改了 `magic_ddim.py`，使其能够正确在 `scheduler` 子目录中定位并加载配置文件。

## 2. 功能增强：迈向自然语言驱动的异常生成

在解决了基础的运行问题后，我们为项目增加了一系列核心功能，使其从一个简单的异常生成工具，演变为一个由自然语言驱动、高度灵活的智能生成框架。

### 2.1. 支持 `DS-MVTec` 数据集的逐样本标题训练

*   **目标**: 使模型能够针对 `DS-MVTec` 数据集中的每一个缺陷样本，学习其对应的详细文本描述，从而实现更精确的缺陷生成。
*   **实现**:
    *   在环境中安装了 `pandas` 和 `openpyxl` 库，用于读取 `captions.xlsx` 文件。
    *   修改了 `train_dreambooth_noise.py`，增加了在训练流程中加载和处理 Excel 文件中样本标题的逻辑。
    *   相应地更新了 `run_train.py` 脚本，以在启动训练时传递必要的参数。

### 2.2. 兼容 `MVTec-AD` 标准数据集

*   **目标**: 扩展项目的适用性，使其不仅支持自定义的 `DS-MVTec` 数据集结构，也完全兼容更为主流的 `MVTec-AD` 数据集。
*   **实现**:
    *   在 `run_train.py` 中引入了 `--dataset_type` 参数（可选 `mvtec` 或 `ds_mvtec`）。
    *   脚本内部使用条件逻辑，根据所选的数据集类型，动态构建正确的数据和掩码目录路径。
    *   更新了 `defect_classification.json` 文件，使其包含 `MVTec-AD` `screw` 类别的正确缺陷名称。

### 2.3. 实现基于语义的动态掩码生成 (Semantic Mask Generation)

这是项目最具创新性的功能，使得用户能够通过自然语言描述来直接控制生成缺陷的位置、形状和大小。

*   **目标**: 让 `inference.py` 脚本摆脱对预定义掩码文件的依赖，根据用户输入的文本提示（Prompt）动态生成用于修复（Inpainting）的掩码。
*   **实现**:
    1.  **鲁棒的掩码处理**: 在实现动态生成前，首先增强了 `inference.py` 对静态掩码的处理能力，包括：
        *   支持多种掩码格式（灰度、RGB、RGBA）。
        *   自动检测并翻转掩码极性（黑底白字 vs 白底黑字）。
        *   在多个预设目录 (`mask/`, `ground_truth/` 等) 中搜索掩码文件。
    2.  **语义解析函数**: 在 `inference.py` 中创建了 `parse_prompt_for_mask` 函数。该函数负责解析文本提示中的关键词。
        *   **尺寸**: "thin", "thick", "细", "宽" 等词决定了缺陷区域的宽度。
        *   **位置**: "top", "head", "center", "顶部" 等词决定了缺陷在物体上的大致位置。
        *   **形状**: "along screw axis", "沿轴向" 等词会生成一个与物体主轴对齐的椭圆形掩码，以模拟划痕。
    3.  **动态掩码创建**:
        *   脚本首先加载一个通用的前景掩码 (`obj_foreground_mask`) 来确定物体轮廓。
        *   接着，根据语义解析的结果，在前景轮廓内生成一个符合描述的“条带”或“椭圆”区域。
        *   这个动态生成的区域将作为最终的修复掩码，引导 Stable Diffusion 模型在此处生成缺陷。

## 3. 最终状态

经过上述一系列的开发和调试，项目已经达到一个稳定且功能强大的状态。**核心能力**包括：

*   **双数据集支持**: 可在 `DS-MVTec` 和 `MVTec-AD` 两种数据集上进行训练。
*   **精细化训练**: 支持使用逐样本的详细文本标题进行模型微调。
*   **自然语言驱动的推理**: 能够仅通过一句文本描述（例如 "在螺丝顶部生成一条细长的划痕"），在无须提供任何具体掩码的情况下，智能地生成符合要求的异常样本。

## 4. 最新命令参考

以下是在 `MVTec-AD` 数据集上进行训练和执行语义推理的最新命令。

### 训练命令

```bash
python run_train.py --base_dir ./dataset/mvtecad --output_name output_mvtec_screw --text_noise_scale 1.0 --category screw --dataset_type mvtec
```

### 语义推理命令

```bash
python inference.py --model_ckpt_root ./model/output_mvtec_screw_noise_1.0 --ddim_scheduler_root "C:/Users/Administrator/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-inpainting/snapshots/81a84f49b15956b60b4272a405ad3daef3da4590" --categories screw --dataset_type mvtec --normal_masks "./obj_foreground_mask" --output_name "./gen_out_semantic" --base_dir "./dataset/mvtecad" --prompt "a photo of a screw, with a thin scratch along screw axis at the top"