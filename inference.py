import argparse
import os
import random
import torch
import re
from PIL import Image
import json
from magic_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_magic import \
    StableDiffusionInpaintPipeline_magic
import types
import numpy as np
import cv2



def parse_args():
    parser = argparse.ArgumentParser(description="Inference-time mask-alignment augmentation.")  
    parser.add_argument("--defect_json", type=str,
                        default="./CAMA_json_file/defect_classification.json")  
    parser.add_argument("--match_json", type=str, default='./CAMA_json_file/matching_result.json',
                        help="Path to JSON containing best matching pixels.")  
    parser.add_argument("--model_ckpt_root", type=str, 
                        help="Root directory of DreamBooth checkpoints.")
    parser.add_argument("--categories", type=str, nargs='+', default=None,
                        help="MVTEC categories (e.g. bottle cable capsule ...).")
    parser.add_argument("--blur_factor", type=int, default=0,
                        help="Gaussian-blur radius applied to the binary mask.")
    parser.add_argument("--text_noise_scale", type=float, default=1.0,
                        help="Std-dev of Gaussian noise added to prompt embeddings (0 = none).") 
    parser.add_argument('--output_name', type=str, default='./generated_image',
                        help="Base directory for saving outputs.")
    parser.add_argument("--anomaly_strength_min", type=float, default=0.0,
                        help="Minimum anomaly_strength fed to the pipeline.")
    parser.add_argument("--anomaly_strength_max", type=float, default=0.6,
                        help="Maximum anomaly_strength fed to the pipeline.")
    parser.add_argument("--anomaly_stop_step", type=int, default=20,
                        help="Apply anomaly_strength only for the first <N> steps.")  
    parser.add_argument("--normal_masks", type=str, default="./normal_masks",
                        help="Folder containing object masks for normal images.")
    parser.add_argument("--mask_dir", type=str,
                        help="Directory that stores pre-generated binary masks.")
    parser.add_argument("--base_dir", type=str, default="./mvtecad",
                        help="Directory that stores mvtecad datasets.")
    parser.add_argument("--CAMA", action="store_true",
                        help="Enable mask-alignment (CAMA). Otherwise use the mask as-is.")  
    parser.add_argument("--use_random_mask", action="store_true",
                        help="Let the pipeline draw a random convex hole in the mask.")  
    return parser.parse_args()


args = parse_args()


def extract_number_from_filename(filename):
    """Return the first integer appearing in a file name (or +∞ if none)."""
    m = re.search(r'\d+', filename)
    return int(m.group()) if m else float('inf')


def monkey_patch_encode_prompt(pipe):
    """Inject text-embedding noise without touching diffusers’ source code."""
    old_encode_prompt = pipe.encode_prompt

    def new_encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None,
        clip_skip=None,
    ):
        prompt_embeds, negative_embeds = old_encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=clip_skip
        )
        # add Gaussian noise to conditional embeddings
        if getattr(self, "text_noise_scale", 0.0) > 0.0:
            scale = self.text_noise_scale
            if do_classifier_free_guidance:
                half = prompt_embeds.shape[0] // 2
                uncond, cond = prompt_embeds[:half], prompt_embeds[half:]
                cond += torch.randn_like(cond) * scale
                prompt_embeds = torch.cat([uncond, cond], dim=0)
            else:  # no guidance
                prompt_embeds += torch.randn_like(prompt_embeds) * scale
        return prompt_embeds, negative_embeds

    pipe.encode_prompt = types.MethodType(new_encode_prompt, pipe)


def inpaint(pipe, image, prompt, mask=None, n_samples=4, device='cuda',
            blur_factor=0, anomaly_strength=0.0, anomaly_stop_step=999999):
    """Run Stable-Diffusion inpainting with anomaly_strength control."""
    from PIL import Image as PilImage

    if isinstance(image, str):
        image_pil = PilImage.open(image).convert('RGB')
    elif isinstance(image, PilImage.Image):
        image_pil = image.convert('RGB') if image.mode != 'RGB' else image
    else:
        raise ValueError("image must be path or PIL.Image")

    if isinstance(mask, str):
        mask_image = PilImage.open(mask).convert('RGB')
    elif isinstance(mask, PilImage.Image):
        mask_image = mask.convert('RGB') if mask.mode != 'RGB' else mask
    else:
        raise ValueError("mask must be path or PIL.Image")

    blurred_mask = pipe.mask_processor.blur(mask_image, blur_factor=blur_factor)

    images = pipe(
        prompt=[prompt] * n_samples,
        image=image_pil,
        mask_image=blurred_mask,
        anomaly_strength=anomaly_strength,
        anomaly_stop_step=anomaly_stop_step,
        use_random_mask=args.use_random_mask,
    ).images
    return images


def get_random_image(image_dir):
    """Pick a random RGB image inside a directory."""
    imgs = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not imgs:
        raise ValueError(f"No images in {image_dir}")
    return os.path.join(image_dir, random.choice(imgs))


def load_object_mask(category, normal_image_path, normal_masks_dir):
    """
    Try per-image SAM mask:   <root>/<cat>/train/good/<name>_mask.png (or _mask_0.png)
    Fallback category mask:  <root>/<cat>/mask.png
    Return a binary (H,W) uint8 array or None on failure.
    """
    cat_dir = os.path.join(normal_masks_dir, category)
    train_dir = os.path.join(cat_dir, "train", "good")
    base = os.path.splitext(os.path.basename(normal_image_path))[0]

    if os.path.exists(train_dir):                                   # per-image mask
        mask_fname = f"{base}_mask.png" if args.normal_masks.endswith('auto_sam') \
                     else f"{base}_mask_0.png"
        mask_path = os.path.join(train_dir, mask_fname)
        if os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return (m > 127).astype(np.uint8) if m is not None else None
    # fallback single representative mask
    rep_path = os.path.join(cat_dir, "mask.png")
    if os.path.exists(rep_path):
        m = cv2.imread(rep_path, cv2.IMREAD_GRAYSCALE)
        return (m > 127).astype(np.uint8) if m is not None else None
    return None

def debug_save_masks(original_mask_bin, min_x, min_y, max_x, max_y,
                     shifted_mask_bin, debug_save_path):
    """
    Save side-by-side view:
      • left  = original mask (white) with red bbox
      • right = shifted / aligned mask
    """
    H, W = original_mask_bin.shape
    left = np.zeros((H, W, 3), np.uint8)
    left[original_mask_bin > 0] = (255, 255, 255)
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(original_mask_bin, 8)
    for lbl in range(1, n_lbl):
        x, y, bw, bh, _ = stats[lbl]
        if bw and bh:
            cv2.rectangle(left, (x, y), (x + bw - 1, y + bh - 1), (0, 0, 255), 2)

    right = np.zeros((H, W, 3), np.uint8)
    right[shifted_mask_bin > 0] = (255, 255, 255)
    cv2.imwrite(debug_save_path, np.concatenate([left, right], axis=1))

###############################################################################
# 3) CAMA: Context-Aware Mask Alignment (returns best_x, best_y, is_shifted)
###############################################################################
def CAMA(
    class_val,             # 0 = keep only overlap, 1 = use full mask
    code_mask_bin,         # (H,W) binary mask to be aligned
    obj_mask_np,           # (H,W) object mask from SAM / template
    normal_image_path,     # current normal image file
    category,
    defect_class,
    defect_data,           # first JSON (class_val)
    match_data,            # second JSON (best_x / best_y)
    debug_save_dir=None,
    debug_name=None
):
    """
    For every connected component, translate it so that its centroid matches
    a randomly selected target (best_x, best_y) from matching_json for the
    same normal image. If the target already lies on the mask → no shift.
    Return (final_mask, best_x, best_y, is_shifted).
    """
    H, W = code_mask_bin.shape
    # ------- 1) choose target (best_x, best_y) in 480-space -------
    base_normal = os.path.basename(normal_image_path)
    cand = [(item["best_x"], item["best_y"])
            for item in match_data.get(category, {}).get(defect_class, [])
            if item["normal_img"] == base_normal]

    if not cand:   # no entry found → disable shift
        fallback = cv2.bitwise_and(code_mask_bin, obj_mask_np) if class_val == 0 else code_mask_bin
        if debug_save_dir and debug_name:            
            debug_save_masks(code_mask_bin, 0, 0, 0, 0,
                             fallback,
                             os.path.join(debug_save_dir, f"{debug_name}_fallback.jpg"))
        return fallback, -1, -1, False

    best_x_480, best_y_480 = random.choice(cand)  

    best_x = int(best_x_480 * W / 480.0)
    best_y = int(best_y_480 * H / 480.0)

    # ------- 1-b) NEW: if best point already on the mask → keep mask ------
    if 0 <= best_x < W and 0 <= best_y < H and code_mask_bin[best_y, best_x] > 0:
        final_keep = cv2.bitwise_and(code_mask_bin, obj_mask_np) if class_val == 0 else code_mask_bin
        if debug_save_dir and debug_name:
            debug_save_masks(code_mask_bin, 0, 0, 0, 0, final_keep,
                             os.path.join(debug_save_dir, f"{debug_name}_nosshift.jpg"))
        return final_keep, best_x, best_y, False

    # ------- 2) translate every connected component -------
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(code_mask_bin, 8)
    shifted = np.zeros_like(code_mask_bin, np.uint8)

    for lbl in range(1, n_lbl):
        x, y, bw, bh, _ = stats[lbl]
        if not bw or not bh:
            continue
        crop = (lbl_map[y:y + bh, x:x + bw] == lbl).astype(np.uint8)

        cx, cy = x + bw // 2, y + bh // 2           # current centroid
        tx, ty = best_x - bw // 2, best_y - bh // 2  # top-left after shift
        for r in range(bh):
            for c in range(bw):
                if crop[r, c]:
                    yy = ty + r + (cy - (y + bh // 2))
                    xx = tx + c + (cx - (x + bw // 2))
                    if 0 <= xx < W and 0 <= yy < H:
                        shifted[yy, xx] = 1

    # ------- 3) AND with object mask if required -------
    final = cv2.bitwise_and(shifted, obj_mask_np) if class_val == 0 else shifted

    if debug_save_dir and debug_name:
        ys, xs = np.where(code_mask_bin)
        debug_save_masks(code_mask_bin,
                         xs.min() if xs.size else 0, ys.min() if ys.size else 0,
                         xs.max() if xs.size else 0, ys.max() if ys.size else 0,
                         final,
                         os.path.join(debug_save_dir, f"{debug_name}.jpg"))
    return final, best_x, best_y, True

###############################################################################
# 4) main()
###############################################################################
def main():
    with open(args.defect_json, "r", encoding="utf-8") as f:
        defect_data = json.load(f)
    with open(args.match_json, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    base_dir = args.base_dir
    categorys = args.categories or [
        'bottle', 'cable', 'capsule', 'carpet', 'grid', 'hazelnut', 'leather',
        'metal_nut', 'pill', 'screw', 'tile', 'toothbrush', 'transistor',
        'wood', 'zipper'
    ]

    for category in categorys:
        device = "cuda"
        gt_path = os.path.join(base_dir, category, "ground_truth")
        if not os.path.exists(gt_path):
            print(f"[WARN] ground_truth path not found: {gt_path}")
            continue

        defect_classes = os.listdir(gt_path)
        for defect_class in defect_classes:
            if defect_class not in defect_data.get(category, {}):
                print(f"[WARN] {defect_class} not in defect_data[{category}] → skip")
                continue

            class_val = defect_data[category][defect_class]
            print(f"Category={category}, Defect={defect_class}, class_val={class_val}")

            current_mask_root = os.path.join(args.mask_dir, category, defect_class)
            if not os.path.exists(current_mask_root):
                print(f"[WARN] {current_mask_root} not found → skip")
                continue

            ckpt_root = os.path.join(args.model_ckpt_root, category, defect_class)
            if not os.path.exists(ckpt_root):
                print(f"[WARN] checkpoint absent: {ckpt_root} → skip")
                continue

            pipe = StableDiffusionInpaintPipeline_magic.from_pretrained(
                ckpt_root, torch_dtype=torch.float16)
            pipe.scheduler = DDIMScheduler.from_pretrained("./scheduler")
            pipe.text_noise_scale = args.text_noise_scale
            monkey_patch_encode_prompt(pipe)
            pipe.to(device)

            mask_images = sorted(
                (os.path.join(current_mask_root, f) for f in os.listdir(current_mask_root)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))),
                key=lambda x: extract_number_from_filename(os.path.basename(x))
            )

            normal_root = os.path.join(base_dir, category, "train", "good")
            if not os.path.exists(normal_root):
                print(f"[WARN] {normal_root} not found → skip")
                continue

            suffix = (f"{os.path.basename(args.model_ckpt_root)}_noise_{args.text_noise_scale}_"
                      f"anomaly_{args.anomaly_strength_min}_{args.anomaly_strength_max}_"
                      f"dynamic_{args.anomaly_stop_step}_"
                      + ("align" if args.CAMA else "no_align"))
            save_root = os.path.join(args.output_name, suffix, category, defect_class)
            img_dir  = os.path.join(save_root, "image");        os.makedirs(img_dir,  exist_ok=True)
            norm_dir = os.path.join(save_root, "normal");       os.makedirs(norm_dir, exist_ok=True)
            msk_dir  = os.path.join(save_root, "masks");        os.makedirs(msk_dir,  exist_ok=True)
            dbg_dir  = os.path.join(save_root, "debug_mask");   os.makedirs(dbg_dir, exist_ok=True)

            target_n = len(mask_images)
            idx = 0
            while idx < target_n:
                mask_image_path = mask_images[idx]
                normal_image_path = get_random_image(normal_root)
                normal_image = Image.open(normal_image_path)
                normal_w, normal_h = normal_image.size

                mask_np = (np.array(Image.open(mask_image_path).convert('L')) > 127).astype(np.uint8)
                obj_mask = load_object_mask(category, normal_image_path, args.normal_masks)
                if obj_mask is None:
                    obj_mask = np.ones_like(mask_np, np.uint8)
                if obj_mask.shape != mask_np.shape:
                    obj_mask = cv2.resize(obj_mask, mask_np.shape[::-1], interpolation=cv2.INTER_NEAREST)

                if args.CAMA:
                    final_mask, best_x, best_y, is_shifted = CAMA(
                        class_val, mask_np, obj_mask, normal_image_path,
                        category, defect_class, defect_data, match_data,
                        debug_save_dir=dbg_dir, debug_name=f"{idx}")
                else:
                    final_mask, best_x, best_y, is_shifted = mask_np, -1, -1, False

                final_mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))

                anomaly_strength = random.uniform(args.anomaly_strength_min, args.anomaly_strength_max)
                images = inpaint(
                    pipe, normal_image, prompt="a photo of a sks defect",
                    mask=final_mask_pil, n_samples=1, device=device,
                    blur_factor=args.blur_factor,
                    anomaly_strength=anomaly_strength,
                    anomaly_stop_step=args.anomaly_stop_step
                )

                out_name = f"{idx}.jpg"
                images[0].save(os.path.join(img_dir, out_name))
                normal_image.save(os.path.join(norm_dir, out_name))
                final_mask_pil.convert('RGB').save(os.path.join(msk_dir, out_name))
                print(f"Saved {out_name}")
                idx += 1


if __name__ == "__main__":
    main()
