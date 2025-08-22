import argparse, os, random, re, json, types, glob
import torch, numpy as np, cv2
from PIL import Image
from magic_ddim import DDIMScheduler
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint_magic import \
     StableDiffusionInpaintPipeline_magic as StableDiffusionInpaintPipeline_dynamic

import cv2
import numpy as np

def draw_mask_interactively(base_bgr: np.ndarray, brush: int = 18) -> np.ndarray:
    """
    左键绘制、右键/CTRL 擦除、+/- 改画笔、R 清空、S 保存继续、Q 退出。
    返回 0/1 的二值掩码（uint8）。
    """
    assert base_bgr.dtype == np.uint8
    h, w = base_bgr.shape[:2]
    # 这里用 0/255 存，显示更方便；最后返回时再转 0/1
    mask = np.zeros((h, w), np.uint8)

    drawing = [False]; erasing = [False]; r = [brush]

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing[0] = True
        elif event == cv2.EVENT_LBUTTONUP:
            drawing[0] = False
        elif event == cv2.EVENT_RBUTTONDOWN or (flags & cv2.EVENT_FLAG_CTRLKEY):
            erasing[0] = True
        elif event == cv2.EVENT_RBUTTONUP:
            erasing[0] = False
        if drawing[0]:
            cv2.circle(mask, (x, y), r[0], 255, -1)   # 画白色（255）
        if erasing[0]:
            cv2.circle(mask, (x, y), r[0], 0, -1)     # 擦成黑色（0）

    win = "Draw mask  (LMB=paint, RMB/CTRL=erase, +/-=brush, R=reset, S=save, Q=quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        # 叠加显示：把 mask 映射到“红色覆盖层”，三通道都是 uint8
        overlay = np.zeros_like(base_bgr, dtype=np.uint8)
        overlay[..., 2] = mask  # 红色通道 = 掩码
        vis = cv2.addWeighted(base_bgr, 1.0, overlay, 0.35, 0)

        cv2.putText(vis, f"brush={r[0]}  keys: +/- R S Q", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2, cv2.LINE_AA)
        cv2.imshow(win, vis)

        k = cv2.waitKey(10) & 0xFF
        if k in (ord('+'), ord('=')): r[0] = min(128, r[0]+2)
        elif k in (ord('-'), ord('_')): r[0] = max(2, r[0]-2)
        elif k in (ord('r'), ord('R')): mask[:] = 0
        elif k in (ord('s'), ord('S')): break
        elif k in (ord('q'), ord('Q')): mask[:] = 0; break

    cv2.destroyWindow(win)
    # 返回 0/1 二值
    return (mask > 0).astype(np.uint8)


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
    else:  # middle / center
        rr = slice(int(H*(0.5-band/2)), int(H*(0.5+band/2)))
        cc = slice(int(W*(0.5-band/2)), int(W*(0.5+band/2)))

    tmp = np.zeros_like(obj_mask_np, np.uint8); tmp[rr, cc] = 1
    return (tmp & (obj_mask_np > 0).astype(np.uint8))

def parse_prompt_for_mask(prompt: str):
    p = prompt.lower()
    # ① 带宽（thin/hairline 更细；deep/thick 更宽）
    if any(k in p for k in ["hairline", "very thin", "极细", "发丝"]):
        band = 0.12
    elif any(k in p for k in ["thin", "细"]):
        band = 0.18
    elif any(k in p for k in ["thick", "宽", "deep", "深"]):
        band = 0.35
    else:
        band = 0.25
    # ② 位置（top/head/default；如需 left/right 也可扩展）
    where = "top" if any(k in p for k in ["top", "head", "顶部", "头部"]) else "center"
    # ③ 形状（直条 or 斜条；含 along axis）
    along = any(k in p for k in ["along screw axis", "沿", "轴向"])
    return band, where, along

def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-time mask-alignment augmentation.")
    parser.add_argument("--defect_json", required=True)
    parser.add_argument("--match_json",  required=True)
    parser.add_argument("--model_ckpt_root", required=True)
    parser.add_argument("--ddim_scheduler_root", required=True)
    parser.add_argument("--categories", nargs='+', default=None)
    parser.add_argument("--defect_class", type=str, default=None, help="Specify a single defect class to run.")
    parser.add_argument("--blur_factor", type=int, default=8,
                        help="Gaussian blur for mask edges (in pixels).")
    parser.add_argument("--text_noise_scale", type=float, default=0.0)
    parser.add_argument("--single_image", type=str, default=None,
                        help="Path to a specific normal image to use as the base.")
    parser.add_argument("--user_mask", type=str, default=None,
                        help="Path to a user-provided binary mask (white=defect).")
    parser.add_argument("--interactive", action="store_true",
                        help="Draw mask with mouse on top of the base image.")
    parser.add_argument("--and_object", action="store_true",
                        help="Intersect user/interactive mask with object foreground mask if available.")
    parser.add_argument("--output_name", default="./")
    parser.add_argument("--anomaly_strength_min", type=float, default=0.0)
    parser.add_argument("--anomaly_strength_max", type=float, default=0.0)
    parser.add_argument("--anomaly_stop_step", type=int, default=999999)
    parser.add_argument("--normal_masks", default="./normal_masks")
    parser.add_argument("--mask_dir",    default="./Aug_mask_3_shot")
    parser.add_argument("--base_dir",    default="./mvtecad")
    parser.add_argument("--CAMA",        action="store_true")
    parser.add_argument("--use_random_mask", action="store_true")
    parser.add_argument("--prompt", type=str, default="a photo of a defect")
    parser.add_argument("--auto_mask_text", type=str, default=None)
    parser.add_argument("--dataset_type", choices=["mvtec_3d", "mvtec", "ds_mvtec"],
                        default="mvtec_3d",
                        help="mvtec_3d: MVTEC-3D Anomaly, mvtec: MVTEC-AD 2-D, ds_mvtec: Defect Spectrum MVTec")
    return parser.parse_args()

args = parse_args()


def extract_number_from_filename(fname):
    m = re.search(r"\d+", fname)
    return int(m.group()) if m else float("inf")


def monkey_patch_encode_prompt(pipe):
    old_encode = pipe.encode_prompt
    def new_encode_prompt(self, prompt, device, num_images_per_prompt,
                          do_classifier_free_guidance, negative_prompt=None,
                          prompt_embeds=None, negative_prompt_embeds=None,
                          lora_scale=None, clip_skip=None):
        prompt_embeds, neg_embeds = old_encode(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance,
            negative_prompt=negative_prompt, prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale, clip_skip=clip_skip)
        if getattr(self, "text_noise_scale", 0.0) > 0.0:
            s = self.text_noise_scale
            if do_classifier_free_guidance:
                half = prompt_embeds.shape[0] // 2
                uncond, cond = prompt_embeds[:half], prompt_embeds[half:]
                cond += torch.randn_like(cond) * s
                prompt_embeds = torch.cat([uncond, cond], 0)
            else:
                prompt_embeds += torch.randn_like(prompt_embeds) * s
        return prompt_embeds, neg_embeds
    pipe.encode_prompt = types.MethodType(new_encode_prompt, pipe)


def inpaint(pipe, image, prompt, mask=None, n_samples=4, device="cuda",
            blur_factor=0, anomaly_strength=0.0, anomaly_stop_step=999999):
    from PIL import Image as PilImage
    if isinstance(image, str):
        image_pil = PilImage.open(image).convert("RGB")
    else:
        image_pil = image.convert("RGB") if image.mode != "RGB" else image
    if isinstance(mask, str):
        mask_pil = PilImage.open(mask).convert("RGB")
    else:
        mask_pil = mask.convert("RGB") if mask.mode != "RGB" else mask
    mask_pil = pipe.mask_processor.blur(mask_pil, blur_factor=blur_factor)
    return pipe(
        prompt=[prompt]*n_samples, image=image_pil, mask_image=mask_pil,
        anomaly_strength=anomaly_strength, anomaly_stop_step=anomaly_stop_step,
        use_random_mask=args.use_random_mask).images


def get_random_image(img_dir):
    imgs = [f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    if not imgs:
        raise ValueError(f"No images in {img_dir}")
    return os.path.join(img_dir, random.choice(imgs))


def load_object_mask(category, normal_img_path, normal_masks_dir):
    cat_dir = os.path.join(normal_masks_dir, category)
    base = os.path.splitext(os.path.basename(normal_img_path))[0]

    # Candidate directories to search first
    cand_dirs = [
        os.path.join(cat_dir, "train", "masks"),
        os.path.join(cat_dir, "masks"),
        cat_dir,
    ]
    candidates = []
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        candidates.append(os.path.join(d, f"{base}_mask.png"))
        candidates.extend(sorted(glob.glob(os.path.join(d, f"{base}_mask_*.png"))))
        candidates.append(os.path.join(d, f"{base}.png"))
        candidates.append(os.path.join(d, "mask.png"))

    # Remove duplicates and try only existing paths
    seen = set()
    ordered_paths = []
    for p in candidates:
        if p not in seen and os.path.exists(p):
            seen.add(p)
            ordered_paths.append(p)

    for mask_path in ordered_paths:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            return (m > 127).astype(np.uint8)

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
# 3) CAMA: Context-Aware Mask Alignment
###############################################################################
def CAMA(
    class_val,
    code_mask_bin,
    obj_mask_np,
    normal_image_path,
    category,
    defect_class,
    defect_data,
    match_data,
    debug_save_dir=None,
    debug_name=None,
):
    """
    Return (final_mask, first_best_x, first_best_y, is_shifted)
    """
    H, W = code_mask_bin.shape
    base_normal = os.path.basename(normal_image_path)

    # ───────── ① Collect coordinates per defect_img ─────────
    by_defect = {}
    for it in match_data.get(category, {}).get(defect_class, []):
        if it["normal_img"] != base_normal:
            continue
        by_defect.setdefault(it["defect_img"], []).append((it["best_x"], it["best_y"]))

    if not by_defect:
        fallback = cv2.bitwise_and(code_mask_bin, obj_mask_np) if class_val == 0 else code_mask_bin
        if debug_save_dir and debug_name:
            debug_save_masks(
                code_mask_bin, 0, 0, 0, 0, fallback,
                os.path.join(debug_save_dir, f"{debug_name}_fallback.jpg")
            )
        return fallback, -1, -1, False

    # ───────── ② Randomly choose one defect_img ─────────
    chosen_defect, coords_all = random.choice(list(by_defect.items()))

    # ───────── ③ Extract components ─────────
    n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(code_mask_bin, 8)
    comps = list(range(1, n_lbl))  # 0 is background
    if not comps:  # mask is empty
        return code_mask_bin, -1, -1, False

    n_comp = len(comps)
    n_coords = len(coords_all)

    # ───────── ④ Match coordinate list size to the number of components ─────────
    def rand_point_inside(mask):
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            # If obj_mask is empty, sample from the whole image
            return random.randint(0, W - 1), random.randint(0, H - 1)
        idx = random.randrange(ys.size)
        # Return in (x, y) order consistently
        return int(xs[idx] * 480.0 / W), int(ys[idx] * 480.0 / H)

    if n_coords >= n_comp:
        coords = random.sample(coords_all, n_comp)
    else:
        coords = list(coords_all)
        # Fill the shortage with random coordinates
        for _ in range(n_comp - n_coords):
            rx, ry = rand_point_inside(obj_mask_np if class_val == 0 else np.ones_like(obj_mask_np))
            coords.append((rx, ry))

    # Now comps and coords have the same length (n_comp)
    target_pairs = list(zip(comps, coords))

    shifted = np.zeros_like(code_mask_bin, np.uint8)
    for lbl, (best_x_480, best_y_480) in target_pairs:
        best_x = int(best_x_480 * W / 480.0)
        best_y = int(best_y_480 * H / 480.0)

        # If the component already contains the best point, keep it as is
        if 0 <= best_x < W and 0 <= best_y < H and code_mask_bin[best_y, best_x]:
            shifted |= (lbl_map == lbl).astype(np.uint8)
            continue

        x, y, bw, bh, _ = stats[lbl]
        if bw == 0 or bh == 0:
            continue

        crop = (lbl_map[y:y + bh, x:x + bw] == lbl).astype(np.uint8)

        # Simple translation without center correction
        tx = best_x - bw // 2
        ty = best_y - bh // 2
        for r in range(bh):
            for c in range(bw):
                if crop[r, c]:
                    yy = ty + r
                    xx = tx + c
                    if 0 <= xx < W and 0 <= yy < H:
                        shifted[yy, xx] = 1

    final = cv2.bitwise_and(shifted, obj_mask_np) if class_val == 0 else shifted

    if debug_save_dir and debug_name:
        ys, xs = np.where(code_mask_bin)
        debug_save_masks(
            code_mask_bin,
            xs.min() if xs.size else 0, ys.min() if ys.size else 0,
            xs.max() if xs.size else 0, ys.max() if ys.size else 0,
            final,
            os.path.join(debug_save_dir, f"{debug_name}.jpg")
        )

    first_best = coords[0]
    return final, first_best[0], first_best[1], True




def main():
    with open(args.defect_json, "r", encoding="utf-8") as f:
        defect_data = json.load(f)
    with open(args.match_json, "r", encoding="utf-8") as f:
        match_data = json.load(f)

    # ───────── Select default category list ─────────
    if args.dataset_type == "mvtec_3d":
        default_cats = ['bagel', 'cable_gland', 'carrot', 'cookie', 'dowel',
                        'foam', 'peach', 'potato', 'rope', 'tire']
    else:  # mvtec
        default_cats = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                        'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                        'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
    categories = args.categories or default_cats


    for category in categories:
        device = "cuda"

        # ───────── Path differences per dataset ─────────
        if args.dataset_type == "mvtec_3d":
            gt_path     = os.path.join(args.base_dir, category, "test")
            normal_root = os.path.join(args.base_dir, category,
                                       "train", "good", "rgb")
        elif args.dataset_type == "mvtec":
            gt_path     = os.path.join(args.base_dir, category, "ground_truth")
            normal_root = os.path.join(args.base_dir, category,
                                       "train", "good")
        else:  # ds_mvtec
            gt_path     = os.path.join(args.base_dir, category, "mask")
            normal_root = os.path.join(args.base_dir, category, "image", "good")

        if not os.path.exists(gt_path):
            print(f"[WARN] ground_truth path not found: {gt_path}")
            continue
        if not os.path.exists(normal_root):
            print(f"[WARN] normal image path not found: {normal_root}")
            continue

        # Determine defect classes to run
        all_defect_classes = [d for d in os.listdir(gt_path)
                              if os.path.isdir(os.path.join(gt_path, d)) and d != "good"]
        
        defect_classes_to_run = []
        if args.single_image or args.interactive or args.user_mask:
            if args.defect_class:
                if args.defect_class in all_defect_classes:
                    defect_classes_to_run.append(args.defect_class)
                else:
                    print(f"[ERROR] --defect_class '{args.defect_class}' not found for category '{category}'.")
                    continue
            else:
                if all_defect_classes:
                    print(f"[WARN] --defect_class not specified for single-image mode. Using first found: {all_defect_classes[0]}")
                    defect_classes_to_run.append(all_defect_classes[0])
                else:
                    print(f"[ERROR] No defect classes found for category {category}.")
                    continue
        else:
            defect_classes_to_run = all_defect_classes

        for defect_class in defect_classes_to_run:
            if defect_class not in defect_data.get(category, {}):
                print(f"[WARN] {defect_class} not in defect_json → skip")
                continue
            
            class_val = defect_data[category][defect_class]
            print(f"Category={category}, Defect={defect_class}, class_val={class_val}")

            ckpt_root = os.path.join(args.model_ckpt_root, category, defect_class)
            if not os.path.exists(ckpt_root):
                print(f"[WARN] checkpoint absent: {ckpt_root} → skip")
                continue

            pipe = StableDiffusionInpaintPipeline_dynamic.from_pretrained(ckpt_root, torch_dtype=torch.float16)
            pipe.scheduler = DDIMScheduler.from_pretrained(args.ddim_scheduler_root)
            pipe.text_noise_scale = args.text_noise_scale
            monkey_patch_encode_prompt(pipe)
            pipe.to(device)

            # --- SINGLE/INTERACTIVE MODE ---
            if args.single_image or args.interactive or args.user_mask:
                print(f"Running in single-generation mode for {category}/{defect_class}...")
                
                if args.single_image and not os.path.exists(args.single_image):
                    raise FileNotFoundError(f"--single_image path not found: {args.single_image}")
                
                normal_img_path = args.single_image if args.single_image else get_random_image(normal_root)
                normal_img = Image.open(normal_img_path).convert("RGB")
                H, W = normal_img.size

                final_mask = None
                if args.interactive:
                    base_bgr = cv2.cvtColor(np.array(normal_img), cv2.COLOR_RGB2BGR)
                    final_mask = draw_mask_interactively(base_bgr)
                    if final_mask.sum() == 0:
                        print("Interactive mask cancelled or empty. Exiting.")
                        return
                elif args.user_mask:
                    if not os.path.exists(args.user_mask):
                        raise FileNotFoundError(f"--user_mask path not found: {args.user_mask}")
                    m = Image.open(args.user_mask).resize((W, H), Image.NEAREST)
                    a = np.array(m.convert("L")) if m.mode in ("L", "P") else np.array(m.convert("RGB")).max(axis=2)
                    final_mask = (a > 127).astype(np.uint8)

                if final_mask is not None:
                    obj_mask = load_object_mask(category, normal_img_path, args.normal_masks)
                    if obj_mask is not None:
                        if obj_mask.shape[0] != H or obj_mask.shape[1] != W:
                            obj_mask = cv2.resize(obj_mask, (W, H), interpolation=cv2.INTER_NEAREST)
                        if args.and_object:
                            final_mask = cv2.bitwise_and(final_mask, obj_mask)
                    
                    if args.blur_factor > 0:
                        k = max(1, int(args.blur_factor // 2) * 2 + 1)
                        final_mask = (cv2.GaussianBlur(final_mask * 255, (k, k), 0) > 127).astype(np.uint8)
                else:
                    print("[ERROR] No mask generated in single/interactive mode.")
                    return

                final_mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))
                a_strength = random.uniform(args.anomaly_strength_min, args.anomaly_strength_max)
                
                imgs = inpaint(pipe, normal_img, prompt=args.prompt, mask=final_mask_pil, n_samples=1, device=device,
                               blur_factor=0,  # Blur already applied by us
                               anomaly_strength=a_strength, anomaly_stop_step=args.anomaly_stop_step)

                suffix = "interactive" if args.interactive else "user_mask"
                save_root = os.path.join(args.output_name, suffix, category, defect_class)
                os.makedirs(save_root, exist_ok=True)
                
                out_base = os.path.splitext(os.path.basename(normal_img_path))[0]
                out_name = f"{out_base}_result.jpg"
                imgs[0].save(os.path.join(save_root, out_name))
                final_mask_pil.convert("RGB").save(os.path.join(save_root, f"{out_base}_mask.jpg"))
                normal_img.save(os.path.join(save_root, f"{out_base}_normal.jpg"))
                print(f"Saved single generation to {os.path.join(save_root, out_name)}")
                return # Exit after single run

            # --- BATCH MODE (original logic) ---
            else:
                candidates = [
                    os.path.join(args.mask_dir, category, "rgb_mask", defect_class),
                    os.path.join(args.mask_dir, category, "mask", defect_class),
                    os.path.join(args.mask_dir, category, "rbg_mask", defect_class),
                    os.path.join(args.base_dir, category, "ground_truth", defect_class),
                ]
                mask_root = next((d for d in candidates if os.path.isdir(d) and any(f.lower().endswith(('.png','.jpg','.jpeg')) for f in os.listdir(d))), None)
                if mask_root is None:
                    print(f"[WARN] No mask dir found for {category}/{defect_class}, skipping.")
                    continue

                mask_imgs = sorted(
                    (os.path.join(mask_root, f) for f in os.listdir(mask_root) if f.lower().endswith((".png", ".jpg", ".jpeg"))),
                    key=lambda x: extract_number_from_filename(os.path.basename(x)))

                suffix = (f"noise_{args.text_noise_scale}_anomaly_{args.anomaly_strength_min}_{args.anomaly_strength_max}_"
                          f"dynamic_{args.anomaly_stop_step}_" + ("align" if args.CAMA else "no_align"))
                save_root = os.path.join(args.output_name, suffix, category, defect_class)
                img_dir  = os.makedirs(os.path.join(save_root, "image"), exist_ok=True) or os.path.join(save_root, "image")
                norm_dir = os.makedirs(os.path.join(save_root, "normal"), exist_ok=True) or os.path.join(save_root, "normal")
                msk_dir  = os.makedirs(os.path.join(save_root, "masks"), exist_ok=True) or os.path.join(save_root, "masks")
                dbg_dir  = os.makedirs(os.path.join(save_root, "debug_mask"), exist_ok=True) or os.path.join(save_root, "debug_mask")

                for idx, mask_path in enumerate(mask_imgs):
                    base = os.path.splitext(os.path.basename(mask_path))[0].replace("_mask","")
                    cand = os.path.join(normal_root, f"{base}.png")
                    if not os.path.exists(cand): cand = os.path.join(normal_root, f"{base}.jpg")
                    normal_img_path = cand if os.path.exists(cand) else get_random_image(normal_root)
                    normal_img = Image.open(normal_img_path)
                    H, W = normal_img.size

                    im = Image.open(mask_path)
                    raw = np.array(im.convert("L"))
                    if raw.max() <= 1: mask_np = (raw > 0).astype(np.uint8)
                    else: _, th = cv2.threshold(raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU); mask_np = (th > 0).astype(np.uint8)
                    
                    obj_mask = load_object_mask(category, normal_img_path, args.normal_masks)
                    if obj_mask is None: obj_mask = np.ones((H, W), np.uint8)
                    if obj_mask.shape[0] != H or obj_mask.shape[1] != W: obj_mask = cv2.resize(obj_mask, (W, H), interpolation=cv2.INTER_NEAREST)

                    if mask_np.sum() == 0:
                        band, where, along_axis = parse_prompt_for_mask(args.prompt)
                        mask_np = region2mask_from_obj(obj_mask, where, band=band).astype(np.uint8)

                    mask_np = cv2.bitwise_and(mask_np, obj_mask)
                    band, where, along_axis = parse_prompt_for_mask(args.prompt)
                    band_mask = region2mask_from_obj(obj_mask, where, band=band).astype(np.uint8)
                    inter = cv2.bitwise_and(mask_np, band_mask)
                    overlap_ratio = inter.sum() / (band_mask.sum() + 1e-6)
                    mask_np = band_mask.copy() if inter.sum() == 0 or overlap_ratio < 0.02 else inter
                    
                    cv2.imwrite(os.path.join(dbg_dir, f"{idx}_obj_mask.png"), (obj_mask*255).astype(np.uint8))
                    cv2.imwrite(os.path.join(dbg_dir, f"{idx}_band_mask.png"), (band_mask*255).astype(np.uint8))
                    cv2.imwrite(os.path.join(dbg_dir, f"{idx}_inter.png"), (inter*255).astype(np.uint8))

                    if along_axis:
                        stroke = np.zeros_like(mask_np, np.uint8)
                        ys, xs = np.where(band_mask > 0)
                        if xs.size > 0:
                            n = 1 + int(band < 0.2)
                            for _ in range(n):
                                i = np.random.randint(xs.size)
                                cx, cy = xs[i], ys[i]
                                ax = max(8, int(0.20 * W)); ay = max(2, int(0.02 * H))
                                cv2.ellipse(stroke, (int(cx), int(cy)), (ax, ay), 0, 0, 360, 1, -1)
                        mask_np = cv2.bitwise_and(stroke, obj_mask)
                        if mask_np.sum() == 0 and band_mask.sum() > 0: mask_np = band_mask

                    if args.CAMA:
                        final_mask, *_ = CAMA(class_val, mask_np, obj_mask, normal_img_path, category, defect_class, defect_data, match_data, debug_save_dir=dbg_dir, debug_name=f"{idx}")
                        if final_mask.sum() == 0: final_mask = mask_np
                    else:
                        final_mask = mask_np

                    final_mask_pil = Image.fromarray((final_mask * 255).astype(np.uint8))
                    a_strength = random.uniform(args.anomaly_strength_min, args.anomaly_strength_max)
                    imgs = inpaint(pipe, normal_img, prompt=args.prompt, mask=final_mask_pil, n_samples=1, device=device,
                                   blur_factor=args.blur_factor, anomaly_strength=a_strength, anomaly_stop_step=args.anomaly_stop_step)

                    out = f"{idx}.jpg"
                    imgs[0].save(os.path.join(img_dir,  out))
                    normal_img.save(os.path.join(norm_dir, out))
                    final_mask_pil.convert("RGB").save(os.path.join(msk_dir, out))
                    cv2.imwrite(os.path.join(dbg_dir, f"{idx}_final_mask.png"), (final_mask * 255).astype(np.uint8))
                    if 'raw' in locals():
                        cv2.imwrite(os.path.join(dbg_dir, f"{idx}_raw_mask.png"), raw)
                    print(f"Saved {out}")


if __name__ == "__main__":
    main()
