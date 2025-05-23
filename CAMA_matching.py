import os, gc, time, glob, json, argparse
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
from PIL import Image, ImageDraw
if not hasattr(Image, "LINEAR"):
    Image.LINEAR = Image.Resampling.BILINEAR
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils_correspondence import resize
from model_utils.extractor_sd import load_model, process_features_and_mask
from model_utils.extractor_dino import ViTExtractor
from model_utils.projection_network import AggregationNetwork
from preprocess_map import set_seed

# ------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------
set_seed(42)
torch.backends.cuda.matmul.allow_tf32 = True
device = torch.device("cuda")
num_patches = 60

sd_model, sd_aug = load_model(
    diffusion_ver="v1-5",
    image_size=num_patches * 16,
    num_timesteps=50,
    block_indices=[2, 5, 8, 11],
)
sd_model = sd_model.half().eval().to(device)

aggre_net = AggregationNetwork(
    feature_dims=[640, 1280, 1280, 768], projection_dim=768, device=device
).half().eval()
aggre_net.load_pretrained_weights(torch.load("results_spair/best_856.PTH"))

extractor_vit = ViTExtractor("dinov2_vitb14", stride=14, device="cpu")
extractor_vit.model.eval().requires_grad_(False)

# ------------------------------------------------------------------
# Feature helpers
# ------------------------------------------------------------------
@torch.no_grad()
def get_processed_features(img_pil):
    with torch.cuda.amp.autocast(dtype=torch.float16):
        fsd = process_features_and_mask(
            sd_model,
            sd_aug,
            resize(img_pil, num_patches * 16, True, True),
            mask=False,
            raw=True,
        )
    del fsd["s2"]

    dino_in = resize(img_pil, num_patches * 14, True, True)
    dino_tensor = extractor_vit.preprocess_pil(dino_in).to("cpu")
    fdino = (
        extractor_vit.extract_descriptors(dino_tensor, layer=11, facet="token")
        .permute(0, 1, 3, 2)
        .reshape(1, -1, num_patches, num_patches)
    )
    fdino = fdino.to(device=device, dtype=torch.float16, non_blocking=True)

    desc = torch.cat(
        [
            fsd["s3"].half(),
            F.interpolate(fsd["s4"].half(), (num_patches, num_patches), mode="bilinear", align_corners=False),
            F.interpolate(fsd["s5"].half(), (num_patches, num_patches), mode="bilinear", align_corners=False),
            fdino,
        ],
        dim=1,
    )
    return F.normalize(aggre_net(desc), dim=1)


def get_center_and_ud(mask_np):
    ys, xs = np.where(mask_np > 127)
    if len(xs) == 0:
        return None, None, None, False
    cx, cy = int(round(xs.mean())), int(round(ys.mean()))
    up_pt = (cx, int(ys.min()))
    dn_pt = (cx, int(ys.max()))
    inside = bool(mask_np[cy, cx] > 127)
    return (cx, cy), up_pt, dn_pt, inside


@torch.no_grad()
def sim_map_single_center(feat_def, feat_norm, center, size=480):
    if center is None:
        return None
    cx, cy = center
    feat_def_up = F.interpolate(feat_def, (size, size), mode="bilinear", align_corners=False)
    feat_norm_up = F.interpolate(feat_norm, (size, size), mode="bilinear", align_corners=False)
    vec = feat_def_up[0, :, cy, cx].view(1, -1, 1, 1)
    return F.cosine_similarity(vec, feat_norm_up, dim=1)[0]


# ------------------------------------------------------------------
# Main JSON generation routine
# ------------------------------------------------------------------
def generate_correspondence_json(
    mvtecad_dir,
    categories,
    img_size=480,
    out_dir="output",
    out_json_name="matching.json",
    viz_dir_name="matched_visuals_line",
    mask_root="./normal_masks",
):
    os.makedirs(out_dir, exist_ok=True)
    viz_dir = os.path.join(out_dir, viz_dir_name)
    os.makedirs(viz_dir, exist_ok=True)
    line_dir = os.path.join(out_dir, viz_dir_name + "_line")
    os.makedirs(line_dir, exist_ok=True)

    result, outside_list = {}, []

    for cat in categories:
        cat_dir = os.path.join(mvtecad_dir, cat)
        test_dir, gt_root = os.path.join(cat_dir, "test"), os.path.join(cat_dir, "ground_truth")
        norm_root = os.path.join(cat_dir, "train", "good")
        if not os.path.exists(test_dir):
            continue

        defect_classes = sorted(
            d for d in os.listdir(test_dir) if d != "good" and os.path.isdir(os.path.join(test_dir, d))
        )
        normals = sorted(glob.glob(os.path.join(norm_root, "*.*")))
        result[cat] = {}

        for dcls in defect_classes:
            dimgs_all = sorted(glob.glob(os.path.join(test_dir, dcls, "*.*")))
            use_cnt = max(1, len(dimgs_all) // 3)
            dimgs_sel = dimgs_all[:use_cnt]

            (
                defect_feats,
                defect_U,
                defect_D,
                defect_C,
                defect_in,
                defect_names,
            ) = ([], [], [], [], [], [])

            for dpath in dimgs_sel:
                base = os.path.splitext(os.path.basename(dpath))[0]
                gt_files = glob.glob(os.path.join(gt_root, dcls, base + "*"))
                if not gt_files:
                    continue
                mask_np = np.array(Image.open(gt_files[0]).convert("L").resize((img_size, img_size), Image.NEAREST))
                center, up_pt, dn_pt, inside = get_center_and_ud(mask_np)
                if center is None:
                    continue
                feat_def = get_processed_features(resize(Image.open(dpath).convert("RGB"), img_size, True, True))
                defect_feats.append(feat_def)
                defect_U.append(up_pt)
                defect_D.append(dn_pt)
                defect_C.append(center)
                defect_in.append(inside)
                defect_names.append(os.path.basename(dpath))
                if not inside:
                    outside_list.append(f"{cat}/{dcls}/{os.path.basename(dpath)}")

            if not defect_feats:
                continue

            result[cat][dcls] = []
            print(f"\n[{cat}/{dcls}] normals={len(normals)}, defects_used={len(defect_feats)}")

            for npath in tqdm(normals, ncols=85, desc=f"{cat}/{dcls}"):
                t0 = time.time()
                base_img = os.path.splitext(os.path.basename(npath))[0]
                obj_mask_path = os.path.join(mask_root, cat, "train", "masks", f"{base_img}.png")
                if not os.path.exists(obj_mask_path):
                    continue
                obj_np = np.array(Image.open(obj_mask_path).convert("L").resize((img_size, img_size), Image.NEAREST))
                obj_mask = torch.from_numpy(obj_np > 127).to(device)
                if obj_mask.sum() == 0:
                    continue
                img_norm = resize(Image.open(npath).convert("RGB"), img_size, True, True)
                feat_norm = get_processed_features(img_norm)

                for fdef, up_pt, dn_pt, ctr, inside, def_name in zip(
                    defect_feats, defect_U, defect_D, defect_C, defect_in, defect_names
                ):
                    sim_up = sim_map_single_center(fdef, feat_norm, up_pt, img_size)
                    sim_dn = sim_map_single_center(fdef, feat_norm, dn_pt, img_size)
                    center_map = sim_map_single_center(fdef, feat_norm, ctr, img_size)
                    if None in (sim_up, sim_dn, center_map):
                        continue

                    up_idx = sim_up.view(-1).argmax().item()
                    dn_idx = sim_dn.view(-1).argmax().item()
                    up_pt_best = (up_idx % img_size, up_idx // img_size)
                    dn_pt_best = (dn_idx % img_size, dn_idx // img_size)

                    if inside:
                        line_np = np.zeros((img_size, img_size), dtype=np.uint8)
                        cv2.line(line_np, up_pt_best, dn_pt_best, 1, thickness=3)
                        line_mask = torch.from_numpy(line_np).to(device, dtype=torch.bool)
                        intersect_mask = line_mask & obj_mask
                    else:
                        intersect_mask = obj_mask

                    if intersect_mask.sum() == 0:
                        continue
                    masked_sim = center_map.float()
                    masked_sim[~intersect_mask] = -1e9
                    best_idx = masked_sim.view(-1).argmax().item()
                    bx_final, by_final = best_idx % img_size, best_idx // img_size

                    vis_name = f"{cat}_{dcls}__{base_img}__{os.path.splitext(def_name)[0]}_best.png"
                    ann = img_norm.copy()
                    draw = ImageDraw.Draw(ann)
                    for p in (up_pt_best, dn_pt_best):
                        draw.ellipse([(p[0] - 4, p[1] - 4), (p[0] + 4, p[1] + 4)], fill="blue", outline="blue")
                    draw.ellipse([(bx_final - 5, by_final - 5), (bx_final + 5, by_final + 5)], fill="red", outline="red")
                    if inside:
                        draw.line([up_pt_best, dn_pt_best], fill="green", width=2)
                    ann.save(os.path.join(viz_dir, vis_name))

                    result[cat][dcls].append(
                        {
                            "normal_img": os.path.basename(npath),
                            "defect_img": def_name,
                            "best_x": int(bx_final),
                            "best_y": int(by_final),
                            "up_pt": [int(up_pt_best[0]), int(up_pt_best[1])],
                            "down_pt": [int(dn_pt_best[0]), int(dn_pt_best[1])],
                            "center_inside": bool(inside),
                        }
                    )

                print(f"[Timing] {os.path.basename(npath)}: {time.time() - t0:.3f}s")
                torch.cuda.empty_cache()
                gc.collect()

    def np_encoder(o):
        if isinstance(o, (np.integer, np.floating, np.bool_)):
            return o.item()
        raise TypeError


    best_json = os.path.join(out_dir, out_json_name)

    with open(best_json, "w") as f:
        json.dump(result, f, indent=4, default=np_encoder)

    print(f"\n[Info] Best-point JSON saved to {best_json}")
    print(f"[Info] Visualizations saved to {viz_dir}")
    print(f"[Info] Line visualizations saved to {line_dir}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate correspondence JSON for MVTec-AD categories.")
    parser.add_argument("--mvtecad_dir", type=str, default="./mvtecad", help="Path to the MVTec-AD dataset root.")
    parser.add_argument(
        "--categories",
        type=str,
        required=True,
        help="Comma-separated list of categories, e.g. 'bottle,screw' (no spaces).",
    )
    parser.add_argument("--img_size", type=int, default=480, help="Image resolution for processing.")
    parser.add_argument("--out_dir", type=str, default="./output", help="Directory to save results.")
    parser.add_argument(
        "--out_json_name",
        type=str,
        default="matching_result_center_all_object_1.json",
        help="Filename for the resulting JSON.",
    )
    parser.add_argument(
        "--viz_dir_name",
        type=str,
        default="matched_visuals_line",
        help="Subdirectory name for visualization outputs.",
    )
    parser.add_argument(
        "--mask_root",
        type=str,
        default="./normal_masks",
        help="Root directory containing foreground masks for normal images.",
    )

    args = parser.parse_args()

    cat_list = [c.strip() for c in args.categories.split(",") if c.strip()]
    generate_correspondence_json(
        mvtecad_dir=args.mvtecad_dir,
        categories=cat_list,
        img_size=args.img_size,
        out_dir=args.out_dir,
        out_json_name=args.out_json_name,
        viz_dir_name=args.viz_dir_name,
        mask_root=args.mask_root,
    )
