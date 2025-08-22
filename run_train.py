import os
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Training script for DreamBooth Inpainting.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory (e.g., ./mvtecad)")
    parser.add_argument("--dataset_type", type=str, choices=["mvtec", "ds_mvtec"], default="ds_mvtec", help="Type of the dataset")
    parser.add_argument("--defect_class", type=str, default=None, help="Specific defect class (e.g., color)")
    parser.add_argument("--category", type=str, nargs="+", default=None)
    parser.add_argument("--soft_mask", default=None, action="store_true")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument(
        "--text_noise_scale",
        type=float,
        default=1.0,
        help="Standard deviation of random noise added to encA/encB",
    )
    args = parser.parse_args()

    base_dir = args.base_dir
    categories = args.category if args.category is not None else os.listdir(base_dir)

    for category in categories:
        if args.dataset_type == 'ds_mvtec':
            defect_root = os.path.join(base_dir, category, "image")
            mask_root_template = os.path.join(base_dir, category, "mask", "{defect}")
            instance_dir_template = os.path.join(base_dir, category, "image", "{defect}")
        else: # mvtec
            defect_root = os.path.join(base_dir, category, "test")
            mask_root_template = os.path.join(base_dir, category, "ground_truth", "{defect}")
            instance_dir_template = os.path.join(base_dir, category, "test", "{defect}")

        defect_classes = (
            [args.defect_class]
            if args.defect_class
            else [
                d for d in os.listdir(defect_root)
                if os.path.isdir(os.path.join(defect_root, d))
                and d.lower() not in {"good", "ok", "normal"}
            ]
        )

        for defect in defect_classes:
            instance_data_dir = instance_dir_template.format(defect=defect)
            class_data_dir = instance_data_dir
            mask_data_dir = mask_root_template.format(defect=defect)
            output_dir = f"./model/{args.output_name}_noise_{args.text_noise_scale}/{category}/{defect}"
            
            command = (
                f"accelerate launch train_dreambooth_noise.py "
                f'--pretrained_model_name_or_path="stabilityai/stable-diffusion-2-inpainting" '
                f'--instance_data_dir="{instance_data_dir}" '
                f'--class_data_dir="{class_data_dir}" '
                f'--output_dir="{output_dir}" '
                f"--prior_loss_weight=1.0 "
                f'--instance_prompt="a photo of sks defect" '
                f'--class_prompt="a photo of defect" '
                f"--resolution=512 "
                f"--train_batch_size=4 "
                f"--gradient_accumulation_steps=1 "
                f"--learning_rate=5e-6 "
                f'--lr_scheduler="constant" '
                f"--lr_warmup_steps=0 "
                f"--num_class_images=200 "
                f"--max_train_steps=5000 "
                f"--text_noise_scale={args.text_noise_scale} "
                f"--mvtecad "
                f'--mask_data_dir="{mask_data_dir}" '
                f'--center_crop '
            )

            if args.dataset_type == 'ds_mvtec':
                ds_caption_xlsx_path = os.path.join(base_dir, "captions.xlsx")
                command += (
                    f"--use_ds_captions "
                    f'--ds_caption_xlsx="{ds_caption_xlsx_path}" '
                    f'--instance_category="{category}" '
                )

            print(f"Running command for category '{category}', defect '{defect}'.")
            subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()
