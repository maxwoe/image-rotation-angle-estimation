"""
Generate qualitative examples showing good and bad predictions.

Runs CGD + MambaOut Base on COCO test images with a fixed random rotation,
picks the best and worst predictions, and exports individual images
for use in a LaTeX figure.

Usage (from repo root):
    python eval/generate_qualitative_figure.py

Optional arguments:
    --test-dir         Test image directory (default: data/datasets/test_coco)
    --checkpoint       Path to checkpoint file
    --n-good           Number of good examples to show (default: 3)
    --n-bad            Number of bad examples to show (default: 3)
    --seed             Random seed for test rotations (default: 0)
    --output-dir       Output directory for images (default: eval/results/qualitative)
"""

import argparse
import glob
import json
import sys
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import transforms
import timm
import timm.data

# Match train.py precision setting
torch.set_float32_matmul_precision('high')

from rotation_utils import rotate_preserve_content
from model_cgd import CGDAngleEstimation


def circular_distance(pred, true):
    """Circular distance in degrees (0-180 range)."""
    diff = abs(pred - true) % 360
    return min(diff, 360 - diff)


def load_model(checkpoint_path):
    """Load CGD model from checkpoint."""
    model = CGDAngleEstimation.load_from_checkpoint(
        checkpoint_path,
        test_dir=None,
        test_rotation_range=360.0,
        test_random_seed=0
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda:0")
    return model


def run_inference(model, test_dir, seed, rotation_range=360.0):
    """Run inference on all test images with deterministic rotations."""
    test_dir = str(test_dir)

    # Match model_cgd.py setup(stage="test") file discovery exactly:
    # same extensions, same order (glob.glob, no sorting)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(test_dir, f"*{ext}")))
        image_paths.extend(glob.glob(os.path.join(test_dir, f"*{ext.upper()}")))

    if not image_paths:
        raise ValueError(f"No images found in {test_dir}")

    # Match data_loader.py _generate_test_angles exactly (global np.random state)
    np.random.seed(seed)
    if rotation_range >= 360:
        angles = np.random.uniform(0, 360, len(image_paths))
    else:
        half = rotation_range / 2
        angles = np.random.uniform(-half, half, len(image_paths))

    # Build transform matching test pipeline (data_loader.py _create_transforms)
    image_size = getattr(model.hparams, 'image_size', None)
    try:
        data_config = timm.data.resolve_model_data_config(model.hparams.model_name)
        data_config['crop_pct'] = 1.0  # Match data_loader: use full image, no center crop
        if image_size is not None:
            data_config['input_size'] = (3, image_size, image_size)
        else:
            # Match data_loader: get model's default input size
            tmp_model = timm.create_model(model.hparams.model_name, pretrained=False)
            actual_size = tmp_model.default_cfg.get('input_size', (3, 224, 224))
            data_config['input_size'] = actual_size
            del tmp_model
        transform = timm.data.create_transform(**data_config, is_training=False)
    except Exception:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    device = next(model.parameters()).device

    results = []
    for i, (img_path, angle_applied) in enumerate(zip(image_paths, angles)):
        # Create rotated image (this is what the model sees)
        rotated_pil = rotate_preserve_content(str(img_path), angle_applied)

        # Run inference directly (avoid predict_angle's device bug)
        image_tensor = transform(rotated_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_distributions = model(image_tensor)
            angle_predicted = model.cgd.distribution_to_angle(
                pred_distributions, method=model.inference_method
            ).item()

        error = circular_distance(angle_predicted, angle_applied)

        results.append({
            "image_path": str(img_path),
            "angle_applied": float(angle_applied),
            "angle_predicted": float(angle_predicted),
            "error": float(error),
            "rotated_pil": rotated_pil,
        })

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(image_paths)} images...")

    return results


def corrected_image(image_path, predicted_angle):
    """Rotate original image by the prediction error: rotate_preserve_content(original, applied - predicted).
    For a perfect prediction this equals the original image."""
    return rotate_preserve_content(image_path, 0)  # placeholder, overridden below


def make_corrected(image_path, angle_applied, angle_predicted):
    """Show what the model's correction looks like: apply (applied - predicted) to the original."""
    residual = angle_applied - angle_predicted
    return rotate_preserve_content(image_path, residual)


def resize_square(img, size=256):
    """Resize image to a square, center-cropping the longer side."""
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((size, size), Image.LANCZOS)


def export_images(good_examples, bad_examples, output_dir, thumb_size=256):
    """Export individual images for LaTeX integration."""
    os.makedirs(output_dir, exist_ok=True)

    all_examples = [("good", good_examples), ("bad", bad_examples)]
    metadata = []

    for category, examples in all_examples:
        for i, ex in enumerate(examples):
            prefix = f"{category}_{i+1}"
            rotated = resize_square(ex["rotated_pil"], thumb_size)
            # GT = original unrotated image (clean, no double-rotation artifacts)
            gt = resize_square(Image.open(ex["image_path"]).convert("RGB"), thumb_size)
            # Corrected = original rotated by residual error (applied - predicted)
            corrected = resize_square(
                make_corrected(ex["image_path"], ex["angle_applied"], ex["angle_predicted"]),
                thumb_size)

            # Save all three versions
            rotated.save(os.path.join(output_dir, f"{prefix}_input.jpg"), quality=95)
            corrected.save(os.path.join(output_dir, f"{prefix}_corrected.jpg"), quality=95)
            gt.save(os.path.join(output_dir, f"{prefix}_gt.jpg"), quality=95)

            metadata.append({
                "prefix": prefix,
                "category": category,
                "source": Path(ex["image_path"]).name,
                "angle_applied": round(ex["angle_applied"], 1),
                "angle_predicted": round(ex["angle_predicted"], 1),
                "error": round(ex["error"], 1),
            })

            print(f"  {prefix}: {Path(ex['image_path']).name}  "
                  f"applied={ex['angle_applied']:.1f}°  "
                  f"predicted={ex['angle_predicted']:.1f}°  "
                  f"error={ex['error']:.1f}°")

    # Save metadata JSON
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Print LaTeX snippet
    n_good = len(good_examples)
    n_bad = len(bad_examples)
    n = n_good + n_bad
    col_width = f"{0.95 / n:.3f}"

    print(f"\n{'='*60}")
    print("LaTeX figure snippet (copy into paper):")
    print(f"{'='*60}\n")

    latex = []
    latex.append(r"\begin{figure*}[t]")
    latex.append(r"\centering")
    latex.append(r"\setlength{\tabcolsep}{1pt}")

    # Two side-by-side subfigures: good (left) and bad (right)
    half_cols = max(n_good, n_bad)
    col_width = f"{0.95 / n:.3f}"

    latex.append(r"\begin{tabular}{" + "c" * (n + 1) + "}")

    # Row: Input
    row = r"\rotatebox{90}{\footnotesize\textbf{Input}} & "
    for entry in metadata:
        row += (r"\includegraphics[width=" + col_width
                + r"\textwidth]{figures/qualitative/"
                + entry["prefix"] + r"_input.jpg} & ")
    latex.append(row.rstrip("& ") + r" \\[1pt]")

    # Row: CGD output
    row = r"\rotatebox{90}{\footnotesize\textbf{CGD}} & "
    for entry in metadata:
        row += (r"\includegraphics[width=" + col_width
                + r"\textwidth]{figures/qualitative/"
                + entry["prefix"] + r"_corrected.jpg} & ")
    latex.append(row.rstrip("& ") + r" \\[1pt]")

    # Row: Ground truth
    row = r"\rotatebox{90}{\footnotesize\textbf{GT}} & "
    for entry in metadata:
        row += (r"\includegraphics[width=" + col_width
                + r"\textwidth]{figures/qualitative/"
                + entry["prefix"] + r"_gt.jpg} & ")
    latex.append(row.rstrip("& ") + r" \\")

    latex.append(r"\end{tabular}")
    latex.append(r"\caption{Qualitative results of CGD (MambaOut Base) on COCO 2014. "
                 r"Left " + str(n_good) + r": accurate predictions; "
                 r"right " + str(n_bad) + r": failure cases.}")
    latex.append(r"\label{fig:qualitative}")
    latex.append(r"\end{figure*}")

    print("\n".join(latex))
    print()

    # Also save the LaTeX snippet to a file
    latex_path = os.path.join(output_dir, "figure_snippet.tex")
    with open(latex_path, "w") as f:
        f.write("\n".join(latex))
    print(f"LaTeX snippet saved to {latex_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate qualitative figure for paper")
    parser.add_argument("--test-dir", type=str, default="data/datasets/test_coco",
                        help="Directory with test images")
    parser.add_argument("--checkpoint", type=str,
                        default="data/saved_models/cgd-kl_divergence-mambaout_base-32/version_0/checkpoints/"
                                "cgd-kl_divergence-mambaout_base-32-version=0-epoch=0051-step=0242164-"
                                "train_loss=0.0274-val_loss=0.2393-train_mae_deg_epoch=1.0981-val_mae_deg=6.1074.ckpt",
                        help="Path to CGD checkpoint")
    parser.add_argument("--n-good", type=int, default=6, help="Number of good examples")
    parser.add_argument("--n-bad", type=int, default=6, help="Number of bad examples")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for test rotations")
    parser.add_argument("--output-dir", type=str, default="eval/results/qualitative",
                        help="Output directory for images")
    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint)

    print(f"Running inference on {args.test_dir} (seed={args.seed})...")
    results = run_inference(model, args.test_dir, args.seed)

    # Sort by error
    results.sort(key=lambda x: x["error"])

    # Print error distribution summary
    errors = [r["error"] for r in results]
    print(f"\nError distribution ({len(errors)} images):")
    bins = [(0, 1), (1, 5), (5, 10), (10, 30), (30, 90), (90, 150), (150, 180)]
    for lo, hi in bins:
        count = sum(1 for e in errors if lo <= e < hi)
        print(f"  [{lo:>3}°, {hi:>3}°): {count:>4}  ({100*count/len(errors):.1f}%)")
    count_180 = sum(1 for e in errors if e >= 170)
    print(f"  [170°, 180°]: {count_180:>4}  ({100*count_180/len(errors):.1f}%)")
    print(f"  Mean: {sum(errors)/len(errors):.2f}°  Median: {sorted(errors)[len(errors)//2]:.2f}°")

    # Save all errors to JSON for further analysis
    all_errors_path = os.path.join(args.output_dir, "all_errors.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(all_errors_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "rotated_pil"} for r in results], f, indent=2)
    print(f"  Full results saved to {all_errors_path}")

    # Pick best and worst
    good = results[:args.n_good]
    bad = results[-args.n_bad:]

    print(f"\nExporting images to {args.output_dir}/...")
    export_images(good, bad, args.output_dir)


if __name__ == "__main__":
    main()
