#!/usr/bin/env python3
"""Create ABC diagnostic panels for images.

Usage:
  python3 scripts/visualize_abc.py --predictions results_abc.csv --images images --masks masks --out visualizations --num 12 --mode top --ground-truth GroundTruth.csv
"""
import argparse
from pathlib import Path
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt


def read_preds(path):
    rows = []
    with open(path, newline='') as f:
        r = csv.DictReader(f)
        for x in r:
            img = Path(x['image']).name
            score = float(x.get('score', 0))
            rows.append({'image': img, 'score': score, **x})
    return rows


def load_mask(mask_path, img_shape=None):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def lesion_center_crop(mask, pad_scale=1.3):
    if mask is None:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    side = int(max(width, height) * pad_scale)
    side = max(side, 1)

    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    x0 = center_x - side // 2
    y0 = center_y - side // 2
    x1 = x0 + side
    y1 = y0 + side

    patch = np.zeros((side, side), dtype=mask.dtype)

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(mask.shape[1], x1)
    src_y1 = min(mask.shape[0], y1)

    if src_x0 >= src_x1 or src_y0 >= src_y1:
        return None

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0

    patch[
        dst_y0:dst_y0 + (src_y1 - src_y0),
        dst_x0:dst_x0 + (src_x1 - src_x0)
    ] = mask[src_y0:src_y1, src_x0:src_x1]

    return patch, (x0, y0, x1, y1)


def asymmetry_diff(mask):
    crop_info = lesion_center_crop(mask)
    if crop_info is None:
        return None

    patch, _ = crop_info

    # Same operation used in asymmetry_score()
    flipped = cv2.flip(patch, -1)

    # Visualize disagreement between original and rotated lesion
    diff = cv2.absdiff(
        (patch > 0).astype(np.uint8) * 255,
        (flipped > 0).astype(np.uint8) * 255
    )

    return diff


def border_overlay(mask):
    if mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    # equivalent-area circle
    area = cv2.countNonZero(mask)
    if area > 0:
        coords = np.column_stack(np.where(mask > 0))
        cy = int(coords[:,0].mean())
        cx = int(coords[:,1].mean())
        r = int(np.sqrt(area / np.pi))
        cv2.circle(overlay, (cx, cy), r, (255, 0, 0), 2)
    return overlay


def color_segmentation(image, mask, k=3):
    # run kmeans on pixels within mask in HSV space
    if image is None or mask is None:
        return None, None
    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked = img[mask > 0]
    if len(masked) == 0:
        return None, None
    Z = masked.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = min(k, max(1, len(Z)))
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.flatten()
    seg = np.full(image.shape[:2], -1, dtype=np.int32)
    seg_indices = np.where(mask > 0)
    seg[seg_indices] = labels
    # map centers back to BGR for visualization
    centers = centers.astype(np.uint8)
    centers_bgr = cv2.cvtColor(centers.reshape(-1,1,3), cv2.COLOR_HSV2BGR).reshape(-1,3)
    return seg, centers_bgr


def make_panel(image_path, mask_path, out_path):
    img = cv2.imread(str(image_path))
    mask = None
    if mask_path is not None:
        mask = load_mask(mask_path, img.shape if img is not None else None)

    # original with mask overlay
    orig = img.copy() if img is not None else np.zeros((256,256,3),dtype=np.uint8)
    if mask is not None:
        overlay = orig.copy()
        overlay[mask>0] = (overlay[mask>0]*0.5 + np.array([0,0,255])*0.5).astype(np.uint8)
        orig_overlay = cv2.addWeighted(orig, 0.7, overlay, 0.3, 0)
    else:
        orig_overlay = orig

    # asymmetry diff
    diff = asymmetry_diff(mask)
    if diff is None:
        diff_rgb = np.zeros_like(orig_overlay)
    else:
        cmap = plt.get_cmap('hot')
        diff_rgb = (cmap(diff)[:,:,:3]*255).astype(np.uint8)

    # border overlay
    bord = border_overlay(mask)
    if bord is None:
        bord_vis = np.zeros_like(orig_overlay)
    else:
        bord_vis = orig.copy()
        mask_idx = bord.sum(axis=2)>0
        bord_vis[mask_idx] = bord[mask_idx]

    # color segmentation
    seg, centers = color_segmentation(img, mask, k=3)
    if seg is None:
        seg_vis = np.zeros_like(orig_overlay)
        palette = []
    else:
        seg_vis = np.zeros_like(orig_overlay)
        for i,c in enumerate(centers):
            seg_vis[seg==i] = c
        palette = [tuple(int(x) for x in c) for c in centers]

    # build figure
    fig, axes = plt.subplots(2,3, figsize=(12,8))
    axes = axes.flatten()
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[1].imshow(cv2.cvtColor(orig_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Mask overlay')
    axes[2].imshow(cv2.cvtColor(diff_rgb, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Asymmetry diff (abs)')
    axes[3].imshow(cv2.cvtColor(bord_vis, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Border + equiv circle')
    axes[4].imshow(cv2.cvtColor(seg_vis, cv2.COLOR_BGR2RGB))
    axes[4].set_title('Color segmentation (kmeans HSV)')

    # palette / proportions
    ax = axes[5]
    ax.axis('off')
    if palette:
        counts = [(seg==i).sum() for i in range(len(palette))]
        total = sum(counts) if sum(counts)>0 else 1
        for i,c in enumerate(palette):
            rect = plt.Rectangle((0, i), 1, 1, color=np.array(c)/255.0)
            ax.add_patch(rect)
            ax.text(1.05, i+0.5, f'{counts[i]} ({counts[i]/total:.2%})', va='center')
        ax.set_xlim(0,2.5)
        ax.set_ylim(0, len(palette))
        ax.set_title('Cluster palette & proportions')
    else:
        ax.text(0.5, 0.5, 'No clusters', ha='center', va='center')

    for a in axes:
        a.axis('off')
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', required=True)
    p.add_argument('--images', required=True)
    p.add_argument('--masks', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--num', type=int, default=12)
    p.add_argument('--mode', choices=['top','bottom','random','fp','fn'], default='top')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--ground-truth', default=None)
    args = p.parse_args()

    preds = read_preds(args.predictions)
    if args.mode == 'top':
        sel = sorted(preds, key=lambda x: x['score'], reverse=True)[:args.num]
    elif args.mode == 'bottom':
        sel = sorted(preds, key=lambda x: x['score'])[:args.num]
    else:
        sel = sorted(preds, key=lambda x: x['score'], reverse=True)[:args.num]

    outdir = Path(args.out)
    imgdir = Path(args.images)
    maskdir = Path(args.masks)

    for r in sel:
        name = r['image']
        img_path = imgdir / name
        # try several mask filename patterns (many datasets use "_segmentation" suffix)
        stem = Path(name).stem
        candidates = [maskdir / name, maskdir / f"{stem}_segmentation.png", maskdir / f"{stem}.png"]
        mask_path = None
        for c in candidates:
            if c.exists():
                mask_path = c
                break
        if mask_path is None:
            print(f'Warning: mask not found for {name}; tried: {[str(x.name) for x in candidates]}')
        out_path = outdir / (Path(name).stem + '.png')
        make_panel(img_path, mask_path, out_path)
        print('Saved', out_path)


if __name__ == '__main__':
    main()
