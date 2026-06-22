#!/usr/bin/env python3
"""
Avaliação simples do critério ABC (Asymmetry, Border, Color)
Uso: python scripts/evaluate_abc.py --images images --masks masks --output results.csv

Dependências: opencv-python, numpy
"""
from pathlib import Path
import argparse
import csv
import math
import cv2
import numpy as np


def load_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"Não foi possível ler máscara: {path}")
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)
    return m


def load_image(path):
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError(f"Não foi possível ler imagem: {path}")
    return im


def centered_square_crop(mask, pad_scale=1.3):
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
    patch[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = mask[src_y0:src_y1, src_x0:src_x1]
    return patch


def asymmetry_score(mask):
    patch = centered_square_crop(mask)
    if patch is None:
        return 0.0
    # flip 180 (equivalente a rota 180)
    flipped = cv2.flip(patch, -1)
    inter = np.logical_and(patch > 0, flipped > 0).sum()
    union = np.logical_or(patch > 0, flipped > 0).sum()
    iou = inter / union if union > 0 else 1.0
    asym = 1.0 - iou
    return float(np.clip(asym, 0.0, 1.0))


def border_irregularity(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    # usar maior contorno
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perim = cv2.arcLength(c, True)
    if perim <= 0 or area <= 0:
        return 0.0
    circularity = (4 * math.pi * area) / (perim * perim + 1e-9)
    score = 1.0 - circularity
    return float(np.clip(score, 0.0, 1.0))


def color_variegation(image, mask, k=3):
    # analisa variação de cor dentro da máscara usando k-means
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    pixels = hsv[ys, xs].astype(np.float32)
    # usar kmeans do OpenCV
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 3
    flags = cv2.KMEANS_PP_CENTERS
    try:
        compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, flags)
    except Exception:
        return 0.0
    labels = labels.flatten()
    counts = np.bincount(labels, minlength=k)
    props = counts / counts.sum()
    # contar clusters significativos (>5%)
    significant = (props > 0.05).sum()
    score = significant / k
    return float(np.clip(score, 0.0, 1.0))


def evaluate(image_path, mask_path):
    img = load_image(image_path)
    mask = load_mask(mask_path)
    a = asymmetry_score(mask)
    b = border_irregularity(mask)
    c = color_variegation(img, mask, k=3)
    # combinação linear com pesos simples
    score = 0.4 * a + 0.3 * b + 0.3 * c
    suspicious = score >= 0.5
    return {
        'image': Path(image_path).name,
        'asymmetry': round(a, 4),
        'border_irregularity': round(b, 4),
        'color_variegation': round(c, 4),
        'score': round(score, 4),
        'suspicious': int(suspicious)
    }


def find_matching_image(images_dir, base_name):
    # normalize base_name: masks often have suffixes like '_segmentation' or '_mask'
    suffixes = ['_segmentation', '_mask', '_seg', '-segmentation']
    base_norm = base_name
    for s in suffixes:
        if base_norm.endswith(s):
            base_norm = base_norm[: -len(s)]
            break

    # procura por ficheiro com base_norm.* (jpg, jpeg, png)
    exts = ['.jpg', '.jpeg', '.png']
    for e in exts:
        p = images_dir / (base_norm + e)
        if p.exists():
            return p
    # também aceitar se base_name já for o nome direto (sem normalizar)
    for e in exts:
        p = images_dir / (base_name + e)
        if p.exists():
            return p

    # tentar qualquer arquivo cujo stem corresponda ou comece com base_norm
    for p in images_dir.iterdir():
        stem = p.stem
        if stem == base_norm or stem == base_name:
            return p
        if stem.startswith(base_norm) or base_norm.startswith(stem):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='pasta com imagens')
    ap.add_argument('--masks', required=True, help='pasta com máscaras binárias')
    ap.add_argument('--output', default='results_abc.csv', help='CSV de saída')
    ap.add_argument('--limit', type=int, default=0, help='limitar número de arquivos processados (0 = todos)')
    args = ap.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)
    out_csv = Path(args.output)

    rows = []
    mask_files = sorted(masks_dir.glob('*'))
    if args.limit > 0:
        mask_files = mask_files[:args.limit]

    for mpath in mask_files:
        if not mpath.is_file():
            continue
        base = mpath.stem
        img_path = find_matching_image(images_dir, base)
        if img_path is None: continue
        try:
            res = evaluate(img_path, mpath)
            rows.append(res)
            print(f"{res['image']}: score={res['score']} suspicious={res['suspicious']}")
        except Exception as e:
            print(f"Erro ao processar {img_path} / {mpath}: {e}")

    # escrever CSV
    if rows:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image', 'asymmetry', 'border_irregularity', 'color_variegation', 'score', 'suspicious'])
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Resultados gravados em: {out_csv}")
    else:
        print('Nenhum resultado para gravar.')


if __name__ == '__main__':
    main()
