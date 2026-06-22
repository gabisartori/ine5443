#!/usr/bin/env python3
import csv
from pathlib import Path
import sys

GROUND = 'GroundTruth.csv'
PREDS = 'results_abc_post_simetry_fix.csv'

def normalize_image_id(name):
    # return basename without extension so it matches GroundTruth image IDs
    return Path(name).stem

def load_ground_truth(path, positive_classes):
    gt = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            img = normalize_image_id(r.get('image') or r.get('Image') or '')
            is_pos = False
            for c in positive_classes:
                try:
                    val = float(r.get(c, 0))
                except:
                    val = 0.0
                if val > 0.5:
                    is_pos = True
                    break
            gt[img] = is_pos
    return gt

def load_predictions(path):
    preds = {}
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            img = normalize_image_id(r.get('image') or r.get('Image') or '')
            try:
                score = float(r.get('score', '0'))
            except:
                score = 0.0
            preds[img] = score
    return preds

def main():
    positive_classes = ['MEL','BCC']
    if not Path(GROUND).is_file():
        print('Ground truth file not found:', GROUND); sys.exit(1)
    if not Path(PREDS).is_file():
        print('Predictions file not found:', PREDS); sys.exit(1)

    gt = load_ground_truth(GROUND, positive_classes)
    preds = load_predictions(PREDS)

    common = set(gt.keys()) & set(preds.keys())
    missing = len(gt) - len(common)

    thresholds = [round(x/100, 2) for x in range(30, 71, 5)]

    out_rows = []
    print(f'Total ground truth images: {len(gt)}, with predictions: {len(common)}, missing: {missing}')
    print('threshold,TP,TN,FP,FN,accuracy,precision,recall,f1')
    for t in thresholds:
        tp = tn = fp = fn = 0
        for img in common:
            y_true = gt[img]
            score = preds[img]
            y_pred = score >= t
            if y_true and y_pred:
                tp += 1
            elif not y_true and not y_pred:
                tn += 1
            elif not y_true and y_pred:
                fp += 1
            elif y_true and not y_pred:
                fn += 1
        total = tp+tn+fp+fn
        acc = (tp+tn)/total if total>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        recall = tp/(tp+fn) if (tp+fn)>0 else 0
        f1 = (2*prec*recall/(prec+recall)) if (prec+recall)>0 else 0
        print(f'{t:.2f},{tp},{tn},{fp},{fn},{acc:.4f},{prec:.4f},{recall:.4f},{f1:.4f}')
        out_rows.append((t,tp,tn,fp,fn,acc,prec,recall,f1))

    # save CSV
    out_path = 'threshold_sweep_results.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['threshold','TP','TN','FP','FN','accuracy','precision','recall','f1'])
        for r in out_rows:
            writer.writerow(r)
    print('Saved sweep results to', out_path)

if __name__ == '__main__':
    main()
