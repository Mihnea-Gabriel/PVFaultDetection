#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path

# === CONFIG ===
ROOT      = Path(__file__).parent.resolve()
DATA_ROOT = ROOT / 'split_solarpanel'
MODEL     = 'yolov8n-cls.pt'
EPOCHS    = 100
BATCH     = 64
IMGSZ     = 224
PROJECT   = 'runs/solarpanel'
RUN_NAME  = 'yolov8n-cls'

def run(cmd):
    print(">>>", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode:
        print(f"Command failed: {' '.join(cmd)}")
        sys.exit(r.returncode)

def main():
    if not DATA_ROOT.exists():
        print(f"Dataset folder not found at {DATA_ROOT}")
        sys.exit(1)

    # TRAIN (reuse same run dir)
    run([
        'yolo', 'classify', 'train',
        f'data={DATA_ROOT}',
        f'model={MODEL}',
        f'epochs={EPOCHS}',
        f'imgsz={IMGSZ}',
        f'batch={BATCH}',
        f'project={PROJECT}',
        f'name={RUN_NAME}',
        'exist_ok=True'       # proper flag to overwrite existing run
    ])

    # VALIDATE on best.pt
    run_dir = Path(PROJECT) / RUN_NAME
    best1 = run_dir / 'weights' / 'best.pt'
    best2 = run_dir / 'best.pt'
    checkpoint = best1 if best1.exists() else best2
    if not checkpoint.exists():
        print(f"Cannot find best.pt in {best1} or {best2}")
        sys.exit(1)

    run([
        'yolo', 'classify', 'val',
        f'data={DATA_ROOT}',
        f'model={checkpoint}',
        f'imgsz={IMGSZ}',
        f'batch={BATCH}',
    ])

if __name__ == '__main__':
    main()
