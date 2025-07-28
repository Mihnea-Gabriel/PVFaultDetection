from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple
import ffmpeg
import os
import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from model import CVModel  


def ensure_h264(input_path: str) -> str : 
    info = ffmpeg.probe(input_path)
    vstream = next(s for s in info["streams"] if s["codec_type"] == "video")
    codec = vstream["codec_name"]
    if codec == "h264":
        return input_path
    base, ext = os.path.splitext(input_path)
    out_path = f"{base}_h264{ext}"
    (
      ffmpeg
        .input(input_path)
        .output(out_path, vcodec="libx264", acodec="copy")
        .run(overwrite_output=True)
    )
    print(f"Re-encoded to H.264 → {out_path}")
    return out_path


def _ensure_head(model: nn.Module, num_classes: int) -> None:
    core = model.model if hasattr(model, "model") else model

    if hasattr(core, "reset_classifier"):
        core.reset_classifier(num_classes)
        return

    for attr in ("head", "classifier"):
        mod = getattr(core, attr, None)
        if isinstance(mod, nn.Linear):
            if mod.out_features != num_classes:
                setattr(
                    core,
                    attr,
                    nn.Linear(mod.in_features, num_classes, bias=mod.bias is not None),
                )
            return


    seq = getattr(core, "model", None)
    if isinstance(seq, nn.Sequential) and isinstance(seq[-1], nn.Linear):
        last = seq[-1]
        if last.out_features != num_classes:
            seq[-1] = nn.Linear(
                last.in_features, num_classes, bias=last.bias is not None
            )
        return


    for name, mod in reversed(list(core.named_modules())):
        if isinstance(mod, nn.Linear):
            if mod.out_features == num_classes:
                return  
            parent = core
            *path, child = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(
                parent,
                child,
                nn.Linear(mod.in_features, num_classes, bias=mod.bias is not None),
            )
            return

    raise RuntimeError("Could not locate a Linear classification head to resize.")


def load_model(
    ckpt_path: Path, device: torch.device
) -> Tuple[nn.Module, List[str]]:

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model_name: str | None = ckpt.get("model_name")
    class_names: List[str] = ckpt.get("class_names", [])
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")

    if not model_name or not class_names or not state_dict:
        raise ValueError(
            "Checkpoint must contain 'model_name', 'class_names', and weights."
        )

    num_classes = len(class_names)
    model = CVModel(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict, strict=False)


    _ensure_head(model, num_classes)

    model.to(device, non_blocking=True)


    _ensure_head(model, num_classes)

    for m in model.modules():
        m.training = False
    return model, class_names


def build_transform(img_size: int = 288) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solar-panel fault demo viewer")
    p.add_argument("--weights", required=True, type=Path, help="Checkpoint .pth")
    p.add_argument("--video", required=True, type=Path, help="H.264 video file")
    p.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    p.add_argument("--imgsz", type=int, default=288, help="Inference resolution")
    p.add_argument("--quit-key", default="q", help="Key to exit viewer")
    return p.parse_args()


@torch.no_grad()
def run_demo(
    model: nn.Module,
    class_names: List[str],
    video_path: Path,
    device: torch.device,
    img_size: int,
    quit_key: str,
) -> None:
    tfm = build_transform(img_size)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    font = cv2.FONT_HERSHEY_SIMPLEX
    green = (0, 255, 0)

    frame_count, t0 = 0, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = tfm(img).unsqueeze(0).to(device)

        logits = model(x)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        probs = logits.softmax(1)[0]
        idx = int(probs.argmax().item())
        conf = float(probs[idx].item())

        label = f"{class_names[idx]}  {conf*100:.1f}%"
        cv2.putText(frame, label, (10, 30), font, 0.9, green, 2, cv2.LINE_AA)

        cv2.imshow("Solar-panel demo  ({} to quit)".format(quit_key), frame)
        if cv2.waitKey(1) & 0xFF == ord(quit_key):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t0
    if elapsed > 0:
        print(f"FPS: {frame_count/elapsed:.2f}  ({frame_count} frames)")


def main() -> None:
    args = parse_args()
    args.video = ensure_h264(args.video)
    args.video = Path(args.video)
    if not args.video.is_file():
        raise SystemExit(f"Video not found: {args.video}")
    if not args.weights.is_file():
        raise SystemExit(f"Weights not found: {args.weights}")

    requested = torch.device(args.device)
    device = (
        requested
        if requested.type == "cpu"
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Running on: {device}")

    model, class_names = load_model(args.weights, device)
    run_demo(
        model,
        class_names,
        args.video,
        device,
        img_size=args.imgsz,
        quit_key=args.quit_key,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
