#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

CLASSES = [
    "Clean", "Dusty", "bird_drop",
    "electrical_damage", "physical_damage", "snow_covered",
]


class VideoClassifier:
    """YOLOv8-CLS wrapper that outputs top-1 label and soft-max confidence."""

    def __init__(self, weights: Path, arch: str, device: torch.device):
        self.device = device
        self.model  = self._build_model(weights, arch).to(device).eval()
        self.prep   = self._build_transform(288)

    @staticmethod
    def _build_transform(sz: int):
        return transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize((sz, sz)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _build_model(weights: Path, arch: str) -> nn.Module:
        global CLASSES                       # we may overwrite it

        # ------------------------------------------------------------------ #
        # 1) load checkpoint robustly
        # ------------------------------------------------------------------ #
        try:
            # first attempt: weights_only=True  (fast, safe)
            state = torch.load(weights, map_location="cpu", weights_only=True)
            extra = {}                       # names may be missing here
        except Exception:
            # fall back: full pickle
            ckpt  = torch.load(weights, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            extra = ckpt                     # to look for 'names' etc.

        # update class list if Ultralytics saved it
        names = extra.get("names") if isinstance(extra, dict) else None
        if names:
            CLASSES = list(names)

        # ------------------------------------------------------------------ #
        # 2) build backbone and patch classifier head
        # ------------------------------------------------------------------ #
        y = YOLO(f"{arch}.pt")
        model: nn.Module = y.model

        for name, mod in reversed(list(model.named_modules())):
            if isinstance(mod, nn.Linear):
                parent = model
                *path, child = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, child, nn.Linear(mod.in_features, len(CLASSES)))
                break

        # ------------------------------------------------------------------ #
        # 3) load weights (state-dict) into the backbone
        # ------------------------------------------------------------------ #
        model.load_state_dict(state, strict=False)
        return model

    @torch.no_grad()
    def __call__(self, frame_bgr):
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        t   = self.prep(img).unsqueeze(0).to(self.device)

        logits = self.model(t)[0] if isinstance(self.model(t), (tuple, list)) else self.model(t)
        probs  = logits.softmax(1)[0]
        idx    = probs.argmax().item()
        return CLASSES[idx], probs[idx].item()


class VideoRunner:
    """Handles video transcoding, streaming and overlay."""

    def __init__(self, src: Path):
        self.play_path = self._h264_copy(src)
        self.cap = cv2.VideoCapture(str(self.play_path))
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video.")

    @staticmethod
    def _h264_copy(src: Path) -> Path:
        tmp_file = Path(tempfile.mkdtemp()) / "video_h264.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "23",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(tmp_file),
        ]
        if subprocess.run(cmd).returncode != 0 or not tmp_file.exists():
            raise RuntimeError("ffmpeg transcoding failed.")
        return tmp_file

    def stream(self, classifier: VideoClassifier):
        font, green = cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0)
        frames, t0 = 0, datetime.now()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            label, conf = classifier(frame)
            cv2.putText(
                frame,
                f"{label}  {conf*100:.1f}%",
                (10, 30),
                font,
                0.9,
                green,
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("PV-fault demo  (q = quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frames += 1

        self.cap.release()
        cv2.destroyAllWindows()

        elapsed = (datetime.now() - t0).total_seconds()
        print(
            f"Frames: {frames} | Time: {elapsed:.1f} s | FPS: {frames/elapsed:.2f}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--arch", default="yolov8x-cls")
    return parser.parse_args()


def main():
    args = parse_args()
    src = args.video.expanduser().resolve()
    wts = args.weights.expanduser().resolve()

    if not src.is_file():
        raise SystemExit(f"Video not found: {src}")
    if not wts.is_file():
        raise SystemExit(f"Weights not found: {wts}")

    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    classifier = VideoClassifier(wts, args.arch, device)
    runner = VideoRunner(src)
    runner.stream(classifier)


if __name__ == "__main__":
    main()
