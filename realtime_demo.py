#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Realtime PV-fault demo
  ▸ adaptive video opening   (probe → optional H.264 transcode)
  ▸ YOLOv8-CLS backbone      (top-1 label + Δ-margin)
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime
import argparse, subprocess, tempfile, cv2, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO                                # ≥ 8.1

CLASSES = [
    "Clean", "Dusty", "bird_drop",
    "electrical_damage", "physical_damage", "snow_covered",
]


class VideoSource:
    """Open a video; if native decode fails, create a temp H.264 copy."""

    def __init__(self, path: Path):
        self._src = path.expanduser().resolve()
        if not self._src.is_file():
            raise FileNotFoundError(self._src)

        self._play = self._probe_or_transcode()
        self.cap   = cv2.VideoCapture(str(self._play))
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open video.")

    # ------------------------------------------------------------------
    def _probe_or_transcode(self) -> Path:
        """Return the path that OpenCV should read (orig or temp)."""
        if self._decodes(self._src):          # happy path
            return self._src

        tmp = Path(tempfile.mkdtemp()) / "reencoded.mp4"
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(self._src),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-movflags", "+faststart",
            str(tmp),
        ]
        if subprocess.run(cmd).returncode != 0 or not tmp.exists():
            raise RuntimeError("ffmpeg transcoding failed.")
        return tmp

    # ------------------------------------------------------------------
    @staticmethod
    def _decodes(path: Path) -> bool:
        cap = cv2.VideoCapture(str(path))
        ok = False
        if cap.isOpened():
            for _ in range(5):
                ret, _ = cap.read()
                if ret:
                    ok = True
                    break
        cap.release()
        return ok


# ─── keep THIS definition ─────────────────────────────────────────
class PVClassifier:
    """YOLOv8-CLS wrapper -> label + softmax score."""
    def __init__(self, weights: Path, arch: str, use_gpu: bool):
        dev           = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.device   = torch.device(dev)
        self.model    = self._init_model(weights, arch).to(self.device).eval()
        self.prep     = self._build_tf(288)

    @staticmethod
    def _build_tf(sz: int):
        return transforms.Compose([
            transforms.Lambda(lambda im: im.convert("RGB")),
            transforms.Resize(sz + 32), transforms.CenterCrop(sz),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def _init_model(self, w: Path, arch: str) -> nn.Module:
        y = YOLO(f"{arch}.pt")
        m: nn.Module = y.model
        # replace final Linear to match class count
        for name, mod in reversed(list(m.named_modules())):
            if isinstance(mod, nn.Linear):
                parent = m
                *path, child = name.split(".")
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, child, nn.Linear(mod.in_features, len(CLASSES)))
                break
        m.load_state_dict(torch.load(w, map_location="cpu"), strict=False)
        return m

    # ---------- inference ----------
    @torch.no_grad()
    def __call__(self, frame_bgr):
        img    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        logits = self.model(self.prep(img).unsqueeze(0).to(self.device))[0]
        probs  = logits.softmax(1)[0]
        idx    = probs.argmax().item()
        return CLASSES[idx], probs[idx].item()
# ────────────────────────────────────────────────────────────────


class RealtimeDemo:
    """Video loop with overlay + FPS stats."""

    def __init__(self, src: Path, clsf: PVClassifier):
        self.video = VideoSource(src)
        self.clsf  = clsf

    def run(self):
        font, green = cv2.FONT_HERSHEY_SIMPLEX, (0, 255, 0)
        frames, t0  = 0, datetime.now()


        fps   = self.video.cap.get(cv2.CAP_PROP_FPS) or 30  # fallback
        delay = int(1000 / fps)                             # milliseconds

        while True:
            ret, frame = self.video.cap.read()
            if not ret:
                break

            label, score = self.clsf(frame)
            cv2.putText(
                frame,
                f"{label}  {score*100:5.1f}%",
                (10, 28), font, 0.8, green, 2, cv2.LINE_AA,
            )
            cv2.imshow("PV-fault | q = quit", frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
            frames += 1

        self.video.cap.release()
        cv2.destroyAllWindows()
        elapsed = (datetime.now() - t0).total_seconds()
        print(f"Frames {frames} | {elapsed:.1f} s | {frames/elapsed:.2f} FPS")

# ────────────────────────────────────────────────────────── CLI
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--video",   required=True,  type=Path)
    p.add_argument("--weights", required=True,  type=Path)
    p.add_argument("--arch",    default="yolov8x-cls")
    p.add_argument("--gpu",     action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    a = parse()
    demo = RealtimeDemo(
        a.video,
        PVClassifier(a.weights, a.arch, a.gpu),
    )
    demo.run()
