# extract_nth_frames.py
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_every_nth_frame(
    video_path: Path,
    out_dir: Path,
    n: int,
    start: int = 0,
    max_frames: int | None = None,
    resize_w: int | None = None,
    resize_h: int | None = None,
    prefix: str = "frame",
) -> None:
    if n <= 0:
        raise ValueError("--nth must be >= 1")
    if start < 0:
        raise ValueError("--start must be >= 0")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max_frames must be >= 1 (or omit it)")

    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    saved = 0
    idx = 0

    # Optional: try to jump close to the start frame (not perfect for all codecs)
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (idx - start) % n == 0:
                if resize_w is not None or resize_h is not None:
                    h, w = frame.shape[:2]
                    new_w = resize_w if resize_w is not None else int(w * (resize_h / h))
                    new_h = resize_h if resize_h is not None else int(h * (resize_w / w))
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                out_path = out_dir / f"{prefix}_{saved:06d}.jpg"
                ok_write = cv2.imwrite(str(out_path), frame)
                if not ok_write:
                    raise RuntimeError(f"Failed to write image: {out_path}")

                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break

            idx += 1

    finally:
        cap.release()

    print(f"Done. Saved {saved} frames to: {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract every Nth frame from a video.")
    p.add_argument("--video", type=str, required=True, help="Path to input video (mp4/mov/etc.)")
    p.add_argument("--out_dir", type=str, required=True, help="Folder to save extracted frames")
    p.add_argument("--nth", type=int, required=True, help="Save every Nth frame (e.g., 10 saves 0,10,20,...)")
    p.add_argument("--start", type=int, default=0, help="Start frame index (default: 0)")
    p.add_argument("--max_frames", type=int, default=None, help="Max number of frames to save (optional)")
    p.add_argument("--resize_w", type=int, default=None, help="Resize width (optional)")
    p.add_argument("--resize_h", type=int, default=None, help="Resize height (optional)")
    p.add_argument("--prefix", type=str, default="frame", help="Filename prefix (default: frame)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    extract_every_nth_frame(
        video_path=Path(args.video),
        out_dir=Path(args.out_dir),
        n=args.nth,
        start=args.start,
        max_frames=args.max_frames,
        resize_w=args.resize_w,
        resize_h=args.resize_h,
        prefix=args.prefix,
    )


if __name__ == "__main__":
    main()
