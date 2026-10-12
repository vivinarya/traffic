"""
Creates real traffic video clip from VisDrone sequential camera frames.
"""
import os
import glob
from pathlib import Path
import cv2

def make_real_traffic_clip(output_path: Path = Path(r"c:\g\tests\real_traffic_video.mp4"), fps: int = 6):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    val_dir = Path(r"c:\g\datasets\VisDrone\images\val")

    # Take sequences 0000289, 0000271, 0000276
    all_imgs = sorted(glob.glob(str(val_dir / "0000289_*.jpg")) + glob.glob(str(val_dir / "0000271_*.jpg")))

    if not all_imgs:
        raise FileNotFoundError("No VisDrone val images found!")

    sample = cv2.imread(all_imgs[0])
    h, w, _ = sample.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    print(f"Stitching {len(all_imgs)} real camera frames ({w}x{h}) into {output_path}...")
    for img_p in all_imgs:
        frame = cv2.imread(img_p)
        writer.write(frame)

    writer.release()
    print(f"Created real traffic video: {output_path} ({output_path.stat().st_size / (1024*1024):.2f} MB)")
    return output_path

if __name__ == "__main__":
    make_real_traffic_clip()
