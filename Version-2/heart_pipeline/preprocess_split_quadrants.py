#!/usr/bin/env python3
import argparse
from pathlib import Path
from PIL import Image
import pandas as pd

# Quadrant mapping (top-left = LA, top-right = RA, bottom-left = LV, bottom-right = RV)
QUAD_ORDER = [("LA", 0, 0), ("RA", 1, 0), ("LV", 0, 1), ("RV", 1, 1)]

def split_into_quadrants_gray(img: Image.Image):
    """Return dict label->PIL.Image for the 4 chambers after grayscale conversion."""
    g = img.convert("L")
    w, h = g.size
    mx, my = w // 2, h // 2
    quads = {
        "LA": g.crop((0,   0,   mx,  my)),
        "RA": g.crop((mx,  0,   w,   my)),
        "LV": g.crop((0,   my,  mx,  h )),
        "RV": g.crop((mx,  my,  w,   h )),
    }
    return quads

def process_split(data_root: Path, out_root: Path, split: str, classes=("ASD","NORM")):
    rows = []
    for cls in classes:
        in_dir = data_root / split / cls
        out_dir = out_root / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in sorted(in_dir.glob("*.png")):
            img = Image.open(p).convert("RGB")  # source may be RGB or grayscale; normalize
            quads = split_into_quadrants_gray(img)
            for qlabel, qimg in quads.items():
                out_name = f"{p.stem}_{qlabel}.png"
                qimg.save(out_dir / out_name)
                rows.append({
                    "split": split,
                    "label_global": cls,
                    "file_full": str(p.relative_to(data_root)),
                    "file_patch": str((out_dir / out_name).relative_to(out_root)),
                    "patch_label": qlabel
                })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="Path to data_orig with train/validation/test")
    ap.add_argument("--out_root", default="data_patches", help="Output root for grayscale quadrant patches")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_root  = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for split in ("train", "validation", "test"):
        split_rows = process_split(data_root, out_root, split)
        all_rows.extend(split_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_root / "patch_mapping.csv", index=False)
    print("Wrote:", out_root / "patch_mapping.csv")
    print("Done. New dataset at:", out_root)

if __name__ == "__main__":
    main()