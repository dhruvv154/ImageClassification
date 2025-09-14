#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)  # data_orig
    ap.add_argument("--pred_csv", required=True)   # runs_quads/test_predictions_patch_agg.csv
    ap.add_argument("--out_dir", default="runs_quads/viz_quadrants")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.pred_csv)

    # Try to load a default font (works even if not present)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        pass

    for _, r in df.iterrows():
        img_path = Path(args.data_root)/r["file"]
        if not img_path.exists(): continue
        im = Image.open(img_path).convert("RGB")
        w, h = im.size; mx, my = w//2, h//2
        d = ImageDraw.Draw(im)

        # Boxes for LA, RA, LV, RV
        quads = {
            "LA": (0, 0, mx, my),
            "RA": (mx, 0, w,  my),
            "LV": (0, my, mx, h ),
            "RV": (mx, my, w,  h ),
        }

        # Draw boxes + labels
        for label, (x1,y1,x2,y2) in quads.items():
            d.rectangle([x1,y1,x2,y2], outline="red", width=3)
            text = f"{label}: {r['pred_'+label]}"
            tx, ty = x1 + 6, y1 + 6
            d.text((tx, ty), text, fill="white", font=font)

        # Title with final aggregated prediction
        title = f"Final: {r['pred_image']}  |  ASD={r.get('prob_ASD', None):.2f}  NORM={r.get('prob_NORM', None):.2f}"
        d.text((10, h-24), title, fill="white", font=font)

        out = Path(args.out_dir)/ (img_path.stem + "_quads.png")
        im.save(out)
    print("Saved visualizations to", args.out_dir)

if __name__ == "__main__":
    main()