#!/usr/bin/env python3
import argparse, json
from pathlib import Path
from typing import Dict
import torch, torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# --- transforms must match training ---
TFM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def split_quads_gray(img: Image.Image) -> Dict[str, Image.Image]:
    g = img.convert("L")
    w, h = g.size; mx, my = w//2, h//2
    return {
        "LA": g.crop((0,  0,  mx, my)),
        "RA": g.crop((mx, 0,  w,  my)),
        "LV": g.crop((0,  my, mx, h )),
        "RV": g.crop((mx, my, w,  h )),
    }

def load_model(weights_path: str):
    ckpt = torch.load(weights_path, map_location="cpu")
    classes = ckpt["classes"]
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, len(classes))
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device, classes

def aggregate(method: str, scores_per_quad: Dict[str, torch.Tensor], classes: list, weights: Dict[str, float]):
    """Return final_class, aggregated_scores (Tensor on CPU)."""
    # stack in fixed order for reproducibility
    order = ["LA","RA","LV","RV"]
    mat = torch.stack([scores_per_quad[q] for q in order], dim=0)  # [4, C]
    if method == "majority":
        votes = mat.argmax(dim=1)                                   # [4]
        # tie-breaker by average probability
        counts = torch.bincount(votes, minlength=len(classes))
        max_count = counts.max()
        winners = (counts == max_count).nonzero(as_tuple=True)[0]
        avg_scores = mat.mean(dim=0)
        final_idx = winners[avg_scores[winners].argmax()].item()
        agg = avg_scores
    elif method == "weighted":
        w = torch.tensor([weights["LA"], weights["RA"], weights["LV"], weights["RV"]], dtype=torch.float32)
        w = w / w.sum()
        agg = (mat * w.view(-1,1)).sum(dim=0)                       # [C]
        final_idx = int(agg.argmax().item())
    else:  # "average"
        agg = mat.mean(dim=0)
        final_idx = int(agg.argmax().item())
    return final_idx, agg.cpu()

def main():
    ap = argparse.ArgumentParser(description="Predict per-patch and stitched image-level labels, export CSV.")
    ap.add_argument("--data_root", required=True, help="Path to data_orig")
    ap.add_argument("--weights", required=True, help="Path to best_patch_cls.pth")
    ap.add_argument("--split", default="test", choices=["train","validation","test"])
    ap.add_argument("--out_csv", default="runs_quads_bal/test_predictions_patch_agg.csv")
    ap.add_argument("--method", default="majority", choices=["majority","average","weighted"])
    ap.add_argument("--w_la", type=float, default=0.15, help="Weight for LA (if method=weighted)")
    ap.add_argument("--w_ra", type=float, default=0.15, help="Weight for RA (if method=weighted)")
    ap.add_argument("--w_lv", type=float, default=0.35, help="Weight for LV (if method=weighted)")
    ap.add_argument("--w_rv", type=float, default=0.35, help="Weight for RV (if method=weighted)")
    args = ap.parse_args()

    model, device, classes = load_model(args.weights)
    cls_to_idx = {c:i for i,c in enumerate(classes)}
    data_root = Path(args.data_root)
    split_root = data_root / args.split

    # collect all image paths under split (expects ASD/NORM subfolders)
    img_paths = []
    for sub in classes:
        img_paths += sorted((split_root/sub).glob("*.png"))

    rows = []
    for p in img_paths:
        img = Image.open(p).convert("RGB")
        quads = split_quads_gray(img)

        scores_per_quad = {}
        per_patch_pred = {}
        per_patch_probs = {}

        for qlabel, qimg in quads.items():
            x = TFM(qimg).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(x), dim=1).squeeze(0).cpu()  # [C]
            scores_per_quad[qlabel] = prob
            per_patch_pred[qlabel] = classes[int(prob.argmax().item())]
            # store probs per patch in dict
            per_patch_probs[qlabel] = {c: float(prob[cls_to_idx[c]]) for c in classes}

        weights = {"LA": args.w_la, "RA": args.w_ra, "LV": args.w_lv, "RV": args.w_rv}
        final_idx, agg_scores = aggregate(args.method, scores_per_quad, classes, weights)
        final_label = classes[final_idx]

        row = {
            "file": str(p.relative_to(data_root)),
            "method": args.method,
            "pred_image": final_label,
        }
        # aggregated probs per class
        for c in classes:
            row[f"prob_{c}"] = float(agg_scores[cls_to_idx[c]])
        # per-patch predicted label + probs
        for q in ["LA","RA","LV","RV"]:
            row[f"pred_{q}"] = per_patch_pred[q]
            for c in classes:
                row[f"p_{q}_{c}"] = per_patch_probs[q][c]

        rows.append(row)

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()