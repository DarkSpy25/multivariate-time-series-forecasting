"""
Phase 9 inference entrypoint (roadmap 9.1).

Usage:
    python predict.py --input_dir /data/input --output_file /output/predictions.csv --checkpoint checkpoint.pt

Reads a covariates-only input directory (either the public
`validation_input.csv` + `forecast_index_validation.csv`, or the graded
`test_input.csv` + `forecast_index_test.csv`), reconstructs everything the
model needs from the checkpoint saved by src/train.py (scaler, series order,
feature list, and the baked-in training-history tail), runs inference for
all series across the full horizon, and writes a prediction CSV with columns
`series_id, timestamp, prediction`.

Compliance note (per forum / roadmap): only train-derived state is baked
into the checkpoint and used here -- never validation- or test-period
target values. The only thing read from the input directory is covariates.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data import prepare_inference_cube
from src.scaling import GlobalScaler
from src.models import build_model_from_config


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def find_input_files(input_dir):
    """Accept either the graded 'test_*' names or the public 'validation_*' names."""
    input_dir = Path(input_dir)
    for stem in ("test", "validation"):
        inp = input_dir / f"{stem}_input.csv"
        idx = input_dir / f"forecast_index_{stem}.csv"
        if inp.exists() and idx.exists():
            return inp, idx
    raise FileNotFoundError(
        f"Could not find <stem>_input.csv + forecast_index_<stem>.csv "
        f"(tried 'test' and 'validation') in {input_dir}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="Directory containing *_input.csv, forecast_index_*.csv, metadata.json")
    ap.add_argument("--output_file", required=True,
                    help="Full path to write the prediction CSV to")
    ap.add_argument("--checkpoint", default="checkpoint.pt",
                    help="Path to the trained checkpoint (default: checkpoint.pt next to this script)")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    dev = a.device or device()
    print(f"Using device: {dev}")

    inp_path, idx_path = find_input_files(a.input_dir)
    print(f"Reading covariates from {inp_path}")
    print(f"Reading forecast index from {idx_path}")
    fi = pd.read_csv(idx_path, parse_dates=["timestamp"])

    print(f"Loading checkpoint from {a.checkpoint}")
    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)

    scaler = GlobalScaler.from_state(ckpt["scaler"])
    series_order = ckpt["series"]
    features = ckpt["features"]
    L, H = ckpt["L"], ckpt["H"]
    cfg = ckpt["config"]

    print(f"Model: {cfg['model']}  |  series: {len(series_order)}  |  L={L}  H={H}")

    # Reconstruct inference tensors: future covariates come from the input
    # directory, past target history comes from the baked checkpoint tail.
    Xh, Xf, series_order = prepare_inference_cube(
        inp_path, fi, features, series_order, scaler, L, H,
        fallback_x=ckpt["history_x"],
        fallback_y=ckpt["history_y"],
        fallback_end=ckpt["history_end"],
    )

    model = build_model_from_config(cfg, n_feat=len(features),
                                     n_series=len(series_order), L=L, H=H)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()

    xh = torch.from_numpy(Xh).to(dev)
    xf = torch.from_numpy(Xf).to(dev)
    sidx = torch.arange(len(series_order), device=dev)

    with torch.no_grad():
        pred = model(xh, xf, sidx).cpu().numpy()   # (S, H)

    assert np.isfinite(pred).all(), "predictions contain NaN or inf"

    # forecast_index_* gives the exact timestamps to predict, in order.
    fut = np.sort(fi.timestamp.unique())[-H:]

    rows = []
    for i, sid in enumerate(series_order):
        for t, ts in enumerate(fut):
            rows.append((sid, ts, float(pred[i, t])))

    out = pd.DataFrame(rows, columns=["series_id", "timestamp", "prediction"])

    out_path = Path(a.output_file)
    os.makedirs(out_path.parent, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out)} rows ({len(series_order)} series x {H} hours) to {out_path}")


if __name__ == "__main__":
    main()