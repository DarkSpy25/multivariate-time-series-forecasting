import argparse, os, time, numpy as np, torch, torch.nn as nn
from .data import load_cube, WindowDataset, make_loader, fold_bounds, validation_batch, H
from .scaling import GlobalScaler
from .metrics import all_metrics
from .models import build_model
from .logbook import log_run


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


LOSSES = {"l1": nn.L1Loss, "huber": nn.SmoothL1Loss, "mse": nn.MSELoss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="linear")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lookback", type=int, default=512)
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--loss", default="huber", choices=list(LOSSES))
    ap.add_argument("--revin", type=int, default=1)
    ap.add_argument("--covariates", type=int, default=0)
    ap.add_argument("--use-trend", type=int, default=1)
    ap.add_argument("--fill", default="ffill")
    ap.add_argument("--add-mask", type=int, default=0)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--patch-len", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="experiments/checkpoints")
    ap.add_argument("--notes", default="")
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--channel-mixing", type=int, default=0)
    ap.add_argument("--target-only", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev, t0 = (a.device or device()), time.time()
    L = a.lookback

    X, Y, series, hours, feats, St = load_cube(
        "data/raw/train.csv", use_trend=bool(a.use_trend),
        fill=a.fill, add_mask=bool(a.add_mask))
    train_end, s0, s1 = fold_bounds(a.fold)

    scaler = GlobalScaler().fit(X, upto=train_end)
    Xs = scaler.transform(X)

    ds = WindowDataset(Xs, Y, L=L, H=H, train_end=train_end, gap=a.gap)
    dl = make_loader(ds, bs=a.batch_size, shuffle=True)

    vx, vxf, vy = validation_batch(Xs, Y, series, L, train_end, s0, s1, gap=a.gap)
    vx, vxf = vx.to(dev), vxf.to(dev)

    model = build_model(a, n_feat=Xs.shape[2], n_series=len(series), L=L, H=H).to(dev)
    print(a.model, "params:", sum(p.numel() for p in model.parameters()))

    crit = LOSSES[a.loss]()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * len(dl))

    best, bad, best_state = 1e9, 0, None
    for ep in range(a.epochs):
        epoch_start = time.time()
        print(f"\nStarting epoch {ep+1}/{a.epochs}...", flush=True)
        model.train()
        tot = n = 0

        for batch_idx, (xh, xf, yf, si) in enumerate(dl):
            xh, xf, yf = xh.to(dev), xf.to(dev), yf.to(dev)
            opt.zero_grad()
            loss = crit(
                model(xh, xf, si.to(dev)),
                yf
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            opt.step()
            sched.step()
            tot += loss.item() * len(yf)
            n += len(yf)

            # Print progress every 10 batches
            if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dl):
                print(
                    f"  batch {batch_idx+1}/{len(dl)} "
                    f"loss {loss.item():.4f}",
                    flush=True
                )

        model.eval()
        with torch.no_grad():
            pv = model(
                vx,
                vxf,
                torch.arange(len(series), device=dev)
            ).cpu().numpy()
        m = all_metrics(vy, pv)

        epoch_time = time.time() - epoch_start

        print(
            f"ep{ep+1:02d}/{a.epochs}  "
            f"train {tot/n:.4f}  "
            f"val WAPE {m['wape']:.4f}  "
            f"time {epoch_time:.1f}s",
            flush=True
        )

        if m["wape"] < best - 1e-4:
            best, bad = m["wape"], 0

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            best_metrics = m

            print(
                f"  ✓ new best WAPE: {best:.4f}",
                flush=True
            )

        else:
            bad += 1

            print(
                f"  no improvement ({bad}/{a.patience})",
                flush=True
            )

            if bad >= a.patience:
                print("early stop", flush=True)
                break

    run_id = f"{a.model}_f{a.fold}_s{a.seed}_L{L}_{a.loss}" \
             f"{'_revin' if a.revin else ''}{'_cov' if a.covariates else ''}"
    os.makedirs(a.out, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "config": vars(a),
        "scaler": scaler.state(),
        "series": list(series),
        "features": list(feats),
        "L": L, "H": H,
    }, f"{a.out}/{run_id}.pt")

    log_run(run_id=run_id, minutes=round((time.time()-t0)/60, 1),
        **{k: v for k, v in vars(a).items() if k in
           ("model", "fold", "seed", "lookback", "loss", "revin", "covariates",
            "patch_len", "d_model", "layers", "lr", "epochs", "notes")},
        **best_metrics)


if __name__ == "__main__":
    main()