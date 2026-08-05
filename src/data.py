H = 336            # forecast horizon, fixed by the task
N_TRAIN_HOURS = 4320

def fold_bounds(fold: int, total: int = N_TRAIN_HOURS, h: int = H):
    """Return (train_end, score_start, score_end) as hour indices."""
    score_end   = total - fold * h
    score_start = score_end - h
    train_end   = score_start   # training must not touch [score_start, score_end)
    return train_end, score_start, score_end