"""Walk-forward backtesting con purga del horizonte.

La purga elimina las últimas `h` filas de train, cuyo target futuro se solapa
con el periodo de test. Sin ella, el corte temporal NO evita la fuga.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

MIN_TRAIN = 1260   # ~5 años antes del primer test
TEST_SIZE = 252    # cada pliegue evalúa 1 año

@dataclass(frozen=True)
class Fold:
    idx: int
    train_end: int      # exclusivo, ya purgado
    test_start: int
    test_end: int       # exclusivo

def make_folds(n: int, h: int, min_train: int = MIN_TRAIN,
               test_size: int = TEST_SIZE) -> list[Fold]:
    folds, k, start = [], 0, min_train
    while start + test_size <= n:
        train_end = start - h              # <- purga
        if train_end >= min_train // 2:
            folds.append(Fold(k, train_end, start, min(start + test_size, n)))
            k += 1
        start += test_size
    return folds

def describe(folds: list[Fold], dates) -> str:
    if not folds:
        return "  (sin pliegues)"
    d = np.asarray(dates)
    lines = []
    for f in folds:
        lines.append(f"  F{f.idx:<2} train <= {str(d[f.train_end - 1])[:10]} | "
                     f"test {str(d[f.test_start])[:10]} → {str(d[f.test_end - 1])[:10]}")
    return "\n".join(lines)
