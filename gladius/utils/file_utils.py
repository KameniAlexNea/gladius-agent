import fcntl
import numpy as np
from pathlib import Path


def write_oof_file(path: str | Path, array: np.ndarray):
    """Write OOF array to file with POSIX advisory lock."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            np.save(f, array)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_oof_file(path: str | Path) -> np.ndarray:
    """Read OOF array from file."""
    return np.load(str(path))
