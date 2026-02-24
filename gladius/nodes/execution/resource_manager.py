import shutil
from pathlib import Path
from gladius.state import GraphState

CACHE_DIR = Path("state/cache")
DISK_THRESHOLD = 0.90  # 90% disk usage triggers cleanup


def resource_manager_node(state: GraphState) -> GraphState:
    disk_usage = _get_disk_usage()

    if disk_usage > DISK_THRESHOLD:
        _evict_old_caches(state)

    return {"next_node": state.get("next_node", "strategy")}


def _get_disk_usage() -> float:
    try:
        usage = shutil.disk_usage("/")
        return usage.used / usage.total
    except Exception:
        return 0.0


def _evict_old_caches(state: GraphState):
    if not CACHE_DIR.exists():
        return
    protected = set()
    best_run = state.get("run_id")
    if best_run:
        protected.add(best_run)

    cache_files = sorted(CACHE_DIR.iterdir(), key=lambda f: f.stat().st_mtime)
    for f in cache_files:
        if any(p in f.name for p in protected):
            continue
        if _get_disk_usage() <= DISK_THRESHOLD * 0.85:
            break
        try:
            if f.is_file():
                f.unlink()
            elif f.is_dir():
                shutil.rmtree(f)
        except Exception:
            pass
