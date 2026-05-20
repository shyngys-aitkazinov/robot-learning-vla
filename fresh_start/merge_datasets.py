"""Merge the per-celebrity datasets into one SmolVLA training corpus.

lerobot's training factory (`make_dataset`) raises `NotImplementedError` for a
list of `repo_id`s, so joint training across the 9 `dataset_v4_*` datasets
needs them merged into a single LeRobot dataset first. This wraps
`lerobot.datasets.aggregate.aggregate_datasets`, which validates the schemas,
de-duplicates task strings, concatenates videos + parquet, and recomputes
normalization stats.

Run directly to merge:

    python fresh_start/merge_datasets.py
    python fresh_start/merge_datasets.py --data.force_remerge=true

`train.py` calls `ensure_merged()` automatically, so a manual run is optional.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# --- bootstrap: make sibling modules importable when run as a script ---
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import groot_shim  # noqa: E402

groot_shim.apply()

import draccus  # noqa: E402

from config import DataConfig, PipelineConfig  # noqa: E402

REPO_ROOT = _HERE.parent


def resolve_source_root(path: str) -> Path:
    """Resolve a (possibly repo-root-relative) dataset path to an absolute Path."""
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p)


def _print_summary(aggr_root: Path) -> None:
    info = json.loads((aggr_root / "meta" / "info.json").read_text())
    print(
        f"[merge] merged dataset: {info.get('total_episodes')} episodes, "
        f"{info.get('total_frames')} frames, {info.get('total_tasks')} tasks, "
        f"fps={info.get('fps')}, robot={info.get('robot_type')}"
    )


def merge(data_cfg: DataConfig) -> Path:
    """Aggregate `data_cfg.source_roots` into a single dataset at `merged_root`."""
    from lerobot.datasets.aggregate import aggregate_datasets

    roots = [resolve_source_root(p) for p in data_cfg.source_roots]
    missing = [str(r) for r in roots if not (r / "meta" / "info.json").exists()]
    if missing:
        raise FileNotFoundError(
            "Source dataset(s) missing meta/info.json — pull them first:\n  "
            + "\n  ".join(missing)
        )

    aggr_root = Path(data_cfg.merged_root)
    if aggr_root.exists():
        if data_cfg.force_remerge:
            print(f"[merge] force_remerge: removing existing {aggr_root}")
            shutil.rmtree(aggr_root)
        else:
            raise FileExistsError(
                f"{aggr_root} already exists. Pass --data.force_remerge=true to rebuild it."
            )

    repo_ids = [f"local/{r.name}" for r in roots]
    print(f"[merge] aggregating {len(roots)} datasets -> {aggr_root}")
    for r in roots:
        print(f"          {r}")
    aggr_root.parent.mkdir(parents=True, exist_ok=True)
    aggregate_datasets(
        repo_ids=repo_ids,
        aggr_repo_id=data_cfg.merged_repo_id,
        roots=roots,
        aggr_root=aggr_root,
    )
    _print_summary(aggr_root)
    return aggr_root


def ensure_merged(data_cfg: DataConfig) -> Path:
    """Return the merged-dataset root, building it if absent (or if forced)."""
    aggr_root = Path(data_cfg.merged_root)
    if (aggr_root / "meta" / "info.json").exists() and not data_cfg.force_remerge:
        print(f"[merge] reusing existing merged dataset at {aggr_root}")
        return aggr_root
    return merge(data_cfg)


def main() -> None:
    cfg = draccus.parse(PipelineConfig)
    ensure_merged(cfg.data)


if __name__ == "__main__":
    main()
