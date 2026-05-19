#!/usr/bin/env python3
"""Fast prompt-collapse check: same image, three celebrity prompts, action L2.

A minimal version of ``tools/eval3_synthetic_ood_test.py`` intended to confirm
the v9 ChArUco "model ignores language" hypothesis with the lowest possible
download / inference cost.

Defaults:
  - one frame per dataset (3 frames total, mid-episode-0 from each celeb repo)
  - "original" scene only (no bg / print augmentation)
  - reports per-frame pairwise action L2 across the 3 canonical prompts

A prompt-collapsed policy returns near-zero pairwise L2; a working
language-following policy returns clearly non-zero values (the v3-fresh
checkpoint baseline is ~30-60 deg on the same gates).

Usage::

    .venv/bin/python tools/eval3_promptswap_quick.py \\
        --policy_path outputs/train/eval3_3way_50k_v3_fresh/checkpoints/050000/pretrained_model \\
        --policy_device mps

    .venv/bin/python tools/eval3_promptswap_quick.py \\
        --policy_path RobotLearningVLA/eval3-vla-v9-smolvla-fresh-charuco-50k \\
        --policy_device mps
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from eval3_lerobot_shim import apply as _shim_apply  # noqa: E402
_shim_apply()

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.utils.control_utils import predict_action  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402
from lerobot.processor.rename_processor import rename_stats  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402


PROMPTS = [
    ("swift", "Place the coke on Taylor Swift"),
    ("lecun", "Place the coke on Yann LeCun"),
    ("obama", "Place the coke on Barack Obama"),
]
DATASETS = [
    ("RobotLearningVLA/taylor_swift_1", "swift"),
    ("RobotLearningVLA/yann_lecun_1",   "lecun"),
    ("RobotLearningVLA/barack_obama_1", "obama"),
]


def _model_config_path(policy_path: str) -> str:
    p = Path(policy_path)
    if p.exists():
        return str(p / "config.json")
    return hf_hub_download(policy_path, "config.json")


def _build_policy(policy_path: str, device: str):
    """Mirror the patient init dance from tools/eval3_synthetic_ood_test.py."""
    import json as _json
    from dataclasses import fields as _dc_fields
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.configs.types import FeatureType, PolicyFeature

    with open(_model_config_path(policy_path), encoding="utf-8") as f:
        raw_cfg = _json.load(f)
    raw_cfg.pop("type", None)
    valid_keys = {f.name for f in _dc_fields(SmolVLAConfig)}
    init_kwargs = {k: v for k, v in raw_cfg.items() if k in valid_keys}
    policy_cfg = SmolVLAConfig(**init_kwargs)

    def _restore_features(d):
        return {k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
                for k, v in (d or {}).items()}

    policy_cfg.input_features = _restore_features(raw_cfg.get("input_features"))
    policy_cfg.output_features = _restore_features(raw_cfg.get("output_features"))
    policy_cfg.device = device
    policy_cfg.push_to_hub = False
    policy_cfg.compile_model = False
    policy_cfg.pretrained_path = policy_path

    rename_map = {"observation.images.front": "observation.images.camera1"}
    ds_meta = LeRobotDatasetMetadata("RobotLearningVLA/taylor_swift_1")
    policy = make_policy(policy_cfg, ds_meta=ds_meta, rename_map=rename_map)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=policy_path,
        dataset_stats=rename_stats(ds_meta.stats, rename_map),
        preprocessor_overrides={
            "device_processor": {"device": policy_cfg.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    dev = get_safe_torch_device(policy.config.device)
    return policy, preprocessor, postprocessor, dev


def _obs_from_row(row: dict) -> dict:
    img_key = "observation.images.front"
    img = row[img_key]
    if isinstance(img, torch.Tensor):
        arr = (img.numpy().transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
    else:
        arr = np.asarray(img)
    state = row["observation.state"]
    state_arr = state.numpy() if isinstance(state, torch.Tensor) else np.asarray(state)
    return {img_key: arr, "observation.state": state_arr}


def _normalize_action(a) -> np.ndarray:
    arr = a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)
    flat = arr.reshape(-1)
    if flat.size > 6:
        flat = arr[0] if arr.ndim >= 2 else flat[:6]
    return flat.astype(float)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy_path", required=True)
    ap.add_argument("--policy_device", default="mps")
    ap.add_argument("--out", default="outputs/eval3_diag/promptswap_quick.json", type=Path)
    ap.add_argument("--frames_per_ds", type=int, default=1)
    args = ap.parse_args()

    print(f"loading policy from {args.policy_path}  (device={args.policy_device})", flush=True)
    policy, preprocessor, postprocessor, device = _build_policy(args.policy_path, args.policy_device)

    rows_info: list[dict] = []
    for repo, slug in DATASETS:
        print(f"  loading dataset {repo}", flush=True)
        ds = LeRobotDataset(repo, video_backend="pyav", episodes=[0])
        n = len(ds)
        idxs = [int(n * (i + 1) / (args.frames_per_ds + 1)) for i in range(args.frames_per_ds)]
        for idx in idxs:
            row = ds[idx]
            obs = _obs_from_row(row)
            per_prompt_actions: dict[str, list[float]] = {}
            for label, prompt in PROMPTS:
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                a = predict_action(
                    observation=obs,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=prompt,
                )
                per_prompt_actions[label] = _normalize_action(a).tolist()
            pair_l2 = {}
            for (l1, a1), (l2, a2) in combinations(per_prompt_actions.items(), 2):
                pair_l2[f"{l1}-{l2}"] = float(np.linalg.norm(np.array(a1) - np.array(a2)))
            rows_info.append({"dataset_slug": slug, "frame_idx": idx,
                              "actions": per_prompt_actions, "pair_l2": pair_l2})
            print(f"    [{slug} f{idx}]  pair_l2: " +
                  "  ".join(f"{k}={v:6.2f}" for k, v in pair_l2.items()), flush=True)

    pair_means = {}
    for pair_name in ("swift-lecun", "swift-obama", "lecun-obama"):
        vals = [r["pair_l2"][pair_name] for r in rows_info]
        pair_means[pair_name] = float(np.mean(vals)) if vals else 0.0
    min_pair = min(pair_means.values())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"policy_path": args.policy_path,
                                    "frames": rows_info,
                                    "pair_means": pair_means,
                                    "min_pair_mean": min_pair}, indent=2))
    print("\n=== mean pairwise action L2 across {} frames ===".format(len(rows_info)))
    for p, v in pair_means.items():
        print(f"  {p:<12} {v:6.2f}")
    print(f"  {'min':<12} {min_pair:6.2f}")
    if min_pair < 5.0:
        print("\nVERDICT: prompt-collapse — min pair L2 < 5 deg. The policy "
              "ignores the celebrity name; same image -> same action regardless of prompt.")
        return 2
    if min_pair < 30.0:
        print("\nVERDICT: weak language conditioning — min pair L2 < 30 deg. "
              "Below the v3-fresh baseline; celebrity selection is unreliable.")
        return 1
    print("\nVERDICT: language is influencing actions (min pair L2 >= 30 deg).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
