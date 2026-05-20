"""SmolVLA fine-tuning launcher — the pipeline entry point.

Fine-tunes `lerobot/smolvla_base` on the merged `dataset_v4` corpus with the
configured visual augmentation. Run it (defaults are sensible; override any
field — see config.py):

    python fresh_start/train.py
    python fresh_start/train.py --training.steps=30000 --finetune.preset=full
    python fresh_start/train.py --augmentation.noise.sigma=0.06
    python fresh_start/train.py --training.batch_size=4 --training.steps=10   # smoke

It runs lerobot's training loop in-process so the GROOT import shim stays
active in the interpreter that imports `lerobot.policies`.
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- bootstrap: make sibling modules importable when run as a script ---
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import groot_shim  # noqa: E402

groot_shim.apply()  # MUST run before any `lerobot.policies` import

import transforms_ext  # noqa: E402

transforms_ext.register()  # expose custom transforms to lerobot by name

import draccus  # noqa: E402

from augmentation import build_image_transforms_config, summarize  # noqa: E402
from config import PipelineConfig, resolve_finetune  # noqa: E402
from merge_datasets import ensure_merged  # noqa: E402


def run(cfg: PipelineConfig) -> Path:
    """Fine-tune SmolVLA according to `cfg`. Returns the output directory."""
    # 1. Ensure the merged training corpus exists (builds it on first run).
    merged_root = ensure_merged(cfg.data)

    # 2. lerobot imports — only after the GROOT shim is applied.
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.scripts.lerobot_train import train as lerobot_train

    output_dir = Path(cfg.training.output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"Output dir {output_dir} already exists — lerobot refuses to overwrite a "
            f"run. Move it or set --training.output_dir to a fresh path."
        )

    # 3. Policy config: load pretrained SmolVLA, then apply the fine-tune preset.
    freeze_vision, expert_only, train_state_proj = resolve_finetune(cfg.finetune)
    policy_cfg = PreTrainedConfig.from_pretrained(cfg.training.policy_base)
    policy_cfg.pretrained_path = Path(cfg.training.policy_base)  # -> load weights
    # smolvla_base ships input_features for its original 3-camera rig
    # (camera1/2/3). Clear them so make_policy re-derives input_features from
    # THIS dataset — a single `observation.images.front` camera. SmolVLA is
    # camera-count agnostic, so the pretrained weights still load cleanly.
    policy_cfg.input_features = {}
    policy_cfg.device = cfg.training.device
    policy_cfg.push_to_hub = False
    policy_cfg.freeze_vision_encoder = freeze_vision
    policy_cfg.train_expert_only = expert_only
    policy_cfg.train_state_proj = train_state_proj
    if cfg.training.lr is not None:
        policy_cfg.optimizer_lr = cfg.training.lr

    # 4. Dataset config: the merged corpus + the visual augmentation stack.
    dataset_cfg = DatasetConfig(
        repo_id=cfg.data.merged_repo_id,
        root=str(merged_root),
        image_transforms=build_image_transforms_config(cfg.augmentation),
        video_backend=cfg.data.video_backend,
    )

    # 5. Assemble the lerobot training config. Passing a real TrainPipelineConfig
    #    instance to the (parser-wrapped) train() makes it skip argv parsing.
    train_cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=cfg.training.job_name,
        seed=cfg.training.seed,
        num_workers=cfg.training.num_workers,
        batch_size=cfg.training.batch_size,
        steps=cfg.training.steps,
        eval_freq=0,  # no simulation env for real-robot data
        log_freq=cfg.training.log_freq,
        save_freq=cfg.training.save_freq,
        wandb=WandBConfig(
            enable=cfg.training.wandb_enable,
            project=cfg.training.wandb_project,
            entity=cfg.training.wandb_entity,
        ),
    )

    print(f"[train] {summarize(cfg.augmentation)}")
    print(
        f"[train] finetune: preset={cfg.finetune.preset} "
        f"freeze_vision_encoder={freeze_vision} train_expert_only={expert_only} "
        f"train_state_proj={train_state_proj}"
    )
    print(
        f"[train] steps={cfg.training.steps} batch_size={cfg.training.batch_size} "
        f"device={cfg.training.device}"
    )
    print(f"[train] base={cfg.training.policy_base}  output_dir={output_dir}")

    # 6. Run lerobot's training loop in-process.
    lerobot_train(train_cfg)
    return output_dir


def main() -> None:
    run(draccus.parse(PipelineConfig))


if __name__ == "__main__":
    main()
