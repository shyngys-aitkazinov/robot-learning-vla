"""Configurable surface for the SmolVLA fine-tuning pipeline.

Everything tunable lives here as plain dataclasses with sensible defaults.
Edit the defaults below, or override any field on the command line — every
entry point parses this with draccus, so nested fields use dotted paths::

    python fresh_start/train.py --training.steps=30000 --finetune.preset=full
    python fresh_start/train.py --augmentation.noise.sigma=0.06
    python fresh_start/train.py --augmentation.spatial_lighting.enable=true

Nothing here is decided for you — `finetune.preset`, the augmentation set, and
the training budget are all knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The 9 real teleop datasets (3 celebrities x 3 can positions), stored as repo-root
# relative paths. merge_datasets.py resolves them against the repo root.
V4_SOURCE_DATASETS: list[str] = [
    "datasets/dataset_v4_taylor_left",
    "datasets/dataset_v4_taylor_middle",
    "datasets/dataset_v4_taylor_right",
    "datasets/dataset_v4_barack_left",
    "datasets/dataset_v4_barack_middle",
    "datasets/dataset_v4_barack_right",
    "datasets/dataset_v4_yann_left",
    "datasets/dataset_v4_yann_middle",
    "datasets/dataset_v4_yann_right",
]

# Maps finetune.preset -> (freeze_vision_encoder, train_expert_only).
# train_state_proj is kept True in every preset (small newly-init projection).
_FINETUNE_PRESETS: dict[str, tuple[bool, bool]] = {
    # Freeze the SmolVLM backbone + SigLIP vision encoder; train the action
    # expert only. SmolVLA's stock recipe — fastest, most robust on small data.
    "expert_only": (True, True),
    # Also train the VLM->expert pathway; vision encoder still frozen.
    "expert_plus_connectors": (True, False),
    # Unfreeze the vision encoder too. Most adaptation, most overfit risk.
    "full": (False, False),
}


@dataclass
class DataConfig:
    """Which datasets to train on and where the merged corpus lives."""

    # Source datasets merged into one training corpus (repo-root relative).
    source_roots: list[str] = field(default_factory=lambda: list(V4_SOURCE_DATASETS))
    # Where the merged corpus is written / read from. Scratch space, uncommitted.
    merged_root: str = "/ephemeral/datasets/dataset_v4_all9"
    # Label only — the merged dataset is local and never pushed.
    merged_repo_id: str = "local/dataset_v4_all9"
    # Video decoding backend. "pyav" is self-contained (PyAV bundles ffmpeg);
    # "torchcodec" is faster but needs system ffmpeg libs installed.
    video_backend: str = "pyav"
    # Re-merge even if merged_root already exists (deletes it first).
    force_remerge: bool = False


@dataclass
class FinetuneConfig:
    """How much of SmolVLA to train."""

    # One of: "expert_only", "expert_plus_connectors", "full". See _FINETUNE_PRESETS.
    preset: str = "expert_only"
    # Direct overrides; leave as None to derive from `preset`.
    freeze_vision_encoder: bool | None = None
    train_expert_only: bool | None = None
    # The small state-projection layer is newly initialised; train it by default.
    train_state_proj: bool = True


@dataclass
class TrainingConfig:
    """Training budget + run bookkeeping."""

    policy_base: str = "lerobot/smolvla_base"  # pretrained checkpoint to fine-tune
    steps: int = 20_000
    batch_size: int = 64
    save_freq: int = 2_000
    log_freq: int = 100
    num_workers: int = 8
    seed: int = 1_000
    device: str = "cuda"
    lr: float | None = None  # None -> SmolVLA preset (1e-4)
    output_dir: str = "/ephemeral/outputs/smolvla_v4_all9"
    job_name: str = "smolvla_v4_all9"
    wandb_enable: bool = False
    wandb_project: str = "smolvla-eval3"
    wandb_entity: str | None = None


@dataclass
class LightingAug:
    """Photometric / lighting jitter (torchvision ColorJitter, built-in).

    Each attribute becomes a separate ColorJitter transform so the sampler can
    pick any subset. Ranges are (min, max) multipliers (hue is additive).
    """

    enable: bool = True
    brightness: list[float] = field(default_factory=lambda: [0.6, 1.4])
    contrast: list[float] = field(default_factory=lambda: [0.7, 1.3])
    saturation: list[float] = field(default_factory=lambda: [0.6, 1.4])
    hue: list[float] = field(default_factory=lambda: [-0.04, 0.04])
    weight: float = 1.0  # sampling weight applied to each lighting transform


@dataclass
class NoiseAug:
    """Additive Gaussian sensor noise (torchvision v2 GaussianNoise)."""

    enable: bool = True
    sigma: float = 0.04  # stddev on the [0, 1] pixel scale (~10/255)
    mean: float = 0.0
    weight: float = 1.0


@dataclass
class BlurAug:
    """Random Gaussian (defocus) blur."""

    enable: bool = True
    kernel_size: int = 3
    sigma: list[float] = field(default_factory=lambda: [0.1, 1.5])
    weight: float = 0.5


@dataclass
class SpatialLightingAug:
    """Spatially-varying illumination (custom transform — see transforms_ext.py).

    Off by default; global ColorJitter covers the basic "lighting" need. Enable
    for vignette + directional brightness gradients.
    """

    enable: bool = False
    vignette: list[float] = field(default_factory=lambda: [0.0, 0.5])
    gradient: list[float] = field(default_factory=lambda: [0.0, 0.4])
    weight: float = 0.6


@dataclass
class AugmentationConfig:
    """The visual augmentation stack applied to camera frames during training."""

    enable: bool = True  # master switch
    max_num_transforms: int = 3  # how many transforms are sampled per frame
    random_order: bool = True
    lighting: LightingAug = field(default_factory=LightingAug)
    noise: NoiseAug = field(default_factory=NoiseAug)
    blur: BlurAug = field(default_factory=BlurAug)
    spatial_lighting: SpatialLightingAug = field(default_factory=SpatialLightingAug)


@dataclass
class PipelineConfig:
    """Top-level config — what every entry point parses."""

    data: DataConfig = field(default_factory=DataConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


def resolve_finetune(finetune: FinetuneConfig) -> tuple[bool, bool, bool]:
    """Resolve (freeze_vision_encoder, train_expert_only, train_state_proj).

    The preset supplies defaults; explicit non-None fields override it.
    """
    if finetune.preset not in _FINETUNE_PRESETS:
        raise ValueError(
            f"Unknown finetune.preset={finetune.preset!r}. "
            f"Choose one of {sorted(_FINETUNE_PRESETS)}."
        )
    freeze_default, expert_only_default = _FINETUNE_PRESETS[finetune.preset]
    freeze = freeze_default if finetune.freeze_vision_encoder is None else finetune.freeze_vision_encoder
    expert_only = (
        expert_only_default if finetune.train_expert_only is None else finetune.train_expert_only
    )
    return freeze, expert_only, finetune.train_state_proj
