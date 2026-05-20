"""Load Eval3 OpenVLA LoRA checkpoints for closed-loop deploy."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openvla_eval3.inference import pick_device, pick_dtype


@dataclass(frozen=True)
class OpenVLACheckpointBundle:
    processor: Any
    model: Any
    unnorm_key: str
    chunk_size: int
    vla_path: str
    checkpoint_dir: Path
    dataset_statistics: dict[str, Any]
    train_config: dict[str, Any]


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _unwrap_stats(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if not payload:
        raise ValueError("dataset_statistics.json is empty.")
    if "action" in payload and "proprio" in payload:
        key = str(payload.get("unnorm_key", "eval3_so101_new66"))
        return key, payload
    if len(payload) == 1:
        key, body = next(iter(payload.items()))
        if isinstance(body, dict) and "action" in body:
            return str(key), body
    raise ValueError("Could not parse OpenVLA dataset statistics (expected unnorm_key → action/proprio).")


def resolve_openvla_checkpoint(checkpoint_path: str) -> tuple[Path, Path | None, Path | None]:
    """Return (adapter_dir, train_config_path, dataset_statistics_path)."""
    path = Path(checkpoint_path).expanduser()
    if path.is_dir():
        if (path / "adapter_config.json").is_file():
            return path, path / "train_config.json", path / "dataset_statistics.json"
        candidates = sorted(path.glob("checkpoints/*"))
        for candidate in reversed(candidates):
            if (candidate / "adapter_config.json").is_file():
                return candidate, candidate / "train_config.json", candidate / "dataset_statistics.json"
        raise FileNotFoundError(f"No OpenVLA LoRA adapter under {path} (expected adapter_config.json).")

    from huggingface_hub import hf_hub_download

    repo_id = str(checkpoint_path)
    for name in ("adapter_config.json", "train_config.json", "dataset_statistics.json"):
        hf_hub_download(repo_id, name)
    cache_root = Path(hf_hub_download(repo_id, "adapter_config.json")).parent
    return cache_root, cache_root / "train_config.json", cache_root / "dataset_statistics.json"


def reshape_eval3_action(raw: np.ndarray, *, chunk_size: int) -> np.ndarray:
    """Flattened or chunked OpenVLA output → ``[chunk_size, 6]`` SO-101 joints (degrees)."""
    vec = np.asarray(raw, dtype=np.float64).reshape(-1)
    expected = int(chunk_size) * 7
    if vec.size == expected:
        return vec.reshape(int(chunk_size), 7)[:, :6]
    if vec.size == 7:
        return vec[:6].reshape(1, 6)
    if vec.size == 6:
        return vec.reshape(1, 6)
    raise ValueError(f"OpenVLA action length {vec.size}; expected {expected}, 7, or 6.")


def load_eval3_openvla(
    checkpoint_path: str,
    *,
    openvla_src: str = "",
    device: str = "",
    dtype: str = "auto",
    merge_lora: bool = False,
) -> OpenVLACheckpointBundle:
    """Load base OpenVLA-7B + Eval3 LoRA adapter and attach training statistics."""
    adapter_dir, train_cfg_path, stats_path = resolve_openvla_checkpoint(checkpoint_path)
    train_config = _read_json(train_cfg_path)
    stats_payload = _read_json(stats_path)
    unnorm_key, stats_body = _unwrap_stats(stats_payload)
    norm_stats = {unnorm_key: stats_body}

    integrations_root = Path(__file__).resolve().parents[1]
    train_scripts = integrations_root / "scripts"
    if str(train_scripts) not in sys.path:
        sys.path.insert(0, str(train_scripts))
    from train_eval3_lora import register_openvla  # noqa: WPS433

    deps = register_openvla(openvla_src)
    torch_device = pick_device(device)
    torch_dtype = pick_dtype(torch_device, dtype)

    vla_path = str(train_config.get("vla_path", "openvla/openvla-7b"))
    chunk_size = int(train_config.get("chunk_size", stats_body.get("chunk_size", 10)))

    processor = deps["AutoProcessor"].from_pretrained(str(adapter_dir), trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if torch_device.type == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"

    try:
        base = deps["AutoModelForVision2Seq"].from_pretrained(vla_path, **model_kwargs)
    except Exception:
        model_kwargs.pop("attn_implementation", None)
        base = deps["AutoModelForVision2Seq"].from_pretrained(vla_path, **model_kwargs)
    try:
        from peft import PeftModel
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("peft is required to load Eval3 OpenVLA LoRA checkpoints.") from exc

    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
    if merge_lora:
        model = model.merge_and_unload()

    # Inject the Eval3 normalization stats so `predict_action(unnorm_key=...)`
    # can resolve our SO-101 key. Critically, `predict_action` /
    # `get_action_dim` run BOUND TO THE BASE OpenVLA model (PeftModel and
    # LoraModel just delegate the attribute via __getattr__), so they read
    # `self.norm_stats` off the base — setting it only on the PeftModel
    # wrapper is not enough. Set it on every layer of the wrapping
    # (PeftModel -> LoraModel -> base OpenVLA) plus each `.config`.
    _norm_targets = []
    _node = model
    for _ in range(4):  # PeftModel -> base_model(LoraModel) -> model(base VLA)
        if _node is None or id(_node) in {id(t) for t in _norm_targets}:
            break
        _norm_targets.append(_node)
        _node = getattr(_node, "model", None) or getattr(_node, "base_model", None)
    for _t in _norm_targets:
        try:
            _t.norm_stats = norm_stats
        except Exception:  # noqa: BLE001 - some wrappers forbid attr set
            pass
        _cfg = getattr(_t, "config", None)
        if _cfg is not None:
            try:
                _cfg.norm_stats = norm_stats
            except Exception:  # noqa: BLE001
                pass
    model = model.to(torch_device)
    model.eval()

    return OpenVLACheckpointBundle(
        processor=processor,
        model=model,
        unnorm_key=unnorm_key,
        chunk_size=chunk_size,
        vla_path=vla_path,
        checkpoint_dir=adapter_dir,
        dataset_statistics=stats_body,
        train_config=train_config,
    )
