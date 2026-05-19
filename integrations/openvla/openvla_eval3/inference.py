"""Shared OpenVLA load + predict helpers for integrations/openvla/scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor


def pick_device(explicit: str) -> torch.device:
    if explicit.strip():
        return torch.device(explicit)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, dtype: str) -> torch.dtype:
    if dtype != "auto":
        return getattr(torch, dtype)
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_openvla(
    model_id: str,
    *,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> tuple[AutoProcessor, Any]:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    kwargs: dict[str, Any] = dict(trust_remote_code=True, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    if device.type == "cuda":
        kwargs["attn_implementation"] = "flash_attention_2"

    try:
        vla = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs).to(device)
    except Exception:
        kwargs.pop("attn_implementation", None)
        vla = AutoModelForVision2Seq.from_pretrained(model_id, **kwargs).to(device)

    vla.eval()
    return processor, vla


def _tensor_to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    x = torch.as_tensor(image)
    if x.ndim != 3:
        raise ValueError(f"Expected HxWx3 or CxHxW image, got shape {tuple(x.shape)}")
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = x.permute(1, 2, 0)
    if x.dtype != torch.uint8:
        if float(x.max()) <= 2.0:
            x = (x * 255.0).round()
        x = x.clamp(0, 255).to(torch.uint8)
    return Image.fromarray(x.detach().cpu().numpy())


def build_inputs(
    processor,
    *,
    prompt: str,
    image_path: Path,
    device: torch.device,
    torch_dtype: torch.dtype,
):
    image = Image.open(image_path).convert("RGB")
    return _processor_inputs(processor, prompt=prompt, image=image, device=device, torch_dtype=torch_dtype)


def build_inputs_from_image(
    processor,
    *,
    prompt: str,
    image,
    device: torch.device,
    torch_dtype: torch.dtype,
):
    """Build model inputs from a PIL image or HWC/CHW tensor (robot camera frame)."""
    return _processor_inputs(
        processor,
        prompt=prompt,
        image=_tensor_to_pil(image),
        device=device,
        torch_dtype=torch_dtype,
    )


def _processor_inputs(
    processor,
    *,
    prompt: str,
    image: Image.Image,
    device: torch.device,
    torch_dtype: torch.dtype,
):
    try:
        inputs = processor(text=prompt, images=image, return_tensors="pt")
    except TypeError:
        inputs = processor(prompt, image, return_tensors="pt")  # type: ignore[arg-type]
    # Normalize to tensor dict on device
    out = {}
    if hasattr(inputs, "items"):
        iterator = inputs.items()
    else:
        iterator = dict(inputs).items()
    for k, v in iterator:
        if not hasattr(v, "to"):
            continue
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                out[k] = v.to(device=device, dtype=torch_dtype)
            else:
                out[k] = v.to(device=device)
        else:
            out[k] = v
    return out


def predict_action_numpy(
    vla,
    model_inputs: dict,
    *,
    unnorm_key: str,
    do_sample: bool,
) -> np.ndarray:
    with torch.inference_mode():
        action = vla.predict_action(**model_inputs, unnorm_key=unnorm_key, do_sample=do_sample)
    return np.asarray(action).reshape(-1)
