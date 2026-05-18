"""Work around LeRobot compatibility issues used by the Eval3 scripts.

Import this and call ``apply()`` before any ``lerobot.policies`` import when SmolVLA needs
``transformers`` installed.

Background: with `transformers` installed, importing `lerobot.policies` cascades into
`lerobot.policies.groot.groot_n1`, whose dataclass inherits from a transformers parent that
breaks Python dataclass field ordering (`non-default argument 'backbone_cfg' follows default
argument`). The naive workaround is to set `lerobot.utils.import_utils._transformers_available
= False` before any policy import, but that also breaks SmolVLA's `TokenizerProcessorStep`
which legitimately needs transformers.

Surgical fix: pre-load the tokenizer processor with the flag True (so transformers is properly
imported and cached in sys.modules), then briefly flip the flag False to load GROOT with stubs
(avoiding the dataclass crash), then restore the flag for the rest of the process. Subsequent
imports of GROOT just reuse the cached stubs from sys.modules; SmolVLA gets real transformers.

The shim also patches a SmolVLA processor/model edge case observed with the PyPI version used
in this project: language token tensors and language attention masks can enter the model with
different sequence lengths when loading pretrained processors. SmolVLA later builds a 2-D
attention mask from both tensors and crashes if those lengths differ. We align the mask to the
token tensor in the policy boundary, deriving validity from the tokenizer pad id when available.
"""

from __future__ import annotations

import logging


_SMOLVLA_LANG_MASK_PATCHED = False
_SMOLVLA_LANG_MASK_LOGGED = False
_SMOLVLA_PREFIX_MASK_LOGGED = False


def _patch_smolvla_language_masks() -> None:
    global _SMOLVLA_LANG_MASK_PATCHED

    if _SMOLVLA_LANG_MASK_PATCHED:
        return

    try:
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    except Exception:
        return

    original_forward = SmolVLAPolicy.forward
    original_get_action_chunk = SmolVLAPolicy._get_action_chunk
    original_embed_prefix = VLAFlowMatching.embed_prefix

    def _align_language_mask(policy: SmolVLAPolicy, batch: dict) -> None:
        global _SMOLVLA_LANG_MASK_LOGGED

        token_key = f"{OBS_LANGUAGE_TOKENS}"
        mask_key = f"{OBS_LANGUAGE_ATTENTION_MASK}"
        tokens = batch.get(token_key)
        masks = batch.get(mask_key)
        if not isinstance(tokens, torch.Tensor) or not isinstance(masks, torch.Tensor):
            return

        if tokens.shape == masks.shape:
            if masks.dtype is not torch.bool:
                batch[mask_key] = masks.to(dtype=torch.bool)
            return

        pad_token_id = None
        try:
            pad_token_id = policy.model.vlm_with_expert.processor.tokenizer.pad_token_id
        except Exception:
            pad_token_id = None

        if pad_token_id is not None:
            aligned = tokens.ne(pad_token_id)
        else:
            aligned = torch.zeros(tokens.shape, dtype=torch.bool, device=tokens.device)
            if tokens.ndim == masks.ndim and tokens.shape[:-1] == masks.shape[:-1]:
                common = min(tokens.shape[-1], masks.shape[-1])
                aligned[..., :common] = masks[..., :common].to(device=tokens.device, dtype=torch.bool)
                if tokens.shape[-1] > common:
                    aligned[..., common:] = True
            else:
                aligned.fill_(True)

        if not _SMOLVLA_LANG_MASK_LOGGED:
            logging.warning(
                "Eval3 SmolVLA shim aligned language mask shape %s -> %s "
                "(token shape %s, pad_token_id=%s)",
                tuple(masks.shape),
                tuple(aligned.shape),
                tuple(tokens.shape),
                pad_token_id,
            )
            _SMOLVLA_LANG_MASK_LOGGED = True

        batch[mask_key] = aligned.to(device=tokens.device, dtype=torch.bool)

    def patched_forward(self, batch, *args, **kwargs):
        _align_language_mask(self, batch)
        return original_forward(self, batch, *args, **kwargs)

    def patched_get_action_chunk(self, batch, *args, **kwargs):
        _align_language_mask(self, batch)
        return original_get_action_chunk(self, batch, *args, **kwargs)

    def _resize_sequence_mask(mask: torch.Tensor, target_len: int, fill_value: bool) -> torch.Tensor:
        if mask.shape[1] == target_len:
            return mask
        if mask.shape[1] > target_len:
            return mask[:, :target_len]
        pad = torch.full(
            (mask.shape[0], target_len - mask.shape[1]),
            fill_value,
            dtype=mask.dtype,
            device=mask.device,
        )
        return torch.cat([mask, pad], dim=1)

    def patched_embed_prefix(self, *args, **kwargs):
        global _SMOLVLA_PREFIX_MASK_LOGGED

        embs, pad_masks, att_masks = original_embed_prefix(self, *args, **kwargs)
        target_len = embs.shape[1]
        if pad_masks.shape[1] == target_len and att_masks.shape[1] == target_len:
            return embs, pad_masks, att_masks

        old_pad_shape = tuple(pad_masks.shape)
        old_att_shape = tuple(att_masks.shape)
        pad_masks = _resize_sequence_mask(pad_masks, target_len, fill_value=False).to(dtype=torch.bool)
        fill_att = bool(att_masks[:, -1:].any().item()) if att_masks.shape[1] else False
        att_masks = _resize_sequence_mask(att_masks, target_len, fill_value=fill_att).to(dtype=torch.bool)

        if not _SMOLVLA_PREFIX_MASK_LOGGED:
            logging.warning(
                "Eval3 SmolVLA shim aligned prefix masks to embeddings: "
                "embs=%s pad=%s->%s att=%s->%s",
                tuple(embs.shape),
                old_pad_shape,
                tuple(pad_masks.shape),
                old_att_shape,
                tuple(att_masks.shape),
            )
            _SMOLVLA_PREFIX_MASK_LOGGED = True

        return embs, pad_masks, att_masks

    SmolVLAPolicy.forward = patched_forward
    SmolVLAPolicy._get_action_chunk = patched_get_action_chunk
    VLAFlowMatching.embed_prefix = patched_embed_prefix
    _SMOLVLA_LANG_MASK_PATCHED = True


def apply() -> None:
    import lerobot.utils.import_utils as import_utils

    saved = import_utils._transformers_available

    # 1) Load TokenizerProcessorStep WITH transformers so it captures the real transformers symbols.
    import lerobot.processor.tokenizer_processor  # noqa: F401

    # 2) Disable transformers detection just long enough to import GROOT modules,
    #    so their dataclass inheritance uses stub parents instead of the broken HF PretrainedConfig.
    import_utils._transformers_available = False
    try:
        import lerobot.policies.groot.groot_n1  # noqa: F401
        import lerobot.policies.groot.action_head.flow_matching_action_head  # noqa: F401
        import lerobot.policies.groot.modeling_groot  # noqa: F401
        import lerobot.policies.groot.processor_groot  # noqa: F401
    finally:
        # 3) Restore the flag so later imports (e.g. SmolVLA's processor pipeline) work normally.
        import_utils._transformers_available = saved

    # Importing ``lerobot.policies`` while transformers is hidden may also import
    # XVLA's config module with ``Florence2Config = None`` cached at module scope.
    # Restore it so the same training wrapper can launch XVLA/Flower-style runs.
    try:
        import lerobot.policies.xvla.configuration_xvla as xvla_cfg
        from lerobot.policies.xvla.configuration_florence2 import Florence2Config
    except Exception:
        pass
    else:
        xvla_cfg.Florence2Config = Florence2Config

    _patch_smolvla_language_masks()
