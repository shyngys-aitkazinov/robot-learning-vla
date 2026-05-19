"""Monkey-patch SmolVLA to add an auxiliary position-classification head.

Goal
----
Force the SmolVLA action expert's hidden state ``suffix_out`` to encode
**language-image binding** by training a small auxiliary 3-way classifier
(left / middle / right print) on top of it. This breaks the
`(observation.state, image) -> action` shortcut that v6_synth_15k learned, by
making the language tokens a necessary feature for minimizing total loss.

Why monkey-patch (not modify lerobot)
-------------------------------------
Same pattern as ``scripts/eval3_lerobot_shim.py`` and
``scripts/eval3_concat_patch.py``: lerobot is a PyPI dependency
(``./install.sh`` re-creates the venv on every fresh clone). All Eval-3
modifications must work *around* lerobot at import time, not edit the
``.venv`` files. CLAUDE.md documents this constraint.

What we patch
-------------
1. ``VLAFlowMatching.__init__`` — instantiate ``self.position_clf_head`` and
   read aux-loss config from env vars.
2. ``VLAFlowMatching.forward`` — accept an optional ``target_position`` kwarg,
   mean-pool ``suffix_out`` across the 50 chunk steps, run the aux head,
   compute CE loss + accuracy, stash on ``self._last_aux_*``.
3. ``SmolVLAPolicy.forward`` — pop ``target_position`` from the training
   batch, pass it to ``self.model.forward``, then add the weighted aux loss
   to the main loss and surface it in ``loss_dict``.

Hidden state choice
-------------------
**Option 1 (mean pool across all chunk steps)** —
``decision_h = suffix_out.mean(dim=1)`` — selected. Language matters at
every chunk step (every step is its own action prediction conditioned on
the prompt), and pooling all 50 positions gives the most stable gradient.

Env vars
--------
``EVAL3_AUX_POS_LOSS_WEIGHT``  (float, default ``0.0``)
    Multiplier on the auxiliary CE loss. ``0`` = patch is a no-op (head
    exists but contributes no gradient). Recommended start: ``0.3``.

``EVAL3_AUX_POS_DROPOUT``      (float, default ``0.1``)
    Dropout inside the classification head.

``EVAL3_AUX_POS_HIDDEN``       (int, default ``256``)
    Hidden width of the classification head.

Idempotency
-----------
``apply()`` is safe to call multiple times; it short-circuits on a second
call. Call it after ``eval3_lerobot_shim.apply()`` (so transformers is
in ``sys.modules``) and after the concat patch (so it doesn't matter which
order — they touch different modules).
"""

from __future__ import annotations

import logging
import os

_APPLIED = False


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


def apply() -> None:
    """Install the auxiliary-head patch on VLAFlowMatching and SmolVLAPolicy."""
    global _APPLIED
    if _APPLIED:
        return

    # Heavy imports deferred so importing this module itself is cheap.
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy,
        VLAFlowMatching,
        make_att_2d_masks,
    )
    from lerobot.utils.constants import (  # noqa: F401
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    # Cache the originals — used inside the patches and for re-importability checks.
    _orig_vlaf_init = VLAFlowMatching.__init__
    _orig_policy_forward = SmolVLAPolicy.forward

    # --- 0. Patch lerobot.processor.converters._extract_complementary_data --
    # The lerobot pipeline drops any batch key that isn't in a fixed allow-list
    # (task, index, task_index, episode_index, *_is_pad). Our `target_position`
    # label needs to survive batch_to_transition() → transition_to_batch() to
    # reach SmolVLAPolicy.forward, so we patch the extractor to also pick it up.
    from lerobot.processor import converters as _conv_mod

    _orig_extract = _conv_mod._extract_complementary_data

    def _patched_extract_complementary_data(batch):
        extra = _orig_extract(batch)
        if "target_position" in batch:
            extra["target_position"] = batch["target_position"]
        return extra

    _conv_mod._extract_complementary_data = _patched_extract_complementary_data

    # --- 1. Patch VLAFlowMatching.__init__ to add the aux head -------------

    def _patched_vlaf_init(self, config, rtc_processor=None):
        _orig_vlaf_init(self, config, rtc_processor=rtc_processor)

        aux_weight = _env_float("EVAL3_AUX_POS_LOSS_WEIGHT", 0.0)
        aux_dropout = _env_float("EVAL3_AUX_POS_DROPOUT", 0.1)
        aux_hidden = _env_int("EVAL3_AUX_POS_HIDDEN", 256)

        in_dim = (
            self.vlm_with_expert.expert_hidden_size
        )  # 720 for SmolVLM2-500M @ 0.75x
        self.position_clf_head = nn.Sequential(
            nn.Linear(in_dim, aux_hidden),
            nn.GELU(),
            nn.Dropout(aux_dropout),
            nn.Linear(aux_hidden, 3),  # left=0, middle=1, right=2
        )
        self._aux_loss_weight = float(aux_weight)
        self._last_aux_loss = None
        self._last_aux_acc = 0.0
        # When train_expert_only=True the parent's set_requires_grad still leaves
        # newly-added Modules trainable (default requires_grad=True), so the head
        # is trained as long as it's referenced in the loss graph.

        logging.info(
            "[eval3_smolvla_aux_head] installed position_clf_head "
            "(in_dim=%d, hidden=%d, dropout=%.2f, loss_weight=%.3f)",
            in_dim,
            aux_hidden,
            aux_dropout,
            aux_weight,
        )

    VLAFlowMatching.__init__ = _patched_vlaf_init

    # --- 2. Patch VLAFlowMatching.forward to compute the aux loss ----------

    def _patched_vlaf_forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        target_position=None,
    ):
        # Replicate the original forward body so we can intercept suffix_out
        # before the action_out_proj. Keeping this in one place beats a
        # double-forward.
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # --- Auxiliary position-classification head ------------------------
        # Mean-pool across the 50 chunk steps. Every step is supervised with
        # the same (constant per-episode) position label, which encourages
        # the expert's cross-attention to bind language tokens to image
        # regions across the whole chunk, not just the first step.
        #
        # We reset BOTH `_last_aux_loss` and `_last_aux_acc` at every call so a
        # subsequent forward without labels can't read stale state from a
        # previous batch.
        self._last_aux_loss = None
        self._last_aux_acc = None  # None = "no valid samples" (vs 0.0 = "perfect random")
        if (
            target_position is not None
            and self._aux_loss_weight > 0.0
            and hasattr(self, "position_clf_head")
        ):
            # Align dtype/device. The head is float32 by default (created on
            # CPU at __init__, moved by policy.to(device)). suffix_out is cast
            # to float32 above, but under CUDA + autocast it can still be
            # bf16 — force-match the head's dtype.
            head_dtype = next(self.position_clf_head.parameters()).dtype
            decision_h = suffix_out.mean(dim=1).to(dtype=head_dtype)  # (B, 720)
            logits = self.position_clf_head(decision_h)
            valid_mask = target_position != -100
            if valid_mask.any():
                aux_loss = F.cross_entropy(
                    logits, target_position, ignore_index=-100, reduction="mean"
                )
                with torch.no_grad():
                    preds = logits[valid_mask].argmax(dim=-1)
                    acc = (preds == target_position[valid_mask]).float().mean()
                self._last_aux_loss = aux_loss
                self._last_aux_acc = float(acc.item())
            # else: leave _last_aux_loss = None — policy.forward will skip
            # adding to the main loss AND skip logging, so we don't pollute
            # metrics with fake 0.0 values when no samples in the batch had a
            # valid target_position.

        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses

    VLAFlowMatching.forward = _patched_vlaf_forward

    # --- 3. Patch SmolVLAPolicy.forward to plumb the label through ----------

    def _patched_policy_forward(
        self, batch, noise=None, time=None, reduction: str = "mean"
    ):
        # Pop the aux label so it doesn't flow into prepare_state / prepare_action.
        target_position = batch.pop("target_position", None)
        if isinstance(target_position, torch.Tensor):
            # collate gives shape (B,) or (B, 1) depending on row shape.
            if target_position.ndim > 1:
                target_position = target_position.view(-1)
            # Cast to long for CE.
            target_position = target_position.to(dtype=torch.long)
            # Ensure it lives on the model's device.
            device = next(self.model.parameters()).device
            target_position = target_position.to(device=device)

        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION_KEY] = self._pi_aloha_encode_actions_inv(batch[ACTION_KEY])

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        actions_is_pad = batch.get("action_is_pad")

        loss_dict = {}
        losses = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            target_position=target_position,
        )
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]
        loss_dict["losses_after_forward"] = losses.clone().mean().item()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone().mean().item()

        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone().mean().item()

        aux_loss = getattr(self.model, "_last_aux_loss", None)
        aux_weight = float(getattr(self.model, "_aux_loss_weight", 0.0))
        aux_acc = getattr(self.model, "_last_aux_acc", None)  # None when no labels in batch
        # Only surface aux metrics in loss_dict if the head actually ran AND
        # produced supervisable output. This keeps metric averages honest:
        # `aux_pos_acc` reflects only batches where the head was supervised.
        aux_active = (
            aux_loss is not None
            and aux_weight > 0.0
            and aux_acc is not None
        )
        # Stash on the policy for the train-loop hook (MetricsTracker uses
        # these to update the per-step AverageMeter; see the loop patch).
        self._last_aux_dict = (
            {"aux_pos_loss": float(aux_loss.item()), "aux_pos_acc": float(aux_acc),
             "aux_pos_weight": aux_weight}
            if aux_active else None
        )

        if reduction == "none":
            per_sample_loss = losses.mean(dim=(1, 2))  # (B,)
            if aux_active:
                per_sample_loss = per_sample_loss + aux_weight * aux_loss.expand_as(
                    per_sample_loss
                )
                loss_dict["aux_pos_loss"] = float(aux_loss.item())
                loss_dict["aux_pos_acc"] = float(aux_acc)
                loss_dict["aux_pos_weight"] = aux_weight
            loss_dict["loss"] = per_sample_loss.mean().item()
            return per_sample_loss, loss_dict
        else:
            loss = losses.mean()
            if aux_active:
                loss = loss + aux_weight * aux_loss
                loss_dict["aux_pos_loss"] = float(aux_loss.item())
                loss_dict["aux_pos_acc"] = float(aux_acc)
                loss_dict["aux_pos_weight"] = aux_weight
            loss_dict["loss"] = loss.item()
            return loss, loss_dict

    SmolVLAPolicy.forward = _patched_policy_forward

    # Imports needed by the patched policy forward — pulled into module scope
    # so each call doesn't re-import. (`ACTION` from lerobot.utils.constants is
    # the canonical action key in the batch.)
    global ACTION_KEY
    from lerobot.utils.constants import ACTION as _ACTION

    ACTION_KEY = _ACTION

    # --- 4. Patch lerobot.utils.logging_utils.MetricsTracker.__init__ -------
    # Add aux_pos_loss + aux_pos_acc AverageMeters so they're tracked in the
    # per-step INFO log line ("loss:0.10 aux_pos_loss:0.03 aux_pos_acc:1.00")
    # and exported to wandb via train_metrics.to_dict() when wandb is enabled.
    from lerobot.utils.logging_utils import MetricsTracker, AverageMeter

    _orig_mt_init = MetricsTracker.__init__

    def _patched_mt_init(
        self,
        batch_size,
        num_frames,
        num_episodes,
        metrics,
        initial_step=0,
        accelerator=None,
    ):
        metrics = dict(metrics)  # don't mutate caller's dict
        metrics.setdefault("aux_pos_loss", AverageMeter("aux_pos_loss", ":.3f"))
        metrics.setdefault("aux_pos_acc", AverageMeter("aux_pos_acc", ":.3f"))
        _orig_mt_init(
            self,
            batch_size,
            num_frames,
            num_episodes,
            metrics,
            initial_step=initial_step,
            accelerator=accelerator,
        )

    MetricsTracker.__init__ = _patched_mt_init

    # --- 5. Wire aux metrics into MetricsTracker via a side channel ---------
    # lerobot's train loop calls `train_tracker.loss = loss` once per step
    # right after policy.forward. MetricsTracker.__setattr__ routes metric
    # assignments to AverageMeter.update(). We register every SmolVLAPolicy
    # instance into a global ref, then piggy-back on the `loss` assignment
    # to also update aux meters from the policy's last forward.
    _aux_pump_state = {"policy_ref": None}

    _orig_policy_init = SmolVLAPolicy.__init__

    def _patched_policy_init(self, *args, **kwargs):
        _orig_policy_init(self, *args, **kwargs)
        _aux_pump_state["policy_ref"] = self
        self._last_aux_dict = None  # ensure attribute always exists

    SmolVLAPolicy.__init__ = _patched_policy_init

    _orig_mt_setattr = MetricsTracker.__setattr__
    # Lazy import — only needed if the user enabled state-aug curriculum.
    _curriculum_counter = None

    def _get_curriculum_counter():
        nonlocal _curriculum_counter
        if _curriculum_counter is None:
            try:
                from eval3_dataset_prep import get_curriculum_step_counter
                _curriculum_counter = get_curriculum_step_counter()
            except ImportError:
                _curriculum_counter = False  # sentinel: import failed, don't retry
        return _curriculum_counter if _curriculum_counter is not False else None

    def _patched_mt_setattr(self, name, value):
        _orig_mt_setattr(self, name, value)
        # When the train loop sets `loss`, do two things:
        #   1. Pump aux metrics from the most recent policy.forward.
        #   2. Advance the state-aug curriculum step counter (shared mp.Value
        #      visible to DataLoader workers; see eval3_dataset_prep.StateAugmenter).
        if name == "loss":
            pol = _aux_pump_state.get("policy_ref")
            aux = getattr(pol, "_last_aux_dict", None) if pol is not None else None
            if aux is not None:
                if "aux_pos_loss" in aux and "aux_pos_loss" in self.metrics:
                    _orig_mt_setattr(self, "aux_pos_loss", aux["aux_pos_loss"])
                if "aux_pos_acc" in aux and "aux_pos_acc" in self.metrics:
                    _orig_mt_setattr(self, "aux_pos_acc", aux["aux_pos_acc"])
            # Curriculum step advance — MetricsTracker.steps is the trainer's
            # source-of-truth step counter, already incremented by the trainer
            # before this setattr fires (via tracker.step() which advances
            # `self.steps` directly).
            counter = _get_curriculum_counter()
            if counter is not None:
                counter.set_step(int(self.steps))

    MetricsTracker.__setattr__ = _patched_mt_setattr

    # --- 6. Sanity guard: detect if upstream forward signature drifted ------
    # The patched VLAFlowMatching.forward copies the body of the original. If
    # lerobot adds new positional args we'll silently break. Check at apply()
    # time and warn loudly.
    import inspect

    try:
        _sig = inspect.signature(_orig_vlaf_init)
        # Just verify __init__ takes (self, config, rtc_processor=None) shape
        # we depend on. Same check for forward:
        _fsig = inspect.signature(VLAFlowMatching.forward)  # this is our patched one now
        _expected_params = {"self", "images", "img_masks", "lang_tokens", "lang_masks",
                            "state", "actions", "noise", "time", "target_position"}
        _got_params = set(_fsig.parameters.keys())
        _missing = _expected_params - _got_params
        if _missing:
            logging.warning(
                "[eval3_smolvla_aux_head] expected VLAFlowMatching.forward to take %s, "
                "but it's missing %s — upstream signature may have drifted",
                _expected_params, _missing,
            )
    except Exception as exc:
        logging.warning("[eval3_smolvla_aux_head] could not inspect upstream signature: %s", exc)

    _APPLIED = True


if __name__ == "__main__":
    # Manual sanity check: ``python scripts/eval3_smolvla_aux_head.py``.
    # Requires the shim to be applied first.
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    import eval3_lerobot_shim

    eval3_lerobot_shim.apply()
    apply()
    print("[eval3_smolvla_aux_head] applied OK")
