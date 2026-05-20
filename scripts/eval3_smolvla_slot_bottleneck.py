"""Monkey-patch SmolVLA with a prefix-side slot-bottleneck head.

Why this exists
---------------
The earlier ``eval3_smolvla_aux_head.py`` put a 3-way position classifier on
``suffix_out`` — the action expert's representation of the *noised ground-truth
action chunk*. That input trivially encodes the trajectory direction, so the
aux head hit 100% accuracy in ~150 steps without ever consulting the image or
the language. It measured nothing.

This module fixes the leak with a structural change:

1. A ``SlotClassifier`` reads ONLY the prefix image tokens + language tokens
   (never the state token, never the action/suffix tokens). Its bottleneck
   hidden state ``feat`` is forced slot-separable by a CE loss on
   ``logit_head(feat)`` vs ``target_position``. ``feat`` IS the slot code.
2. ``h_slot`` = ``slot_proj(feat.detach())`` is **appended as one extra token
   to the prefix**, so the action expert attends to it. Raw language is kept
   (concat, not replace). The detach keeps ``feat`` a pure CE-trained code;
   ``slot_proj`` is the action-trained adapter the expert reads through.
3. Because every training scene contains all three eval celebrities, the slot
   label is *ambiguous given the image alone* — the classifier is forced to
   use the language tokens to drive its CE loss down. That is the property the
   ``suffix_out`` head never had.

Knobs (env vars; ``apply()`` is a no-op unless the weight > 0):

  EVAL3_SLOT_LOSS_WEIGHT   float, default 0.0  — CE loss multiplier; 0 = disabled
  EVAL3_SLOT_HIDDEN        int,   default 256  — classifier MLP width
  EVAL3_SLOT_BOTTLENECK    int,   default 64   — penultimate width (info bottleneck)
  EVAL3_SLOT_DROPOUT       float, default 0.1
  EVAL3_SLOT_STOPGRAD      "1"/"0", default 1  — feed the expert h_slot.detach()
                            so the classifier is trained ONLY by the CE loss
  EVAL3_SLOT_GUMBEL        "1"/"0", default 0  — strict variant: the prefix token
                            is a straight-through Gumbel pick over 3 learned
                            slot embeddings (channel == exactly a 3-way categorical)

Same import-order contract as eval3_smolvla_aux_head: call ``apply()`` after
``eval3_lerobot_shim.apply()``. Mutually exclusive with the aux head — the
entry script picks one.
"""
from __future__ import annotations

import logging
import os

_APPLIED = False


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def slot_loss_weight() -> float:
    return _env_float("EVAL3_SLOT_LOSS_WEIGHT", 0.0)


def apply() -> None:
    """Install the slot-bottleneck patch. No-op if EVAL3_SLOT_LOSS_WEIGHT <= 0."""
    global _APPLIED
    if _APPLIED:
        return
    if slot_loss_weight() <= 0.0:
        return  # disabled — leaves SmolVLA untouched

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, VLAFlowMatching

    weight = slot_loss_weight()
    hidden = _env_int("EVAL3_SLOT_HIDDEN", 256)
    bottleneck = _env_int("EVAL3_SLOT_BOTTLENECK", 64)
    dropout = _env_float("EVAL3_SLOT_DROPOUT", 0.1)
    stopgrad = _env_bool("EVAL3_SLOT_STOPGRAD", True)
    gumbel = _env_bool("EVAL3_SLOT_GUMBEL", False)
    # v16: read the slot classifier off the frame-0 image (camera2 slot), and
    # count the slot CE loss only on pre-grasp frames.
    frame0_mode = _env_bool("EVAL3_SLOT_FRAME0", False)
    ce_pregrasp_only = _env_bool("EVAL3_SLOT_CE_PREGRASP_ONLY", False)

    # ---- the slot classifier module -------------------------------------
    class SlotClassifier(nn.Module):
        """language-queries-image cross-attention -> slot logits + h_slot.

        Reads only image tokens + language tokens. The language vector queries
        the image tokens (spatial selection), then a bottlenecked MLP emits
        the 3-way logits and the h_slot embedding.
        """

        def __init__(self, d_model: int):
            super().__init__()
            self.d_model = int(d_model)
            self.q_proj = nn.Linear(d_model, d_model)
            self.mlp = nn.Sequential(
                nn.Linear(2 * d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, bottleneck),
                nn.GELU(),
            )
            self.logit_head = nn.Linear(bottleneck, 3)
            # slot_proj turns the CE-trained bottleneck `feat` into the prefix
            # token. It is applied AFTER the stop-grad (see make_token), so it
            # is trained by the ACTION loss — the expert's adapter — while
            # `feat` itself stays a pure CE-trained slot code. (The earlier
            # design projected `feat` BEFORE the detach, leaving that layer
            # with no gradient from either loss.)
            self.slot_proj = nn.Linear(bottleneck, d_model)
            self.gumbel = gumbel
            if gumbel:
                # 3 learned slot embeddings; the prefix token is a (ST-Gumbel) pick.
                self.slot_embed = nn.Parameter(torch.randn(3, d_model) * 0.02)

        def forward(self, img_embs, lang_embs, lang_masks):
            """Returns (logits, feat). `feat` is the bottleneck hidden state
            that the CE loss makes slot-separable — embed_prefix turns it into
            the h_slot prefix token via make_token()."""
            dt = self.q_proj.weight.dtype
            img = img_embs.to(dt)
            lang = lang_embs.to(dt)
            lm = lang_masks.to(dt).unsqueeze(-1)  # (B, n_lang, 1)
            lang_vec = (lang * lm).sum(1) / lm.sum(1).clamp_min(1.0)  # (B, D)
            q = self.q_proj(lang_vec)  # (B, D)
            scores = torch.einsum("bnd,bd->bn", img, q) / (self.d_model ** 0.5)
            attn = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, n_img, 1)
            ctx = (img * attn).sum(1)  # (B, D)
            feat = self.mlp(torch.cat([ctx, lang_vec], dim=-1))  # (B, bottleneck)
            logits = self.logit_head(feat)  # (B, 3)
            return logits, feat

        def make_token(self, logits, feat, stopgrad):
            """Build the h_slot prefix token from the classifier outputs.

            soft   : detach the CE-trained bottleneck `feat`, then project it
                     with the action-trained slot_proj.
            gumbel : straight-through 3-way pick over the learned slot_embed
                     table (slot_embed is action-trained; gradient also reaches
                     `logits` via the ST estimator).
            """
            if self.gumbel:
                sel = F.gumbel_softmax(logits, tau=1.0, hard=True)  # (B, 3)
                return sel @ self.slot_embed.to(sel.dtype)
            code = feat.detach() if stopgrad else feat
            return self.slot_proj(code)

    # ---- 0. keep target_position alive through the lerobot pipeline -----
    from lerobot.processor import converters as _conv_mod

    _orig_extract = _conv_mod._extract_complementary_data

    def _patched_extract(batch):
        extra = _orig_extract(batch)
        if "target_position" in batch:
            extra["target_position"] = batch["target_position"]
        if "is_pregrasp" in batch:
            extra["is_pregrasp"] = batch["is_pregrasp"]
        return extra

    _conv_mod._extract_complementary_data = _patched_extract

    # ---- 1. VLAFlowMatching.__init__ — add the slot classifier ----------
    _orig_vlaf_init = VLAFlowMatching.__init__

    def _patched_vlaf_init(self, config, rtc_processor=None):
        _orig_vlaf_init(self, config, rtc_processor=rtc_processor)
        d_model = int(self.state_proj.out_features)  # prefix hidden dim
        self.slot_clf = SlotClassifier(d_model)
        self._slot_stopgrad = stopgrad
        self._last_slot_logits = None
        logging.info(
            "[eval3_slot_bottleneck] installed SlotClassifier "
            "(d_model=%d, hidden=%d, bottleneck=%d, dropout=%.2f, stopgrad=%s, gumbel=%s, "
            "weight=%.3f, frame0=%s, ce_pregrasp_only=%s)",
            d_model, hidden, bottleneck, dropout, stopgrad, gumbel, weight,
            frame0_mode, ce_pregrasp_only,
        )

    VLAFlowMatching.__init__ = _patched_vlaf_init

    # ---- 2. embed_prefix — append the h_slot token ----------------------
    # Wrapping (not replicating) the original keeps us robust to upstream
    # changes. prefix_length=0 for this config -> embed_prefix does NOT pad,
    # so the returned prefix is exactly [image .. , language .. , state] and
    # slicing by token counts is exact.
    _orig_embed_prefix = VLAFlowMatching.embed_prefix

    def _patched_embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state=None):
        embs, pad_masks, att_masks = _orig_embed_prefix(
            self, images, img_masks, lang_tokens, lang_masks, state=state
        )
        if not hasattr(self, "slot_clf"):
            return embs, pad_masks, att_masks

        n_lang = int(lang_tokens.shape[1])
        n_state = 1  # state_proj emits a single token
        n_img = int(embs.shape[1]) - n_lang - n_state
        if n_img <= 0:
            # unexpected prefix layout (e.g. add_image_special_tokens / padding) —
            # skip slot injection rather than corrupt the sequence.
            self._last_slot_logits = None
            logging.warning("[eval3_slot_bottleneck] unexpected prefix layout; slot token skipped")
            return embs, pad_masks, att_masks

        lang_embs = embs[:, n_img : n_img + n_lang, :]
        if frame0_mode:
            # Cameras embed in order [cam1=frame-t, cam2=frame-0, cam3=empty];
            # the classifier must read ONLY the cam2 (frame-0) token slice so
            # it never sees the coke's motion.
            n_cams = max(1, len(getattr(self.config, "image_features", []) or [1]))
            tok_per_cam = (n_img // n_cams) if n_cams else n_img
            if n_cams >= 2 and tok_per_cam > 0:
                img_embs = embs[:, tok_per_cam : 2 * tok_per_cam, :]
            else:
                img_embs = embs[:, :n_img, :]
                logging.warning(
                    "[eval3_slot_bottleneck] frame0 mode: cannot isolate cam2 "
                    "(n_img=%d n_cams=%d) — using all image tokens", n_img, n_cams,
                )
        else:
            img_embs = embs[:, :n_img, :]
        logits, feat = self.slot_clf(img_embs, lang_embs, lang_masks)
        self._last_slot_logits = logits  # (B, 3), fp32 — picked up by SmolVLAPolicy.forward
        # h_slot token: soft path detaches the CE-trained bottleneck then
        # projects (slot_proj is action-trained); gumbel path is a ST pick.
        h_slot = self.slot_clf.make_token(logits, feat, self._slot_stopgrad)
        tok = h_slot.to(dtype=embs.dtype).unsqueeze(1)  # (B, 1, D)
        B = embs.shape[0]
        one_pad = torch.ones(B, 1, dtype=pad_masks.dtype, device=pad_masks.device)
        one_att = torch.ones(B, 1, dtype=att_masks.dtype, device=att_masks.device)
        embs = torch.cat([embs, tok], dim=1)
        pad_masks = torch.cat([pad_masks, one_pad], dim=1)
        att_masks = torch.cat([att_masks, one_att], dim=1)
        return embs, pad_masks, att_masks

    VLAFlowMatching.embed_prefix = _patched_embed_prefix

    # ---- 3. SmolVLAPolicy.forward — add the slot CE loss ----------------
    _orig_policy_forward = SmolVLAPolicy.forward

    def _patched_policy_forward(self, batch, noise=None, time=None, reduction="mean"):
        target_position = batch.pop("target_position", None)
        is_pregrasp = batch.pop("is_pregrasp", None)
        out = _orig_policy_forward(self, batch, noise=noise, time=time, reduction=reduction)
        loss, loss_dict = out

        self._last_slot_dict = None
        logits = getattr(self.model, "_last_slot_logits", None)
        if logits is not None and isinstance(target_position, torch.Tensor) and weight > 0.0:
            tp = target_position.view(-1).to(dtype=torch.long, device=logits.device)
            mask = tp != -100
            if ce_pregrasp_only and isinstance(is_pregrasp, torch.Tensor):
                # Count the slot CE loss only on pre-grasp frames — the slot
                # must be decided before the carry begins (the gradient always
                # propagates; there is no expert-freeze phase).
                pg = is_pregrasp.view(-1).to(device=logits.device)
                mask = mask & (pg > 0)
            if bool(mask.any()):
                ce = F.cross_entropy(logits.float(), tp, ignore_index=-100, reduction="none")
                m = mask.float()
                denom = m.sum().clamp_min(1.0)
                slot_loss = (ce * m).sum() / denom
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    slot_acc = float(((preds == tp).float() * m).sum().item() / float(denom.item()))
                loss = loss + weight * slot_loss
                loss_dict["slot_loss"] = float(slot_loss.item())
                loss_dict["slot_acc"] = slot_acc
                loss_dict["slot_weight"] = float(weight)
                loss_dict["slot_ce_n"] = float(denom.item())
                loss_dict["loss"] = (
                    float(loss.item()) if reduction == "mean" else float(loss.mean().item())
                )
                self._last_slot_dict = {
                    "slot_loss": float(slot_loss.item()),
                    "slot_acc": slot_acc,
                    "slot_ce_n": float(denom.item()),
                }
        return loss, loss_dict

    SmolVLAPolicy.forward = _patched_policy_forward

    # ---- 4. MetricsTracker — slot_loss / slot_acc meters + pump ---------
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker

    _orig_mt_init = MetricsTracker.__init__

    def _patched_mt_init(self, batch_size, num_frames, num_episodes, metrics,
                         initial_step=0, accelerator=None):
        metrics = dict(metrics)
        metrics.setdefault("slot_loss", AverageMeter("slot_loss", ":.3f"))
        metrics.setdefault("slot_acc", AverageMeter("slot_acc", ":.3f"))
        metrics.setdefault("slot_ce_n", AverageMeter("slot_ce_n", ":.1f"))
        _orig_mt_init(self, batch_size, num_frames, num_episodes, metrics,
                      initial_step=initial_step, accelerator=accelerator)

    MetricsTracker.__init__ = _patched_mt_init

    _pump = {"policy_ref": None}
    _orig_policy_init = SmolVLAPolicy.__init__

    def _patched_policy_init(self, *args, **kwargs):
        _orig_policy_init(self, *args, **kwargs)
        _pump["policy_ref"] = self
        self._last_slot_dict = None

    SmolVLAPolicy.__init__ = _patched_policy_init

    # state-aug curriculum counter (shared mp.Value read by DataLoader workers)
    _curr = {"counter": None}

    def _get_curriculum_counter():
        if _curr["counter"] is None:
            try:
                from eval3_dataset_prep import get_curriculum_step_counter
                _curr["counter"] = get_curriculum_step_counter()
            except Exception:
                _curr["counter"] = False
        return _curr["counter"] if _curr["counter"] is not False else None

    _orig_mt_setattr = MetricsTracker.__setattr__

    def _patched_mt_setattr(self, name, value):
        _orig_mt_setattr(self, name, value)
        if name == "loss":
            pol = _pump.get("policy_ref")
            sd = getattr(pol, "_last_slot_dict", None) if pol is not None else None
            if sd is not None:
                if "slot_loss" in self.metrics:
                    _orig_mt_setattr(self, "slot_loss", sd["slot_loss"])
                if "slot_acc" in self.metrics:
                    _orig_mt_setattr(self, "slot_acc", sd["slot_acc"])
                if "slot_ce_n" in self.metrics and "slot_ce_n" in sd:
                    _orig_mt_setattr(self, "slot_ce_n", sd["slot_ce_n"])
            counter = _get_curriculum_counter()
            if counter is not None:
                counter.set_step(int(self.steps))

    MetricsTracker.__setattr__ = _patched_mt_setattr

    _APPLIED = True
    logging.info("[eval3_slot_bottleneck] applied (weight=%.3f)", weight)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    _here = Path(__file__).resolve().parent
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))
    os.environ.setdefault("EVAL3_SLOT_LOSS_WEIGHT", "0.5")
    import eval3_lerobot_shim

    eval3_lerobot_shim.apply()
    apply()
    print("[eval3_slot_bottleneck] applied OK")
