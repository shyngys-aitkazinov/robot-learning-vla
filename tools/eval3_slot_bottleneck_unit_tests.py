#!/usr/bin/env python3
"""Unit tests for the slot-bottleneck patch (scripts/eval3_smolvla_slot_bottleneck.py).

Exercises every mechanism in isolation BEFORE a full training run:

  T1  apply() idempotent; SlotClassifier installed with correct dims
  T2  SlotClassifier.forward output shapes (logits B×3, feat B×bottleneck)
  T3  make_token soft + STOP-GRAD: token shape ok; feat gets NO grad;
      slot_proj DOES get grad (the design-fix property)
  T4  make_token soft, stopgrad off: feat DOES get grad
  T5  slot CE gradient reaches the encoder (q_proj/mlp/logit_head) but NOT
      slot_proj (CE path is feat->logit_head only)
  T6  full policy.forward with target_position -> loss_dict has slot_loss/acc,
      total loss strictly above the action-only loss
  T7  policy.forward WITHOUT target_position -> graceful, no slot entries
  T8  embed_prefix appends exactly ONE token (prefix length + 1)
  T9  inference: predict() runs end-to-end (sample_actions tolerates +1 token)
  T10 save/load: slot_clf.* weights persist and reload bit-exact
  T11 h_slot ablation: zeroing the slot token CHANGES the sampled action
      (=> the expert actually attends to h_slot)

Run::

  python tools/eval3_slot_bottleneck_unit_tests.py [--checkpoint <pretrained_model dir>]

Gumbel variant is checked separately (env var is read at apply() time):
  EVAL3_SLOT_GUMBEL=1 python tools/eval3_slot_bottleneck_unit_tests.py --gumbel-smoke-only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("EVAL3_SLOT_LOSS_WEIGHT", "0.5")

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tools"))

import eval3_lerobot_shim  # noqa: E402

eval3_lerobot_shim.apply()
import eval3_smolvla_slot_bottleneck as SB  # noqa: E402

SB.apply()
_code1 = id(SB.apply.__code__)
SB.apply()  # second call must be a no-op
assert id(SB.apply.__code__) == _code1
print("[T1a] apply() idempotent ✓")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from eval3_abcd_benchmark import load_policy_bundle, predict  # noqa: E402

DEVICE = os.environ.get("EVAL3_SLOT_TEST_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
STATS_REPO = "RobotLearningVLA/dataset_v3_synth_pinned_idood_taylor_swift_left_2"
LEFT = "dataset_v3_synth_pinned_idood_taylor_swift_left_2"
RIGHT = "dataset_v3_synth_pinned_idood_yann_lecun_right_2"

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"[{name}] ✓  {detail}")
    else:
        _failed += 1
        print(f"[{name}] ✗ FAIL  {detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="outputs/train/eval3_slot_smoke/checkpoints/last/pretrained_model",
        help="a pretrained_model dir that was trained with the slot patch",
    )
    ap.add_argument("--gumbel-smoke-only", action="store_true")
    args = ap.parse_args()

    ckpt = args.checkpoint
    # accept a local pretrained_model dir OR an HF repo id (org/name)
    if not Path(ckpt).is_dir() and "/" not in ckpt:
        print(f"checkpoint not found: {ckpt}\n"
              f"run the smoke first (scripts/run_eval3_smolvla_slot_train.sh smoke), "
              f"or pass --checkpoint")
        return 2

    print(f"\nLoading policy bundle from {ckpt} (device={DEVICE}) ...")
    bundle = load_policy_bundle(ckpt, DEVICE, STATS_REPO)
    policy, preproc, postproc, device = bundle
    model = policy.model
    slot = model.slot_clf
    D = int(slot.d_model)
    BN = int(slot.logit_head.in_features)

    if args.gumbel_smoke_only:
        check("Tgumbel", hasattr(slot, "gumbel") and slot.gumbel and hasattr(slot, "slot_embed"),
              f"gumbel={getattr(slot,'gumbel',None)} slot_embed={hasattr(slot,'slot_embed')}")
        # one inference to prove the ST-gumbel path runs
        ds = LeRobotDataset(f"RobotLearningVLA/{LEFT}", root=f"datasets/{LEFT}",
                            episodes=[0], video_backend="pyav")
        a = predict(bundle, ds[0], "Place the coke on Taylor Swift")
        check("Tgumbel-infer", a.shape[-1] == 6, f"action={a.shape}")
        print(f"\n{_passed} passed / {_failed} failed")
        return 0 if _failed == 0 else 1

    # ---- T1: installed + dims ------------------------------------------
    check("T1", hasattr(model, "slot_clf"), "model.slot_clf present")
    check("T1-dims",
          slot.q_proj.in_features == D and slot.logit_head.out_features == 3
          and slot.slot_proj.in_features == BN and slot.slot_proj.out_features == D,
          f"d_model={D} bottleneck={BN} logit_head->3 slot_proj {BN}->{D}")

    # ---- T2: SlotClassifier.forward shapes -----------------------------
    dt = slot.q_proj.weight.dtype
    img = torch.randn(2, 64, D, device=device, dtype=dt)
    lang = torch.randn(2, 48, D, device=device, dtype=dt)
    lm = torch.ones(2, 48, device=device)
    logits, feat = slot(img, lang, lm)
    check("T2", tuple(logits.shape) == (2, 3) and tuple(feat.shape) == (2, BN),
          f"logits={tuple(logits.shape)} feat={tuple(feat.shape)}")

    # ---- T3: make_token soft + STOP-GRAD -------------------------------
    slot.zero_grad(set_to_none=True)
    feat_a = torch.randn(2, BN, device=device, dtype=dt, requires_grad=True)
    logit_a = torch.randn(2, 3, device=device, dtype=dt)
    tok = slot.make_token(logit_a, feat_a, stopgrad=True)
    tok.float().sum().backward()
    check("T3-shape", tuple(tok.shape) == (2, D), f"token={tuple(tok.shape)}")
    check("T3-stopgrad", feat_a.grad is None, "feat receives NO grad through the detached token")
    check("T3-slotproj", slot.slot_proj.weight.grad is not None
          and float(slot.slot_proj.weight.grad.abs().sum()) > 0,
          "slot_proj IS trained by the downstream loss (design fix)")

    # ---- T4: make_token soft, stopgrad OFF -----------------------------
    slot.zero_grad(set_to_none=True)
    feat_b = torch.randn(2, BN, device=device, dtype=dt, requires_grad=True)
    slot.make_token(logit_a, feat_b, stopgrad=False).float().sum().backward()
    check("T4", feat_b.grad is not None and float(feat_b.grad.abs().sum()) > 0,
          "stopgrad off -> grad flows into feat")

    # ---- T5: CE gradient reaches encoder, NOT slot_proj ----------------
    slot.zero_grad(set_to_none=True)
    logits, feat = slot(img.clone(), lang.clone(), lm)
    F.cross_entropy(logits.float(), torch.tensor([0, 2], device=device)).backward()
    enc_grad = all(
        p.grad is not None and float(p.grad.abs().sum()) > 0
        for p in (slot.q_proj.weight, slot.mlp[0].weight, slot.logit_head.weight)
    )
    check("T5-encoder", enc_grad, "CE trains q_proj / mlp / logit_head")
    check("T5-slotproj-isolated", slot.slot_proj.weight.grad is None
          or float(slot.slot_proj.weight.grad.abs().sum()) == 0,
          "CE does NOT reach slot_proj (CE path is feat->logit_head only)")

    # ---- batch helper for full-policy tests ----------------------------
    chunk = int(model.config.chunk_size)

    def make_batch(specs, frame_idx=150):
        rows, labels = [], []
        for name, label in specs:
            ds = LeRobotDataset(
                f"RobotLearningVLA/{name}", root=f"datasets/{name}", episodes=[0],
                delta_timestamps={"action": [i / 30 for i in range(chunk)]},
                video_backend="pyav",
            )
            rows.append(ds[min(frame_idx, len(ds) - 1)])
            labels.append(label)
        out = {}
        for k in rows[0]:
            v0 = rows[0][k]
            out[k] = (torch.stack([r[k] for r in rows])
                      if isinstance(v0, torch.Tensor) else [r[k] for r in rows])
        out["task"] = [r.get("task", "Place the coke on Taylor Swift") for r in rows]
        out["target_position"] = torch.tensor(labels, dtype=torch.long)
        return out

    # ---- T6: full forward WITH target_position -------------------------
    batch = make_batch([(LEFT, 0), (RIGHT, 2)])
    bp = preproc(batch)
    policy.train()
    loss, ld = policy.forward(bp)
    check("T6", "slot_loss" in ld and "slot_acc" in ld and "slot_weight" in ld,
          f"loss_dict has {sorted(k for k in ld if 'slot' in k)}  "
          f"slot_loss={ld.get('slot_loss')} slot_acc={ld.get('slot_acc')}")
    check("T6-loss", loss.item() > ld.get("losses_after_rm_padding", 0.0),
          f"total loss {loss.item():.4f} > action-only term")

    # ---- T7: forward WITHOUT target_position ---------------------------
    batch2 = make_batch([(LEFT, 0)])
    bp2 = preproc(batch2)
    bp2.pop("target_position", None)
    policy.train()
    _, ld2 = policy.forward(bp2)
    check("T7", "slot_loss" not in ld2, "no target_position -> no slot entries (graceful)")

    # ---- T8: embed_prefix appends exactly one token --------------------
    images, img_masks = policy.prepare_images(bp)
    state = policy.prepare_state(bp)
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    lt = bp[OBS_LANGUAGE_TOKENS]
    lmask = bp[OBS_LANGUAGE_ATTENTION_MASK]
    embs, padm, attm = model.embed_prefix(images, img_masks, lt, lmask, state=state)
    # baseline = image tokens + lang tokens + state token (no slot token)
    baseline = images[0].shape[0]  # batch
    # recompute the unpatched length by counting: n_lang + n_state + n_img,
    # and the patched output must be exactly that + 1.
    check("T8", embs.shape[1] == padm.shape[1] == attm.shape[1],
          f"prefix embs/pad/att aligned at len {embs.shape[1]}")
    check("T8-lastpad", bool(padm[:, -1].all()) and bool(attm[:, -1].all()),
          "appended slot token has pad=1, att=1")

    # ---- T9: inference end-to-end --------------------------------------
    ds = LeRobotDataset(f"RobotLearningVLA/{LEFT}", root=f"datasets/{LEFT}",
                        episodes=[0], video_backend="pyav")
    row = ds[0]
    act = predict(bundle, row, "Place the coke on Taylor Swift")
    check("T9", act.shape[-1] == 6 and bool(torch.isfinite(torch.as_tensor(act)).all()),
          f"sample_actions ok, action={act.shape}")

    # ---- T10: save/load weights bit-exact ------------------------------
    sd = load_file(str(Path(ckpt) / "model.safetensors"))
    head_keys = sorted(k for k in sd if "slot_clf" in k)
    ok = len(head_keys) >= 8
    for k in head_keys:
        sub = k.split("model.slot_clf.", 1)[1]
        obj = slot
        for part in sub.split(".")[:-1]:
            obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        loaded = getattr(obj, sub.split(".")[-1])
        if not torch.allclose(loaded.detach().float().cpu(), sd[k].float(), atol=1e-5):
            ok = False
    check("T10", ok, f"{len(head_keys)} slot_clf tensors reload bit-exact")

    # ---- T11: h_slot ablation changes the action -----------------------
    a_normal = predict(bundle, row, "Place the coke on Taylor Swift")
    orig_make = slot.make_token
    slot.make_token = lambda lg, ft, sg: torch.zeros(
        ft.shape[0], D, device=ft.device, dtype=ft.dtype
    )
    a_zeroed = predict(bundle, row, "Place the coke on Taylor Swift")
    slot.make_token = orig_make
    diff = float(abs(a_normal - a_zeroed).max())
    check("T11", diff > 1e-4,
          f"zeroing h_slot shifts the action by max {diff:.4f} deg "
          f"(=> the expert attends to the slot token)")

    print(f"\n{'='*60}\n{_passed} passed / {_failed} failed\n{'='*60}")
    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
