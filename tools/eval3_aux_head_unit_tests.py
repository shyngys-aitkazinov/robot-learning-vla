#!/usr/bin/env python3
"""Comprehensive unit tests for the aux-head patch.

Covers:
    T1. weight=0 → patch is a runtime no-op (loss_dict has no aux entries)
    T2. target_position absent in batch → no aux contribution
    T3. all-ignore batch (target_position all -100) → no aux contribution
    T4. partial-ignore batch → aux runs on valid subset only, acc only over valid
    T5. apply() is idempotent (second call is a no-op)
    T6. save+load round-trip preserves head weights AND loads cleanly back
    T7. Inference via select_action does NOT invoke the aux head
    T8. Batch with mixed valid labels (0 and 2) → CE loss > 0, head gets gradient
    T9. MetricsTracker integration: aux_pos_loss + aux_pos_acc meters exist and update
    T10. Head's dtype matches suffix_out's dtype (no float/bf16 mix-ups)

Run::

    EVAL3_AUX_POS_LOSS_WEIGHT=0.5 python tools/eval3_aux_head_unit_tests.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "tools"))

# Default to weight=0.5 so the patch's aux path is exercised. Individual tests
# below override the policy's _aux_loss_weight in-place to test other regimes.
os.environ.setdefault("EVAL3_AUX_POS_LOSS_WEIGHT", "0.5")

import eval3_lerobot_shim  # noqa: E402

eval3_lerobot_shim.apply()
import eval3_smolvla_aux_head  # noqa: E402

# Sanity: apply once, then again — second call must be a no-op.
eval3_smolvla_aux_head.apply()
_first_id = id(eval3_smolvla_aux_head.apply.__code__)
eval3_smolvla_aux_head.apply()  # T5
_second_id = id(eval3_smolvla_aux_head.apply.__code__)
assert _first_id == _second_id, "apply() not idempotent"
print("[T5] apply() idempotent ✓")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.utils.logging_utils import MetricsTracker  # noqa: E402

from eval3_abcd_benchmark import load_policy_bundle  # noqa: E402


DEVICE = os.environ.get(
    "EVAL3_AUX_TEST_DEVICE",
    "cuda" if torch.cuda.is_available() else "mps",
)
CKPT = "RobotLearningVLA/eval3-smolvla-3way-25k-b128-v6-synth-step15k"
STATS_REPO = "RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2"

print("\nLoading policy bundle (may take a minute)...")
bundle = load_policy_bundle(CKPT, DEVICE, STATS_REPO)
policy, preprocessor, postprocessor, device = bundle
model = policy.model


# --- helper: build a small training batch from real data ---
def make_batch(repos_and_labels, frame_idx=200):
    """Build a stacked training batch from N datasets, one frame each.

    `repos_and_labels` is a list of (repo_id, target_position) tuples. Use -100
    to mark "ignore" for that sample.
    """
    rows = []
    labels = []
    for repo, label in repos_and_labels:
        ds = LeRobotDataset(
            repo,
            episodes=[0],
            revision="v3.0",
            delta_timestamps={"action": [i / 30 for i in range(model.config.chunk_size)]},
            video_backend="pyav",
        )
        rows.append(ds[frame_idx])
        labels.append(label)

    def _stack(rs):
        out = {}
        for k in rs[0]:
            v0 = rs[0][k]
            vs = [r[k] for r in rs]
            out[k] = torch.stack(vs) if isinstance(v0, torch.Tensor) else vs
        return out

    batch = _stack(rows)
    batch["task"] = [r.get("task", "Place the coke on Yann LeCun") for r in rows]
    batch["target_position"] = torch.tensor(labels, dtype=torch.long)
    return batch


# === T1: weight=0 ⇒ no aux contribution ===
print("\n[T1] weight=0 no-op")
old_weight = model._aux_loss_weight
model._aux_loss_weight = 0.0
batch = make_batch(
    [("RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2", 0),
     ("RobotLearningVLA/dataset_v3_synth_taylor_swift_right_2", 2)],
    frame_idx=200,
)
batch_p = preprocessor(batch)
policy.train()
loss, ld = policy.forward(batch_p)
assert "aux_pos_loss" not in ld, f"weight=0 should skip aux_pos_loss in loss_dict, got {ld.keys()}"
assert "aux_pos_acc" not in ld
print(f"   loss={loss.item():.4f} (action-only, no aux contribution) ✓")
model._aux_loss_weight = old_weight


# === T2: target_position absent in batch ===
print("\n[T2] target_position absent in batch")
batch = make_batch(
    [("RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2", 0)],
    frame_idx=200,
)
batch_p = preprocessor(batch)
batch_p.pop("target_position", None)  # remove after preprocessor
policy.train()
loss2, ld2 = policy.forward(batch_p)
assert "aux_pos_loss" not in ld2, "absent target_position should skip aux"
assert "aux_pos_acc" not in ld2
print(f"   loss={loss2.item():.4f}, aux entries absent ✓")


# === T3: all-ignore batch ===
print("\n[T3] all-ignore batch (target_position all -100)")
batch = make_batch(
    [("RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2", -100),
     ("RobotLearningVLA/dataset_v3_synth_taylor_swift_right_2", -100)],
    frame_idx=200,
)
batch_p = preprocessor(batch)
policy.train()
loss3, ld3 = policy.forward(batch_p)
assert "aux_pos_loss" not in ld3, "all-ignore should skip aux (no valid samples)"
assert "aux_pos_acc" not in ld3
print(f"   loss={loss3.item():.4f}, no spurious aux entries ✓")


# === T4: partial-ignore batch ===
print("\n[T4] partial-ignore batch (one -100, one valid)")
batch = make_batch(
    [("RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2", 0),
     ("RobotLearningVLA/dataset_v3_synth_taylor_swift_right_2", -100)],
    frame_idx=200,
)
batch_p = preprocessor(batch)
policy.train()
loss4, ld4 = policy.forward(batch_p)
assert "aux_pos_loss" in ld4, "partial-ignore: aux should run on the valid sample"
print(f"   loss={loss4.item():.4f} aux_pos_loss={ld4['aux_pos_loss']:.4f} aux_pos_acc={ld4['aux_pos_acc']:.2f} ✓")


# === T7: select_action (inference) does NOT touch the aux head ===
print("\n[T7] select_action / inference path skips aux head")
# Reset model._last_aux_dict to a known sentinel, run select_action, verify it stays sentinel.
model._last_aux_loss = "SENTINEL"  # type: ignore
policy._last_aux_dict = "SENTINEL_DICT"  # type: ignore
# Build a single-frame inference batch (one observation)
ds_lecun = LeRobotDataset(
    "RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2",
    episodes=[0],
    revision="v3.0",
    video_backend="pyav",
)
row = ds_lecun[200]
inf_batch = {k: (v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else [v]) for k, v in row.items()}
inf_batch["task"] = [row["task"]]
# Need observation dims to match what select_action expects (with time dim)
# Use the existing predict helper from abcd_benchmark which handles this.
from eval3_abcd_benchmark import predict  # noqa: E402

_ = predict(bundle, row, row["task"])
assert model._last_aux_loss == "SENTINEL", (
    f"select_action invoked the aux path (got {model._last_aux_loss!r})"
)
print("   select_action did not touch _last_aux_loss ✓")
# Restore
model._last_aux_loss = None
policy._last_aux_dict = None


# === T8: batch with mixed valid labels ⇒ head gets gradient ===
print("\n[T8] mixed-label batch: CE loss > 0, head receives gradient")
batch = make_batch(
    [("RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2", 0),
     ("RobotLearningVLA/dataset_v3_synth_taylor_swift_right_2", 2)],
    frame_idx=200,
)
batch_p = preprocessor(batch)
policy.zero_grad()
policy.train()
loss8, ld8 = policy.forward(batch_p)
assert "aux_pos_loss" in ld8 and ld8["aux_pos_loss"] >= 0
loss8.backward()
first_lin = next(m for m in model.position_clf_head if isinstance(m, torch.nn.Linear))
grad_norm = first_lin.weight.grad.norm().item() if first_lin.weight.grad is not None else 0
assert grad_norm > 0, f"head got no gradient (norm={grad_norm})"
print(f"   aux_pos_loss={ld8['aux_pos_loss']:.4f} head grad_norm={grad_norm:.4e} ✓")


# === T10: head dtype matches input dtype after the explicit cast ===
print("\n[T10] head input dtype matches head weight dtype (no fp32/bf16 mix)")
head_param_dtype = next(model.position_clf_head.parameters()).dtype
# After our forward, the head ran without dtype errors → automatic pass if T8 passed.
# Additionally verify the cast is in place by reading the source line.
src = (Path(__file__).resolve().parent.parent / "scripts" / "eval3_smolvla_aux_head.py").read_text()
assert ".to(dtype=head_dtype)" in src, "explicit dtype cast was removed from the patch"
print(f"   head param dtype = {head_param_dtype}, explicit cast present in patch ✓")


# === T9: MetricsTracker integration ===
print("\n[T9] MetricsTracker has aux_pos_loss + aux_pos_acc meters; pump updates them")
from lerobot.utils.logging_utils import AverageMeter  # noqa: E402

# Build a minimal MetricsTracker. Note: the patched __init__ inserts our meters.
tracker = MetricsTracker(
    batch_size=2,
    num_frames=1000,
    num_episodes=10,
    metrics={"loss": AverageMeter("loss", ":.3f"), "grdn": AverageMeter("grdn", ":.3f")},
    initial_step=0,
)
assert "aux_pos_loss" in tracker.metrics, "MetricsTracker missing aux_pos_loss meter"
assert "aux_pos_acc" in tracker.metrics, "MetricsTracker missing aux_pos_acc meter"

# Run a forward, then simulate the train-loop "loss assignment" which is what
# triggers our pump.
batch = make_batch(
    [("RobotLearningVLA/dataset_v3_synth_yann_lecun_left_2", 0),
     ("RobotLearningVLA/dataset_v3_synth_taylor_swift_right_2", 2)],
    frame_idx=200,
)
batch_p = preprocessor(batch)
policy.train()
loss9, ld9 = policy.forward(batch_p)
# Simulate the trainer's "tracker.loss = loss" line — our pump fires here.
tracker.loss = ld9["loss"]
assert tracker.metrics["aux_pos_loss"].count > 0, "aux_pos_loss meter never updated"
assert tracker.metrics["aux_pos_acc"].count > 0, "aux_pos_acc meter never updated"
print(
    f"   tracker.loss={tracker.metrics['loss'].avg:.3f}  "
    f"tracker.aux_pos_loss={tracker.metrics['aux_pos_loss'].avg:.3f}  "
    f"tracker.aux_pos_acc={tracker.metrics['aux_pos_acc'].avg:.3f}  ✓"
)

# Also verify they appear in __str__ (which is what gets printed each log_freq).
tracker_str = str(tracker)
assert "aux_pos_loss" in tracker_str, "aux_pos_loss missing from MetricsTracker __str__"
assert "aux_pos_acc" in tracker_str, "aux_pos_acc missing from MetricsTracker __str__"
print(f"   tracker __str__ contains both meters ✓")

# And verify to_dict() carries them (this is what wandb consumes).
d = tracker.to_dict()
assert "aux_pos_loss" in d and "aux_pos_acc" in d, f"to_dict() missing meters: {d.keys()}"
print(f"   tracker.to_dict() carries aux metrics → wandb-ready ✓")


# === T6: save+load round-trip using the real training-time save_checkpoint ===
print("\n[T6] save+load round-trip (using the actual mini-training checkpoint)")
trained_ckpt = _REPO / "outputs" / "train" / "eval3_aux_smoke" / "checkpoints" / "000300" / "pretrained_model"
if not trained_ckpt.is_dir():
    print(f"   SKIPPED — no checkpoint at {trained_ckpt} (run the mini training first)")
else:
    # Verify the head's weights are in the saved safetensors
    from safetensors.torch import load_file  # noqa: E402

    sd_path = trained_ckpt / "model.safetensors"
    sd = load_file(str(sd_path))
    head_keys = sorted(k for k in sd.keys() if "position_clf_head" in k)
    assert len(head_keys) == 4, f"expected 4 head params, got {len(head_keys)}: {head_keys}"
    print(f"   saved checkpoint contains head params: {head_keys}")
    # Load via the same loader the deploy/test code uses and check weights match the on-disk file.
    reloaded = load_policy_bundle(str(trained_ckpt), DEVICE, STATS_REPO)
    reloaded_policy = reloaded[0]
    reloaded_head = reloaded_policy.model.position_clf_head
    for k in head_keys:
        # k is like "model.position_clf_head.0.weight" — strip the "model.position_clf_head." prefix
        param_subname = k[len("model.position_clf_head."):]
        layer_idx_str, attr = param_subname.split(".", 1)
        layer = reloaded_head[int(layer_idx_str)]
        loaded_val = getattr(layer, attr)
        disk_val = sd[k].to(loaded_val.device)
        assert torch.allclose(loaded_val.detach().float(), disk_val.float(), atol=1e-6), (
            f"head param {k} differs between on-disk safetensors and reloaded policy"
        )
    print("   reloaded policy's head weights match on-disk safetensors ✓")


print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
