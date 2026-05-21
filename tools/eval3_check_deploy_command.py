#!/usr/bin/env python3
"""Pre-flight diagnostic for the Eval 3 deploy command.

Loads a SmolVLA-class checkpoint from the Hub or a local path, inspects its
``input_features`` keys + baked-in normalizer stats, and reports whether a
proposed ``--dataset.rename_map`` + camera config would actually feed real
data to the policy.

This catches the single most common Eval 3 deploy failure: forgetting to
pass ``--dataset.rename_map`` to ``lerobot-record`` so the robot's
``observation.images.front`` never aliases to the policy's expected
``observation.images.camera1`` and the policy silently sees zero-padded
black frames for every camera key.

Usage:
    # Failing case (no rename map, like the eval-day command that failed):
    python tools/eval3_check_deploy_command.py \\
        --policy-pretrained-path RobotLearningVLA/eval3-vla-v7-A-smolvla-new-10k \\
        --rename-map '{}' \\
        --task "Place the coke on Taylor Swift"

    # Corrected case:
    python tools/eval3_check_deploy_command.py \\
        --policy-pretrained-path RobotLearningVLA/eval3-vla-v7-A-smolvla-new-10k \\
        --rename-map '{"observation.images.front":"observation.images.camera1"}' \\
        --task "Place the coke on Taylor Swift"

By default the diagnostic assumes the recording config provides exactly one
camera key called ``observation.images.front`` (the standard SO-101 + OpenCV
setup); override with ``--robot-camera-keys`` if you have a different rig.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Canonical-form task strings the training pipeline rewrites every task to
# at EVAL3_TASK_AUG_CANONICAL_P=1.0 (see scripts/eval3_dataset_prep.py:48-78).
CANONICAL_TASKS = {
    "Place the coke on Taylor Swift",
    "Place the coke on Yann LeCun",
    "Place the coke on Barack Obama",
}
KNOWN_CELEBS = ("Taylor Swift", "Yann LeCun", "Barack Obama")


def _green(s): return f"\033[32m{s}\033[0m"
def _red(s): return f"\033[31m{s}\033[0m"
def _yellow(s): return f"\033[33m{s}\033[0m"
def _bold(s): return f"\033[1m{s}\033[0m"


def load_checkpoint_config(path_or_repo: str) -> dict:
    """Fetch config.json from a local checkpoint dir or an HF repo id."""
    local = Path(path_or_repo)
    if local.is_dir() and (local / "config.json").is_file():
        return json.loads((local / "config.json").read_text())
    # Fall back to HF Hub.
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(path_or_repo, "config.json")
    return json.loads(Path(p).read_text())


def load_baked_normalizer(path_or_repo: str) -> dict | None:
    """Return a small subset of the baked-in normalizer stats for sanity-check.

    Returns None if the safetensors file isn't found (e.g. for older checkpoints).
    """
    fname = "policy_preprocessor_step_5_normalizer_processor.safetensors"
    local = Path(path_or_repo)
    if local.is_dir() and (local / fname).is_file():
        path = local / fname
    else:
        try:
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(path_or_repo, fname))
        except Exception:
            return None
    try:
        from safetensors.torch import load_file
    except ImportError:
        return None
    state = load_file(str(path))
    out: dict = {}
    for key in ("action", "observation.state"):
        sub = {k.split(".", 1)[1]: v for k, v in state.items() if k.startswith(f"{key}.")}
        if not sub:
            continue
        out[key] = {
            "count":  int(sub["count"].item()) if "count" in sub else None,
            "min":    sub["min"].tolist() if "min" in sub else None,
            "max":    sub["max"].tolist() if "max" in sub else None,
            "mean":   sub["mean"].tolist() if "mean" in sub else None,
            "std":    sub["std"].tolist() if "std" in sub else None,
        }
    return out


def parse_rename_map(s: str) -> dict[str, str]:
    if not s.strip():
        return {}
    try:
        d = json.loads(s)
    except json.JSONDecodeError as e:
        sys.exit(f"--rename-map is not valid JSON: {e}")
    if not isinstance(d, dict):
        sys.exit(f"--rename-map must be a JSON object, got {type(d).__name__}")
    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--policy-pretrained-path", required=True,
                    help="HF repo id or local directory of the checkpoint.")
    ap.add_argument("--rename-map", default="{}",
                    help='JSON dict, e.g. \'{"observation.images.front":"observation.images.camera1"}\'. '
                         "Default '{}' simulates the user forgetting to pass --dataset.rename_map.")
    ap.add_argument("--robot-camera-keys", nargs="+",
                    default=["observation.images.front"],
                    help="Camera observation keys the robot config will produce "
                         "(default: ['observation.images.front'] for the standard SO-101 OpenCV rig).")
    ap.add_argument("--task", default="",
                    help='Optional task string to validate against canonical form '
                         '(e.g. "Place the coke on Taylor Swift"). Skips check if empty.')
    ap.add_argument("--robot-port", default="<your-port>",
                    help="Robot port for the emitted command (default placeholder).")
    ap.add_argument("--robot-id", default="my_awesome_follower_arm",
                    help="Robot calibration id for the emitted command.")
    ap.add_argument("--dataset-repo-id", default="RobotLearningVLA/eval_v7_A_taylor_swift_1",
                    help="--dataset.repo_id for the emitted command (must start with 'eval_' "
                         "when --policy is set).")
    args = ap.parse_args()

    print(_bold(f"\n== Pre-flight check for {args.policy_pretrained_path} ==\n"))

    cfg = load_checkpoint_config(args.policy_pretrained_path)
    in_features = cfg.get("input_features") or {}
    cam_keys = sorted(k for k in in_features if k.startswith("observation.images."))
    non_cam = sorted(k for k in in_features if not k.startswith("observation.images."))
    n_action_steps = cfg.get("n_action_steps")
    chunk_size = cfg.get("chunk_size")
    empty_cameras = cfg.get("empty_cameras")
    resize = cfg.get("resize_imgs_with_padding")

    print(f"  policy.empty_cameras           : {empty_cameras}")
    print(f"  policy.n_action_steps          : {n_action_steps}")
    print(f"  policy.chunk_size              : {chunk_size}")
    print(f"  policy.resize_imgs_with_padding: {resize}")
    print(f"  non-camera input_features      : {non_cam}")
    print(f"  camera input_features ({len(cam_keys)}):")
    for k in cam_keys:
        print(f"    - {k}")

    # ----- Camera-key resolution -----
    print(_bold("\n== Camera key resolution after --dataset.rename_map ==\n"))
    rename_map = parse_rename_map(args.rename_map)
    print(f"  rename_map provided   : {rename_map or '(empty)'}")
    print(f"  robot camera keys     : {args.robot_camera_keys}")

    # After lerobot applies rename_map, each robot key K becomes rename_map.get(K, K).
    aliased = {rename_map.get(k, k): k for k in args.robot_camera_keys}
    print(f"  -> visible to policy  : {sorted(aliased.keys())}")

    # v16 frame-0 checkpoints: eval3_vla_deploy.py injects a synthetic
    # observation.images.*_frame0 key every step and the rename_map routes it to
    # camera2. That camera is fed by the deploy script, NOT a robot camera — so a
    # rename_map entry whose KEY ends in "_frame0" satisfies its target camera.
    frame0_fed = {v for k, v in rename_map.items() if k.endswith("_frame0")}
    if frame0_fed:
        print(f"  -> v16 frame-0 fed    : {sorted(frame0_fed)} "
              "(deploy-injected by eval3_vla_deploy.py, not a robot camera)")

    # Classify each policy-expected camera key.
    # - The "empty_camera_N" keys are synthetic black-pads added by SmolVLA's
    #   processor pipeline based on policy.empty_cameras; they NEVER need a
    #   robot source.
    # - Of the remaining named cameras (camera1, camera2, ...), the runtime
    #   uses policy.empty_cameras to auto-pad the HIGHEST-numbered ones with
    #   zeros. So if there are 3 named cameras and empty_cameras=2, only
    #   camera1 actually needs a real source after the rename_map is applied.
    all_named_cams = sorted(k for k in cam_keys
                             if not k.startswith("observation.images.empty_camera_"))
    synthetic_empties = sorted(k for k in cam_keys
                                if k.startswith("observation.images.empty_camera_"))
    emp_count = int(cfg.get("empty_cameras") or 0)
    n_need_real = max(0, len(all_named_cams) - emp_count)
    needed_real = all_named_cams[:n_need_real]
    runtime_padded = all_named_cams[n_need_real:]

    print(f"\n  named camera input_features ({len(all_named_cams)}):  {all_named_cams}")
    print(f"  synthetic empty_camera pads ({len(synthetic_empties)}):  {synthetic_empties}")
    print(f"  policy.empty_cameras = {emp_count}  -> runtime auto-pads "
          f"the last {emp_count} named camera(s): {runtime_padded}")
    print(f"  -> {len(needed_real)} camera(s) need a real source:   {needed_real}")

    missing_real = [k for k in needed_real if k not in aliased and k not in frame0_fed]
    fed_real = [k for k in needed_real if k in aliased or k in frame0_fed]
    cam_pass = (len(missing_real) == 0)
    if cam_pass:
        print(_green(f"\n  ✓ All {len(needed_real)} real camera key(s) are fed by a robot key after rename."))
    else:
        print(_red(f"\n  ✗ {len(missing_real)} real camera key(s) have NO source after rename:"))
        for k in missing_real:
            print(_red(f"      - {k}  (no robot key would alias to this)"))
        print(_red("    The policy will receive zero-padded BLACK frames for those keys."))
        print(_red("    This is the most common Eval 3 deploy failure."))
        # Recommend the fix.
        if needed_real and args.robot_camera_keys:
            suggested = {args.robot_camera_keys[0]: needed_real[0]}
            print(_yellow(f"\n    Suggested fix: --dataset.rename_map='{json.dumps(suggested)}'"))

    # ----- Task-string canonicalization -----
    print(_bold("\n== Task-string canonicalization ==\n"))
    task_pass = True
    if args.task:
        print(f"  task              : {args.task!r}")
        if args.task in CANONICAL_TASKS:
            print(_green(f"  ✓ exact match against the training canonical form."))
        else:
            task_pass = False
            print(_red(f"  ✗ NOT a canonical form. Training rewrites every task to one of:"))
            for c in sorted(CANONICAL_TASKS):
                print(_red(f"      {c!r}"))
            # Try to suggest a fix.
            for celeb in KNOWN_CELEBS:
                if celeb.lower() in args.task.lower():
                    print(_yellow(f"\n    Suggested fix: --dataset.single_task='Place the coke on {celeb}'"))
                    break
    else:
        print("  (skipped; no --task provided)")

    # ----- Baked-in normalizer sanity -----
    print(_bold("\n== Baked-in normalizer (action / observation.state) ==\n"))
    stats = load_baked_normalizer(args.policy_pretrained_path)
    if stats is None:
        print(_yellow("  (could not load policy_preprocessor_step_5_normalizer_processor.safetensors)"))
    else:
        for key, st in stats.items():
            print(f"  {key}:")
            print(f"    frame count: {st['count']}")
            if st['mean']:
                print(f"    mean : {[round(v, 2) for v in st['mean']]}")
                print(f"    std  : {[round(v, 2) for v in st['std']]}")
                print(f"    range: [{[round(v, 2) for v in st['min']]} .. {[round(v, 2) for v in st['max']]}]")

    # ----- Verdict + corrected command -----
    print(_bold("\n== Verdict ==\n"))
    if cam_pass and task_pass:
        verdict = "PASS"
        color = _green
    elif cam_pass and not task_pass:
        verdict = "WARN"
        color = _yellow
    else:
        verdict = "FAIL"
        color = _red
    print(color(f"  {verdict}  (cameras={'OK' if cam_pass else 'FAIL'}, task={'OK' if task_pass else 'FAIL'})"))

    if not cam_pass or not task_pass:
        # Build a corrected command.
        suggested_rename = rename_map.copy()
        if not cam_pass and needed_real and args.robot_camera_keys:
            suggested_rename[args.robot_camera_keys[0]] = needed_real[0]
        suggested_task = args.task
        if not task_pass:
            for celeb in KNOWN_CELEBS:
                if celeb.lower() in args.task.lower():
                    suggested_task = f"Place the coke on {celeb}"
                    break
        print(_bold("\n  Suggested corrected lerobot-record command:\n"))
        cam_dict = (f"{{ {args.robot_camera_keys[0].split('.')[-1]}: "
                    f"{{type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}} }}")
        rename_arg = json.dumps(suggested_rename, separators=(",", ":"))
        n_step_arg = ""
        if n_action_steps and n_action_steps > 25:
            n_step_arg = " \\\n      --policy.n_action_steps=25"
        print(f"""    lerobot-record \\
      --robot.type=so101_follower \\
      --robot.port={args.robot_port} \\
      --robot.id={args.robot_id} \\
      --robot.cameras='{cam_dict}' \\
      --policy.type=smolvla \\
      --policy.pretrained_path={args.policy_pretrained_path} \\
      --policy.device=mps{n_step_arg} \\
      --dataset.repo_id={args.dataset_repo_id} \\
      --dataset.rename_map='{rename_arg}' \\
      --dataset.num_episodes=1 \\
      --dataset.single_task='{suggested_task}' \\
      --dataset.streaming_encoding=true \\
      --dataset.encoder_threads=4 \\
      --dataset.vcodec=h264_videotoolbox \\
      --display_data=true""")

    print()
    return 0 if cam_pass and task_pass else (2 if not cam_pass else 1)


if __name__ == "__main__":
    sys.exit(main())
