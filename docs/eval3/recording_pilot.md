# Eval 3 — recording pilot (regime A permutations)

Hardware-dependent pilot your teammates execute; this doc is the **repeatable procedure**.

## Preconditions

- Calibration + ports per [README.md](../../README.md#calibration-files).
- Scene checklist in [scene_spec.md](scene_spec.md).
- Prompt strings per [prompt_protocol.md](prompt_protocol.md).

## Permutation matrix (3 identities × positions)

Label positions **L**, **C**, **R** from **robot camera** perspective.

For **each** assignment below, record **≥1 episode** with matching `single_task` naming the **target** celebrity (the one to place the can **on**):

| Episode tag | Swift | Obama | LeCun |
|-------------|-------|-------|-------|
| perm_01 | L | C | R |
| perm_02 | L | R | C |
| perm_03 | C | L | R |
| perm_04 | C | R | L |
| perm_05 | R | L | C |
| perm_06 | R | C | L |

Also vary **which name is the instruction target** across episodes so each identity appears as target **balanced** across permutations.

## `lerobot-record` skeleton

Replace ports, camera dict, and repo id.

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=<FOLLOWER_USB_PORT> \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=<LEADER_USB_PORT> \
  --teleop.id=my_awesome_leader_arm \
  --dataset.repo_id=RobotLearningVLA/eval3_toy_v1 \
  --dataset.num_episodes=1 \
  --dataset.single_task="Place the coke on Taylor Swift" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=4 \
  --dataset.vcodec=h264_videotoolbox
```

On Linux with NVIDIA encoding you may switch `--dataset.vcodec` per LeRobot docs.

## Post-session QA

1. `lerobot-replay` **≥2** episodes on hardware with **same layout** as record.
2. Run `python tools/inspect_lerobot_dataset.py --repo-id ...` after Hub sync.
3. Add **`v3.0`** tag if new dataset ([dataset_matrix.md](dataset_matrix.md)).

## HG field block

After replay QC passes in the lab, schedule **≥1** capture block in **HG-like** lighting/table ([scene_spec.md](scene_spec.md)).
