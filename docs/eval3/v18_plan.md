# v18 — KL Attention Loss (KLAL) — Implementation Plan

> Status: **plan only** (no code written yet). This is the design doc for the
> v18 run, sibling to `v16_playbook.md` / `v17_playbook.md`. Implementation is
> deferred until this plan is signed off.

## 1. Context — why v18

v16 (slot bottleneck) and v17 (cam1 drop) both attack one failure *indirectly*:
the action expert grounds the trajectory in a **shortcut** (can-drift, proprio,
live-cam pixels) instead of the **language → correct-face** binding. They make
the shortcut harder to take (v17) and give the language a privileged token
(v16's `h_slot`), but neither tells the model *where to look*.

**KLAL adds that signal directly.** We know the pixel regions of the 3
celebrity face-prints in every scene. KLAL adds an auxiliary KL loss that pushes
the model's attention onto those print pixels — "look at the faces, not the
table." It is complementary to v16/v17, not a replacement.

Decisions locked with the user:

- **Supervise BOTH attentions**: (1) the `SlotClassifier`'s existing
  language-conditioned softmax over image tokens, and (2) the action expert's
  real cross-attention probabilities.
- **Target distribution = union of all 3 print regions** (foreground vs table;
  prompt-agnostic). No target-slot binding required.
- **Coverage = synthetic slates AND real `dataset_v4_*`**; any frame with no
  print-region info is *ignored* (mirrors the aux head's `-100` and the slot CE
  pre-grasp mask).

KLAL is **off by default** (both weights `0` → patches inert → bit-exactly v17),
enabled per-run via env vars. **Train-only**: a v18 checkpoint is a v16/v17
checkpoint for `eval3_vla_deploy.py` (deploy never sets the capture flag, so the
attention hook is dormant and the new modules carry no inference weight).

## 2. Key facts established during exploration

### 2.1 Token geometry (deterministic, content-independent)

- Input to vision encoder: **512×512** (`configuration_smolvla.py:48`
  `resize_imgs_with_padding=(512,512)`), patch size **32** → 16×16=256 patches,
  then `pixel_shuffle(scale=2)` (`.venv/.../smolvlm/modeling_smolvlm.py:424`) →
  **64 tokens per camera = 8×8 row-major grid**.
- Token `t` (0..63) → grid cell `(t//8, t%8)` → pixel box
  `[(t//8)*64 : +64, (t%8)*64 : +64]` of the 512×512 letterboxed image.
- **`tok_per_cam = 64` is fixed.** Camera ordering in the prefix is
  `[cam1, cam2, cam3]`, so camera *k* (1-indexed) occupies image-token columns
  `[(k-1)*64 : k*64]`. (Confirmed by the slot bottleneck's "v16 prefix check":
  `n_img=192 n_cams=3 tok_per_cam=64`.)

### 2.2 Prefix layout & attention path

- Prefix sequence: `[cam1·64][cam2·64][cam3·64][lang·n][state·1][h_slot·K]`
  (`eval3_smolvla_slot_bottleneck.py:_patched_embed_prefix`). Image tokens are
  the first `n_cams·64` positions.
- Expert cross-attention is computed in
  `SmolVLMWithExpertModel.forward_cross_attn_layer`
  (`.venv/.../smolvla/smolvlm_with_expert.py:374`): `expert_query_states`
  (q_len = suffix/chunk length) attend to `expert_key_states` (k_len = prefix
  length). The call goes through `eager_attention_forward` (`:504`), which
  **always materializes** `probs = softmax(...)` of shape
  `(B, heads, q_len, k_len)` (`:540`) but discards it (returns only the context).
- `get_attention_interface()` (`:500`) is hardcoded to `eager_attention_forward`
  — no SDPA/flash to fight. **No `.venv` edits needed**; we monkey-patch
  `eager_attention_forward` to stash `probs` when a capture flag is set.
- **Disambiguating the expert-cross call** among all `eager_attention_forward`
  invocations (prefix self-attn at `:316`, per-layer self-attn in
  `forward_attn_layer`, expert-cross at `:374`): the expert-cross probs are the
  ones with `q_len < k_len` and `q_len == suffix length` (the action-chunk
  region) and `k_len == prefix length` (contains the 192 image cols). Capture
  every such tensor across the cross-attn layers and average them.

### 2.3 Composition with existing patches (critical)

- The slot bottleneck wraps `embed_prefix` + `SmolVLAPolicy.forward` +
  `_extract_complementary_data` + `MetricsTracker`. The aux head wraps
  `VLAFlowMatching.forward` + same others, but **aux head and slot bottleneck
  are mutually exclusive** (`train_eval3_smolvla.py:47-54`): v17/v18 use the
  slot bottleneck, so `VLAFlowMatching.forward` is the **stock lerobot** one.
- KLAL therefore does **not** need to replicate `VLAFlowMatching.forward`.
  It only needs two already-available intermediates:
  1. `model.slot_clf._last_attn` — add a one-line stash to
     `SlotClassifier.forward` (it already computes `attn`, shape `(B, n_img,1)`).
  2. the expert-cross `probs` — captured via the `eager_attention_forward`
     patch above.
  KLAL **wraps `SmolVLAPolicy.forward`** (applied *after* the slot bottleneck so
  it sits outermost): set capture flag → call inner forward (slot version) →
  read the two intermediates → add weighted KL → clear flag.
- `MetricsTracker.__init__/__setattr__` chaining: capture the *current* methods
  at `apply()` time and reassign, exactly like the existing two patches. Applied
  after slot bottleneck, the chain is `klal → slot → lerobot`, all fire.

### 2.4 Print-region provenance

- **Synth** (`tools/eval3_synth_dataset_gen.py:733/749`,
  `eval3_charuco_episode_demo.py:696` `compose_multi`): the 3 board quads
  `board_img = cv2.perspectiveTransform(board_corners_mm, H)` are already
  computed per frame; homography is **locked on frame 0** → quads are constant
  per episode. Currently not persisted.
- **Real `dataset_v4_*`**: fixed camera + fixed board ⇒ print regions are
  constant per dataset. `tools/eval3_extract_masks.py:36` has hardcoded per-slug
  print polygons (swift/lecun/obama); ChArUco detection is the general fallback.
- Slot/identity from repo name: `_slot_from_repo` (`eval3_concat_patch.py:524`).
- **Consequence**: a single `(64,)` token-mask **per episode** covers every
  camera (board is static within the episode). Cheap.

## 3. Design

Decouple **"where are the prints"** (data side → per-episode 64-token target
mask) from **"supervise attention there"** (model side → two KL terms). The
shared contract is a new batch key **`attn_target_mask`**: a `(64,)` float vector
(mass on print-overlapping grid cells). A row whose mask is all-zeros / absent
is *ignored*.

### 3.1 Target distribution

`p_target = mask / mask.sum()` (uniform over the union of the 3 print tokens).
Forward KL `KL(p_target ‖ p_attn) = −Σ p_target·log p_attn + const` ⇒ pushes
attention mass onto the print tokens (mode-covering; correct for "cover the 3
faces"). Optional `EVAL3_KLAL_TARGET_SMOOTH` adds a small uniform floor for
stability. Diagnostic `klal_*_mass` = Σ over print tokens of `p_attn` (fraction
of attention landing on prints; should climb toward 1).

## 4. Implementation (deferred — for reference)

### 4.1 New `scripts/eval3_attn_target.py` — geometry helper

`quads_to_token_mask(quads_px, img_hw, grid=8) -> np.ndarray[(64,)]`: rasterize
the 3 print quads, apply the **same** letterbox resize as
`resize_imgs_with_padding` (so token cells line up with what the model sees),
pool to 8×8, normalize. Single source of truth shared by training and any audit.

### 4.2 Data side — per-episode sidecar + prep wiring

- **Sidecar** `datasets/<name>/meta/print_regions.json`:
  `{"episodes": {"<ep>": {"prints_px": [[[x,y]×4]×3], "img_hw": [H,W]}}}`
  (or one `"default"` entry when the board is dataset-fixed). Lazy-loaded,
  picklable (same pattern as masks/backgrounds).
- **Producers**:
  - Synth generators (`tools/eval3_synth_dataset_gen.py`,
    `tools/eval3_synth_pins_dataset_gen.py`) emit the already-computed quads at
    gen time (≈0 cost).
  - **Back-fill tool** `tools/eval3_emit_print_regions.py` recomputes sidecars
    for *existing* datasets — synth: re-lock homography on frame 0; real v4:
    ChArUco detect on frame 0 (hardcoded-polygon fallback for legacy slugs).
    Avoids regenerating the ~1.8 M-frame corpus.
- **`Eval3PrepDataset`** (`scripts/eval3_dataset_prep.py`): opt-in augmenter
  layer (env `EVAL3_KLAL_TARGET=1`) that, in `__getitem__`, looks up the
  episode quads, builds the `(64,)` mask via `eval3_attn_target`, and attaches
  `row["attn_target_mask"]`. Missing sidecar → no key → ignored downstream.
  Picklable via the existing `__reduce__`; per-repo enable logged like
  `cam_drop=` in startup diagnostics.
- **`eval3_concat_patch._patched_make_dataset`**: read the env, resolve the
  sidecar path per repo (same site as `EVAL3_LOCAL_REPOS` / `cam_drop`), pass to
  the augmenter.

### 4.3 New `scripts/eval3_smolvla_klal.py` — model patch (`apply()`)

Called from `train_eval3_smolvla.py` **after** the slot bottleneck. Steps:

1. `_extract_complementary_data` patch → pass `attn_target_mask` through
   `batch_to_transition` (compose with, don't clobber, the slot/aux patch).
2. `SmolVLMWithExpertModel.eager_attention_forward` wrap → when
   `self._klal_capture` is set and the call is the expert-cross one
   (`q_len < k_len`, `q_len == suffix len`), append `probs` to
   `self._klal_attn_buf`. Negligible overhead; **inference never sets the flag**.
3. `SlotClassifier.forward` (edit in `eval3_smolvla_slot_bottleneck.py`):
   one line `self._last_attn = attn.squeeze(-1)  # (B, n_img)`.
4. `SmolVLAPolicy.forward` **wrapper** (outermost): pop `attn_target_mask`;
   set capture flag + clear buffer; call inner forward → `(loss, loss_dict)`;
   then:
   - **Slot KL** (`EVAL3_KLAL_SLOT_WEIGHT`): take `model.slot_clf._last_attn`
     (already a softmax over the cam2 slice in frame-0 mode = 64 tokens; if it
     is 192-wide, slice the `EVAL3_KLAL_CAMERA` columns and renormalize),
     `KL(p_target ‖ attn)` over valid rows.
   - **Expert KL** (`EVAL3_KLAL_EXPERT_WEIGHT`): stack `_klal_attn_buf`, mean
     over heads + over the chunk query dim, slice key cols
     `[(cam-1)*64 : cam*64]` (`EVAL3_KLAL_CAMERA`, default 2 = frame-0, robust to
     v17 cam1-drop), renormalize, mean over the selected layers
     (`EVAL3_KLAL_EXPERT_LAYERS`, default all), `KL(p_target ‖ p_attn)`.
   - Add `w_slot·klal_slot + w_expert·klal_expert` to `loss` (per-sample when
     `reduction="none"`); surface `klal_slot_loss`, `klal_expert_loss`,
     `klal_slot_mass`, `klal_expert_mass`, `klal_n` in `loss_dict`; stash
     `self._last_klal_dict`; clear the capture flag.
5. `MetricsTracker.__init__` add the meters; `__setattr__` pump them on the
   `loss` assignment (mirror the existing patches; chain preserved).

### 4.4 Launcher + entry hook

- `scripts/run_eval3_smolvla_v18_klal_train.sh` — copy of the v17 launcher with
  the KLAL env block documented in the header.
- `scripts/train_eval3_smolvla.py` — after the slot/aux block, if
  `EVAL3_KLAL_SLOT_WEIGHT>0` or `EVAL3_KLAL_EXPERT_WEIGHT>0`, call
  `eval3_smolvla_klal.apply()` (so it wraps the slot bottleneck).

### 4.5 Env-var contract (all default to off / no-op)

| Env var | Default | Effect |
|---|---|---|
| `EVAL3_KLAL_TARGET` | `0` | Master switch: attach `attn_target_mask` in prep. |
| `EVAL3_KLAL_SLOT_WEIGHT` | `0` | Weight on the SlotClassifier-attn KL term. |
| `EVAL3_KLAL_EXPERT_WEIGHT` | `0` | Weight on the expert cross-attn KL term. |
| `EVAL3_KLAL_CAMERA` | `2` | Camera whose tokens to supervise (1=cam1/live, 2=cam2/frame-0). cam2 is robust to v17 cam1-drop. |
| `EVAL3_KLAL_EXPERT_LAYERS` | `all` | Comma list of expert cross-attn layer indices to average (or `all`). |
| `EVAL3_KLAL_TARGET_SMOOTH` | `0.0` | Uniform floor added to `p_target` before normalize. |
| `EVAL3_KLAL_SIDECAR_NAME` | `print_regions.json` | Sidecar filename under `meta/`. |

## 5. Critical files

**New**: `scripts/eval3_attn_target.py`, `scripts/eval3_smolvla_klal.py`,
`tools/eval3_emit_print_regions.py`, `scripts/run_eval3_smolvla_v18_klal_train.sh`,
`tools/eval3_klal_test.py`.
**Edit**: `scripts/eval3_smolvla_slot_bottleneck.py` (1-line `_last_attn` stash),
`scripts/eval3_dataset_prep.py` (augmenter layer),
`scripts/eval3_concat_patch.py` (env + sidecar wiring),
`scripts/train_eval3_smolvla.py` (apply hook), synth generators (gen-time emit).

## 6. Verification ladder

1. **Geometry unit test** (`tools/eval3_klal_test.py`): `quads_to_token_mask`
   maps a known bbox to the expected 8×8 cells; token `t`↔`(t//8,t%8)`;
   ignore/validity handling. Pure CPU.
2. **Back-fill smoke**: run `eval3_emit_print_regions.py` on one synth + one
   `dataset_v4_*`; overlay the quads (reuse `tools/eval3_render_overlay.py`) to
   confirm the 3 prints are covered.
3. **No-op invariance**: both weights `0` → a 4-step run reproduces the v17 loss
   curve (KLAL inert).
4. **KLAL-on end-to-end smoke** (MPS, 4 steps): `klal_slot_loss`,
   `klal_expert_loss`, `klal_*_mass`, `klal_n>0` appear; `klal_n` ≈ the synth/v4
   fraction of the batch; the v16 prefix check + `slot_acc` unchanged.
5. **Attention-mass climb** (val watcher §4.1 of v17 playbook): `klal_*_mass`
   rises; `slot_acc` / `cross_prompt_delta` improve vs v17 at matched steps.
6. **Deploy unchanged**: a v18 checkpoint loads via the `v16` deploy battery arm
   (`eval3_check_deploy_command.py` passes); KLAL adds no inference deps.

## 7. Open questions / risks

- **Expert attention is diffuse early** — the KL may fight the flow loss before
  features form. Mitigation: small `EVAL3_KLAL_EXPERT_WEIGHT` (≈0.1–0.3) and/or
  a warmup; the slot-attn KL is the safer/cheaper of the two.
- **cam2 semantics under v16**: pre-grasp cam2 = current frame, post-grasp =
  cached frame-0 — both show the same static board, so the per-episode mask is
  valid throughout. cam1-drop (v17) is why cam2 is the default supervised camera.
- **Real v4 quad accuracy** depends on ChArUco detection on frame 0; the
  hardcoded polygons cover only the legacy slugs. Back-fill tool logs per-dataset
  detection success so misses are visible before training.
