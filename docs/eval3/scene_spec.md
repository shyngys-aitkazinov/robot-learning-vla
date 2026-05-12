# Eval 3 — physical scene specification

Measurable checklist so teleop, replay, and eval day stay aligned.

## Objects

| Item | Spec | Notes |
|------|------|--------|
| Celebrity prints | **3× DIN A5** (~148 × 210 mm), **semicircle** in front of robot | No tape to table (course rule). |
| Coke | **330 ml slim** can, **empty**, may be **lightly dented** but must **stand** | Use the **same physical can** on demo day if possible. |
| Workspace | **White table**, robot at edge | Final eval in **HG**; collect **some** data there (non-perfect white). |

## Layout (operational)

1. **Arc**: Three prints rest on the table forming a shallow semicircle opening toward the robot base (consistent “left / center / right” from **camera / robot** perspective — **document which frame you use** in dataset README).
2. **Can start pose**: **Near geometric center** of the three prints (spec: “middle”). Record **approximate XY** of can base relative to table fiducial (optional tape **under** table or edge ruler — **not** tape on prints).
3. **±5 cm tolerance** (course): Between episodes, jitter **each print centroid** and **can base** independently within **±50 mm** in the table plane (small rotations optional). Stratify so models do not memorize absolute pixels only.

## Camera (single stream)

- **Framing**: Full **FOV contains all three prints + entire can + approach corridor** for the arm.
- **Height / distance**: Fixed across Eval 3 train and deploy once chosen (Eval 3 allows different mount than Eval 1/2).
- **Exposure**: Prefer **manual exposure** after pilot; if auto, pair with **strong illumination augmentation** in training.

### Acceptance checklist (before recording batch)

- [ ] All three faces recognizable at **256×256** resize (simulate in preview).
- [ ] Can not clipped by frame edge at extremes of **±5 cm** layout.
- [ ] Glare on prints acceptable after rotating lamp once — avoid saturated blobs on faces.

## TOY prints (regime A)

- Print Slack PDF **in color**.
- **Cut flush** — **no white border** (TA match requirement).
- Store unused duplicates sealed (avoid sun fade).

## Success semantics (for internal QA)

- **Functional**: After placement, can **majority overlaps** the **named** print; stable enough not to fall within ~2 s.
- **Temporal**: First motion within rollout starts after TA **Enter**; total policy-controlled segment **≤ 20 s** (see rollout harness).
