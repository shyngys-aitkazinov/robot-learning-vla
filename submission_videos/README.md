# Submission videos (Eval 3)

These MP4s are tracked in git (~20 MB total) so teammates can clone them.
Also zip for the course Azure upload: `./scripts/package_submission_videos.sh teamXX`

Rename files before zipping if you know the exact eval (examples below).

| File | Duration | Rename suggestion (after you verify content) |
|------|----------|-----------------------------------------------|
| `eval3_video_01_36s.mp4` | ~36 s | e.g. `eval3_id_taylor_swift.mp4` or `eval3_id_multi.mp4` |
| `eval3_video_02_75s.mp4` | ~75 s | e.g. `eval3_id_yann_lecun.mp4` or `eval3_ood_messi.mp4` |

## Course requirements (each eval you claim)

- Leader **and** follower arm visible (not teleoperating the follower)
- **≥3 rollouts back-to-back** without a cut
- One video per eval condition (or clearly separated segments)

## Upload

```bash
cd ..
TEAM=teamXX   # your team number
zip -r "${TEAM}-videos.zip" submission_videos/*.mp4
# then use the videos curl command from docs/PROJECT_SUBMISSION.md
```

Or: `./scripts/package_course_submission.sh teamXX` only zips repo — use a separate videos zip as above.
