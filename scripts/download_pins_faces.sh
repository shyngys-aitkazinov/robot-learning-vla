#!/usr/bin/env bash
# Download the Pins Face Recognition dataset (Kaggle) into ./datasets/ and unzip it
# into the layout the ChArUco post-processor expects.
#
# Source : https://www.kaggle.com/datasets/hereisburak/pins-face-recognition
# Layout : datasets/pins-face-recognition/105_classes_pins_dataset/pins_<Name>/<Name>*.jpg
# Doc    : datasets/README.md  ::  Pins Face Recognition (celebrity face pool)
# Used by: docs/eval3/charuco_pipeline.md  §5 (Post-process)
#
# Idempotent — re-running is cheap:
#   * skips the download if the zip is already on disk,
#   * skips extraction if the target directory already contains JPGs.
# Pass -f / --force to refetch and re-extract from scratch.
#
# Usage:
#   ./scripts/download_pins_faces.sh
#   ./scripts/download_pins_faces.sh --force

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

URL="https://www.kaggle.com/api/v1/datasets/download/hereisburak/pins-face-recognition"
ZIP="datasets/pins-face-recognition.zip"
EXT_DIR="datasets/pins-face-recognition"
INNER_DIR="${EXT_DIR}/105_classes_pins_dataset"

FORCE=0
for arg in "$@"; do
  case "$arg" in
    -f|--force) FORCE=1 ;;
    -h|--help)
      sed -n '2,18p' "${BASH_SOURCE[0]}" | sed -E 's/^# ?//'
      exit 0
      ;;
    *)
      echo "unknown arg: $arg  (use --help)" >&2
      exit 1
      ;;
  esac
done

mkdir -p datasets

# --- 1. Download --------------------------------------------------------
if [[ -f "$ZIP" && "$FORCE" -eq 0 ]]; then
  size=$(du -h "$ZIP" | awk '{print $1}')
  echo ">> $ZIP already exists ($size); skipping download. Use --force to refetch."
else
  [[ -f "$ZIP" && "$FORCE" -eq 1 ]] && { echo ">> --force: removing existing $ZIP"; rm -f "$ZIP"; }
  echo ">> Downloading Pins Face Recognition from Kaggle (~372 MB)"
  echo "     $URL"
  echo "     -> $ZIP"
  curl -L -f --progress-bar -o "$ZIP" "$URL"
  size=$(du -h "$ZIP" | awk '{print $1}')
  echo ">> Downloaded $size"
fi

# --- 2. Extract ---------------------------------------------------------
if [[ -d "$INNER_DIR" && "$FORCE" -eq 0 ]]; then
  n=$(find "$INNER_DIR" -type f -name '*.jpg' 2>/dev/null | wc -l | tr -d ' ')
  echo ">> $INNER_DIR already populated ($n JPGs); skipping extract. Use --force to re-extract."
else
  [[ -d "$EXT_DIR" && "$FORCE" -eq 1 ]] && { echo ">> --force: removing existing $EXT_DIR"; rm -rf "$EXT_DIR"; }
  mkdir -p "$EXT_DIR"
  echo ">> Unzipping $ZIP -> $EXT_DIR"
  unzip -q "$ZIP" -d "$EXT_DIR"
  echo ">> Extract complete."
fi

# --- 3. Verify layout ---------------------------------------------------
if [[ ! -d "$INNER_DIR" ]]; then
  echo "ERROR: expected $INNER_DIR after extraction but it's missing." >&2
  echo "       The Kaggle archive layout may have changed; inspect $ZIP manually." >&2
  exit 2
fi

n_classes=$(find "$INNER_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
n_images=$(find "$INNER_DIR" -type f -name '*.jpg' | wc -l | tr -d ' ')
size=$(du -sh "$EXT_DIR" | awk '{print $1}')

cat <<EOM

==========================================================================
 Pins Face Recognition ready.

   classes (celebrities) : $n_classes
   total images          : $n_images
   extracted size        : $size
   path                  : $INNER_DIR

 Each celebrity has its own directory: $INNER_DIR/pins_<Name>/

 Reminder when using as an OOD pool for ChArUco post-processing
 (docs/eval3/charuco_pipeline.md §5):
   * Hold out 'Taylor Swift', 'Yann LeCun', 'Barack Obama' from the
     pool — those are the TOY identities scored in runs 1–6 and must
     not appear in OOD training data.
   * Filter to portrait-aspect crops (or re-crop with a face detector)
     before warping into the board area for consistent compositing.
==========================================================================
EOM
