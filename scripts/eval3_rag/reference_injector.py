"""Reference-face retrieval for the Eval3 RAG training variant.

Fully generic — no hardcoded person names.  Works for any person whose name
appears in the task string or whose slug directory exists in a search dir.

At training time a *random* face image is sampled per step so the model learns
visual matching instead of memorising one specific reference photo.
At inference time the canonical (first sorted) image is used.

The injector searches local dataset directories in priority order:

    1. datasets/celeb_refdb/               (built by build_celeb_refdb.py)
    2. datasets/out-distribution-eval-3/
    3. datasets/in-distribution-eval-3/
    4. datasets/out-distribution-eval-3-pins/

When no image is found locally and ``auto_fetch=True``, the Wikipedia REST
Summary API is queried (no API key) and the result is cached to celeb_refdb/.
This makes inference work for any person with a Wikipedia article — no manual
DB setup required.

The returned tensor is ``(3, H, W)`` float in ``[0, 1]`` — the same format as
camera frames decoded by the LeRobot dataset backend.
"""
from __future__ import annotations

import glob
import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Search directories
# ---------------------------------------------------------------------------

def _default_search_dirs(root: str | Path | None = None) -> list[str]:
    """Return ordered local directories to search for person reference images."""
    if root is None:
        root = Path(__file__).resolve().parent.parent.parent
    root = Path(root)
    return [str(root / "datasets" / d) for d in (
        "celeb_refdb",
        "out-distribution-eval-3",
        "in-distribution-eval-3",
        "out-distribution-eval-3-pins",
    )]


# ---------------------------------------------------------------------------
# Task / repo-id → slug helpers  (zero hardcoded names)
# ---------------------------------------------------------------------------

# Phrases that mark a canonical placeholder task — no real person name.
_CANONICAL_TASK_PHRASES: frozenset[str] = frozenset({
    "person in the reference photo",
    "person in reference photo",
})

# Prefixes that precede the person name in the standard task format.
_TASK_PREFIXES: tuple[str, ...] = (
    "place the coke on the ",
    "place the coke on ",
    "put the coke on the ",
    "put the coke on ",
)

# Dataset-naming prefixes to strip from repo IDs.
_REPO_PREFIXES: tuple[str, ...] = (
    "dataset_v3_synth_rag1_",
    "dataset_v3_synth_pinned_idood_",
    "dataset_v4_synth_",
    "dataset_v3_synth_",
    "dataset_v4_",
    "dataset_v3_",
    "dataset_v2_",
    "dataset_v1_",
)

# Position / version suffixes to strip from repo IDs.
_REPO_SUFFIXES: tuple[str, ...] = (
    "_left_3", "_middle_3", "_right_3",
    "_left_2", "_middle_2", "_right_2",
    "_left_1", "_middle_1", "_right_1",
    "_left",   "_middle",   "_right",
    "_1", "_2", "_3", "_4", "_5",
)


def task_to_celeb_slug(task: str) -> str | None:
    """Extract a person slug from any task string — no hardcoded names.

    Examples::

        "Place the coke on Marc Pollefeys"               → "marc_pollefeys"
        "Place the coke on Elon Musk"                    → "elon_musk"
        "Place the coke on Fei-Fei Li"                   → "fei_fei_li"
        "Place the coke on the person in the reference photo"  → None
        ""                                               → None

    Returns ``None`` for canonical/placeholder tasks or strings that don't
    start with a recognised task prefix.
    """
    tl = task.lower().strip()

    for phrase in _CANONICAL_TASK_PHRASES:
        if phrase in tl:
            return None

    for prefix in _TASK_PREFIXES:
        if tl.startswith(prefix):
            name_part = tl[len(prefix):].strip()
            if name_part:
                slug = re.sub(r"[^a-z0-9]+", "_", name_part).strip("_")
                return slug or None

    return None


def _repo_id_to_celeb_slug(
    repo_id: str,
    search_dirs: list[str] | None = None,
) -> str | None:
    """Extract a slug from a dataset repo ID — no hardcoded names.

    Strips dataset-naming prefixes and position/version suffixes.  If the
    result is a short abbreviated name (e.g. ``"taylor"`` from
    ``"dataset_v4_taylor_left"``), tries to resolve it to a full slug by
    prefix-matching against directories that actually exist in *search_dirs*
    (e.g. ``"taylor"`` → ``"taylor_swift"`` when ``celeb_refdb/taylor_swift/``
    is present).  Falls back to the stripped name if no match is found.

    Examples::

        "dataset_v4_taylor_left"               → "taylor_swift"  (if dir exists)
        "dataset_v3_synth_rag1_elon_musk_left_2" → "elon_musk"
        "dataset_v3_synth_rag1_marc_pollefeys_right_2" → "marc_pollefeys"
    """
    name = repo_id.lower().rsplit("/", 1)[-1]
    for prefix in _REPO_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    for suffix in _REPO_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    if not name:
        return None

    # Resolve abbreviated names via filesystem prefix-matching so that e.g.
    # "taylor" resolves to "taylor_swift" when that directory exists — without
    # any hardcoded mapping.
    if search_dirs is None:
        search_dirs = _default_search_dirs()
    for db_dir in search_dirs:
        if not os.path.isdir(db_dir):
            continue
        if os.path.isdir(os.path.join(db_dir, name)):
            return name  # exact match
        try:
            matches = sorted(
                d for d in os.listdir(db_dir)
                if os.path.isdir(os.path.join(db_dir, d))
                and d.startswith(name + "_")
            )
        except OSError:
            continue
        if matches:
            return matches[0]  # best prefix match (shortest / first alphabetically)

    return name  # return as-is; caller handles missing images


# ---------------------------------------------------------------------------
# Open-source face dataset lookup  (offline pre-fetch via build_celeb_refdb.py)
# ---------------------------------------------------------------------------
#
# Recommended HuggingFace face recognition repos (run once offline):
#   leondgarse/celeba_hq_face_recognition  — 30k HQ face crops, 7k+ people
#   HuggingFaceM4/VGGFace2_sample          — VGGFace2 sample, many celebrities
#
# Usage (one-time offline setup for any OOD person):
#   python scripts/eval3_rag/build_celeb_refdb.py \
#       --from-hf leondgarse/celeba_hq_face_recognition \
#       --slugs marc_pollefeys elon_musk ...
#
# Portraits from face recognition datasets are guaranteed face crops — no
# quality validation needed.  Wikipedia fetch (below) is the runtime fallback
# for people not yet in the local DB.

# ---------------------------------------------------------------------------
# Portrait quality — face detection
# ---------------------------------------------------------------------------

def _has_detectable_face(img_path: str | Path) -> bool:
    """Return True if at least one face is detected in the image.

    Tries three OpenCV Haar cascades in order — frontal (default), frontal
    alt2, and profile — so the check passes for slightly off-axis portraits
    that the strict frontal-only cascade misses.  Histogram equalisation is
    applied first to handle uneven lighting.

    Returns True if cv2 is unavailable so the caller is not blocked.
    """
    try:
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            return False
        # Downscale very large images to speed up detection
        h, w = img.shape[:2]
        scale = min(1.0, 1200 / max(h, w, 1))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)   # normalise contrast

        _cascades = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        ]
        for cascade_file in _cascades:
            path = cv2.data.haarcascades + cascade_file
            cascade = cv2.CascadeClassifier(path)
            if cascade.empty():
                continue
            faces = cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(30, 30)
            )
            if len(faces) > 0:
                return True
        return False
    except Exception:
        return True  # cv2 unavailable or model file missing — don't block


# ---------------------------------------------------------------------------
# Portrait validation — inference-safe helpers
# ---------------------------------------------------------------------------

_SENTINEL_FILENAME  = ".portrait_validated"
_CANONICAL_FILENAME = ".canonical_portrait"  # pre-computed best-resolution filename


def _write_canonical_portrait(slug_dir: str | Path, image_paths: list[str | Path]) -> None:
    """Pre-compute and cache the highest-resolution portrait filename.

    Called offline (fetch_wiki_portraits.py / download_refdb.py) after all
    portraits for a celebrity are saved.  Writes the filename of the
    highest-resolution validated image to .canonical_portrait so that
    _best_portrait_path() can return it instantly at inference without
    opening any image files.

    Safe to call with an empty list (no-op).
    """
    if not image_paths:
        return
    try:
        from PIL import Image as _PILImage
        best_name, best_px = "", 0
        for p in image_paths:
            try:
                with _PILImage.open(str(p)) as im:
                    px = im.width * im.height
                if px > best_px:
                    best_px = px
                    best_name = Path(p).name
            except Exception:
                pass
        if best_name:
            Path(slug_dir).joinpath(_CANONICAL_FILENAME).write_text(best_name + "\n")
    except Exception:
        pass


def _write_sentinel(slug_dir: str | Path, filename: str | None = None) -> None:
    """Record a Haar-validated filename in the sentinel file.

    Called offline (fetch_wiki_portraits.py / download_refdb.py) after face
    detection passes — never at inference.

    If *filename* is given, the name is appended to the sentinel so that
    _load_validated_filenames() can return exactly which files were validated.
    If *filename* is None (legacy call-sites), the sentinel is touched as a
    boolean marker for backward compatibility.
    """
    try:
        sentinel = Path(slug_dir) / _SENTINEL_FILENAME
        if filename:
            with sentinel.open("a") as f:
                f.write(filename + "\n")
        else:
            sentinel.touch()
    except Exception:
        pass


def _load_validated_filenames(slug_dir: str | Path) -> set[str] | None:
    """Return the set of Haar-validated filenames for *slug_dir*, or None.

    Returns None when no sentinel exists.
    Returns an empty set when the sentinel exists but has no listed files
    (legacy touch-only sentinel) — callers treat this as "all files validated".
    Returns a non-empty set when the sentinel lists specific filenames — only
    those files are considered validated.

    This is a plain text read — no pixel inspection, no image recognition.
    """
    sentinel = Path(slug_dir) / _SENTINEL_FILENAME
    if not sentinel.exists():
        return None
    try:
        names = {ln.strip() for ln in sentinel.read_text().splitlines() if ln.strip()}
        return names  # empty set = legacy boolean sentinel
    except Exception:
        return set()  # unreadable sentinel → treat as legacy boolean


def _has_portrait_sentinel(slug_dir: str | Path) -> bool:
    """Backward-compat alias — True if any sentinel exists for this directory."""
    return _load_validated_filenames(slug_dir) is not None


def _is_portrait_aspect(path: str, min_ratio: float = 0.5) -> bool:
    """True if the image has a portrait (or square) aspect ratio.

    Reads only the image header for dimensions — no pixel decoding, no
    neural network, no image recognition.  Portrait/headshot images are
    almost always taller than wide; wide landscapes (H/W < 0.5) are likely
    group photos, flags, or banners.

    ``min_ratio=0.5`` accepts anything taller than half its width, including
    tight square face crops.  Returns True on read errors so callers are not
    accidentally blocked by corrupt metadata.
    """
    try:
        from PIL import Image as _PILImage
        with _PILImage.open(path) as img:
            w, h = img.size
        return h >= w * min_ratio
    except Exception:
        return True  # can't read header — don't reject


# ---------------------------------------------------------------------------
# Online portrait fetch (Wikipedia REST API)
# ---------------------------------------------------------------------------

def _slug_to_display_name(slug: str) -> str:
    """Convert a filesystem slug to a display name suitable for Wikipedia lookup.

    Pure title-casing — no hardcoded dictionary::

        "marc_pollefeys" → "Marc Pollefeys"
        "fei_fei_li"     → "Fei Fei Li"
        "elon_musk"      → "Elon Musk"
    """
    return slug.replace("_", " ").title()


def _fetch_portrait_online(
    slug: str,
    save_root: str | Path,
    *,
    validate_face: bool = True,
) -> str | None:
    """Download a portrait from Wikipedia and cache it under *save_root*/<slug>/.

    Strategy
    --------
    1. Query the Wikipedia REST Summary API for the person's article.
    2. Download ``originalimage`` (full resolution) or ``thumbnail`` as fallback.
    3. If ``validate_face=True`` (default when called from download_refdb.py):
       run face detection (OpenCV Haar cascade) to verify the image is a
       portrait.  Images with no detectable face are discarded.
       Pass ``validate_face=False`` at deploy/inference time to avoid any
       image-recognition call — the image is accepted as-is and cached.
    4. Cache the validated image as ``wiki_portrait.<ext>``; subsequent calls
       return the cached path instantly without hitting the network.

    Returns the saved path, or ``None`` when offline / no article found.
    """
    import json
    import urllib.parse
    import urllib.request

    display_name = _slug_to_display_name(slug)
    save_dir = Path(save_root) / slug
    save_dir.mkdir(parents=True, exist_ok=True)

    # Filename fragments that strongly indicate non-portrait images.
    # Applied at runtime when validate_face=False (inference auto-fetch)
    # so we at least reject obvious event/logo/map images by URL.
    _URL_SKIP = frozenset([
        "logo", "flag", "signature", "map", "icon", "stadium", "trophy",
        "album", "poster", "chart", "banner", "wikidata", "wikipedia",
        "premiere", "ceremony", "gala", "festival", "awards",
        "red_carpet", "conference", "concert", "_vs_", "_vs.",
        "figure", "tussaud", "statue",
    ])

    def _url_looks_like_portrait(url: str) -> bool:
        """Heuristic: reject URLs whose filename contains known non-portrait terms."""
        fname = urllib.parse.unquote(url.split("/")[-1]).lower().replace(" ", "_")
        return not any(frag in fname for frag in _URL_SKIP)

    def _download_and_cache(img_url: str, filename: str) -> str | None:
        # At inference (validate_face=False) apply URL-fragment filter so
        # obvious non-portrait images are skipped before any download.
        if not validate_face and not _url_looks_like_portrait(img_url):
            log.debug("fetch_portrait: skipping non-portrait URL: %s", img_url.split("/")[-1])
            return None

        parsed_ext = Path(urllib.parse.urlparse(img_url).path).suffix.lower()
        ext = parsed_ext if parsed_ext in {".jpg", ".jpeg", ".png", ".webp"} else ".jpg"
        save_path = save_dir / f"{filename}{ext}"
        if save_path.exists():
            if not validate_face or _has_detectable_face(save_path):
                log.info("fetch_portrait: using cached portrait for %r", slug)
                return str(save_path)
            else:
                # Cached image failed face detection — discard and re-fetch.
                save_path.unlink(missing_ok=True)
        try:
            req = urllib.request.Request(
                img_url,
                headers={"User-Agent": "eval3-rag/1.0 (educational robot-learning)"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                save_path.write_bytes(resp.read())
        except Exception as exc:
            log.warning("fetch_portrait: download failed for %r: %s", slug, exc)
            save_path.unlink(missing_ok=True)
            return None

        if validate_face and not _has_detectable_face(save_path):
            log.warning(
                "fetch_portrait: image for %r has no detectable face — discarding. "
                "Pre-download with: python scripts/eval3_rag/download_refdb.py --slugs %s",
                slug, slug,
            )
            save_path.unlink(missing_ok=True)
            return None

        # Aspect-ratio check (inference path): Wikipedia lead images are usually
        # portrait-oriented; reject clear landscape images (H < 0.7*W).
        if not validate_face and not _is_portrait_aspect(str(save_path), min_ratio=0.7):
            log.warning(
                "fetch_portrait: %r lead image has landscape aspect ratio — discarding. "
                "Pre-download with: python scripts/eval3_rag/download_refdb.py --slugs %s",
                slug, slug,
            )
            save_path.unlink(missing_ok=True)
            return None

        log.info("fetch_portrait: saved portrait %s → %s", display_name, save_path)
        if validate_face:
            # Offline validation path: record exact filename in sentinel.
            _write_sentinel(save_dir, save_path.name)
        return str(save_path)

    # ── query Wikipedia REST Summary API ─────────────────────────────────────
    title_raw = display_name.replace(" ", "_")
    title = urllib.parse.quote(title_raw, safe="")
    api_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
    try:
        req = urllib.request.Request(
            api_url,
            headers={"User-Agent": "eval3-rag/1.0 (educational robot-learning)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        log.warning("fetch_portrait: Wikipedia API failed for %r: %s", slug, exc)
        return None

    # ── Step 1: try original lead image ──────────────────────────────────────
    original_url = (data.get("originalimage") or {}).get("source")
    thumbnail_url = (data.get("thumbnail") or {}).get("source")

    if original_url:
        result = _download_and_cache(original_url, "wiki_portrait")
        if result:
            return result

    # ── Step 2: try thumbnail (smaller but same subject) ─────────────────────
    if thumbnail_url and thumbnail_url != original_url:
        log.info("fetch_portrait: original image failed, trying thumbnail for %r", slug)
        result = _download_and_cache(thumbnail_url, "wiki_thumbnail")
        if result:
            return result

    # ── Step 3: search article images (only when validate_face=True) ──────────
    # When face validation is off (deploy mode) we stop here — lead image and
    # thumbnail are sufficient.  The article search only matters for quality
    # filtering (finding a portrait among multiple non-portrait images), which
    # only applies during offline pre-download (download_refdb.py).
    if not validate_face:
        log.warning("fetch_portrait: no image found in Wikipedia lead for %r (%s).",
                    slug, display_name)
        return None

    log.info("fetch_portrait: lead image failed for %r — searching article images …", slug)
    try:
        images_api = (
            "https://en.wikipedia.org/w/api.php"
            f"?action=query&titles={urllib.parse.quote(title_raw, safe='')}"
            "&prop=images&imlimit=20&format=json"
        )
        req = urllib.request.Request(
            images_api,
            headers={"User-Agent": "eval3-rag/1.0 (educational robot-learning)"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            img_data = json.loads(resp.read().decode())

        pages = img_data.get("query", {}).get("pages", {})
        image_titles: list[str] = []
        for page in pages.values():
            for img_entry in page.get("images", []):
                fname = img_entry.get("title", "")
                if fname.lower().endswith((".svg", ".ogg", ".ogv", ".webm", ".pdf")):
                    continue
                lower = fname.lower()
                if any(kw in lower for kw in ("logo", "icon", "flag", "seal", "map", "signature")):
                    continue
                image_titles.append(fname)

        if image_titles:
            titles_param = "|".join(urllib.parse.quote(t, safe="") for t in image_titles[:10])
            info_api = (
                "https://en.wikipedia.org/w/api.php"
                f"?action=query&titles={titles_param}"
                "&prop=imageinfo&iiprop=url&format=json"
            )
            req = urllib.request.Request(
                info_api,
                headers={"User-Agent": "eval3-rag/1.0 (educational robot-learning)"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                info_data = json.loads(resp.read().decode())

            pages2 = info_data.get("query", {}).get("pages", {})
            for idx, page in enumerate(pages2.values()):
                url = (page.get("imageinfo") or [{}])[0].get("url", "")
                if not url:
                    continue
                result = _download_and_cache(url, f"wiki_article_{idx:02d}")
                if result:
                    log.info("fetch_portrait: found portrait in article images for %r", slug)
                    return result
    except Exception as exc:
        log.warning("fetch_portrait: article image search failed for %r: %s", slug, exc)

    log.warning(
        "fetch_portrait: no portrait found for %r (%s). "
        "Place a photo manually in %s/%s/",
        slug, display_name, save_root, slug,
    )
    return None


# ---------------------------------------------------------------------------
# ReferenceImageInjector
# ---------------------------------------------------------------------------

class ReferenceImageInjector:
    """Load and return a reference face image from local dataset directories.

    Works for any person — no pre-registration required.  Given a slug like
    ``"marc_pollefeys"``, searches the local DB directories and, when
    ``auto_fetch=True``, falls back to downloading from Wikipedia.

    **Training** (``random_sample=True``): samples a different image each call.
    **Inference** (``random_sample=False``): always returns the first sorted image.

    The returned tensor is ``(3, H, W)`` float in ``[0, 1]``.
    Picklable: stores only primitive types; images are lazy-loaded per worker.
    """

    def __init__(
        self,
        celeb_slug: str,
        db_dir: str | None = None,
        image_hw: tuple[int, int] = (480, 640),
        seed: int = 0,
        random_sample: bool = True,
        search_dirs: list[str] | None = None,
        auto_fetch: bool = False,
        validate_face: bool = False,
    ):
        self._celeb_slug    = str(celeb_slug)
        self._image_hw      = tuple(image_hw)
        self._seed          = int(seed)
        self._random_sample = bool(random_sample)
        self._auto_fetch    = bool(auto_fetch)
        self._validate_face = bool(validate_face)
        self._rng           = random.Random(seed)

        if search_dirs is not None:
            self._search_dirs: list[str] = [str(d) for d in search_dirs]
        elif db_dir is not None:
            defaults = _default_search_dirs()
            seen = {str(db_dir)}
            self._search_dirs = [str(db_dir)] + [d for d in defaults if d not in seen]
        else:
            self._search_dirs = _default_search_dirs()

        self._image_paths: list[str] | None = None
        self._canonical_tensor: torch.Tensor | None = None
        self._canonical_idx: int | None = None  # best-resolution portrait index for inference

    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._image_paths is not None:
            return
        paths: list[str] = []
        celeb_dir_found: str | None = None
        for db_dir in self._search_dirs:
            celeb_dir = os.path.join(db_dir, self._celeb_slug)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
                paths.extend(glob.glob(os.path.join(celeb_dir, ext)))
            if paths:
                celeb_dir_found = celeb_dir
                break

        if not paths and self._auto_fetch:
            # No local images — fall back to Wikipedia.  validate_face=False at
            # deploy time so no image-recognition runs during inference.
            # validate_face=True only when called from download_refdb.py (pre-demo).
            save_root = self._search_dirs[0] if self._search_dirs else "datasets/celeb_refdb"
            fetched = _fetch_portrait_online(
                self._celeb_slug, save_root, validate_face=self._validate_face
            )
            if fetched:
                paths = [fetched]
                celeb_dir_found = os.path.join(save_root, self._celeb_slug)

        # ── Portrait quality gate (inference-safe, no image recognition) ────────
        # Priority 1: sentinel with filenames — filter to exactly the files that
        #   passed Haar detection offline.  Pure text-file read; zero pixels.
        # Priority 2: legacy boolean sentinel (empty file) — all images in dir
        #   were validated as a batch; use all of them.
        # Priority 3: no sentinel — aspect-ratio fallback (PIL header read only).
        if celeb_dir_found:
            validated = _load_validated_filenames(celeb_dir_found)
            if validated is None:
                # No sentinel — aspect-ratio filter as best-effort guard.
                total = len(paths)
                portrait_paths = [p for p in paths if _is_portrait_aspect(p)]
                if portrait_paths:
                    paths = portrait_paths
                    log.info(
                        "ReferenceImageInjector: %r — no sentinel; aspect-ratio filter "
                        "kept %d/%d image(s).",
                        self._celeb_slug, len(portrait_paths), total,
                    )
                else:
                    log.warning(
                        "ReferenceImageInjector: %r — no sentinel and no portrait-aspect "
                        "images found; using all %d image(s) as-is.",
                        self._celeb_slug, total,
                    )
            elif validated:
                # Sentinel lists specific filenames — keep only those.
                total = len(paths)
                paths = [p for p in paths if os.path.basename(p) in validated]
                log.info(
                    "ReferenceImageInjector: %r — sentinel filter kept %d/%d image(s).",
                    self._celeb_slug, len(paths), total,
                )
                if not paths:
                    log.warning(
                        "ReferenceImageInjector: %r — sentinel listed %d file(s) but "
                        "none found on disk; falling back to all images.",
                        self._celeb_slug, len(validated),
                    )
                    paths = [p for p in glob.glob(
                        os.path.join(celeb_dir_found, "*")) if os.path.isfile(p)]
            # else: validated is an empty set = legacy boolean sentinel; use all paths

        paths.sort()
        if not paths:
            searched = "\n  ".join(self._search_dirs)
            fetch_note = (
                "Auto-fetch attempted (Wikipedia returned no image)."
                if self._auto_fetch
                else "Pass auto_fetch=True or add images manually."
            )
            raise FileNotFoundError(
                f"ReferenceImageInjector: no images found for {self._celeb_slug!r}.\n"
                f"Searched:\n  {searched}\n"
                f"{fetch_note}"
            )
        self._image_paths = paths

    def _load_tensor(self, path: str) -> torch.Tensor:
        from PIL import Image as _PILImage
        h, w = self._image_hw
        # LANCZOS (antialias) for downsampling: much sharper than BILINEAR for
        # high-res portraits (e.g. 2924×3843 → 640×480). Preserves facial detail.
        img = _PILImage.open(path).convert("RGB").resize((w, h), _PILImage.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _best_portrait_path(self) -> str:
        """Return the path of the highest-resolution portrait for canonical inference.

        Fast path (O(1)): if .canonical_portrait was written offline by
        fetch_wiki_portraits.py or download_refdb.py, read the filename from
        that text file and return immediately — no image file is opened.

        Slow path (fallback): scan all validated image paths with PIL header
        reads.  Only used when the canonical file is absent or names a file
        not in the validated set (e.g. first run before offline pre-download).
        """
        if self._canonical_idx is None:
            # ── Fast path: pre-computed canonical file ───────────────────────
            for db_dir in self._search_dirs:
                canon_file = os.path.join(db_dir, self._celeb_slug, _CANONICAL_FILENAME)
                if os.path.isfile(canon_file):
                    try:
                        name = open(canon_file).read().strip()
                        for i, p in enumerate(self._image_paths):
                            if os.path.basename(p) == name:
                                self._canonical_idx = i
                                break
                    except Exception:
                        pass
                    if self._canonical_idx is not None:
                        break

            # ── Slow path: PIL header scan (offline fallback) ────────────────
            if self._canonical_idx is None:
                from PIL import Image as _PILImage
                best_idx, best_px = 0, 0
                for i, p in enumerate(self._image_paths):
                    try:
                        with _PILImage.open(p) as im:
                            px = im.width * im.height
                        if px > best_px:
                            best_px, best_idx = px, i
                    except Exception:
                        pass
                self._canonical_idx = best_idx
                # Persist result so future restarts use the fast path instead.
                try:
                    slug_dir = os.path.dirname(self._image_paths[best_idx])
                    _write_canonical_portrait(slug_dir, self._image_paths)
                except Exception:
                    pass

        return self._image_paths[self._canonical_idx]

    # ------------------------------------------------------------------

    def __call__(self) -> torch.Tensor:
        """Return a reference face as a ``(3, H, W)`` float tensor in ``[0, 1]``."""
        self._ensure_loaded()
        if not self._random_sample:
            if self._canonical_tensor is None:
                self._canonical_tensor = self._load_tensor(self._best_portrait_path())
            return self._canonical_tensor
        path = self._image_paths[self._rng.randrange(len(self._image_paths))]
        return self._load_tensor(path)

    def load_numpy(self) -> np.ndarray:
        """Return the canonical reference as a ``(H, W, 3)`` uint8 numpy array."""
        self._ensure_loaded()
        from PIL import Image as _PILImage
        h, w = self._image_hw
        img = _PILImage.open(self._best_portrait_path()).convert("RGB").resize((w, h), _PILImage.LANCZOS)
        return np.array(img, dtype=np.uint8)

    @property
    def n_images(self) -> int:
        self._ensure_loaded()
        return len(self._image_paths)

    def __reduce__(self):
        return (
            self.__class__,
            (self._celeb_slug, None, self._image_hw, self._seed, self._random_sample,
             self._search_dirs, self._auto_fetch, self._validate_face),
        )

    def __repr__(self) -> str:
        return (
            f"ReferenceImageInjector(slug={self._celeb_slug!r}, "
            f"search_dirs={self._search_dirs!r}, hw={self._image_hw}, "
            f"random={self._random_sample}, auto_fetch={self._auto_fetch}, "
            f"validate_face={self._validate_face})"
        )
