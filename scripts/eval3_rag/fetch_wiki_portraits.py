#!/usr/bin/env python3
"""Fetch multiple high-resolution Wikipedia portrait images per celebrity.

Queries each celebrity's Wikipedia article for all images, filters by aspect
ratio and filename heuristics, validates with OpenCV face detection, and saves
up to --max-per-celeb images into datasets/celeb_refdb/<slug>/.

Uses Wikipedia's thumbnail API (iiurlwidth=800) to avoid CDN rate limits on
full-resolution downloads.

Face quality gate: rejects images where no single face is prominent (face area
< 1.5% of image) or where multiple prominent faces appear (group shots).

Existing images are never overwritten; already-satisfied celebrities are
skipped.  Run build_rag_pool_json.py after this script.

Usage:
    python scripts/eval3_rag/fetch_wiki_portraits.py \\
        --slugs cristiano_ronaldo lionel_messi elon_musk rihanna \\
                leonardo_dicaprio tom_cruise dwayne_johnson robert_downey_jr \\
                scarlett_johansson jennifer_lawrence marc_pollefeys \\
        --max-per-celeb 10

    # dry-run (no downloads):
    python scripts/eval3_rag/fetch_wiki_portraits.py --slugs elon_musk --dry-run
"""
from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_ROOT       = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DB = _ROOT / "datasets" / "celeb_refdb"

# Filename fragments that indicate non-portrait images.
# Checked on the lower-cased file title (spaces replaced with underscores).
_SKIP_FRAGMENTS = frozenset([
    # Generic non-photo
    "logo", "flag", "signature", "map", "icon", "crest", "shield",
    "coat_of_arms", "coat-of-arms", "badge", "jersey", "shirt", "kit",
    "stadium", "pitch", "field", "arena", "ground", "trophy", "award",
    "album", "cover", "poster", "chart", "graph", "diagram", "symbol",
    "banner", "commons-logo", "wikidata", "wikipedia", "wikimedia",
    "edit-clear", "question_mark", "disambig",
    # Wax figures / statues / non-photo artwork
    "figure", "tussaud", "statue", "wax_",
    "street_art", "_art_of_", "mural", "graffiti", "caricature",
    "doodle", "painting_", "illustration", "cartoon_", "monument",
    "sculpture", "bust_", "drawing_",
    # Sports events / group shots
    "_vs_", "_vs.", "-vs-", "world_cup", "_cup_", "league", "_final_",
    "championship", "tournament", "match", "fans_",
    # Explicit multi-person filenames
    "_and_", "_with_",
    # Awards / ceremonies / public appearances
    "premiere", "ceremony", "gala", "festival", "awards", "award_show",
    "photocall", "red_carpet", "carpets",
    # Press / media events
    "interview", "press_", "_press", "conference", "summit", "forum",
    # Speeches / performances / rallies / tours
    "speech", "address_", "podium", "concert", "performance_",
    "rally_", "protest", "tour_", "_tour_", "_tour.",
    # Full-body / action shots
    "full_body", "full-body", "playing", "training_", "practice_",
    "shooting_", "filming",
    # Visit / event context
    "visit_", "_visit", "appearance", "arrival",
])

_IMG_EXTS = frozenset([".jpg", ".jpeg", ".png", ".webp"])

# Manual title overrides for slugs whose .title() doesn't match Wikipedia
_TITLE_OVERRIDES: dict[str, str] = {
    "robert_downey_jr": "Robert Downey Jr.",
    "marc_pollefeys":   "Marc Pollefeys",
}

# Thumbnail width to request.  Wikipedia's thumbnail CDN is far less
# rate-limited than full-resolution direct downloads.
_THUMB_WIDTH = 800

# Wikimedia Commons API (supplemental portrait source for people with few
# Wikipedia-article images — especially academics and researchers).
_COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Minimum fraction of image area a face must cover.
# 8 % rejects distant figures and full-body shots where the face is small.
_MIN_FACE_AREA_RATIO = 0.08   # 8 %

# Haar cascade bounding boxes typically underestimate the visual face by
# ~10–15 % (hair/forehead not included).  0.30 requires the box to be at
# least 30 % of image height, which corresponds to roughly a head+shoulder
# crop where the face visually fills ~35–40 % of the frame.
_MIN_FACE_HEIGHT_RATIO = 0.30


# ---------------------------------------------------------------------------
# Wikipedia API helpers
# ---------------------------------------------------------------------------

def _slug_to_title(slug: str) -> str:
    if slug in _TITLE_OVERRIDES:
        return _TITLE_OVERRIDES[slug]
    return slug.replace("_", " ").title()


def _api_get(session, params: dict, *, max_retries: int = 6, base_delay: float = 3.0):
    """GET Wikipedia API with exponential back-off on 429 / transient errors."""
    for attempt in range(max_retries):
        try:
            r = session.get(
                "https://en.wikipedia.org/w/api.php",
                params=params,
                timeout=20,
            )
            if r.status_code == 429:
                wait = base_delay * (2 ** attempt)
                log.warning("API 429 — sleeping %.0fs then retrying (%d/%d)",
                            wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                log.warning("API error (%s) — retrying in %.0fs", exc, wait)
                time.sleep(wait)
            else:
                raise
    return None


def _cdn_get(session, url: str, *, max_retries: int = 6, base_delay: float = 3.0):
    """GET a CDN URL with exponential back-off on 429."""
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=30, stream=True)
            if r.status_code == 429:
                wait = base_delay * (2 ** attempt)
                log.warning("CDN 429 — sleeping %.0fs then retrying (%d/%d)",
                            wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                log.warning("CDN error (%s) — retrying in %.0fs", exc, wait)
                time.sleep(wait)
            else:
                raise
    return None


def _commons_get(session, params: dict, *, max_retries: int = 6, base_delay: float = 3.0):
    """GET Wikimedia Commons API with exponential back-off on 429."""
    for attempt in range(max_retries):
        try:
            r = session.get(_COMMONS_API, params=params, timeout=20)
            if r.status_code == 429:
                wait = base_delay * (2 ** attempt)
                log.warning("Commons 429 — sleeping %.0fs then retrying (%d/%d)",
                            wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except Exception as exc:
            if attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                log.warning("Commons error (%s) — retrying in %.0fs", exc, wait)
                time.sleep(wait)
            else:
                raise
    return None


def _get_article_images(title: str, session) -> list[str]:
    """Return all File: titles listed in the article."""
    try:
        r = _api_get(session, {
            "action": "query", "prop": "images",
            "titles": title, "format": "json",
            "imlimit": "max", "redirects": 1,
        })
        if r is None:
            return []
        pages = r.json().get("query", {}).get("pages", {})
        return [img["title"] for page in pages.values()
                for img in page.get("images", [])]
    except Exception as exc:
        log.warning("article-images fetch failed for %r: %s", title, exc)
        return []


def _get_image_infos(file_titles: list[str], session) -> list[dict]:
    """Batch-fetch thumbnail URL / original size / mime for up to 50 files.

    Requests iiurlwidth=_THUMB_WIDTH so downloads go through Wikipedia's
    thumbnail CDN path (much higher rate limit than full-res files).
    """
    results: list[dict] = []
    for i in range(0, len(file_titles), 50):
        batch = file_titles[i:i + 50]
        try:
            r = _api_get(session, {
                "action": "query", "prop": "imageinfo",
                "titles": "|".join(batch),
                "iiprop": "url|size|mime|thumburl",
                "iiurlwidth": str(_THUMB_WIDTH),
                "format": "json",
            })
            if r is None:
                continue
            for page in r.json().get("query", {}).get("pages", {}).values():
                for ii in page.get("imageinfo", []):
                    ii["_title"] = page.get("title", "")
                    results.append(ii)
        except Exception as exc:
            log.warning("imageinfo batch failed: %s", exc)
        time.sleep(0.5)
    return results


# ---------------------------------------------------------------------------
# Wikimedia Commons helpers (supplemental portrait source)
# ---------------------------------------------------------------------------

def _get_commons_category_files(category: str, session, limit: int = 100) -> list[str]:
    """Return File: titles listed in a Wikimedia Commons category."""
    try:
        r = _commons_get(session, {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "file",
            "cmlimit": str(limit),
            "format": "json",
        })
        if r is None:
            return []
        return [m["title"] for m in r.json().get("query", {}).get("categorymembers", [])]
    except Exception as exc:
        log.debug("commons category %r: %s", category, exc)
        return []


def _search_commons_category(display_name: str, session) -> list[str]:
    """Search Commons for a person category when pattern-based lookup finds nothing.

    Searches the Category namespace and returns files from the first category
    whose name contains the person's surname.
    """
    try:
        r = _commons_get(session, {
            "action": "query",
            "list": "search",
            "srsearch": display_name,
            "srnamespace": "14",
            "srlimit": "5",
            "format": "json",
        })
        if r is None:
            return []
        surname = display_name.split()[-1].lower()
        for result in r.json().get("query", {}).get("search", []):
            cat_title = result["title"]
            if surname in cat_title.lower():
                files = _get_commons_category_files(cat_title, session)
                if files:
                    log.info("commons search: %d files in %r", len(files), cat_title)
                    return files
    except Exception as exc:
        log.debug("commons search failed for %r: %s", display_name, exc)
    return []


def _get_commons_portrait_files(display_name: str, session) -> list[str]:
    """Return File: titles from Wikimedia Commons for *display_name*.

    Tries well-known category name patterns in order, then falls back to a
    Commons category search.  Returns the first non-empty result.
    """
    patterns = [
        f"Category:Photographs of {display_name}",
        f"Category:Photos of {display_name}",
        f"Category:{display_name}",
    ]
    for cat in patterns:
        files = _get_commons_category_files(cat, session)
        if files:
            log.info("commons: %d files in %r", len(files), cat)
            return files
        time.sleep(1.0)

    files = _search_commons_category(display_name, session)
    if files:
        return files

    log.debug("commons: no category found for %r", display_name)
    return []


def _get_commons_image_infos(file_titles: list[str], session) -> list[dict]:
    """Like _get_image_infos but queries the Wikimedia Commons API endpoint."""
    results: list[dict] = []
    for i in range(0, len(file_titles), 50):
        batch = file_titles[i:i + 50]
        try:
            r = _commons_get(session, {
                "action": "query", "prop": "imageinfo",
                "titles": "|".join(batch),
                "iiprop": "url|size|mime|thumburl",
                "iiurlwidth": str(_THUMB_WIDTH),
                "format": "json",
            })
            if r is None:
                continue
            for page in r.json().get("query", {}).get("pages", {}).values():
                for ii in page.get("imageinfo", []):
                    ii["_title"] = page.get("title", "")
                    results.append(ii)
        except Exception as exc:
            log.warning("commons imageinfo batch failed: %s", exc)
        time.sleep(0.5)
    return results


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def _is_portrait_candidate(file_title: str) -> bool:
    # Wikipedia API returns titles with spaces; normalize to underscores so all
    # fragments (which use underscores) match regardless of API representation.
    low = file_title.lower().replace(" ", "_")
    if not any(low.endswith(ext) for ext in _IMG_EXTS):
        return False
    for frag in _SKIP_FRAGMENTS:
        if frag in low:
            return False
    return True


def _filename_names_wrong_person(file_title: str, slug: str) -> bool:
    """True if the filename explicitly names a different person than *slug*.

    Wikipedia person-photo filenames often start with "Firstname_Lastname_by_…"
    or "Firstname_Lastname_YEAR.jpg".  This function extracts the leading run of
    Title-cased alphabetic words (stopping at the first lowercase word, digit,
    or the word "by") and checks whether any of them overlap with the target
    slug's words.

    Returns False (keep) when:
    - No unambiguous name can be extracted (generic filenames, digit-prefixed).
    - 4+ title-cased words at the start → likely an event / location title.
    - At least one word overlaps with the slug → same person.
    """
    # Strip "File:" prefix, extension, and normalise separators to spaces.
    stem = re.sub(r"^File:", "", file_title, flags=re.IGNORECASE)
    stem = re.sub(r"\.[a-zA-Z]+$", "", stem).replace("_", " ").replace("-", " ")

    name_words: list[str] = []
    for word in stem.split():
        # Strip non-alphabetic characters so punctuation like trailing commas
        # (e.g. "Pollefeys," in "ETH-BIB-Pollefeys, Marc ...") doesn't break
        # word parsing early.
        clean = re.sub(r"[^a-zA-Z]", "", word)
        if clean.lower() == "by":
            break
        if not clean or not clean[0].isupper() or len(clean) < 2:
            break
        name_words.append(clean.lower())

    # Exactly 1–2 title-cased words → person name ("Rihanna", "Mariah Carey").
    # 3+ words → likely an event / location title ("American Music Awards",
    # "White House Visit") → don't filter, let face gate handle it.
    if not name_words or len(name_words) > 2:
        return False

    slug_words = set(slug.split("_"))
    # Require the surname (last extracted word) to appear in the slug.
    # Checking only the surname avoids false matches between people who share
    # a common first name (e.g. "Jennifer Lopez" vs slug "jennifer_lawrence").
    return name_words[-1] not in slug_words


def _is_portrait_shape(info: dict) -> bool:
    w, h = info.get("width", 0), info.get("height", 0)
    if w < 150 or h < 150:
        return False
    aspect = w / h if h > 0 else 0
    # Prefer portrait/square orientation; reject wide landscape shots
    # (event/action photos are typically wider than 1.4:1).
    return 0.4 <= aspect <= 1.4


# dhash size — 16 gives 256 bits; calibrated on Royal Society crop pair:
# same source = 86/256 (33.6%), different photos = 112–143/256 (43–55%).
_DHASH_SIZE      = 16
_DHASH_THRESHOLD = 100   # bits; below this → near-duplicate, reject


def _dhash(path: Path) -> "np.ndarray | None":
    """256-bit difference hash — pure cv2/numpy, no external library."""
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(path))
        if img is None:
            return None
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (_DHASH_SIZE + 1, _DHASH_SIZE),
                           interpolation=cv2.INTER_AREA)
        return (small[:, 1:] > small[:, :-1]).flatten()
    except Exception:
        return None


def _is_near_duplicate(path: Path, existing_paths: list[Path]) -> bool:
    """True if *path* is a near-duplicate of any image in *existing_paths*.

    Uses difference hash (dhash) — pure cv2/numpy.  Catches different-named
    crops of the same Wikipedia source photo (e.g. _cropped_ vs _crop2_).
    """
    try:
        import numpy as np
    except ImportError:
        return False
    new_h = _dhash(path)
    if new_h is None:
        return False
    for ep in existing_paths:
        existing_h = _dhash(ep)
        if existing_h is None:
            continue
        if int(np.sum(new_h != existing_h)) <= _DHASH_THRESHOLD:
            return True
    return False


def _is_solo_portrait_face(path: Path) -> bool:
    """Return True only if the image has exactly one prominent, close-up face.

    Uses only OpenCV Haar cascade (offline only — never called at inference).
    At inference time portrait quality is guaranteed by the .portrait_validated
    sentinel written by fetch_for_slug after this function passes.

    Gates applied:
      - Exactly one detection with area >= _MIN_FACE_AREA_RATIO (8 %) —
        rejects distant figures and group shots.
      - Face height >= _MIN_FACE_HEIGHT_RATIO (30 %) of image height —
        rejects half-body and full-body shots.  Haar boxes underestimate by
        ~10–15 %, so 30 % here ≈ 35–40 % visual face fill.
      - Face centre within 15–85 % of image width (not extreme edge).
      - Face centre in the upper 80 % of image height.

    Falls back to True only when cv2 is not installed.
    Returns False on any other failure so bad images are rejected.
    """
    try:
        import cv2
    except ImportError:
        return True   # cv2 unavailable — skip gate

    try:
        img = cv2.imread(str(path))
        if img is None:
            return False
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clf  = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4,
                                     minSize=(30, 30))
        if len(faces) == 0:
            log.debug("%s: no frontal face detected", path.name)
            return False

        prominent = [(x, y, fw, fh) for (x, y, fw, fh) in faces
                     if fw * fh / img_area >= _MIN_FACE_AREA_RATIO]

        if len(prominent) == 0:
            log.debug("%s: face too small (area gate) — distant/full-body shot", path.name)
            return False
        if len(prominent) > 1:
            log.debug("%s: %d prominent faces — group shot", path.name, len(prominent))
            return False

        x, y, fw, fh = prominent[0]
        face_cx = (x + fw / 2) / img_w
        face_cy = (y + fh / 2) / img_h

        if fh / img_h < _MIN_FACE_HEIGHT_RATIO:
            log.debug("%s: half-body shot (fh/img_h=%.2f < %.2f)",
                      path.name, fh / img_h, _MIN_FACE_HEIGHT_RATIO)
            return False
        if not (0.15 <= face_cx <= 0.85):
            log.debug("%s: face off-centre horizontally (cx=%.2f)", path.name, face_cx)
            return False
        if face_cy > 0.80:
            log.debug("%s: face too low in frame (cy=%.2f)", path.name, face_cy)
            return False

        return True
    except Exception as exc:
        log.debug("%s: face-check error — %s, rejecting", path.name, exc)
        return False


# ---------------------------------------------------------------------------
# Per-celebrity fetcher
# ---------------------------------------------------------------------------

def fetch_for_slug(
    slug: str,
    db_root: Path,
    max_per_celeb: int,
    session,
    dry_run: bool = False,
) -> int:
    celeb_dir = db_root / slug
    celeb_dir.mkdir(parents=True, exist_ok=True)

    existing = [p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp")
                for p in celeb_dir.glob(ext)]
    if len(existing) >= max_per_celeb:
        log.info("%s: already has %d/%d images — skipping",
                 slug, len(existing), max_per_celeb)
        return 0

    # Build set of safe-name suffixes already on disk to detect filename duplicates
    existing_safe_names = {p.name.split("_", 2)[-1] for p in existing
                           if p.name.startswith("wiki_multi_")}
    # Track actual saved paths (existing + new) for dhash near-duplicate detection
    saved_portrait_paths: list[Path] = list(existing)

    need  = max_per_celeb - len(existing)
    title = _slug_to_title(slug)
    log.info("%s: %d existing, need %d more (article: %r)",
             slug, len(existing), need, title)

    # Step 1: list images from the Wikipedia article
    wiki_files = _get_article_images(title, session)
    time.sleep(1.0)

    wiki_pre_cands = [f for f in wiki_files
                      if _is_portrait_candidate(f)
                      and not _filename_names_wrong_person(f, slug)]

    # Step 2: supplement from Wikimedia Commons only when Wikipedia doesn't have
    # enough portrait candidates to satisfy `need`.  Commons carries a higher
    # rate-limit cost, so we skip it when Wikipedia is already sufficient.
    if len(wiki_pre_cands) < need:
        log.info("%s: only %d Wikipedia candidates for %d needed — checking Commons",
                 slug, len(wiki_pre_cands), need)
        commons_files = _get_commons_portrait_files(title, session)
        time.sleep(1.0)
    else:
        log.info("%s: %d Wikipedia candidates — skipping Commons lookup", slug, len(wiki_pre_cands))
        commons_files = []

    # Merge: Wikipedia-article files first, then Commons-only additions
    wiki_file_set = set(wiki_files)
    commons_only  = [f for f in commons_files if f not in wiki_file_set]
    if commons_only:
        log.info("%s: +%d Commons-only files added", slug, len(commons_only))

    candidates = wiki_pre_cands + [f for f in commons_only
                                   if _is_portrait_candidate(f)
                                   and not _filename_names_wrong_person(f, slug)]
    if not candidates:
        log.warning("%s: no portrait candidates found in Wikipedia article or Commons", slug)
        return 0

    # Step 3: get image metadata — Wikipedia API for article files, Commons API
    # for files only found in the Commons category
    wiki_cands    = [f for f in candidates if f in wiki_file_set]
    commons_cands = [f for f in candidates if f not in wiki_file_set]
    infos = _get_image_infos(wiki_cands, session) if wiki_cands else []
    if commons_cands:
        infos += _get_commons_image_infos(commons_cands, session)
    time.sleep(1.0)

    def _download_url(info: dict) -> str:
        return info.get("thumburl") or info.get("url", "")

    # Filter + sort by original resolution (largest first)
    good = sorted(
        [i for i in infos if _is_portrait_shape(i) and _download_url(i) and
         (i.get("mime", "") in ("image/jpeg", "image/png", "image/webp", ""))],
        key=lambda x: x.get("width", 0) * x.get("height", 0),
        reverse=True,
    )

    if not good:
        log.warning("%s: no images passed shape filter", slug)
        return 0

    log.info("%s: %d portrait candidates after shape filter", slug, len(good))

    if dry_run:
        for info in good[:need]:
            dl = _download_url(info)
            print(f"  [dry-run] {slug}: would download {dl.split('/')[-1]} "
                  f"(orig {info.get('width')}×{info.get('height')})")
        return 0

    # Step 3: download + face-quality gate
    saved = 0
    for info in good:
        if saved >= need:
            break

        dl_url   = _download_url(info)
        raw_name = urllib.parse.unquote(info.get("url", dl_url).split("/")[-1])
        safe     = re.sub(r"[^\w.-]", "_", raw_name)

        # Skip if same source image already on disk (dedup across runs)
        if safe in existing_safe_names:
            log.debug("%s: %s already on disk — skipping duplicate", slug, safe)
            continue

        dest = celeb_dir / f"wiki_multi_{len(existing)+saved+1:02d}_{safe}"

        try:
            resp = _cdn_get(session, dl_url)
            if resp is None:
                continue
            dest.write_bytes(resp.content)
            w, h = info.get("width", 0), info.get("height", 0)
            if not _is_solo_portrait_face(dest):
                dest.unlink()
                log.info("%s: ✗ %s  — no solo face, discarded", slug, safe)
                continue
            if _is_near_duplicate(dest, saved_portrait_paths):
                dest.unlink()
                log.info("%s: ✗ %s  — near-duplicate of existing image, discarded", slug, safe)
                continue
            log.info("%s: ✓ %s  (orig %d×%d)", slug, dest.name, w, h)
            existing_safe_names.add(safe)
            saved_portrait_paths.append(dest)
            saved += 1
            # Record this specific filename as Haar-validated so inference loads
            # only validated files (pure text read at inference, no pixel ops).
            try:
                from eval3_rag.reference_injector import _write_sentinel
                _write_sentinel(celeb_dir, dest.name)
            except Exception:
                pass
        except Exception as exc:
            log.warning("%s: download failed %s: %s", slug, dl_url, exc)
            if dest.exists():
                dest.unlink()
        time.sleep(random.uniform(2.5, 4.5))

    # Pre-compute highest-resolution portrait so _best_portrait_path() is
    # instant at inference (text file read, no PIL scan).
    try:
        from eval3_rag.reference_injector import _write_canonical_portrait
        all_valid = [p for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp")
                     for p in celeb_dir.glob(ext)]
        _write_canonical_portrait(celeb_dir, all_valid)
    except Exception:
        pass

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--slugs", nargs="+", required=True,
                    help="Space-separated celebrity slugs, e.g. elon_musk cristiano_ronaldo")
    ap.add_argument("--max-per-celeb", type=int, default=10,
                    help="Target total images per celebrity including existing ones (default 10)")
    ap.add_argument("--db-root", type=Path, default=_DEFAULT_DB,
                    help=f"celeb_refdb directory (default: {_DEFAULT_DB})")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel fetch workers (default 1; Wikipedia rate-limits aggressively)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be downloaded without saving anything")
    args = ap.parse_args()

    try:
        import requests
    except ImportError:
        sys.exit("requests not installed — run: pip install requests")

    session = requests.Session()
    session.headers["User-Agent"] = (
        "RobotLearningVLA/1.0 (academic robot learning research; "
        "eval3-rag portrait fetch; contact: research)"
    )

    print(f">> Wikipedia multi-portrait fetch (thumbnail={_THUMB_WIDTH}px, "
          f"min-face-area={_MIN_FACE_AREA_RATIO:.1%})")
    print(f"   slugs         : {args.slugs}")
    print(f"   max-per-celeb : {args.max_per_celeb}")
    print(f"   db-root       : {args.db_root}")
    print(f"   dry-run       : {args.dry_run}")
    print()

    total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {
            pool.submit(fetch_for_slug, slug, args.db_root,
                        args.max_per_celeb, session, args.dry_run): slug
            for slug in args.slugs
        }
        for fut in as_completed(futs):
            slug = futs[fut]
            try:
                n = fut.result()
                print(f"  {slug:<30s}  +{n} new images")
                total += n
            except Exception as exc:
                print(f"  {slug:<30s}  ERROR: {exc}")

    print(f"\nTotal new images saved: {total}")
    if not args.dry_run and total > 0:
        print("Next step:")
        print("  python scripts/eval3_rag/build_rag_pool_json.py --out-json datasets/rag_pool.json")


if __name__ == "__main__":
    main()
