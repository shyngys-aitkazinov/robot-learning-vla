#!/usr/bin/env python3
"""Build the algvr.com conference dataset.

Downloads 1-4 portrait photos per organizer/speaker listed on
https://algvr.com/conference/ and emits a JSON manifest that mirrors
``datasets/out-distribution-eval-3.json``.

Photo #1 is always the conference-site portrait; extras are sourced from
Wikimedia Commons or the person's lab / institution page (license tracked
in the per-image ``image_sources`` block). No Google Images fallback.

Usage::

    uv run python tools/build_algvr_conference_dataset.py --dry-run
    uv run python tools/build_algvr_conference_dataset.py
    uv run python tools/build_algvr_conference_dataset.py --skip-existing
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Reuse helpers from the PINS builder.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_pins_metadata import (  # noqa: E402
    TOY_HOLDOUT,
    _to_ascii,
    image_stats,
    to_slug,
)

log = logging.getLogger("build_algvr")

USER_AGENT = (
    "algvr-conference-dataset-builder/1.0 (research; chikaphys9@gmail.com)"
)
DOWNLOAD_TIMEOUT_S = 15.0


# -- People table -------------------------------------------------------------
#
# Schema per entry:
#   role:           "organizer" or "speaker"
#   name:           canonical ASCII display name (used in task strings)
#   affiliation:    institution shown on the conference page (None if absent)
#   external_url:   personal / lab page linked from the conference site
#   photos:         list of (url, license_str, author_str). Index 0 is always
#                   the conference-site portrait — it lands as ``<slug>_01.jpg``.
#                   Extras are Wikimedia Commons or institution pages with a
#                   known license.
#
# Names follow algvr.com/conference/ verbatim EXCEPT "Aude Billard" — the
# page spells it "Aude Billiard" (two i's); we use her real spelling.
PEOPLE: list[dict] = [
    # ------ Organizers (6) ------
    {
        "role": "organizer",
        "name": "Marc Pollefeys",
        "affiliation": "ETH Zurich",
        "external_url": "https://inf.ethz.ch/people/person-detail.pollefeys.html",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/09/Marc_Pollefeys.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://commons.wikimedia.org/wiki/Special:FilePath/ETH-BIB-Pollefeys,_Marc_(1971-)-Portr_19969.jpg",
             "CC BY-SA 4.0",
             "ETH-Bibliothek Zürich, Bildarchiv (via Wikimedia Commons)"),
        ],
    },
    {
        "role": "organizer",
        "name": "Jeannette Bohg",
        "affiliation": "Stanford University",
        "external_url": "https://web.stanford.edu/~bohg/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/09/Jeannette_Bohg.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://web.stanford.edu/~bohg/img/portrait_square.png",
             "personal/lab page portrait",
             "Jeannette Bohg (Stanford personal page)"),
        ],
    },
    {
        "role": "organizer",
        "name": "Xi Wang",
        "affiliation": None,
        "external_url": "https://xiwang1212.github.io/homepage/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/08/Xi_Wang.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://xiwang1212.github.io/homepage/imgs/xi.jpg",
             "personal/lab page portrait",
             "Xi Wang (personal site)"),
        ],
    },
    {
        "role": "organizer",
        "name": "Alexey Gavryushin",
        "affiliation": None,
        "external_url": "https://algvr.com/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/08/Alexey_Gavryushin_Portrait.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
        ],
    },
    {
        "role": "organizer",
        "name": "Roy Yang",
        "affiliation": None,
        "external_url": "https://royyang0714.github.io",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/04/Roy_Yang.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://royyang0714.github.io/images/roy.jpg",
             "personal/lab page portrait",
             "Roy Yang (personal site)"),
        ],
    },
    {
        "role": "organizer",
        "name": "Ayse Johannes",
        "affiliation": "ETH Zurich",
        "external_url": "https://inf.ethz.ch/de/personen/person-detail.MjUxNTI1.TGlzdC8zMDQsLTg3NDc3NjI0MQ==.html",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/04/Ayse_Johannes.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
        ],
    },

    # ------ Invited speakers (28) ------
    {
        "role": "speaker",
        "name": "Andrea Vedaldi",
        "affiliation": "University of Oxford",
        "external_url": "https://www.robots.ox.ac.uk/~vedaldi/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Andrea_Vedaldi.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://www.robots.ox.ac.uk/~vedaldi/images/vedaldi.jpg",
             "personal/lab page portrait",
             "Andrea Vedaldi (Oxford VGG)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Andy Zeng",
        "affiliation": None,
        "external_url": "https://andyzeng.github.io/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/05/Andy_Zeng.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://andyzeng.github.io/images/profile.jpg",
             "personal/lab page portrait",
             "Andy Zeng (personal site)"),
        ],
    },
    {
        # conference page spells it "Aude Billiard" — that's a typo. Real name has one 'i'.
        "role": "speaker",
        "name": "Aude Billard",
        "affiliation": "EPFL",
        "external_url": "https://people.epfl.ch/aude.billard",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Aude_Billiard.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://commons.wikimedia.org/wiki/Special:FilePath/200903 EPFL Aude Billard Portrait.jpg",
             "CC BY-SA",
             "EPFL Mediacom (via Wikimedia Commons)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Cordelia Schmid",
        "affiliation": "INRIA",
        "external_url": "https://thoth.inrialpes.fr/~schmid/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/12/Cordelia_Schmid.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://thoth.inrialpes.fr/~schmid/images/photoCordelia.gif",
             "personal/lab page portrait",
             "Cordelia Schmid (INRIA Thoth team page)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Danfei Xu",
        "affiliation": "Georgia Tech",
        "external_url": "https://faculty.cc.gatech.edu/~danfei/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/12/Danfei_Xu.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://faculty.cc.gatech.edu/~danfei/profile_2026.jpg",
             "institution page portrait",
             "Danfei Xu (Georgia Tech faculty page)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Dima Damen",
        "affiliation": None,
        "external_url": "https://dimadamen.github.io/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/08/Dima_Damen_equisize.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://dimadamen.github.io/Dima2019_s.jpg",
             "personal/lab page portrait",
             "Dima Damen (personal site)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Edward Johns",
        "affiliation": None,
        "external_url": "http://www.robot-learning.uk/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/05/Edward_Johns.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://static.wixstatic.com/media/20f657_c4a0892ff46d421386482fc25da69df7~mv2_d_1659_2212_s_2.jpg/v1/fill/w_400,h_534,al_c,q_80,enc_avif,quality_auto/edward_johns_2019.jpg",
             "personal/lab page portrait",
             "Edward Johns (Imperial Robot Learning Lab site)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Georgia Chalvatzaki",
        "affiliation": "TU Darmstadt",
        "external_url": "https://www.ias.informatik.tu-darmstadt.de/Team/GeorgiaChalvatzaki",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Georgia_Chalvatzaki.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://www.ias.informatik.tu-darmstadt.de/uploads/Team/GeorgiaChalvatzaki/profile_geo.jpg",
             "institution page portrait",
             "TU Darmstadt IAS lab page"),
        ],
    },
    {
        "role": "speaker",
        "name": "Jakob Engel",
        "affiliation": None,
        "external_url": "https://jakobengel.github.io/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/05/Jakob_Engel.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://jakobengel.github.io/images/JakobEngel.JPG",
             "personal/lab page portrait",
             "Jakob Engel (personal site)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Javier Romero",
        "affiliation": None,
        "external_url": "https://linkedin.com/in/javier-romero-38b87331",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/05/Javier_Romero.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://scholar.googleusercontent.com/citations?view_op=view_photo&user=Wx62iOsAAAAJ&citpid=4",
             "Google Scholar profile photo",
             "Javier Romero (Google Scholar)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Jiajun Wu",
        "affiliation": None,
        "external_url": "https://jiajunwu.com/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Jiajun_Wu.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://jiajunwu.com/images/Jiajun_Wu.jpg",
             "personal/lab page portrait",
             "Jiajun Wu (personal site)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Jitendra Malik",
        "affiliation": "UC Berkeley",
        "external_url": "https://people.eecs.berkeley.edu/~malik/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Jitendra_Malik.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://commons.wikimedia.org/wiki/Special:FilePath/180906-N-PO203-0045 (43874899564).jpg",
             "Public domain (US Navy)",
             "U.S. Navy photo (via Wikimedia Commons)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Jonathan Tremblay",
        "affiliation": "NVIDIA",
        "external_url": "https://research.nvidia.com/person/jonathan-tremblay",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/04/Jonathan_Tremblay.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://www.cs.mcgill.ca/~jtremb59/img/profile.jpg",
             "personal/lab page portrait",
             "Jonathan Tremblay (McGill personal page)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Josef Sivic",
        "affiliation": "ENS Paris",
        "external_url": "https://www.di.ens.fr/~josef/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Josef_Sivic.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://www.di.ens.fr/~josef/images/image2.jpg",
             "institution page portrait",
             "Josef Sivic (ENS DI page)"),
        ],
    },
    {
        # conference page header "Judy Fan" — Stanford profile lists "Judith Ellen Fan" (Judy).
        "role": "speaker",
        "name": "Judy Fan",
        "affiliation": "Stanford University",
        "external_url": "https://psychology.stanford.edu/people/judith-ellen-fan",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/12/Judith_Fan.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://cogtoolslab.github.io/images/people/FanJE_photo.jpg",
             "personal/lab page portrait",
             "Judy Fan (Cognitive Tools Lab page)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Leslie Kaelbling",
        "affiliation": "MIT CSAIL",
        "external_url": "https://www.csail.mit.edu/person/leslie-kaelbling",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/08/Leslie_Kaelbling_equisize.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://people.csail.mit.edu/lpk/lpkLadder.jpg",
             "personal/lab page portrait",
             "Leslie Kaelbling (MIT CSAIL personal page)"),
        ],
    },
    {
        # conference page lists him as "Li Yi" but the photo file is "Eric_Yi.jpg" (English name Eric).
        "role": "speaker",
        "name": "Li Yi",
        "affiliation": "Tsinghua University",
        "external_url": "https://iiis.tsinghua.edu.cn/en/People/Faculty/YiLi.htm",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/12/Eric_Yi.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://iiis.tsinghua.edu.cn/zpyx/yeli.jpg",
             "institution page portrait",
             "Tsinghua IIIS faculty page"),
        ],
    },
    {
        "role": "speaker",
        "name": "Oier Mees",
        "affiliation": None,
        "external_url": "https://www.oiermees.com/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Oier_Mees.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://www.oiermees.com/authors/admin/avatar_huf77bb6f3481ea674f51f312221d30a5e_499601_270x270_fill_q75_lanczos_center.jpg",
             "personal/lab page portrait",
             "Oier Mees (personal site)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Phillip Isola",
        "affiliation": "MIT",
        "external_url": "https://web.mit.edu/phillipi/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Phillip_Isola.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://web.mit.edu/phillipi/www/images/photo_of_me_korea.jpeg",
             "personal/lab page portrait",
             "Phillip Isola (MIT personal page)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Pulkit Agrawal",
        "affiliation": "MIT CSAIL",
        "external_url": "https://people.csail.mit.edu/pulkitag/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Pulkit_Agrawal.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://people.csail.mit.edu/pulkitag/images/pulkit.jpg",
             "institution page portrait",
             "Pulkit Agrawal (MIT CSAIL page)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Robert Katzschmann",
        "affiliation": "ETH Zurich",
        "external_url": "https://mavt.ethz.ch/de/personen/person-detail.katzschmann.html",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/04/Robert_Katzschmann.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://srl.ethz.ch/the-group/_jcr_content/par/fullwidthimage/image.imageformat.1286.1024892551.jpg",
             "institution page portrait",
             "ETH Zurich Soft Robotics Lab page"),
        ],
    },
    {
        "role": "speaker",
        "name": "Shuran Song",
        "affiliation": None,
        "external_url": "https://shurans.github.io/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/08/Shuran_Song_equisize.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://shurans.github.io/images/people/shuran_song.jpg",
             "personal/lab page portrait",
             "Shuran Song (personal site)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Siyu Tang",
        "affiliation": "ETH Zurich",
        "external_url": "https://vlg.inf.ethz.ch/team/Prof-Dr-Siyu-Tang.html",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/05/Siyu_Tang.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://vlg.inf.ethz.ch/assets/img/members/Siyu-Tang.jpg",
             "institution page portrait",
             "ETH Zurich VLG team page"),
        ],
    },
    {
        "role": "speaker",
        "name": "Vladlen Koltun",
        "affiliation": None,
        "external_url": "https://vladlen.info/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Vladlen_Koltun.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("http://vladlen.info/images/vladlen.jpg",
             "personal/lab page portrait",
             "Vladlen Koltun (personal site)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Wenzhen Yuan",
        "affiliation": "University of Illinois",
        "external_url": "https://siebelschool.illinois.edu/about/people/faculty/yuanwz",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/12/Wenzhen_Yuan.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://ws.engr.illinois.edu/directory/viewphoto.aspx?photo=19131&s=300",
             "institution page portrait",
             "UIUC Siebel School faculty directory"),
        ],
    },
    {
        "role": "speaker",
        "name": "Yann LeCun",
        "affiliation": None,
        "external_url": "http://yann.lecun.com/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2026/05/Yann_LeCun_22.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://commons.wikimedia.org/wiki/Special:FilePath/Laura Chaubard & Yann Le Cun - 2024 (53814052697) (cropped).jpg",
             "CC BY-SA 2.0",
             "Jeremy Barande / Ecole polytechnique (via Wikimedia Commons)"),
            ("https://commons.wikimedia.org/wiki/Special:FilePath/Yann LeCun - 2018 (cropped).jpg",
             "CC BY-SA 2.0",
             "Jeremy Barande / Ecole polytechnique (via Wikimedia Commons)"),
            ("https://commons.wikimedia.org/wiki/Special:FilePath/Yann LeCun (29146901108).jpg",
             "CC BY-SA 2.0",
             "Jeremy Barande / Ecole polytechnique (via Wikimedia Commons)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Yuke Zhu",
        "affiliation": None,
        "external_url": "https://yukezhu.me/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/12/Yuke_Zhu.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://www.cs.utexas.edu/~yukez/images/yukezhu.jpg",
             "institution page portrait",
             "Yuke Zhu (UT Austin faculty page)"),
        ],
    },
    {
        "role": "speaker",
        "name": "Yoichi Sato",
        "affiliation": None,
        "external_url": "https://sites.google.com/ut-vision.org/ysato/",
        "photos": [
            ("https://algvr.com/conference/wp-content/uploads/2025/10/Yoichi_Sato.jpg",
             "website portrait (algvr.com)",
             "algvr.com (Vision & Robotics for Embodied AI Conference)"),
            ("https://lh3.googleusercontent.com/sitesv/AA5AbUCIlU8dSN66sp7IewRrJHv9Ugt0GjKucwYzLPNhbDldH4cH2aABV2mSOSoN9ti7eqbnZ4t39UUrjRfO6KI8iOJrI67GI3ya0YX5lGdnNO1uCJBBsOZKYIoJWnoTW3YDB1jVPfg3dpWgMti_ZcKIn2LKPEvalVKLyi5v4sz_NJzBs1KdPl8kKnOTPpiVqKKZ9OUOEa9zTHxhCawTJRoWWpNNgLLmrc_zmy_X=w1280",
             "personal/lab page portrait",
             "Yoichi Sato (UT-Vision Google Sites page)"),
        ],
    },
]


def _ext_for(url: str) -> str:
    """Pick a file extension from the URL. Default .jpg."""
    lower = url.lower().split("?", 1)[0]
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if lower.endswith(ext):
            return ".jpg" if ext == ".jpeg" else ext
    return ".jpg"


def _encode_url(url: str) -> str:
    """Percent-encode unsafe characters in the path component (spaces, parens, etc.).

    Wikimedia Special:FilePath URLs include literal spaces in the filename which
    urllib refuses to send raw. ``quote(..., safe='/:%')`` leaves already-encoded
    sequences alone and the URL structure intact.
    """
    parts = urllib.parse.urlsplit(url)
    encoded_path = urllib.parse.quote(parts.path, safe="/:%")
    return urllib.parse.urlunsplit(parts._replace(path=encoded_path))


def _download(url: str, dest: Path, timeout: float = DOWNLOAD_TIMEOUT_S) -> bool:
    """Download ``url`` to ``dest``. Returns True on success, False on any failure."""
    safe_url = _encode_url(url)
    req = urllib.request.Request(safe_url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            log.warning("empty response: %s", url)
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
        log.warning("download failed: %s (%s)", url, e)
        return False


def build(
    out_dir: Path,
    out_json: Path,
    dry_run: bool = False,
    skip_existing: bool = False,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()

    image_sources: dict[str, dict] = {}
    celebrities: list[dict] = []
    total_images = 0
    total_bytes = 0

    for person in PEOPLE:
        canonical = _to_ascii(person["name"])
        slug = to_slug(canonical)
        person_dir = out_dir / slug
        person_dir.mkdir(parents=True, exist_ok=True)

        n_downloaded = 0
        for idx, (url, lic, author) in enumerate(person["photos"], start=1):
            fname = f"{slug}_{idx:02d}{_ext_for(url)}"
            rel_key = f"{slug}/{fname}"
            dest = person_dir / fname

            if dry_run:
                log.info("[dry-run] %s <- %s", rel_key, url)
                # Still record the source so the JSON is illustrative.
                image_sources[rel_key] = {
                    "source_url": url, "license": lic, "author": author,
                }
                continue

            if skip_existing and dest.exists() and dest.stat().st_size > 0:
                log.info("skip-existing: %s", rel_key)
            else:
                ok = _download(url, dest)
                if not ok:
                    if dest.exists():
                        try:
                            dest.unlink()
                        except OSError:
                            pass
                    continue
                log.info("downloaded: %s (%d bytes)", rel_key, dest.stat().st_size)
            image_sources[rel_key] = {
                "source_url": url, "license": lic, "author": author,
            }
            n_downloaded += 1

        images = sorted(person_dir.glob("*.jpg")) + sorted(person_dir.glob("*.png")) + sorted(person_dir.glob("*.webp"))
        # de-dup while keeping order
        seen, images_unique = set(), []
        for p in images:
            if p.name not in seen:
                seen.add(p.name)
                images_unique.append(p)
        images = images_unique

        size_bytes = sum(p.stat().st_size for p in images) if images else 0
        try:
            rel_dir = person_dir.relative_to(cwd)
        except ValueError:
            rel_dir = person_dir
        entry: dict = {
            "name": canonical,
            "slug": slug,
            "dir": str(rel_dir),
            "n_images": len(images),
            "total_bytes": size_bytes,
            "held_out": canonical in TOY_HOLDOUT,
        }
        stats = image_stats(images) if images and not dry_run else None
        if stats is not None:
            entry.update(stats)
        else:
            entry["aspect_counts"] = {"portrait": 0, "landscape": 0, "square": 0}
            entry["aspect_portrait_frac"] = 0.0
            entry["long_edge_px"] = {"min": 0, "median": 0, "max": 0}
        entry["rank"] = None
        entry["category"] = person["role"]
        entry["affiliation"] = person["affiliation"]
        entry["external_url"] = person["external_url"]

        celebrities.append(entry)
        total_images += len(images)
        total_bytes += size_bytes

    role_counts: dict[str, int] = {}
    for p in PEOPLE:
        role_counts[p["role"]] = role_counts.get(p["role"], 0) + 1

    payload = {
        "dataset": "algvr.com Conference — Organizers + Invited Speakers",
        "source": (
            "Conference portraits from algvr.com plus Wikimedia Commons / "
            "personal / institution pages linked from the conference site"
        ),
        "source_url": "https://algvr.com/conference/",
        "source_json": None,
        "root": str(out_dir.relative_to(cwd)) if out_dir.is_absolute() and cwd in out_dir.parents else str(out_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_celebrities": len(celebrities),
        "total_images": total_images,
        "total_bytes": total_bytes,
        "category_counts": role_counts,
        "selection_rationale": (
            "Every organizer and invited speaker on https://algvr.com/conference/. "
            "Photo #1 is always the conference-site portrait; additional photos are "
            "Wikimedia Commons portraits or institution-page photos (license + author "
            "recorded per image). Hotel hospitality staff (Sebastiano, Maria Petrillo) "
            "are intentionally excluded because the page provides no photo."
        ),
        "held_out_identities_target": sorted(TOY_HOLDOUT),
        "held_out_identities_present": sorted(
            c["name"] for c in celebrities if c["held_out"]
        ),
        "name_overrides_applied": {
            # conference page spells her name "Aude Billiard" — corrected to her real spelling.
            "aude billiard": "Aude Billard",
        },
        "celebrities": celebrities,
        "image_sources": image_sources,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        log.info("wrote %s (%d celebs, %d images, %d bytes)",
                 out_json, len(celebrities), total_images, total_bytes)
    else:
        log.info("[dry-run] would write %s (%d celebs)", out_json, len(celebrities))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out-dir", type=Path,
                    default=Path("datasets/algvr-conference"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("datasets/algvr-conference.json"))
    ap.add_argument("--dry-run", action="store_true",
                    help="Log every URL without downloading anything.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Don't re-download files already on disk (idempotent re-runs).")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    build(
        out_dir=args.out_dir,
        out_json=args.out_json,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
