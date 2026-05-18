#!/usr/bin/env python3
"""Minimal HTTP client for upstream OpenVLA ``vla-scripts/deploy.py`` REST servers.

Expected JSON response keys are deployment-specific — adjust ``--response-action-key``
after inspecting your server implementation.

Example::

    OPENVLA_SERVER=http://127.0.0.1:8777 \\
    integrations/openvla/scripts/http_client.py --image frame.png \\
      --instruction \"Pick up the can\" --unnorm-key bridge_orig
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", type=str, default="", help="Base URL or use OPENVLA_SERVER env")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--instruction", type=str, required=True)
    p.add_argument("--unnorm-key", type=str, default="bridge_orig")
    p.add_argument("--endpoint", type=str, default="/predict", help="POST path appended to server base")
    return p.parse_args()


def main() -> int:
    import os

    args = _parse_args()
    base = (args.server or os.environ.get("OPENVLA_SERVER", "")).rstrip("/")
    if not base:
        print("Set --server or OPENVLA_SERVER", file=sys.stderr)
        return 2

    img_bytes = args.image.read_bytes()
    payload = {
        "instruction": args.instruction,
        "unnorm_key": args.unnorm_key,
        "image_b64": base64.b64encode(img_bytes).decode("ascii"),
        "image_name": args.image.name,
    }
    url = base + args.endpoint
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"), file=sys.stderr)
        return 1

    print(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
