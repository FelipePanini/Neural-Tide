"""Download the MediaPipe models Neural Tide needs into ``models/``.

Usage::

    python tools/download_model.py              # hand landmarker + selfie segmenter
    python tools/download_model.py --hand-only  # skip the segmenter (mode 7)
    python tools/download_model.py --force      # download again even if present
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CONFIG, ModelFile  # noqa: E402


def download(model: ModelFile, force: bool = False) -> Path:
    """Fetch ``model`` to its configured path (atomic: writes ``.part`` then renames)."""
    dest = model.path
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[ok] {dest.name} já existe ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    digest = hashlib.sha256()
    received = 0
    print(f"[..] baixando {dest.name}")
    with urllib.request.urlopen(model.url, timeout=CONFIG.models.download_timeout_s) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        with partial.open("wb") as out:
            while chunk := resp.read(CONFIG.models.download_chunk_bytes):
                out.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if total:
                    print(f"\r     {received / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    if received == 0:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"download vazio: {model.url}")
    partial.replace(dest)
    print(f"[ok] {dest.name}  sha256={digest.hexdigest()[:16]}…")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hand-only", action="store_true", help="skip the selfie segmenter used by mode 7")
    parser.add_argument("--all", action="store_true", help=argparse.SUPPRESS)  # old flag, now the default
    parser.add_argument("--force", action="store_true", help="download again even if the file exists")
    args = parser.parse_args()

    models = [CONFIG.models.hand_landmarker]
    if not args.hand_only:
        models.append(CONFIG.models.selfie_segmenter)
    try:
        for model in models:
            download(model, force=args.force)
    except OSError as exc:  # URLError, HTTPError and disk errors all derive from OSError
        print(f"[erro] falha no download: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
