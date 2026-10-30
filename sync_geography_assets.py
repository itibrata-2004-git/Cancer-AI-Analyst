"""Download selected, checksum-pinned public boundary assets for deployment."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

MANIFEST = Path(__file__).resolve().parent / "manifest.json"
DEFAULT_URL = "https://raw.githubusercontent.com/Hari31416/india-geojson-assets"


def download(entry, destination: Path, base_url: str):
    target = destination / entry["path"]
    if (
        target.exists()
        and hashlib.sha256(target.read_bytes()).hexdigest() == entry["gzip_sha256"]
    ):
        return
    with urlopen(f"{base_url.rstrip('/')}/{entry['path']}", timeout=60) as response:
        data = response.read(entry["gzip_bytes"] + 1)
    if (
        len(data) != entry["gzip_bytes"]
        or hashlib.sha256(data).hexdigest() != entry["gzip_sha256"]
    ):
        raise ValueError(f"Boundary checksum mismatch: {entry['path']}")
    raw = gzip.decompress(data)
    if len(raw) != entry["bytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError(f"Invalid boundary content: {entry['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset",
        action="append",
        help="Asset ID; repeat to select several. Default: all display assets.",
    )
    parser.add_argument(
        "--original",
        action="store_true",
        help="Also download full-resolution originals.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("GEOGRAPHY_ASSET_DIR") or MANIFEST.parent),
    )
    parser.add_argument(
        "--base-url", help="Override the versioned repository URL, e.g. a local mirror."
    )
    parser.add_argument(
        "--version", help="Download an older bundled manifest version for saved charts."
    )
    args = parser.parse_args()
    manifest = json.loads(
        (MANIFEST.parent / "manifests" / f"{args.version}.json").read_bytes()
        if args.version
        else MANIFEST.read_bytes()
    )
    base_url = args.base_url or f"{DEFAULT_URL}/{manifest['version']}"
    for asset_id in args.asset or manifest["assets"]:
        if asset_id not in manifest["assets"]:
            parser.error(f"Unknown asset: {asset_id}")
        entry = manifest["assets"][asset_id]
        for kind in (["display", "original"] if args.original else ["display"]):
            download(entry[kind], args.output, base_url)
        print(asset_id)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_bytes(MANIFEST.read_bytes())
    (args.output / "manifests").mkdir(exist_ok=True)
    for path in (MANIFEST.parent / "manifests").glob("v*.json"):
        (args.output / "manifests" / path.name).write_bytes(path.read_bytes())


if __name__ == "__main__":
    main()
