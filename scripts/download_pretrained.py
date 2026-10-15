"""Download the pinned official YOLO11n checkpoint, without training."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen

from app.config import BASE_DIR
from app.model_registry import sha256_file

URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
EXPECTED_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=BASE_DIR / "models/pretrained/yolo11n.pt")
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists():
        if sha256_file(target) != EXPECTED_SHA256:
            parser.error(f"Existing file checksum differs; inspect it or choose another --output: {target}")
        print(f"Already downloaded; SHA-256 verified: {target}")
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(URL, headers={"User-Agent": "PCB-YOLO11-setup"})
        digest = hashlib.sha256()
        with urlopen(request, timeout=30) as source, tempfile.NamedTemporaryFile(
                dir=target.parent, prefix="yolo11n_", suffix=".download", delete=False) as stream:
            temporary = Path(stream.name)
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
                stream.write(block)
        if digest.hexdigest() != EXPECTED_SHA256:
            raise ValueError("Downloaded checkpoint SHA-256 does not match the pinned official asset")
        if target.exists():
            raise FileExistsError(f"Output appeared during download; refusing to replace it: {target}")
        temporary.rename(target)
        metadata = {"source": URL, "downloaded_at": datetime.now().astimezone().isoformat(),
                    "file": target.name, "size_bytes": target.stat().st_size,
                    "sha256": EXPECTED_SHA256, "training_started": False}
        target.with_suffix(".source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Downloaded and SHA-256 verified: {target}")
        return 0
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
