"""Convert a verified PyTorch UI bundle into a separate FP32 OpenVINO CPU bundle."""
import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile

from app.model_registry import read_metadata, sha256_file, validate_names
from app.runtime import prepare_ultralytics


def export_bundle(weights: Path, output: Path):
    weights, output = Path(weights).resolve(), Path(output).resolve()
    if weights.suffix.lower() != ".pt":
        raise ValueError("Source must be a .pt UI bundle with metadata.json")
    source_metadata = read_metadata(weights)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {output}")
    prepare_ultralytics()
    from ultralytics import YOLO

    output.parent.mkdir(parents=True, exist_ok=True)
    # Export a temporary copy so pretrained/trained source artifacts remain intact.
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".openvino-export-") as temporary:
        source = Path(temporary) / "best.pt"
        shutil.copy2(weights, source)
        model = YOLO(str(source), task="detect")
        validate_names(model.names)
        exported = Path(model.export(format="openvino", device="cpu", imgsz=source_metadata["imgsz"],
                                     batch=1, dynamic=True, quantize=32, nms=False))
        # This Ultralytics version identifies OpenVINO by '_openvino_model' in the filename.
        xml = exported / "best_openvino_model.xml"
        (exported / "best.xml").rename(xml)
        (exported / "best.bin").rename(xml.with_suffix(".bin"))
        files = [xml, xml.with_suffix(".bin"), exported / "metadata.yaml"]
        metadata = dict(source_metadata)
        metadata.update(model_id=source_metadata["model_id"] + "-openvino-fp32",
                        format="openvino", precision="FP32", device="intel:cpu", dynamic=True,
                        created_at=datetime.now().astimezone().isoformat(),
                        source_weights=str(weights), source_sha256=source_metadata["sha256"],
                        sha256=sha256_file(xml), files={p.name: sha256_file(p) for p in files})
        metadata["versions"] = dict(metadata.get("versions", {}),
                                    openvino=importlib.metadata.version("openvino"))
        (exported / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        read_metadata(xml)
        # Same-volume rename publishes the complete bundle without overwriting another bundle.
        exported.rename(output)
    return output / "best_openvino_model.xml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="Bundled .pt with metadata.json alongside")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    print(f"OpenVINO FP32 bundle ready: {export_bundle(args.weights, args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
