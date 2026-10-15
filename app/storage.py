"""Store event images and compensate if a SQLite transaction fails."""
from datetime import datetime
import logging
from pathlib import Path
import re
from uuid import uuid4

from PIL import Image

from . import config
from .db import Repository
from .models import DetectionResult, SaveResult

LOG = logging.getLogger(__name__)


class EventStorage:
    def __init__(self, db_path=config.DB_PATH, output_dir=config.OUTPUT_DIR):
        self.db_path, self.output_dir = Path(db_path), Path(output_dir)

    def save(self, result: DetectionResult) -> SaveResult:
        if result.is_mock:
            return SaveResult("mock", result.event_id)
        if not result.detections:
            return SaveResult("not_required", result.event_id)
        temporary, target, created = None, None, False
        try:
            result.validate()
            if not re.fullmatch(r"[0-9a-f]{32}", result.event_id):
                raise ValueError("Invalid event ID")
            when = datetime.fromisoformat(result.detected_at)
            if when.utcoffset() is None:
                raise ValueError("Event time must include a timezone")
            folder = self.output_dir / when.strftime("%Y/%m/%d")
            target = folder / f"{when:%Y%m%d_%H%M%S}_{result.event_id}_annotated.png"
            with Repository(self.db_path) as repo:
                old = repo.event(result.event_id)
                if old:
                    if old["result_path"] != str(target) or not target.is_file():
                        raise OSError("Existing event has missing or inconsistent evidence")
                    return SaveResult("saved", result.event_id, str(target))
                folder.mkdir(parents=True, exist_ok=True)
                temporary = folder / f".{result.event_id}.{uuid4().hex}.tmp"
                with temporary.open("xb") as stream:
                    result.annotated.save(stream, format="PNG")
                    stream.flush()
                with Image.open(temporary) as check:
                    check.verify()
                # Reserve exclusively; no retry can overwrite another event's evidence.
                with target.open("xb"):
                    pass
                created = True
                temporary.replace(target)
                repo.save_event(result, target)
            return SaveResult("saved", result.event_id, str(target))
        except Exception as exc:
            LOG.exception("Evidence save failed for %s", result.event_id)
            if created and target is not None:
                try:
                    target.unlink(missing_ok=True)
                except OSError:
                    LOG.exception("Orphan evidence retained: %s", target)
            return SaveResult("failed", result.event_id, error=str(exc))
        finally:
            if temporary:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    LOG.exception("Temporary evidence retained: %s", temporary)

    def consistency_report(self) -> dict:
        with Repository(self.db_path) as repo:
            referenced = repo.referenced_paths()
        files = {str(p) for p in self.output_dir.glob("*/*/*/*_annotated.png")}
        return {"missing": sorted(p for p in referenced if not Path(p).is_file()),
                "orphan": sorted(files - referenced),
                "temporary": sorted(str(p) for p in self.output_dir.glob("*/*/*/.*.tmp"))}
