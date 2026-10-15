"""SQLite repository; each instance owns a connection in its calling thread."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from . import config
from .models import Detection, DetectionResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS damage_classes (
 id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, display_name_th TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS images (
 id INTEGER PRIMARY KEY, original_name TEXT NOT NULL, source_path TEXT NOT NULL,
 width INTEGER NOT NULL CHECK(width>0), height INTEGER NOT NULL CHECK(height>0)
);
CREATE TABLE IF NOT EXISTS detection_runs (
 id INTEGER PRIMARY KEY, event_id TEXT NOT NULL UNIQUE,
 image_id INTEGER NOT NULL REFERENCES images(id), detected_at TEXT NOT NULL,
 model_id TEXT NOT NULL, model_sha256 TEXT NOT NULL,
 conf_threshold REAL NOT NULL CHECK(conf_threshold BETWEEN 0 AND 1),
 iou_threshold REAL NOT NULL CHECK(iou_threshold BETWEEN 0 AND 1),
 imgsz INTEGER NOT NULL CHECK(imgsz>0), inference_ms REAL NOT NULL CHECK(inference_ms>=0),
 total_ms REAL NOT NULL CHECK(total_ms>=0), result_path TEXT NOT NULL UNIQUE,
 detection_count INTEGER NOT NULL CHECK(detection_count>0),
 max_confidence REAL NOT NULL CHECK(max_confidence BETWEEN 0 AND 1)
);
CREATE TABLE IF NOT EXISTS detections (
 id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES detection_runs(id) ON DELETE CASCADE,
 class_id INTEGER NOT NULL REFERENCES damage_classes(id), class_name TEXT NOT NULL,
 confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
 x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
 CHECK(x1>=0 AND y1>=0 AND x2>x1 AND y2>y1)
);
CREATE INDEX IF NOT EXISTS idx_runs_time ON detection_runs(detected_at);
CREATE INDEX IF NOT EXISTS idx_detection_run ON detections(run_id);
CREATE INDEX IF NOT EXISTS idx_detection_class ON detections(class_id);
PRAGMA user_version=1;
"""


class Repository:
    def __init__(self, db_path: Path = config.DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=10)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        with self._conn:
            self._conn.executemany("INSERT OR IGNORE INTO damage_classes VALUES (?, ?, ?)",
                                   [(i, info[0], info[1]) for i, info in config.DAMAGE_CLASSES.items()])
        actual = self._conn.execute("SELECT id,name FROM damage_classes ORDER BY id").fetchall()
        if [(r[0], r[1]) for r in actual] != list(enumerate(config.CLASS_NAMES)):
            self.close()
            raise ValueError("Database class mapping does not match PCB classes")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._conn.close()

    def event(self, event_id: str):
        return self._conn.execute("SELECT * FROM detection_runs WHERE event_id=?", (event_id,)).fetchone()

    def save_event(self, result: DetectionResult, result_path: Path) -> int:
        if result.is_mock or not result.detections:
            raise ValueError("Only real detections may be stored")
        result.validate()
        existing = self.event(result.event_id)
        if existing:
            if existing["result_path"] != str(result_path):
                raise ValueError("Event ID already refers to a different file")
            return existing["id"]
        with self._conn:
            image_id = self._conn.execute(
                "INSERT INTO images(original_name,source_path,width,height) VALUES(?,?,?,?)",
                (result.request.source_path.name, str(result.request.source_path), *result.original.size),
            ).lastrowid
            cursor = self._conn.execute(
                """INSERT INTO detection_runs(event_id,image_id,detected_at,model_id,model_sha256,
                conf_threshold,iou_threshold,imgsz,inference_ms,total_ms,result_path,
                detection_count,max_confidence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (result.event_id, image_id, result.detected_at, result.model_id, result.model_sha256,
                 result.request.conf, result.request.iou, result.request.imgsz, result.inference_ms,
                 result.total_ms, str(result_path), len(result.detections),
                 max(d.confidence for d in result.detections)),
            )
            run_id = cursor.lastrowid
            self._conn.executemany(
                "INSERT INTO detections(run_id,class_id,class_name,confidence,x1,y1,x2,y2) VALUES(?,?,?,?,?,?,?,?)",
                [(run_id, d.class_id, d.class_name, d.confidence, d.x1, d.y1, d.x2, d.y2)
                 for d in result.detections],
            )
        return run_id

    def history(self, keyword="", class_id=None, date_from=None, date_to=None):
        clauses, values = [], []
        if keyword:
            clauses.append("i.original_name LIKE ?")
            values.append(f"%{keyword}%")
        if class_id is not None:
            clauses.append("EXISTS(SELECT 1 FROM detections dd WHERE dd.run_id=r.id AND dd.class_id=?)")
            values.append(class_id)
        # Compare the recorded local calendar date; SQLite date() would convert to UTC.
        if date_from:
            clauses.append("substr(r.detected_at,1,10)>=?")
            values.append(date_from)
        if date_to:
            clauses.append("substr(r.detected_at,1,10)<=?")
            values.append(date_to)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        rows = self._conn.execute(
            """SELECT r.*, i.original_name, i.source_path, i.width, i.height,
            (SELECT group_concat(DISTINCT d.class_name) FROM detections d WHERE d.run_id=r.id) AS classes
            FROM detection_runs r JOIN images i ON i.id=r.image_id""" + where +
            " ORDER BY r.detected_at DESC,r.id DESC", values,
        ).fetchall()
        return [dict(row) for row in rows]

    def detections_of(self, run_id: int) -> tuple[Detection, ...]:
        rows = self._conn.execute(
            "SELECT class_id,confidence,x1,y1,x2,y2 FROM detections WHERE run_id=? ORDER BY confidence DESC",
            (run_id,),
        ).fetchall()
        return tuple(Detection(*row) for row in rows)

    def referenced_paths(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT result_path FROM detection_runs")}
