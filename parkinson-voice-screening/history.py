"""
history.py — Module Lưu trữ & Quản lý Lịch sử Sàng lọc Bệnh nhân (SQLite).

Chức năng:
  - Lưu mỗi lần sàng lọc vào SQLite database (data/screening_history.db)
  - Truy vấn lịch sử: tìm kiếm, lọc theo ngày, theo kết quả
  - Thống kê tổng quan: tổng số lần sàng lọc, phân bố kết quả
  - Xuất báo cáo CSV/JSON
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = BASE_DIR / "data" / "screening_history.db"

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _get_conn() -> sqlite3.Connection:
    """Tạo kết nối SQLite, tự động tạo DB nếu chưa có."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Tạo bảng lịch sử sàng lọc nếu chưa tồn tại."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS screening_history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
            prediction      INTEGER NOT NULL,
            label           TEXT    NOT NULL,
            probability     REAL    NOT NULL,
            model_name      TEXT    DEFAULT 'svm',
            gender          TEXT,
            gender_conf     REAL,
            emotion         TEXT,
            emotion_conf    REAL,
            pitch_range     TEXT,
            f0_mean         REAL,
            audio_duration  REAL,
            features_json   TEXT,
            notes           TEXT DEFAULT ''
        )
    """)
    conn.commit()
    conn.close()


# Tự khởi tạo DB khi import
init_db()


# ══════════════════════════════════════════════════════════════════════════════
#  INSERT — Lưu kết quả sàng lọc
# ══════════════════════════════════════════════════════════════════════════════

def save_screening(
    prediction: int,
    label: str,
    probability: float,
    model_name: str = "svm",
    gender: Optional[str] = None,
    gender_conf: Optional[float] = None,
    emotion: Optional[str] = None,
    emotion_conf: Optional[float] = None,
    pitch_range: Optional[str] = None,
    f0_mean: Optional[float] = None,
    audio_duration: Optional[float] = None,
    features: Optional[Dict] = None,
    notes: str = "",
) -> int:
    """Lưu 1 kết quả sàng lọc vào database. Trả về ID bản ghi."""
    conn = _get_conn()
    cur = conn.execute("""
        INSERT INTO screening_history 
            (prediction, label, probability, model_name,
             gender, gender_conf, emotion, emotion_conf,
             pitch_range, f0_mean, audio_duration,
             features_json, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        prediction, label, probability, model_name,
        gender, gender_conf, emotion, emotion_conf,
        pitch_range, f0_mean, audio_duration,
        json.dumps(features, ensure_ascii=False) if features else None,
        notes,
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


# ══════════════════════════════════════════════════════════════════════════════
#  QUERY — Truy vấn lịch sử
# ══════════════════════════════════════════════════════════════════════════════

def get_all_screenings(limit: int = 200) -> List[Dict[str, Any]]:
    """Lấy toàn bộ lịch sử sàng lọc, sắp xếp mới nhất trước."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM screening_history ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_screening_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Lấy chi tiết 1 bản ghi theo ID."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM screening_history WHERE id = ?", (record_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def search_screenings(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    label_filter: Optional[str] = None,
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """Tìm kiếm lịch sử sàng lọc theo ngày và/hoặc kết quả."""
    query = "SELECT * FROM screening_history WHERE 1=1"
    params = []

    if date_from:
        query += " AND timestamp >= ?"
        params.append(date_from)
    if date_to:
        query += " AND timestamp <= ?"
        params.append(date_to + " 23:59:59")
    if label_filter and label_filter != "Tất cả":
        query += " AND label = ?"
        params.append(label_filter)

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    conn = _get_conn()
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_screening(record_id: int) -> bool:
    """Xóa 1 bản ghi sàng lọc."""
    conn = _get_conn()
    cur = conn.execute("DELETE FROM screening_history WHERE id = ?", (record_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    conn.close()
    return deleted


def clear_all_history() -> int:
    """Xóa toàn bộ lịch sử. Trả về số bản ghi đã xóa."""
    conn = _get_conn()
    cur = conn.execute("DELETE FROM screening_history")
    conn.commit()
    count = cur.rowcount
    conn.close()
    return count


# ══════════════════════════════════════════════════════════════════════════════
#  STATISTICS — Thống kê tổng quan
# ══════════════════════════════════════════════════════════════════════════════

def get_statistics() -> Dict[str, Any]:
    """Thống kê tổng quan về lịch sử sàng lọc."""
    conn = _get_conn()

    total = conn.execute("SELECT COUNT(*) FROM screening_history").fetchone()[0]
    n_parkinson = conn.execute(
        "SELECT COUNT(*) FROM screening_history WHERE prediction = 1"
    ).fetchone()[0]
    n_healthy = total - n_parkinson

    avg_prob = conn.execute(
        "SELECT AVG(probability) FROM screening_history"
    ).fetchone()[0] or 0.0

    # Phân bố giới tính
    gender_dist = {}
    rows = conn.execute(
        "SELECT gender, COUNT(*) as cnt FROM screening_history WHERE gender IS NOT NULL GROUP BY gender"
    ).fetchall()
    for r in rows:
        gender_dist[r["gender"]] = r["cnt"]

    # 5 lần sàng lọc gần nhất
    recent = conn.execute(
        "SELECT * FROM screening_history ORDER BY id DESC LIMIT 5"
    ).fetchall()

    conn.close()
    return {
        "total": total,
        "n_parkinson": n_parkinson,
        "n_healthy": n_healthy,
        "avg_probability": round(avg_prob, 2),
        "gender_distribution": gender_dist,
        "recent": [dict(r) for r in recent],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT — Xuất dữ liệu
# ══════════════════════════════════════════════════════════════════════════════

def export_csv(output_path: str | Path) -> int:
    """Xuất toàn bộ lịch sử sang file CSV. Trả về số bản ghi."""
    import csv

    records = get_all_screenings(limit=10000)
    if not records:
        return 0

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cols = [k for k in records[0].keys() if k != "features_json"]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for rec in records:
            row = {k: rec[k] for k in cols}
            writer.writerow(row)

    return len(records)
