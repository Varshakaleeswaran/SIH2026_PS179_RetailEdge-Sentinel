"""
RetailEdge Sentinel - Local SQLite Database Module
Stores aggregate operational metrics and system events.
Guarantees NO biometric or personal identifying information is persisted.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "data/retail_edge.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initialize database tables for events and metrics snapshots."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Events table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        event_type TEXT NOT NULL,
                        zone TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details_json TEXT
                    )
                """)
                # Periodic metrics snapshot
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS metrics_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        footfall INTEGER,
                        occupancy INTEGER,
                        queue_length INTEGER,
                        risk_score REAL,
                        risk_level TEXT,
                        store_state TEXT,
                        fps REAL
                    )
                """)
                conn.commit()
                logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing SQLite database: {e}")

    def log_event(self, event_type: str, zone: str, severity: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Log a high-level operational event (e.g. QUEUE_BUILDUP, SHELF_LOW)."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            details_json = json.dumps(details or {})
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO events (timestamp, event_type, zone, severity, message, details_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (timestamp, event_type, zone, severity, message, details_json))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log event to database: {e}")

    def log_metrics_snapshot(self, metrics: Dict[str, Any]):
        """Store periodic aggregate metrics snapshot."""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO metrics_snapshots (timestamp, footfall, occupancy, queue_length, risk_score, risk_level, store_state, fps)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    metrics.get("footfall", 0),
                    metrics.get("occupancy", 0),
                    metrics.get("queue_length", 0),
                    metrics.get("risk_score", 0.0),
                    metrics.get("risk_level", "NORMAL"),
                    metrics.get("store_state", "NORMAL"),
                    metrics.get("fps", 0.0)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log metrics snapshot: {e}")

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve recent events for the dashboard."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, event_type, zone, severity, message, details_json
                    FROM events
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                events = []
                for row in rows:
                    events.append({
                        "timestamp": row["timestamp"],
                        "event_type": row["event_type"],
                        "zone": row["zone"],
                        "severity": row["severity"],
                        "message": row["message"],
                        "details": json.loads(row["details_json"] or "{}")
                    })
                return events
        except Exception as e:
            logger.error(f"Error fetching recent events: {e}")
            return []

    def get_metrics_history(self, limit: int = 40) -> List[Dict[str, Any]]:
        """Retrieve historical metric snapshots for trend charts."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, footfall, occupancy, queue_length, risk_score, risk_level, store_state, fps
                    FROM metrics_snapshots
                    ORDER BY id DESC
                    LIMIT ?
                """, (limit,))
                rows = cursor.fetchall()
                data = [dict(row) for row in rows]
                data.reverse() # chronological order
                return data
        except Exception as e:
            logger.error(f"Error fetching metrics history: {e}")
            return []
