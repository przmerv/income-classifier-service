import json
import logging
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """Turns each log message into one line of JSON.

    Every line gets the time, level, message, and the current request ID.
    Extra details passed as extra={"fields": {...}} are added as their own
    keys, so log tools can filter and search on them.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Build the JSON line for a single log message."""
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict):
            for key, value in fields.items():
                entry[key] = value
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def setup_logging(level: str) -> None:
    """Send all log messages out as JSON lines at the given level.

    Replaces any existing log setup so every message in the app, from any
    file, comes out in the same JSON format. Call once at startup.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
