"""
utils/logger.py
Lightweight system logger with console output and optional file logging.
"""

import datetime
import os


class SystemLogger:
    LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "system.log")

    def _write(self, level: str, message: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level:7s}] {message}"
        print(line)
        try:
            os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)
            with open(self.LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def info(self, msg):    self._write("INFO",    msg)
    def warning(self, msg): self._write("WARNING", msg)
    def error(self, msg):   self._write("ERROR",   msg)
    def debug(self, msg):   self._write("DEBUG",   msg)
