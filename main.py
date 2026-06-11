#!/usr/bin/env python3
"""
Real-Time Waste Detection System v4.0
Professional AI-Powered Environmental Monitoring Platform
"""

import sys
import os

# Ensure local packages are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.app import WasteDetectionApp
from core.database import WasteDatabase
from utils.logger import SystemLogger


def main():
    log = SystemLogger()
    log.info("System initializing...")

    try:
        import tkinter as tk
        try:
            import ttkbootstrap as tb
            root = tb.Window(themename="darkly")
            log.info("ttkbootstrap theme loaded: darkly")
        except ImportError:
            root = tk.Tk()
            log.warning("ttkbootstrap not found. Using standard tkinter.")

        app = WasteDetectionApp(root, log)

        def on_close():
            log.info("Shutdown requested.")
            app.shutdown()
            root.destroy()

        root.protocol("WM_DELETE_WINDOW", on_close)
        log.info("Application running.")
        root.mainloop()

    except KeyboardInterrupt:
        log.info("Interrupted by user.")
    except Exception as e:
        log.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("""
    +--------------------------------------------------+
    |   REAL-TIME WASTE DETECTION SYSTEM  v4.0         |
    |   Advanced AI-Powered Environmental Monitor      |
    +--------------------------------------------------+
    """)
    main()
