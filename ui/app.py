"""
ui/app.py
Main application window.
Layout: left camera feed | right tabbed dashboard
- Tab 1: Live object detail + metrics
- Tab 2: Real-time charts (toxicity timeline, category distribution)
- Tab 3: Session statistics
Controls: pause/resume, sound, radiation toggle, report, history, export CSV
Performance: frame resize only when canvas dimension changes; chart redraws throttled.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import datetime
import time
import math
import uuid
import csv
import os
from collections import deque, Counter

import cv2
import numpy as np
from PIL import Image, ImageTk

from core.database import WasteDatabase
from core.detector import WasteDetector
from core.renderer import draw_detection, draw_hud, _color

try:
    import ttkbootstrap as tb
    MODERN = True
except ImportError:
    MODERN = False

try:
    import winsound
    _SOUND = True
except ImportError:
    _SOUND = False

# Colors
BG       = "#0f1117"
BG2      = "#181b24"
BG3      = "#1e2230"
ACCENT   = "#00c896"
ACCENT2  = "#4a9eff"
DANGER   = "#ff4444"
WARN     = "#ffaa00"
FG       = "#d0d8e8"
FG_DIM   = "#6b7a96"
MONO     = "Consolas"
UI_FONT  = "Segoe UI"

CHART_HISTORY = 60   # data points in timeline charts


class WasteDetectionApp:
    FRAME_INTERVAL_MS   = 33   # ~30 fps UI refresh
    CHART_REFRESH_MS    = 800  # chart redraw rate

    def __init__(self, root, logger):
        self.root    = root
        self.log     = logger
        self.db      = WasteDatabase()
        self.detector = WasteDetector()

        # Session
        self.session_id   = str(uuid.uuid4())[:8]
        self.session_start = time.monotonic()
        self.db.upsert_session(self.session_id, datetime.datetime.now().isoformat())

        # Camera state
        self.camera         = None
        self.using_sim      = False
        self.sim_frame      = np.zeros((480, 640, 3), dtype=np.uint8)
        self._last_canvas_w = 0
        self._last_canvas_h = 0

        # Runtime state
        self.is_running         = False
        self.detection_active   = True
        self.sound_enabled      = True
        self.radiation_monitor  = True
        self.conf_threshold     = tk.DoubleVar(value=self.detector.CONFIDENCE_THRESHOLD)
        self.roi_enabled        = tk.BooleanVar(value=True)
        self.roi_rect_norm      = (0.18, 0.16, 0.82, 0.84)
        self._scan_phase        = 0.0
        self._last_alert_ts     = 0.0
        self._last_alert_name   = None
        self._summary_open      = False

        # Detection history
        self.history       = deque(maxlen=200)
        self.current_dets  = []
        self.last_det_name = None

        # Stats
        self.total_count     = 0
        self.hazardous_count = 0
        self.recycle_count   = 0
        self.radiation_max   = 0.10
        self.category_counts = Counter()

        # FPS tracking
        self.fps            = 0.0
        self.frame_count    = 0
        self.fps_ts         = time.monotonic()

        # Chart data (deques for O(1) append/pop)
        self.chart_tox  = deque([0.0] * CHART_HISTORY, maxlen=CHART_HISTORY)
        self.chart_rad  = deque([0.0] * CHART_HISTORY, maxlen=CHART_HISTORY)
        self.chart_fps  = deque([0.0] * CHART_HISTORY, maxlen=CHART_HISTORY)
        self.chart_count = deque([0] * CHART_HISTORY, maxlen=CHART_HISTORY)
        self._last_chart_update = 0.0

        self._build_ui()
        self._start_camera()

    # ==================================================================
    # UI Construction
    # ==================================================================
    def _build_ui(self):
        self.root.title("Real-Time Waste Detection System  v1.0.0")
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        ww, wh = min(1500, sw - 40), min(920, sh - 60)
        self.root.geometry(f"{ww}x{wh}+{(sw-ww)//2}+{(sh-wh)//2}")
        self.root.configure(bg=BG)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=2)

        self._build_left_panel()
        self._build_right_panel()
        self._apply_styles()

    # ------------------------------------------------------------------
    # Left panel: camera + status bar
    # ------------------------------------------------------------------
    def _build_left_panel(self):
        lf = tk.Frame(self.root, bg=BG2, relief="flat")
        lf.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=8)
        lf.grid_rowconfigure(1, weight=1)
        lf.grid_columnconfigure(0, weight=1)

        # Title bar
        hdr = tk.Frame(lf, bg=BG3, height=36)
        hdr.grid(row=0, column=0, sticky="ew")
        tk.Label(hdr, text="LIVE CAMERA FEED", font=(UI_FONT, 11, "bold"),
                 fg=ACCENT, bg=BG3).pack(side="left", padx=12, pady=6)
        self.lbl_mode = tk.Label(hdr, text="INITIALIZING", font=(UI_FONT, 9),
                                  fg=WARN, bg=BG3)
        self.lbl_mode.pack(side="right", padx=12, pady=6)

        # Canvas
        cam_frame = tk.Frame(lf, bg="#000000", relief="sunken", bd=2)
        cam_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        cam_frame.grid_rowconfigure(0, weight=1)
        cam_frame.grid_columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(cam_frame, bg="#000000", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        # Bottom status strip
        status_bar = tk.Frame(lf, bg=BG3, height=28)
        status_bar.grid(row=2, column=0, sticky="ew")
        self.lbl_status = tk.Label(
            status_bar, text="System initializing...",
            font=(MONO, 9), fg=FG_DIM, bg=BG3
        )
        self.lbl_status.pack(side="left", padx=10, pady=4)

    # ------------------------------------------------------------------
    # Right panel: header + notebook
    # ------------------------------------------------------------------
    def _build_right_panel(self):
        rf = tk.Frame(self.root, bg=BG2)
        rf.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=8)
        rf.grid_rowconfigure(1, weight=1)
        rf.grid_columnconfigure(0, weight=1)

        # Header
        hdr = tk.Frame(rf, bg=BG3, height=70)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        tk.Label(hdr, text="WASTE DETECTION SYSTEM",
                 font=(UI_FONT, 15, "bold"), fg=ACCENT, bg=BG3
                 ).pack(pady=(10, 0))
        tk.Label(hdr, text="AI-Powered Environmental Monitoring Platform",
                 font=(UI_FONT, 9), fg=FG_DIM, bg=BG3
                 ).pack(pady=(0, 8))

        # Stat chips row
        chips = tk.Frame(rf, bg=BG2)
        chips.grid(row=1, column=0, sticky="ew", padx=8, pady=(6, 0))
        for i in range(4):
            chips.grid_columnconfigure(i, weight=1)

        self.chip_fps      = self._chip(chips, "FPS",        "0.0",  ACCENT,  0)
        self.chip_total    = self._chip(chips, "DETECTIONS",  "0",    ACCENT2, 1)
        self.chip_hazard   = self._chip(chips, "HAZARDOUS",   "0",    DANGER,  2)
        self.chip_recycle  = self._chip(chips, "RECYCLABLE",  "0",    "#44bb66", 3)

        # Live category counter sidebar strip
        counter_frame = tk.LabelFrame(rf, text=" Live Category Counter ",
                                      font=(UI_FONT, 8, "bold"), fg=ACCENT, bg=BG2, relief="flat")
        counter_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(2, 0))
        self.lbl_category_counts = tk.Label(
            counter_frame, text="No detections yet", justify="left", anchor="w",
            font=(MONO, 8), fg=FG, bg=BG, padx=8, pady=5
        )
        self.lbl_category_counts.pack(fill="x", padx=4, pady=3)

        # Detection controls for ROI and confidence filtering
        filter_frame = tk.Frame(rf, bg=BG2)
        filter_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 0))
        filter_frame.grid_columnconfigure(1, weight=1)
        self.chk_roi = tk.Checkbutton(
            filter_frame, text="ROI zone", variable=self.roi_enabled,
            bg=BG2, fg=FG, selectcolor=BG3, activebackground=BG2,
            activeforeground=ACCENT, font=(UI_FONT, 8), cursor="hand2"
        )
        self.chk_roi.grid(row=0, column=0, sticky="w", padx=(0, 8))
        tk.Label(filter_frame, text="Confidence", font=(UI_FONT, 8), fg=FG_DIM, bg=BG2).grid(row=0, column=1, sticky="w")
        self.conf_value_lbl = tk.Label(filter_frame, text=f"{self.conf_threshold.get():.0%}",
                                       font=(MONO, 8, "bold"), fg=ACCENT, bg=BG2)
        self.conf_value_lbl.grid(row=0, column=3, sticky="e", padx=(6, 0))
        self.conf_slider = tk.Scale(
            filter_frame, from_=0.10, to=0.95, resolution=0.05, orient="horizontal",
            variable=self.conf_threshold, command=self._on_conf_threshold_changed,
            showvalue=False, bg=BG2, fg=FG_DIM, troughcolor=BG3,
            highlightthickness=0, activebackground=ACCENT
        )
        self.conf_slider.grid(row=0, column=2, sticky="ew", padx=6)
        filter_frame.grid_columnconfigure(2, weight=1)

        # Notebook
        self.nb = ttk.Notebook(rf)
        self.nb.grid(row=4, column=0, sticky="nsew", padx=8, pady=6)
        rf.grid_rowconfigure(4, weight=1)

        self._build_tab_detail()
        self._build_tab_charts()
        self._build_tab_stats()
        self._build_controls(rf)

    def _chip(self, parent, label, value, color, col):
        """Small metric chip widget. Returns value label for updates."""
        f = tk.Frame(parent, bg=BG3, relief="flat")
        f.grid(row=0, column=col, sticky="ew", padx=3, pady=3)
        val_lbl = tk.Label(f, text=value, font=(UI_FONT, 18, "bold"),
                           fg=color, bg=BG3)
        val_lbl.pack(pady=(6, 0))
        tk.Label(f, text=label, font=(UI_FONT, 7), fg=FG_DIM, bg=BG3).pack(pady=(0, 6))
        return val_lbl

    # ------------------------------------------------------------------
    # Tab 1: Detection detail
    # ------------------------------------------------------------------
    def _build_tab_detail(self):
        tab = tk.Frame(self.nb, bg=BG2)
        self.nb.add(tab, text=" Detection Detail ")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Scrollable text area
        frame = tk.Frame(tab, bg=BG2)
        frame.pack(fill="both", expand=True, padx=6, pady=6)

        sb = tk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        self.txt_detail = tk.Text(
            frame, font=(MONO, 9), bg=BG, fg=FG,
            relief="flat", bd=0, wrap="word",
            yscrollcommand=sb.set, state="disabled",
            selectbackground=BG3
        )
        self.txt_detail.pack(fill="both", expand=True)
        sb.config(command=self.txt_detail.yview)

        # Tag colors
        self.txt_detail.tag_config("header",   foreground=ACCENT,   font=(MONO, 9, "bold"))
        self.txt_detail.tag_config("label",    foreground=ACCENT2,  font=(MONO, 9))
        self.txt_detail.tag_config("value",    foreground=FG,       font=(MONO, 9))
        self.txt_detail.tag_config("danger",   foreground=DANGER,   font=(MONO, 9, "bold"))
        self.txt_detail.tag_config("good",     foreground="#44cc66", font=(MONO, 9))
        self.txt_detail.tag_config("warn",     foreground=WARN,     font=(MONO, 9))
        self.txt_detail.tag_config("dim",      foreground=FG_DIM,   font=(MONO, 8))

        self._detail_write_idle()

    def _detail_write_idle(self):
        self._txt_write(self.txt_detail, [
            ("header", "  Waiting for detection...\n\n"),
            ("dim",    "  Point the camera at any object.\n"),
            ("dim",    "  The system will classify, analyse\n"),
            ("dim",    "  and report environmental metrics.\n"),
        ])

    def _txt_write(self, widget, segments):
        """Write tagged segments to a Text widget."""
        widget.config(state="normal")
        widget.delete("1.0", "end")
        for tag, text in segments:
            widget.insert("end", text, tag)
        widget.config(state="disabled")

    def _update_detail(self, det: dict):
        s = det.get("science", {})
        wn = det["waste_name"].replace("_", " ").title()
        conf = det["confidence"]
        cat  = s.get("category", "Unknown")
        tox  = s.get("toxicity", 0.0)
        rec  = s.get("recycling_score", 0.0)
        bio  = s.get("biodegradability", 0.0)
        rad  = s.get("radiation_level", 0.0)
        rad_sim = det.get("radiation_sim", 0.1)
        decomp  = s.get("decomposition_years", 0)
        co2     = s.get("carbon_footprint_kg", 0.0)
        resin   = s.get("resin_code", "N/A")
        un_cls  = s.get("un_hazard_class", "None")
        dist_m  = det.get("distance_m")
        ar      = det.get("aspect_ratio", 0.0)
        area_pct = det.get("area_pct", 0.0)
        bw      = det.get("width_px", 0)
        bh      = det.get("height_px", 0)
        cx, cy  = det.get("norm_center", (0, 0))

        def bar(v, w=16):
            filled = int(v * w)
            return "[" + "#" * filled + "." * (w - filled) + f"] {v:.0%}"

        tox_tag = "danger" if tox > 0.7 else ("warn" if tox > 0.4 else "good")
        rad_tag = "danger" if rad_sim > 1.0 else "value"

        decomp_str = (f"{decomp:.0f} years" if decomp >= 1 else
                      f"{decomp*365:.0f} days" if decomp >= 1/365 else "< 1 day")

        segs = [
            ("header",  f"  OBJECT DETECTED\n"),
            ("dim",     "  " + "-" * 40 + "\n"),
            ("label",   "  Object      : "), ("value", f"{wn}\n"),
            ("label",   "  Confidence  : "), ("value", f"{conf:.1%}\n"),
            ("label",   "  Category    : "), ("value", f"{cat}\n"),
            ("label",   "  Resin Code  : "), ("value", f"{resin}\n"),
            ("label",   "  UN Hazard   : "), ("value", f"{un_cls}\n"),
            ("dim",     "\n  GEOMETRY\n"),
            ("dim",     "  " + "-" * 40 + "\n"),
            ("label",   "  Size (WxH)  : "), ("value", f"{bw} x {bh} px  ({area_pct:.1f}% of frame)\n"),
            ("label",   "  Aspect Ratio: "), ("value", f"{ar:.3f}\n"),
            ("label",   "  Center (rel): "), ("value", f"({cx:.3f}, {cy:.3f})\n"),
        ]
        if dist_m:
            segs += [("label", "  Est. Dist.  : "), ("value", f"{dist_m:.2f} m\n")]

        segs += [
            ("dim",     "\n  ENVIRONMENTAL METRICS\n"),
            ("dim",     "  " + "-" * 40 + "\n"),
            ("label",   "  Toxicity     : "), (tox_tag, f"{bar(tox)}\n"),
            ("label",   "  Recyclability: "), ("good",  f"{bar(rec)}\n"),
            ("label",   "  Biodegradable: "), ("good",  f"{bar(bio)}\n"),
            ("label",   "  Radiation    : "), (rad_tag, f"{rad_sim:.4f} uSv/h\n"),
            ("label",   "  Decomposition: "), ("value", f"{decomp_str}\n"),
            ("label",   "  Carbon Fprint: "), ("value", f"{co2:.2f} kg CO2e\n"),
            ("dim",     "\n  HEALTH RISK\n"),
            ("dim",     "  " + "-" * 40 + "\n"),
            ("value",   f"  {s.get('health_risk','N/A')}\n"),
            ("dim",     "\n  ENVIRONMENTAL IMPACT\n"),
            ("dim",     "  " + "-" * 40 + "\n"),
            ("value",   f"  {s.get('environmental_impact','N/A')}\n"),
            ("dim",     "\n  SCIENTIFIC NOTES\n"),
            ("dim",     "  " + "-" * 40 + "\n"),
            ("value",   f"  {s.get('scientific_notes','N/A')}\n"),
            ("dim",     "\n  DISPOSAL METHOD\n"),
            ("dim",     "  " + "-" * 40 + "\n"),
            ("good",    f"  {s.get('disposal_method','Check local guidelines.')}\n"),
        ]

        self._txt_write(self.txt_detail, segs)

    # ------------------------------------------------------------------
    # Tab 2: Charts
    # ------------------------------------------------------------------
    def _build_tab_charts(self):
        tab = tk.Frame(self.nb, bg=BG2)
        self.nb.add(tab, text=" Live Charts ")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Timeline chart
        tl_frame = tk.LabelFrame(tab, text=" Toxicity / Radiation Timeline ",
                                  font=(UI_FONT, 9, "bold"),
                                  fg=ACCENT, bg=BG2, relief="flat")
        tl_frame.grid(row=0, column=0, sticky="nsew", padx=6, pady=(6, 3))
        tl_frame.grid_rowconfigure(0, weight=1)
        tl_frame.grid_columnconfigure(0, weight=1)
        self.chart_tl = tk.Canvas(tl_frame, bg=BG, height=130, highlightthickness=0)
        self.chart_tl.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # FPS chart
        fps_frame = tk.LabelFrame(tab, text=" FPS Performance ",
                                   font=(UI_FONT, 9, "bold"),
                                   fg=ACCENT2, bg=BG2, relief="flat")
        fps_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(3, 3))
        fps_frame.grid_rowconfigure(0, weight=1)
        fps_frame.grid_columnconfigure(0, weight=1)
        self.chart_fps_c = tk.Canvas(fps_frame, bg=BG, height=100, highlightthickness=0)
        self.chart_fps_c.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        # Category bar chart
        cat_frame = tk.LabelFrame(tab, text=" Category Distribution ",
                                   font=(UI_FONT, 9, "bold"),
                                   fg="#ffaa44", bg=BG2, relief="flat")
        cat_frame.grid(row=2, column=0, sticky="nsew", padx=6, pady=(3, 6))
        tab.grid_rowconfigure(2, weight=1)
        cat_frame.grid_rowconfigure(0, weight=1)
        cat_frame.grid_columnconfigure(0, weight=1)
        self.chart_cat = tk.Canvas(cat_frame, bg=BG, height=120, highlightthickness=0)
        self.chart_cat.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

    def _redraw_charts(self):
        self._draw_timeline()
        self._draw_fps_chart()
        self._draw_category_chart()
        self._draw_session_dashboard()

    def _draw_line_chart(self, canvas, datasets, title, y_max=1.0):
        """Generic polyline chart with grid."""
        canvas.delete("all")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 10 or H < 10:
            return

        PAD = {"l": 36, "r": 8, "t": 16, "b": 22}
        cw = W - PAD["l"] - PAD["r"]
        ch = H - PAD["t"] - PAD["b"]

        # Grid
        for gi in range(5):
            gy = PAD["t"] + ch * gi // 4
            canvas.create_line(PAD["l"], gy, W - PAD["r"], gy,
                               fill="#1e2a3a", dash=(3, 4))
            val = y_max * (1 - gi / 4)
            canvas.create_text(PAD["l"] - 4, gy, text=f"{val:.2f}",
                               anchor="e", fill=FG_DIM, font=(MONO, 7))

        n = len(datasets[0][1]) if datasets else 0
        if n < 2:
            return

        for color, data in datasets:
            pts = []
            for i, v in enumerate(data):
                x = PAD["l"] + int(i * cw / (n - 1))
                y = PAD["t"] + ch - int((min(v, y_max) / y_max) * ch)
                pts.extend([x, y])
            if len(pts) >= 4:
                canvas.create_line(pts, fill=color, width=1, smooth=True)

        canvas.create_text(W // 2, H - 6, text=title,
                           fill=FG_DIM, font=(UI_FONT, 7))

    def _draw_timeline(self):
        self._draw_line_chart(
            self.chart_tl,
            [(DANGER, self.chart_tox), (ACCENT2, self.chart_rad)],
            "Blue = Radiation (norm)    Red = Toxicity",
            y_max=1.0
        )

    def _draw_fps_chart(self):
        self._draw_line_chart(
            self.chart_fps_c,
            [(ACCENT, self.chart_fps)],
            "Frames per second",
            y_max=60.0
        )

    def _draw_category_chart(self):
        canvas = self.chart_cat
        canvas.delete("all")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 10 or H < 10:
            return

        counts = Counter(d["science"].get("category", "Unknown") for d in self.history)
        if not counts:
            canvas.create_text(W // 2, H // 2, text="No data yet",
                               fill=FG_DIM, font=(UI_FONT, 9))
            return

        total = sum(counts.values())
        items = counts.most_common(8)
        bar_h = max(12, (H - 20) // len(items) - 4)
        y = 10
        max_cnt = items[0][1]

        from core.renderer import CATEGORY_COLORS
        for cat, cnt in items:
            bgr = CATEGORY_COLORS.get(cat, (140, 140, 140))
            r, g, b = bgr[2], bgr[1], bgr[0]  # BGR -> RGB
            hex_c = f"#{r:02x}{g:02x}{b:02x}"
            bw = int((cnt / max_cnt) * (W - 120))
            canvas.create_rectangle(80, y, 80 + bw, y + bar_h,
                                    fill=hex_c, outline="")
            canvas.create_text(6, y + bar_h // 2, text=cat[:10],
                               anchor="w", fill=FG, font=(MONO, 7))
            canvas.create_text(82 + bw, y + bar_h // 2,
                               text=f"{cnt}  ({100*cnt/total:.0f}%)",
                               anchor="w", fill=FG_DIM, font=(MONO, 7))
            y += bar_h + 4

    def _draw_session_dashboard(self):
        """Draw historical detections-per-session-sample bars for the current session."""
        if not hasattr(self, "chart_session"):
            return
        canvas = self.chart_session
        canvas.delete("all")
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        if W < 10 or H < 10:
            return

        data = list(self.chart_count)
        max_v = max(max(data), 1)
        pad_l, pad_t, pad_b = 34, 14, 22
        cw = W - pad_l - 10
        ch = H - pad_t - pad_b

        for gi in range(4):
            gy = pad_t + ch * gi // 3
            canvas.create_line(pad_l, gy, W - 8, gy, fill="#1e2a3a", dash=(2, 4))
            val = int(max_v * (1 - gi / 3))
            canvas.create_text(pad_l - 5, gy, text=str(val), anchor="e", fill=FG_DIM, font=(MONO, 7))

        if len(data) >= 2:
            bar_w = max(2, cw // len(data) - 1)
            for i, v in enumerate(data):
                x1 = pad_l + int(i * cw / len(data))
                x2 = x1 + bar_w
                y1 = pad_t + ch - int((v / max_v) * ch)
                canvas.create_rectangle(x1, y1, x2, pad_t + ch, fill=ACCENT2, outline="")

        canvas.create_text(W // 2, H - 7,
                           text="Historical detections per refresh window (current session)",
                           fill=FG_DIM, font=(UI_FONT, 7))

    # ------------------------------------------------------------------
    # Tab 3: Stats
    # ------------------------------------------------------------------
    def _build_tab_stats(self):
        tab = tk.Frame(self.nb, bg=BG2)
        self.nb.add(tab, text=" Session Stats ")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        dash = tk.LabelFrame(tab, text=" Detection Statistics Dashboard ",
                              font=(UI_FONT, 9, "bold"), fg=ACCENT, bg=BG2, relief="flat")
        dash.pack(fill="x", padx=6, pady=(6, 3))
        self.chart_session = tk.Canvas(dash, bg=BG, height=120, highlightthickness=0)
        self.chart_session.pack(fill="x", expand=False, padx=4, pady=4)

        text_frame = tk.Frame(tab, bg=BG2)
        text_frame.pack(fill="both", expand=True, padx=6, pady=(3, 6))
        sb2 = tk.Scrollbar(text_frame)
        sb2.pack(side="right", fill="y")
        self.txt_stats = tk.Text(
            text_frame, font=(MONO, 9), bg=BG, fg=FG,
            relief="flat", bd=0, wrap="word",
            yscrollcommand=sb2.set, state="disabled"
        )
        self.txt_stats.pack(fill="both", expand=True)
        sb2.config(command=self.txt_stats.yview)
        self.txt_stats.tag_config("h",  foreground=ACCENT,  font=(MONO, 9, "bold"))
        self.txt_stats.tag_config("v",  foreground=FG,      font=(MONO, 9))
        self.txt_stats.tag_config("d",  foreground=FG_DIM,  font=(MONO, 8))

    def _refresh_stats(self):
        elapsed = time.monotonic() - self.session_start
        mins, secs = divmod(int(elapsed), 60)
        hrs,  mins = divmod(mins, 60)

        cat_counts = Counter(d["science"].get("category", "Unknown") for d in self.history)
        total = max(self.total_count, 1)

        segs = [
            ("h", "  SESSION STATISTICS\n"),
            ("d", "  " + "=" * 38 + "\n"),
            ("h", "\n  RUNTIME\n"),
            ("v", f"  Session ID   : {self.session_id}\n"),
            ("v", f"  Elapsed      : {hrs:02d}:{mins:02d}:{secs:02d}\n"),
            ("v", f"  Current FPS  : {self.fps:.1f}\n"),
            ("v", f"  Avg FPS      : {sum(self.chart_fps)/max(len(self.chart_fps),1):.1f}\n"),
            ("h", "\n  DETECTION TOTALS\n"),
            ("v", f"  Total        : {self.total_count}\n"),
            ("v", f"  Unique Types : {len(set(d['waste_name'] for d in self.history))}\n"),
            ("v", f"  Hazardous    : {self.hazardous_count}  ({100*self.hazardous_count/total:.1f}%)\n"),
            ("v", f"  Recyclable   : {self.recycle_count}  ({100*self.recycle_count/total:.1f}%)\n"),
            ("v", f"  Peak Rad.    : {self.radiation_max:.4f} uSv/h\n"),
            ("h", "\n  CATEGORY BREAKDOWN\n"),
        ]

        for cat, cnt in cat_counts.most_common():
            pct = 100 * cnt / total
            bar_w = int(pct / 5)
            bar = "[" + "#" * bar_w + "." * (20 - bar_w) + "]"
            segs.append(("v", f"  {cat:<14} {bar}  {cnt:3d}  ({pct:.1f}%)\n"))

        avg_tox = sum(d["science"].get("toxicity", 0) for d in self.history) / max(len(self.history), 1)
        avg_rec = sum(d["science"].get("recycling_score", 0) for d in self.history) / max(len(self.history), 1)
        avg_bio = sum(d["science"].get("biodegradability", 0) for d in self.history) / max(len(self.history), 1)

        segs += [
            ("h", "\n  SESSION AVERAGES\n"),
            ("v", f"  Avg Toxicity : {avg_tox:.3f}\n"),
            ("v", f"  Avg Recycle  : {avg_rec:.3f}\n"),
            ("v", f"  Avg Biodeg.  : {avg_bio:.3f}\n"),
        ]

        self._txt_write(self.txt_stats, segs)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------
    def _build_controls(self, parent):
        ctrl = tk.Frame(parent, bg=BG2)
        ctrl.grid(row=5, column=0, sticky="ew", padx=8, pady=(0, 8))
        for i in range(6):
            ctrl.grid_columnconfigure(i, weight=1)

        btn_cfg = {"font": (UI_FONT, 8, "bold"), "relief": "flat",
                   "activeforeground": "#ffffff", "cursor": "hand2",
                   "padx": 4, "pady": 6}

        self.btn_toggle = self._btn(ctrl, "PAUSE",       self._toggle_detection, DANGER,    0, 0, btn_cfg)
        self._btn(ctrl, "REPORT",      self._generate_report,    "#2255aa",  0, 1, btn_cfg)
        self._btn(ctrl, "EXPORT CSV",  self._export_csv,         "#225533",  0, 2, btn_cfg)
        self._btn(ctrl, "HISTORY",     self._show_history,       "#554422",  0, 3, btn_cfg)
        self._btn(ctrl, "REFRESH",     self._refresh_stats,      "#334466",  0, 4, btn_cfg)
        self._btn(ctrl, "SOUND",       self._toggle_sound,       "#333355",  0, 5, btn_cfg)

    def _btn(self, parent, text, cmd, bg, row, col, cfg):
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg="#ffffff",
                      activebackground=bg, **cfg)
        b.grid(row=row, column=col, sticky="ew", padx=2, pady=2)
        return b

    # ------------------------------------------------------------------
    # Style
    # ------------------------------------------------------------------
    def _apply_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TNotebook",          background=BG2, borderwidth=0)
        style.configure("TNotebook.Tab",      background=BG3, foreground=FG_DIM,
                        padding=[12, 5], font=(UI_FONT, 9))
        style.map("TNotebook.Tab",
                  background=[("selected", BG)],
                  foreground=[("selected", ACCENT)])

    # ==================================================================
    # Camera
    # ==================================================================
    def _start_camera(self):
        for idx in range(3):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS,           30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE,     1)  # reduce latency
                self.camera    = cap
                self.using_sim = False
                self.log.info(f"Camera opened at index {idx}.")
                break
        else:
            self.using_sim = True
            self.log.warning("No camera found. Running in simulation mode.")

        self.is_running = True
        self._update_frame()

    def _read_frame(self):
        if self.using_sim:
            return self._sim_frame()
        ret, frame = self.camera.read()
        if not ret:
            return self._sim_frame()
        return frame

    def _sim_frame(self):
        """Generate a synthetic frame with random noise + shapes."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        noise = np.random.randint(0, 25, frame.shape, dtype=np.uint8)
        frame = cv2.add(frame, noise)
        if np.random.random() < 0.6:
            x = np.random.randint(80, 480)
            y = np.random.randint(80, 360)
            w = np.random.randint(50, 160)
            h = np.random.randint(50, 160)
            color = tuple(int(c) for c in np.random.randint(80, 230, 3).tolist())
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        return frame

    # ==================================================================
    # Detection UI helpers
    # ==================================================================
    def _on_conf_threshold_changed(self, value):
        threshold = float(value)
        self.detector.CONFIDENCE_THRESHOLD = threshold
        self.conf_value_lbl.config(text=f"{threshold:.0%}")

    def _roi_pixels(self, width: int, height: int):
        x1n, y1n, x2n, y2n = self.roi_rect_norm
        return (int(x1n * width), int(y1n * height), int(x2n * width), int(y2n * height))

    def _det_in_roi(self, det: dict, width: int, height: int) -> bool:
        if not self.roi_enabled.get():
            return True
        x1, y1, x2, y2 = self._roi_pixels(width, height)
        cx, cy = det.get("center", ((det["bbox"][0] + det["bbox"][2]) // 2,
                                      (det["bbox"][1] + det["bbox"][3]) // 2))
        return x1 <= cx <= x2 and y1 <= cy <= y2

    def _draw_roi_overlay(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._roi_pixels(w, h)
        color = (60, 210, 160) if self.roi_enabled.get() else (90, 90, 90)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 1)
        if self.roi_enabled.get():
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "ROI MONITOR ZONE", (x1 + 8, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        return frame

    def _draw_scanning_line(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self._roi_pixels(w, h) if self.roi_enabled.get() else (0, 38, w, h - 30)
        span = max(1, y2 - y1)
        self._scan_phase = (self._scan_phase + 0.015) % 1.0
        y = y1 + int(self._scan_phase * span)
        overlay = frame.copy()
        cv2.line(overlay, (x1, y), (x2, y), (0, 255, 180), 2)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.line(frame, (x1, max(y1, y - 6)), (x2, max(y1, y - 6)), (0, 120, 80), 1)
        cv2.line(frame, (x1, min(y2, y + 6)), (x2, min(y2, y + 6)), (0, 120, 80), 1)
        return frame

    def _draw_minimap(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        h, w = frame.shape[:2]
        mw, mh = 150, 105
        x0, y0 = w - mw - 12, 48
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + mw, y0 + mh), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + mw, y0 + mh), (70, 90, 120), 1)
        cv2.putText(frame, "MINI-MAP", (x0 + 8, y0 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (210, 220, 230), 1, cv2.LINE_AA)

        if self.roi_enabled.get():
            rx1, ry1, rx2, ry2 = self._roi_pixels(w, h)
            mm = lambda px, py: (x0 + int(px * mw / w), y0 + 20 + int(py * (mh - 25) / h))
            p1 = mm(rx1, ry1)
            p2 = mm(rx2, ry2)
            cv2.rectangle(frame, p1, p2, (60, 210, 160), 1)

        for det in detections[:20]:
            s = det.get("science", {})
            color = _color(s.get("category", "Unknown"))
            cx, cy = det.get("center", (0, 0))
            px = x0 + int(cx * mw / max(w, 1))
            py = y0 + 20 + int(cy * (mh - 25) / max(h, 1))
            cv2.circle(frame, (px, py), 4, color, -1)
            cv2.circle(frame, (px, py), 6, (230, 230, 230), 1)
        return frame

    def _update_category_counter(self):
        if not self.category_counts:
            text = "No detections yet"
        else:
            parts = [f"{cat}: {cnt}" for cat, cnt in self.category_counts.most_common(6)]
            text = "   •   ".join(parts)
        self.lbl_category_counts.config(text=text)

    def _show_threat_alert(self, det: dict):
        now = time.monotonic()
        name = det.get("waste_name", "object")
        if self._last_alert_name == name and now - self._last_alert_ts < 6.0:
            return
        self._last_alert_name = name
        self._last_alert_ts = now
        s = det.get("science", {})
        msg = (f"Object Threat Alert\n\n"
               f"Object: {name.replace('_', ' ').title()}\n"
               f"Category: {s.get('category', 'Unknown')}\n"
               f"Toxicity: {s.get('toxicity', 0):.2f}\n"
               f"Confidence: {det.get('confidence', 0):.0%}\n\n"
               f"Recommended action: isolate item and follow hazardous disposal procedure.")
        self.root.after(0, lambda: messagebox.showwarning("Object Threat Alert", msg, parent=self.root))

    def _session_summary_text(self):
        elapsed = time.monotonic() - self.session_start
        mins, secs = divmod(int(elapsed), 60)
        hrs, mins = divmod(mins, 60)
        cat = ", ".join(f"{k}: {v}" for k, v in self.category_counts.most_common(5)) or "No categories"
        return (f"Session Summary\n\n"
                f"Session ID: {self.session_id}\n"
                f"Duration: {hrs:02d}:{mins:02d}:{secs:02d}\n"
                f"Total detections: {self.total_count}\n"
                f"Hazardous: {self.hazardous_count}\n"
                f"Recyclable: {self.recycle_count}\n"
                f"Peak radiation: {self.radiation_max:.4f} uSv/h\n"
                f"Categories: {cat}")

    def _show_session_summary(self):
        if self._summary_open:
            return
        self._summary_open = True
        try:
            messagebox.showinfo("Detection Session Summary", self._session_summary_text(), parent=self.root)
        finally:
            self._summary_open = False

    # ==================================================================
    # Main update loop
    # ==================================================================
    def _update_frame(self):
        if not self.is_running:
            return

        try:
            frame = self._read_frame()
            fh, fw = frame.shape[:2]
            session_time = time.monotonic() - self.session_start

            # Detection
            if self.detection_active:
                new_dets = self.detector.detect(frame)
                if new_dets:
                    new_dets = [det for det in new_dets if self._det_in_roi(det, fw, fh)]
                    self.current_dets = new_dets
                    for det in new_dets:
                        if det["waste_name"] != self.last_det_name:
                            self._on_new_detection(det, fw, fh)
                            self.last_det_name = det["waste_name"]

            # Render overlay
            frame = self._draw_roi_overlay(frame)
            frame = self._draw_scanning_line(frame)
            for idx, det in enumerate(self.current_dets):
                frame = draw_detection(frame, det, idx)
            frame = self._draw_minimap(frame, self.current_dets)

            rad_max = max((d.get("radiation_sim", 0.1) for d in self.current_dets), default=self.radiation_max)
            self.radiation_max = max(self.radiation_max, rad_max)

            frame = draw_hud(frame, self.fps, self.total_count,
                             self.hazardous_count, self.recycle_count,
                             self.detection_active, rad_max, session_time)

            # FPS
            self.frame_count += 1
            now = time.monotonic()
            elapsed = now - self.fps_ts
            if elapsed >= 1.0:
                self.fps = self.frame_count / elapsed
                self.frame_count = 0
                self.fps_ts = now
                self.chart_fps.append(self.fps)
                self.chart_count.append(self.total_count)

            # Display
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw > 10 and ch > 10:
                if cw != self._last_canvas_w or ch != self._last_canvas_h:
                    self._last_canvas_w = cw
                    self._last_canvas_h = ch
                resized = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_LINEAR)
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                imgtk   = ImageTk.PhotoImage(image=img_pil)
                self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
                self.canvas.imgtk = imgtk  # prevent GC

            # Update chips
            self.chip_fps.config(text=f"{self.fps:.1f}")
            self.chip_total.config(text=str(self.total_count))
            self.chip_hazard.config(text=str(self.hazardous_count))
            self.chip_recycle.config(text=str(self.recycle_count))
            self._update_category_counter()

            mode_txt = "LIVE" if not self.using_sim else "SIMULATION"
            mode_col = ACCENT if not self.using_sim else WARN
            self.lbl_mode.config(text=mode_txt, fg=mode_col)

            self.lbl_status.config(
                text=(f"Session {self.session_id}  |  "
                      f"FPS {self.fps:.1f}  |  "
                      f"Detections {self.total_count}  |  "
                      f"Radiation {rad_max:.4f} uSv/h  |  "
                      f"{datetime.datetime.now().strftime('%H:%M:%S')}")
            )

            # Throttled chart refresh
            if now - self._last_chart_update > self.CHART_REFRESH_MS / 1000:
                self._redraw_charts()
                self._last_chart_update = now

        except Exception as exc:
            self.log.error(f"Frame update: {exc}")

        self.root.after(self.FRAME_INTERVAL_MS, self._update_frame)

    def _on_new_detection(self, det: dict, fw: int, fh: int):
        s = det.get("science", {})
        det_record = {**det, "timestamp": datetime.datetime.now()}
        self.history.append(det_record)
        self.total_count += 1

        tox = s.get("toxicity", 0)
        rec = s.get("recycling_score", 0)
        rad = det.get("radiation_sim", 0.1)

        if tox > 0.5:   self.hazardous_count += 1
        if rec > 0.5:   self.recycle_count += 1
        self.category_counts[s.get("category", "Unknown")] += 1

        # Chart data
        self.chart_tox.append(tox)
        self.chart_rad.append(min(1.0, rad / 2.0))  # normalise for display

        # DB log
        try:
            self.db.log_detection(
                self.session_id,
                det["waste_name"], det.get("raw_class", ""),
                det["confidence"],
                s.get("category", "Unknown"),
                rad, tox,
                det["bbox"], (fw, fh)
            )
        except Exception as e:
            self.log.error(f"DB log: {e}")

        # UI detail update
        self._update_detail(det)

        # Sound and popup alert
        if tox > 0.7:
            self._show_threat_alert(det)
            if self.sound_enabled and _SOUND:
                threading.Thread(target=self._beep, daemon=True).start()

    def _beep(self):
        try:
            import winsound
            winsound.Beep(1000, 150)
            winsound.Beep(1400, 150)
        except Exception:
            pass

    # ==================================================================
    # Controls
    # ==================================================================
    def _toggle_detection(self):
        self.detection_active = not self.detection_active
        if self.detection_active:
            self.btn_toggle.config(text="PAUSE", bg=DANGER)
            self.current_dets = []
        else:
            self.btn_toggle.config(text="RESUME", bg="#226622")
            self._show_session_summary()

    def _toggle_sound(self):
        self.sound_enabled = not self.sound_enabled

    def _generate_report(self):
        from ui.report_window import ReportWindow
        ReportWindow(self.root, self)

    def _export_csv(self):
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default = os.path.join(os.path.expanduser("~"), f"waste_export_{now_str}.csv")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile=default
        )
        if not path:
            return
        rows = self.db.get_detection_history(1000)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ID", "Session", "Timestamp", "Object", "Raw Class",
                        "Confidence", "Category", "Radiation", "Toxicity",
                        "BBox X1", "BBox Y1", "BBox X2", "BBox Y2",
                        "Area px", "Frame W", "Frame H"])
            for r in rows:
                w.writerow(list(r))
        messagebox.showinfo("Export Complete", f"Saved to:\n{path}")

    def _show_history(self):
        from ui.history_window import HistoryWindow
        HistoryWindow(self.root, self)

    # ==================================================================
    # Shutdown
    # ==================================================================
    def shutdown(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        try:
            elapsed = time.monotonic() - self.session_start
            self.db.upsert_session(
                self.session_id,
                datetime.datetime.now().isoformat(),
                end_time=datetime.datetime.now().isoformat(),
                total_detections=self.total_count,
                hazardous_count=self.hazardous_count,
                recyclable_count=self.recycle_count,
                avg_fps=self.fps,
            )
        except Exception as e:
            self.log.error(f"Shutdown DB: {e}")
        self.db.close()
        self.log.info("Application shutdown complete.")
