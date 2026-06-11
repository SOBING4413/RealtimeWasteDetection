"""
ui/report_window.py
Popup window that generates and displays a structured session report.
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import datetime
from collections import Counter

BG    = "#0f1117"
BG2   = "#181b24"
BG3   = "#1e2230"
ACCENT = "#00c896"
FG    = "#d0d8e8"
FG_DIM = "#6b7a96"
MONO  = "Consolas"
UI_FONT = "Segoe UI"


class ReportWindow:
    def __init__(self, parent, app):
        self.app = app
        win = tk.Toplevel(parent)
        win.title("Session Report")
        win.geometry("820x640")
        win.configure(bg=BG)
        win.grab_set()

        tk.Label(win, text="SESSION REPORT", font=(UI_FONT, 13, "bold"),
                 fg=ACCENT, bg=BG).pack(pady=(12, 4))
        tk.Label(win, text="Real-Time Waste Detection System  v1.0.0",
                 font=(UI_FONT, 9), fg=FG_DIM, bg=BG).pack()

        frame = tk.Frame(win, bg=BG)
        frame.pack(fill="both", expand=True, padx=14, pady=8)

        sb = tk.Scrollbar(frame)
        sb.pack(side="right", fill="y")
        txt = tk.Text(frame, font=(MONO, 9), bg=BG2, fg=FG,
                      relief="flat", bd=0, wrap="word",
                      yscrollcommand=sb.set, state="disabled")
        txt.pack(fill="both", expand=True)
        sb.config(command=txt.yview)

        report_text = self._build(app)
        txt.config(state="normal")
        txt.insert("1.0", report_text)
        txt.config(state="disabled")

        btn_frame = tk.Frame(win, bg=BG)
        btn_frame.pack(pady=8)

        def save():
            now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = filedialog.asksaveasfilename(
                defaultextension=".txt",
                filetypes=[("Text files", "*.txt")],
                initialfile=f"waste_report_{now}.txt"
            )
            if path:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                messagebox.showinfo("Saved", f"Report saved:\n{path}", parent=win)

        tk.Button(btn_frame, text="Save Report", command=save,
                  font=(UI_FONT, 9, "bold"), bg="#225533", fg="#ffffff",
                  relief="flat", padx=18, pady=6).pack(side="left", padx=8)
        tk.Button(btn_frame, text="Close", command=win.destroy,
                  font=(UI_FONT, 9, "bold"), bg=BG3, fg=FG_DIM,
                  relief="flat", padx=18, pady=6).pack(side="left", padx=8)

    @staticmethod
    def _build(app) -> str:
        now = datetime.datetime.now()
        total = max(app.total_count, 1)
        cat_counts = Counter(d["science"].get("category", "Unknown") for d in app.history)
        unique = len(set(d["waste_name"] for d in app.history))

        avg_tox = sum(d["science"].get("toxicity", 0) for d in app.history) / max(len(app.history), 1)
        avg_rec = sum(d["science"].get("recycling_score", 0) for d in app.history) / max(len(app.history), 1)
        avg_bio = sum(d["science"].get("biodegradability", 0) for d in app.history) / max(len(app.history), 1)

        sep = "=" * 62

        lines = [
            sep,
            "  REAL-TIME WASTE DETECTION SYSTEM  v1.0.0",
            "  Comprehensive Session Analysis Report",
            sep,
            f"  Generated   : {now.strftime('%Y-%m-%d  %H:%M:%S')}",
            f"  Session ID  : {app.session_id}",
            f"  AI Model    : YOLOv8n (COCO 80-class) + Environmental Mapping",
            "",
            sep,
            "  EXECUTIVE SUMMARY",
            sep,
            f"  Total Detections    : {app.total_count}",
            f"  Unique Object Types : {unique}",
            f"  Hazardous Found     : {app.hazardous_count}  "
            f"({100*app.hazardous_count/total:.1f}%)",
            f"  Recyclable Found    : {app.recycle_count}  "
            f"({100*app.recycle_count/total:.1f}%)",
            f"  Peak Radiation      : {app.radiation_max:.4f} uSv/h",
            f"  Average FPS         : {app.fps:.1f}",
            f"  Radiation Monitor   : {'Active' if app.radiation_monitor else 'Disabled'}",
            "",
            sep,
            "  CATEGORY DISTRIBUTION",
            sep,
        ]

        for cat, cnt in cat_counts.most_common():
            pct = 100 * cnt / total
            bar = "#" * int(pct / 4) + "." * (25 - int(pct / 4))
            lines.append(f"  {cat:<16} [{bar}]  {cnt:4d}  ({pct:.1f}%)")

        lines += [
            "",
            sep,
            "  ENVIRONMENTAL METRICS (SESSION AVERAGES)",
            sep,
            f"  Average Toxicity Index     : {avg_tox:.3f}",
            f"  Average Recycling Score    : {avg_rec:.3f}",
            f"  Average Biodegradability   : {avg_bio:.3f}",
            "",
            sep,
            "  DETECTION LOG (LAST 30)",
            sep,
        ]

        recent = list(app.history)[-30:]
        for i, d in enumerate(reversed(recent), 1):
            ts  = d["timestamp"].strftime("%H:%M:%S")
            wn  = d["waste_name"].replace("_", " ").title()
            cat = d["science"].get("category", "?")
            tox = d["science"].get("toxicity", 0)
            rad = d.get("radiation_sim", 0.1)
            conf = d["confidence"]
            dist = d.get("distance_m")
            dist_s = f"  ~{dist:.2f}m" if dist else ""
            lines.append(
                f"  {i:2d}. [{ts}]  {wn:<22} "
                f"Cat:{cat:<12} Conf:{conf:.0%}  "
                f"Tox:{tox:.2f}  Rad:{rad:.4f}{dist_s}"
            )

        lines += [
            "",
            sep,
            "  HAZARDOUS ITEMS",
            sep,
        ]

        hazardous = [d for d in app.history if d["science"].get("toxicity", 0) > 0.5]
        if hazardous:
            for item, cnt in Counter(d["waste_name"] for d in hazardous).items():
                lines.append(f"  {item.replace('_',' ').title():<28} x {cnt} detection(s)")
        else:
            lines.append("  No hazardous materials detected in this session.")

        lines += [
            "",
            sep,
            "  RECOMMENDATIONS",
            sep,
            "  1. Separate recyclable materials (metal, glass, paper) from general waste.",
            "  2. Dispose of hazardous items (batteries, paint, electronics) at certified",
            "     collection facilities. Never place in curbside bins.",
            "  3. Compost all eligible organic waste to reduce landfill methane emissions.",
            "  4. Reduce single-use plastic consumption wherever possible.",
            "  5. Support and use local e-waste take-back programs.",
            "  6. If radiation levels exceed 1.0 uSv/h, contact environmental authorities.",
            "",
            sep,
            "  END OF REPORT",
            sep,
        ]

        return "\n".join(lines)
