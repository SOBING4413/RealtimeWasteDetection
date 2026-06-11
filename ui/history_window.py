"""
ui/history_window.py
Detection history window with sortable, colour-coded treeview.
"""

import tkinter as tk
from tkinter import ttk
from collections import Counter

BG    = "#0f1117"
BG2   = "#181b24"
BG3   = "#1e2230"
ACCENT = "#00c896"
DANGER = "#ff4444"
WARN   = "#ffaa00"
FG    = "#d0d8e8"
FG_DIM = "#6b7a96"
MONO  = "Consolas"
UI_FONT = "Segoe UI"

CATEGORY_TAG_COLORS = {
    "Hazardous":  ("#ff6666", "#220000"),
    "Nuclear":    ("#ff4444", "#1a0000"),
    "Biohazard":  ("#ff9944", "#221100"),
    "E-Waste":    ("#dd66ff", "#1a0022"),
    "Plastic":    ("#44cccc", "#001a1a"),
    "Organic":    ("#66ff88", "#001a00"),
    "Metal":      ("#bbbbbb", "#1a1a1a"),
    "Glass":      ("#ffffff", "#1a1a1a"),
    "Paper":      ("#aabb66", "#111600"),
    "Cardboard":  ("#cc9944", "#1a1000"),
    "Rubber":     ("#6688aa", "#0a0e14"),
    "Textile":    ("#bb8855", "#1a1008"),
}


class HistoryWindow:
    COLUMNS = ("Time", "Object", "Confidence", "Category",
                "Toxicity", "Recycling", "Radiation", "Distance", "Area%")

    def __init__(self, parent, app):
        self.app = app
        win = tk.Toplevel(parent)
        win.title("Detection History")
        win.geometry("1100x560")
        win.configure(bg=BG)
        win.grab_set()

        tk.Label(win, text="DETECTION HISTORY", font=(UI_FONT, 12, "bold"),
                 fg=ACCENT, bg=BG).pack(pady=(10, 2))
        tk.Label(win, text=f"Session  {app.session_id}",
                 font=(UI_FONT, 8), fg=FG_DIM, bg=BG).pack()

        # Treeview frame
        tf = tk.Frame(win, bg=BG)
        tf.pack(fill="both", expand=True, padx=10, pady=8)

        vsb = tk.Scrollbar(tf, orient="vertical")
        hsb = tk.Scrollbar(tf, orient="horizontal")
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("History.Treeview",
                        background=BG2, foreground=FG,
                        fieldbackground=BG2, rowheight=22,
                        font=(MONO, 8))
        style.configure("History.Treeview.Heading",
                        background=BG3, foreground=ACCENT,
                        font=(UI_FONT, 9, "bold"), relief="flat")
        style.map("History.Treeview",
                  background=[("selected", BG3)],
                  foreground=[("selected", ACCENT)])

        tree = ttk.Treeview(
            tf, columns=self.COLUMNS, show="headings",
            yscrollcommand=vsb.set, xscrollcommand=hsb.set,
            style="History.Treeview"
        )
        col_widths = [72, 170, 80, 100, 72, 72, 90, 72, 72]
        for col, w in zip(self.COLUMNS, col_widths):
            tree.heading(col, text=col,
                         command=lambda c=col: self._sort(tree, c, False))
            tree.column(col, width=w, anchor="center", stretch=False)

        tree.pack(fill="both", expand=True)
        vsb.config(command=tree.yview)
        hsb.config(command=tree.xview)

        # Category color tags
        for cat, (fg_c, bg_c) in CATEGORY_TAG_COLORS.items():
            tree.tag_configure(cat, foreground=fg_c, background=bg_c)

        # Populate
        for det in reversed(list(app.history)):
            s    = det.get("science", {})
            cat  = s.get("category", "Unknown")
            ts   = det["timestamp"].strftime("%H:%M:%S")
            wn   = det["waste_name"].replace("_", " ").title()
            conf = f"{det['confidence']:.0%}"
            tox  = f"{s.get('toxicity', 0):.3f}"
            rec  = f"{s.get('recycling_score', 0):.3f}"
            rad  = f"{det.get('radiation_sim', 0.1):.4f}"
            dist = f"{det.get('distance_m'):.2f} m" if det.get("distance_m") else "-"
            area = f"{det.get('area_pct', 0):.1f}%"
            tag  = cat if cat in CATEGORY_TAG_COLORS else ""
            tree.insert("", "end",
                        values=(ts, wn, conf, cat, tox, rec, rad, dist, area),
                        tags=(tag,))

        # Summary
        total    = max(len(app.history), 1)
        cat_dist = Counter(d.get("science", {}).get("category", "Unknown")
                           for d in app.history)
        summary  = "  |  ".join(f"{c}: {n}" for c, n in cat_dist.most_common(6))
        tk.Label(win, text=f"  {len(app.history)} entries   |   {summary}",
                 font=(MONO, 8), fg=FG_DIM, bg=BG).pack(pady=(0, 8))

    @staticmethod
    def _sort(tree, col, reverse):
        data = [(tree.set(child, col), child) for child in tree.get_children("")]
        try:
            data.sort(key=lambda x: float(x[0].rstrip(" m%")), reverse=reverse)
        except ValueError:
            data.sort(key=lambda x: x[0], reverse=reverse)
        for index, (_, child) in enumerate(data):
            tree.move(child, "", index)
        tree.heading(col, command=lambda: HistoryWindow._sort(tree, col, not reverse))
