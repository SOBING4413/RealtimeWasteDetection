"""
core/renderer.py
HUD overlay renderer.
Draws bounding boxes, class labels, metric bars, and system overlay
directly onto OpenCV BGR frames.
All drawing is done with standard cv2 calls (no font dependencies).
"""

import cv2
import numpy as np
import time


# Category -> BGR color
CATEGORY_COLORS = {
    "Plastic":   (0, 230, 230),     # cyan-yellow
    "Paper":     (200, 230, 100),   # lime
    "Cardboard": (60, 180, 255),    # orange
    "Metal":     (200, 200, 200),   # silver
    "Glass":     (255, 255, 255),   # white
    "Organic":   (60, 200, 60),     # green
    "Hazardous": (40, 40, 255),     # red
    "E-Waste":   (200, 60, 200),    # magenta
    "Nuclear":   (30, 230, 30),     # bright green
    "Biohazard": (0, 165, 255),     # orange-red
    "Rubber":    (80, 130, 180),    # muted blue
    "Textile":   (180, 120, 60),    # brown
    "Unknown":   (140, 140, 140),   # grey
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL = 0.40
FONT_MED   = 0.52
FONT_LARGE = 0.70
THICK_THIN = 1
THICK_NORM = 2


def _color(category: str) -> tuple:
    return CATEGORY_COLORS.get(category, CATEGORY_COLORS["Unknown"])


def _draw_bar(img, x, y, w, value, color, label, h=8):
    """Draw a filled progress bar."""
    bg_c = (50, 50, 50)
    fill_w = int(w * max(0.0, min(1.0, value)))
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_c, -1)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.putText(img, f"{label}: {value:.0%}", (x, y - 3),
                FONT, FONT_SMALL, (210, 210, 210), THICK_THIN, cv2.LINE_AA)


def draw_detection(frame: np.ndarray, det: dict, index: int) -> np.ndarray:
    """Draw a single detection's bounding box and detail panel."""
    x1, y1, x2, y2 = det["bbox"]
    conf        = det["confidence"]
    waste_name  = det["waste_name"].replace("_", " ").title()
    science     = det.get("science", {})
    category    = science.get("category", "Unknown")
    color       = _color(category)
    dist_m      = det.get("distance_m")
    ar          = det.get("aspect_ratio", 1.0)
    area_pct    = det.get("area_pct", 0.0)
    tox         = science.get("toxicity", 0.0)
    rec         = science.get("recycling_score", 0.0)
    bio         = science.get("biodegradability", 0.0)
    rad_sim     = det.get("radiation_sim", 0.1)

    # --- Bounding box ---
    # Outer glow effect (thicker darker box)
    glow_c = tuple(int(c * 0.4) for c in color)
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), glow_c, 3)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Corner brackets for precision feel
    L = max(10, min(20, (x2 - x1) // 5))
    T = 2
    for px, py, sx, sy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (px, py), (px + sx * L, py), color, T)
        cv2.line(frame, (px, py), (px, py + sy * L), color, T)

    # --- Top label strip ---
    label   = f"{waste_name}  {conf:.0%}"
    (lw, lh), _ = cv2.getTextSize(label, FONT, FONT_MED, THICK_NORM)
    strip_y1 = max(0, y1 - lh - 12)
    strip_y2 = max(0, y1)
    cv2.rectangle(frame, (x1, strip_y1), (x1 + lw + 16, strip_y2), color, -1)
    cv2.putText(frame, label, (x1 + 6, strip_y2 - 4),
                FONT, FONT_MED, (10, 10, 10), THICK_NORM, cv2.LINE_AA)

    # --- Info sidebar (right of box if space, else left) ---
    H, W = frame.shape[:2]
    side_x = x2 + 8 if (x2 + 180) < W else x1 - 188
    base_y = y1

    def txt(msg, dy, color_=None):
        nonlocal base_y
        cv2.putText(frame, msg, (side_x, base_y + dy),
                    FONT, FONT_SMALL, color_ or (210, 210, 210), THICK_THIN, cv2.LINE_AA)

    overlay = frame.copy()
    panel_h = 130
    panel_w = 180
    cv2.rectangle(overlay, (side_x - 4, base_y - 2), (side_x + panel_w, base_y + panel_h),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    txt(f"Cat : {category}",               12,  color)
    txt(f"Tox : {tox:.2f}",                26,  (80, 80, 255) if tox > 0.5 else (140, 210, 140))
    txt(f"Rec : {rec:.2f}",                40,  (80, 210, 80))
    txt(f"Bio : {bio:.2f}",                54,  (80, 200, 120))
    txt(f"Rad : {rad_sim:.3f} uSv/h",      68,  (80, 80, 255) if rad_sim > 1.0 else (200, 200, 200))
    if dist_m:
        txt(f"Dist: {dist_m:.2f} m",       82,  (200, 180, 100))
    txt(f"AR  : {ar:.2f}",                 96,  (180, 180, 180))
    txt(f"Size: {area_pct:.1f}% frame",   110,  (160, 160, 160))

    # --- Bottom decomp bar ---
    cy = y2 + 6
    if cy + 22 < H:
        decomp = min(1.0, science.get("decomposition_years", 50) / 1000)
        _draw_bar(frame, x1, cy + 10, x2 - x1, decomp, (80, 80, 255), "Decomp")

    return frame


def draw_hud(frame: np.ndarray, fps: float, total: int, hazardous: int,
             recyclable: int, active: bool, radiation_max: float,
             session_time: float) -> np.ndarray:
    """Draw top/bottom HUD bars with system-wide stats."""
    H, W = frame.shape[:2]
    overlay = frame.copy()

    # Top bar
    cv2.rectangle(overlay, (0, 0), (W, 38), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    status_txt = "ACTIVE" if active else "PAUSED"
    status_col = (60, 210, 60) if active else (60, 60, 200)
    mins, secs = divmod(int(session_time), 60)

    cv2.putText(frame, "WASTE DETECTION v1.0.0",     (10, 25), FONT, FONT_MED, (60, 210, 160), THICK_NORM, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:5.1f}",         (W - 320, 25), FONT, FONT_MED, (200, 200, 200), THICK_THIN, cv2.LINE_AA)
    cv2.putText(frame, f"[{status_txt}]",           (W - 210, 25), FONT, FONT_MED, status_col, THICK_NORM, cv2.LINE_AA)
    cv2.putText(frame, f"{mins:02d}:{secs:02d}",    (W - 80, 25),  FONT, FONT_MED, (160, 160, 160), THICK_THIN, cv2.LINE_AA)

    # Bottom bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, H - 30), (W, H), (10, 10, 10), -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

    bottom_txt = (f"  Total: {total}   Hazardous: {hazardous}"
                  f"   Recyclable: {recyclable}"
                  f"   Radiation: {radiation_max:.3f} uSv/h")
    cv2.putText(frame, bottom_txt, (6, H - 9), FONT, FONT_SMALL, (60, 210, 160), THICK_THIN, cv2.LINE_AA)

    # Radiation alert flash
    if radiation_max > 1.0:
        flash_col = (40, 40, 255) if int(time.time() * 2) % 2 == 0 else (0, 0, 180)
        cv2.putText(frame, "RADIATION ALERT", (W // 2 - 100, H // 2),
                    FONT, FONT_LARGE, flash_col, 3, cv2.LINE_AA)

    return frame
