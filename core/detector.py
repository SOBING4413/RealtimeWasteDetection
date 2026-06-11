"""
core/detector.py
Optimized detection engine.
- YOLOv8n (fast) when ultralytics is available
- OpenCV contour fallback with shape heuristics
- Detection throttling to avoid overloading the UI thread
- Precise bounding-box metrics: area, aspect ratio, relative size, estimated distance
"""

import cv2
import numpy as np
import math
import random
import time
from core.science import YOLO_TO_WASTE, lookup

try:
    from ultralytics import YOLO as _YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


# Approximate known real-world widths (metres) for distance estimation
KNOWN_WIDTH_M = {
    "plastic_bottle": 0.07,
    "aluminum_can":   0.065,
    "glass_bottle":   0.08,
    "paper":          0.21,
    "cardboard":      0.30,
    "electronics":    0.30,
    "battery":        0.045,
    "textile":        0.40,
}
FOCAL_LENGTH_PX = 700  # typical webcam approximate focal length


def _estimate_distance(waste_name: str, bbox_width_px: int) -> float | None:
    """Estimate object distance in metres using pinhole camera model."""
    known_w = KNOWN_WIDTH_M.get(waste_name)
    if known_w and bbox_width_px > 0:
        dist = (known_w * FOCAL_LENGTH_PX) / bbox_width_px
        return round(dist, 2)
    return None


class WasteDetector:
    CONFIDENCE_THRESHOLD = 0.40
    MAX_DETECTIONS = 8          # cap per-frame to protect UI thread
    THROTTLE_INTERVAL_MS = 80   # minimum ms between YOLO inferences

    def __init__(self):
        self._model = None
        self._last_inference_ms = 0
        self._scientific_cache: dict = {}
        self.radiation_base = 0.10  # background μSv/h
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self):
        if YOLO_AVAILABLE:
            try:
                self._model = _YOLO("yolov8n.pt")
                # Warm-up on a blank frame to avoid first-frame lag
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                self._model(dummy, verbose=False, imgsz=320)
                return
            except Exception as e:
                print(f"[Detector] YOLO load error: {e}")
                self._model = None
        print("[Detector] Using OpenCV contour fallback.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> list[dict]:
        """Run detection on a BGR frame. Returns list of detection dicts."""
        now_ms = int(time.monotonic() * 1000)
        if now_ms - self._last_inference_ms < self.THROTTLE_INTERVAL_MS:
            return []  # skip frame to maintain target FPS

        self._last_inference_ms = now_ms

        if self._model is not None:
            detections = self._yolo_detect(frame)
        else:
            detections = self._contour_detect(frame)

        # Enrich each detection with scientific + geometric metadata
        enriched = []
        for det in detections[:self.MAX_DETECTIONS]:
            det = self._enrich(det, frame)
            enriched.append(det)

        return enriched

    # ------------------------------------------------------------------
    # YOLO inference
    # ------------------------------------------------------------------
    def _yolo_detect(self, frame: np.ndarray) -> list[dict]:
        results = []
        # Use smaller imgsz for speed while retaining quality
        yolo_out = self._model(frame, verbose=False, imgsz=416, conf=self.CONFIDENCE_THRESHOLD)

        for r in yolo_out:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                raw_name = (r.names.get(cls_id) or "unknown").lower()
                waste_name = YOLO_TO_WASTE.get(raw_name, raw_name.replace(" ", "_"))
                results.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "raw_class": raw_name,
                    "waste_name": waste_name,
                })

        return results

    # ------------------------------------------------------------------
    # OpenCV contour fallback
    # ------------------------------------------------------------------
    FALLBACK_CLASSES = [
        "plastic_bottle", "cardboard", "paper", "aluminum_can",
        "food_waste", "battery", "electronics", "glass_bottle",
    ]

    def _contour_detect(self, frame: np.ndarray) -> list[dict]:
        results = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        dilated = cv2.dilate(edges, None, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]
        min_area = w * h * 0.005  # 0.5% of frame

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            rx, ry, rw, rh = cv2.boundingRect(cnt)
            ar = rw / max(rh, 1)

            # Shape heuristics
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull) or 1
            solidity = area / hull_area

            if solidity > 0.9 and 0.3 < ar < 3.0:
                waste_name = "cardboard"
            elif ar > 0.4 and ar < 0.7:
                waste_name = "plastic_bottle"
            elif ar > 2.5:
                waste_name = "paper"
            else:
                waste_name = random.choice(self.FALLBACK_CLASSES)

            conf = min(0.75, area / (w * h * 0.4))

            results.append({
                "bbox": (rx, ry, rx + rw, ry + rh),
                "confidence": conf,
                "raw_class": "contour_object",
                "waste_name": waste_name,
            })

        # Sort by area descending (largest = most prominent)
        results.sort(key=lambda d: (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1]), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------
    def _enrich(self, det: dict, frame: np.ndarray) -> dict:
        x1, y1, x2, y2 = det["bbox"]
        bw = x2 - x1
        bh = y2 - y1
        h, w = frame.shape[:2]

        waste_name = det["waste_name"]
        science = self.get_science(waste_name)

        # Geometry
        area_px = bw * bh
        area_pct = round(100 * area_px / max(w * h, 1), 2)
        aspect_ratio = round(bw / max(bh, 1), 3)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Normalised centre (0-1 from top-left)
        norm_cx = round(cx / w, 3)
        norm_cy = round(cy / h, 3)
        # Estimated distance
        dist_m = _estimate_distance(waste_name, bw)

        # Radiation simulation
        rad_sim = self.simulate_radiation(science)

        det.update({
            "center":       (cx, cy),
            "width_px":     bw,
            "height_px":    bh,
            "area_px":      area_px,
            "area_pct":     area_pct,
            "aspect_ratio": aspect_ratio,
            "norm_center":  (norm_cx, norm_cy),
            "distance_m":   dist_m,
            "science":      science,
            "radiation_sim":rad_sim,
        })
        return det

    def get_science(self, waste_name: str) -> dict:
        if waste_name not in self._scientific_cache:
            self._scientific_cache[waste_name] = lookup(waste_name)
        return self._scientific_cache[waste_name]

    def simulate_radiation(self, science: dict) -> float:
        base = self.radiation_base
        rad = science.get("radiation_level", 0.0)
        # Add ±15% noise for realism
        jitter = random.uniform(0.85, 1.15)
        return round(base + rad * jitter * 10, 4)
