"""
core/database.py
Persistent SQLite database for waste records, detection logs, and session statistics.
Thread-safe via a single connection and mutex-style commit.
"""

import sqlite3
import datetime
import os
from threading import Lock

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "waste_detection.db")


class WasteDatabase:
    def __init__(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self._lock = Lock()
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._setup()
        self._seed()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def _setup(self):
        with self._lock:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS waste_types (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    name                TEXT UNIQUE NOT NULL,
                    category            TEXT,
                    biodegradability    REAL DEFAULT 0,
                    toxicity            REAL DEFAULT 0,
                    radiation_level     REAL DEFAULT 0,
                    recycling_score     REAL DEFAULT 0,
                    decomposition_years REAL DEFAULT 0,
                    environmental_impact TEXT,
                    health_risk         TEXT,
                    scientific_notes    TEXT,
                    disposal_method     TEXT,
                    resin_code          TEXT,
                    carbon_footprint_kg REAL DEFAULT 0,
                    un_hazard_class     TEXT
                );

                CREATE TABLE IF NOT EXISTS detection_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT,
                    timestamp       TEXT,
                    object_name     TEXT,
                    raw_class       TEXT,
                    confidence      REAL,
                    category        TEXT,
                    radiation_level REAL,
                    toxicity        REAL,
                    bbox_x1         INTEGER,
                    bbox_y1         INTEGER,
                    bbox_x2         INTEGER,
                    bbox_y2         INTEGER,
                    area_px         INTEGER,
                    frame_width     INTEGER,
                    frame_height    INTEGER
                );

                CREATE TABLE IF NOT EXISTS session_stats (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT UNIQUE,
                    start_time      TEXT,
                    end_time        TEXT,
                    total_frames    INTEGER DEFAULT 0,
                    total_detections INTEGER DEFAULT 0,
                    hazardous_count INTEGER DEFAULT 0,
                    recyclable_count INTEGER DEFAULT 0,
                    avg_fps         REAL DEFAULT 0
                );
            """)
            self.conn.commit()

    # ------------------------------------------------------------------
    # Seed scientific data
    # ------------------------------------------------------------------
    def _seed(self):
        rows = [
            # name, category, bio, tox, rad, rec, decomp, env_impact, health_risk, sci_notes, disposal, resin, co2_kg, un_class
            ("plastic_bottle",  "Plastic",   0.01, 0.70, 0.00, 0.80,  450,
             "PET bottles persist 450+ years. Fragmentation releases microplastics into soil and oceans.",
             "BPA and phthalates act as endocrine disruptors. Linked to hormonal imbalance.",
             "Global PET production: ~70 Mt/yr. Recycling rate ~30%. Downcycled to polyester fibers.",
             "Recycle stream #1 (PET). Rinse before disposal. Avoid single-use where possible.",
             "#1 PET", 3.4, "Not regulated"),

            ("plastic_bag",     "Plastic",   0.005,0.60, 0.00, 0.30,  500,
             "Thin-film LDPE breaks into microplastics. Entangles marine megafauna.",
             "Ingested by seabirds and turtles. Airway obstruction in wildlife.",
             "Less than 5% of plastic bags are recycled globally. Banned in 60+ countries.",
             "Return to store drop-off bins. Switch to reusable fabric bags.",
             "#4 LDPE", 1.6, "Not regulated"),

            ("aluminum_can",    "Metal",     0.00, 0.10, 0.00, 0.95,  200,
             "Mining bauxite destroys ecosystems. Smelting releases fluoride emissions.",
             "Minimal health risk post-consumer. Aluminum dust hazardous in manufacturing.",
             "Recycling saves 95% of production energy. Infinite quality retention.",
             "Crush, rinse, place in metal recycling bin.",
             "N/A", 0.9, "Not regulated"),

            ("glass_bottle",    "Glass",     0.00, 0.00, 0.00, 0.90, 4000,
             "Inert in landfills but occupies volume. Glass production emits ~0.9 kg CO2/kg.",
             "Sharp edges cause lacerations. Chemically inert post-consumer.",
             "100% recyclable. Closed-loop recycling commonplace. Made from SiO2, Na2CO3, CaCO3.",
             "Separate by color (clear/green/brown). Recycle in glass bins.",
             "N/A", 0.9, "Not regulated"),

            ("paper",           "Paper",     0.80, 0.05, 0.00, 0.70,    0.08,
             "Deforestation pressure. Paper mills are top industrial water consumers.",
             "Inks may contain isopropanol VOCs. Generally low consumer risk.",
             "Recycling 1 tonne saves 17 trees, 26,000 L water, 4,000 kWh energy.",
             "Recycle dry, clean paper. Compost soiled paper. Avoid laminated stock.",
             "N/A", 1.1, "Not regulated"),

            ("cardboard",       "Cardboard", 0.75, 0.05, 0.00, 0.85,    0.08,
             "High fiber content. Corrugated boxes widely recycled in most municipalities.",
             "Low health risk. Staples and tape must be removed for clean recycling.",
             "Up to 80% recycled content in new boxes. 7-9 recycling cycles possible.",
             "Flatten and bundle. Keep dry. Remove tape, foam, and staples.",
             "N/A", 0.9, "Not regulated"),

            ("food_waste",      "Organic",   0.95, 0.20, 0.00, 0.40,    0.05,
             "Landfill anaerobic decomposition produces CH4 — 28x GWP of CO2 over 100 yrs.",
             "Attracts vectors (rodents, insects). Mold produces mycotoxins.",
             "Food waste = 8% of global GHG. Home composting reduces landfill methane.",
             "Compost at home or use municipal green-waste collection.",
             "N/A", 2.5, "Not regulated"),

            ("battery",         "Hazardous", 0.00, 0.95, 0.30, 0.20,  100,
             "Li-ion thermal runaway causes landfill fires. Pb, Cd, Hg leach into groundwater.",
             "Lead: neurological damage. Cadmium: kidney carcinogen. Lithium: caustic burns.",
             "Li-ion: Co sourced from DRC conflict mines. Only 5% of Li-ion recycled globally.",
             "NEVER in regular trash. Drop off at designated battery collection points.",
             "N/A", 12.5, "UN3480 / Class 9"),

            ("electronics",     "E-Waste",   0.00, 0.85, 0.10, 0.60, 1000,
             "Contains Au, Ag, Cu, but also Pb, As, Hg, Cd. Informal recycling burns toxins.",
             "Pb solder causes neurological harm. Flame retardants are persistent organic pollutants.",
             "53.6 Mt e-waste generated in 2019. Only 17.4% formally recycled. Value: $57B.",
             "Take to certified e-waste facility (R2/e-Stewards certified). Wipe data.",
             "N/A", 300.0, "Class 9 (certain components)"),

            ("rubber_tire",     "Rubber",    0.01, 0.40, 0.00, 0.50, 2000,
             "Tire fires burn for years, release dioxins and furans. Mosquito breeding sites.",
             "Combustion releases benzene, toluene, and polycyclic aromatic hydrocarbons.",
             "Tire-derived fuel, playground rubber, asphalt crumb rubber are recovery pathways.",
             "Return to tire retailer or municipal tire collection event.",
             "N/A", 7.0, "Not regulated"),

            ("medical_waste",   "Biohazard", 0.10, 0.95, 0.20, 0.10,  100,
             "Infectious agents, sharps, cytotoxic drugs, radioactive isotopes in clinical waste.",
             "Needlestick injury transmits HIV, HBV, HCV. Cytotoxic drugs are carcinogenic.",
             "WHO: 85% of clinical waste is non-hazardous; 15% is infectious or hazardous.",
             "Segregated yellow sharps containers. High-temperature incineration (>850 C).",
             "N/A", 0.0, "Class 6.2 Infectious"),

            ("radioactive_waste","Nuclear",  0.00, 1.00, 0.95, 0.00, 100000,
             "High-level waste remains hazardous >10,000 years. Deep geological repositories required.",
             "Acute: ARS, death. Chronic: cancer, leukemia, hereditary effects per ICRP models.",
             "Fission products Cs-137 (T1/2=30yr), Sr-90 (T1/2=29yr) dominate medium-term hazard.",
             "Encapsulate in borosilicate glass. Store in licensed deep geological disposal facility.",
             "N/A", 0.0, "Class 7 Radioactive"),

            ("styrofoam",       "Plastic",   0.001,0.50, 0.00, 0.10,  500,
             "EPS does not biodegrade. Breaks into persistent white beads. Marine ingestion risk.",
             "Styrene monomer — IARC Group 2A possible carcinogen. Leaches into hot food.",
             "Only ~10% of EPS is recycled due to low density economics. Banned in 50+ US cities.",
             "Avoid use. Drop off at specialty EPS compactors. Never place in curbside recycling.",
             "#6 PS", 3.0, "Not regulated"),

            ("textile",         "Textile",   0.30, 0.20, 0.00, 0.40,   40,
             "Fast fashion = 92 Mt textile waste/yr. Synthetic fabrics shed 500,000 t microplastics/yr.",
             "Azo dyes, formaldehyde finishes can cause dermatitis and respiratory issues.",
             "Only 12-15% of textiles are recycled globally. Most ends in landfill or incinerated.",
             "Donate wearable items. Use textile-specific drop boxes. Avoid landfill.",
             "N/A", 15.0, "Not regulated"),

            ("organic_waste",   "Organic",   0.90, 0.10, 0.00, 0.50,    0.05,
             "Garden and yard waste; rapid anaerobic decomposition in landfills emits CH4.",
             "Generally safe. May contain pesticide residues on treated plant matter.",
             "Composting reduces volume 50-70%. Vermicomposting produces premium soil amendment.",
             "Compost on-site or use green-waste collection bin.",
             "N/A", 0.5, "Not regulated"),

            ("paint_can",       "Hazardous", 0.00, 0.80, 0.00, 0.15,  200,
             "VOC emissions contribute to ground-level ozone and smog formation.",
             "Solvents: neurotoxic. Lead-based paint: severe neurological damage in children.",
             "Water-based latex: lower VOC. Alkyd/oil-based: high VOC. Lead phased out 1978 (US).",
             "Dry out latex paint before disposal. Oil-based: hazardous waste facility only.",
             "N/A", 2.8, "Class 3 Flammable (oil-based)"),

            ("aerosol_can",     "Metal",     0.00, 0.50, 0.00, 0.60,  200,
             "Propellant VOCs degrade air quality. Some propellants are greenhouse gases.",
             "Flammable propellants (butane/propane) pose explosion risk if punctured.",
             "Steel or aluminum shell is 100% recyclable after full discharge.",
             "Fully empty the can. Recycle the metal shell. Never puncture or incinerate.",
             "N/A", 1.5, "Class 2.1 Flammable Gas"),

            ("motor_oil",       "Hazardous", 0.00, 0.85, 0.05, 0.30,  200,
             "One litre of used oil contaminates 1 million litres of groundwater.",
             "Polyaromatic hydrocarbons (PAHs) are carcinogenic. Neurotoxic heavy metals.",
             "Re-refining used oil produces base oil equivalent to virgin; saves 85% energy.",
             "Take to automotive retailer or municipal HHW collection event. Never pour down drains.",
             "N/A", 4.0, "Not regulated (but controlled)"),
        ]

        with self._lock:
            self.conn.executemany("""
                INSERT OR IGNORE INTO waste_types
                (name, category, biodegradability, toxicity, radiation_level, recycling_score,
                 decomposition_years, environmental_impact, health_risk, scientific_notes,
                 disposal_method, resin_code, carbon_footprint_kg, un_hazard_class)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            self.conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def get_waste_info(self, name: str):
        row = self.conn.execute(
            "SELECT * FROM waste_types WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None

    def get_all_waste_types(self):
        return self.conn.execute("SELECT name, category FROM waste_types ORDER BY category, name").fetchall()

    def log_detection(self, session_id, obj_name, raw_class, confidence, category,
                      radiation, toxicity, bbox=(0,0,0,0), frame_size=(0,0)):
        x1, y1, x2, y2 = bbox
        fw, fh = frame_size
        area = (x2 - x1) * (y2 - y1)
        ts = datetime.datetime.now().isoformat()
        with self._lock:
            self.conn.execute("""
                INSERT INTO detection_log
                (session_id, timestamp, object_name, raw_class, confidence, category,
                 radiation_level, toxicity, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                 area_px, frame_width, frame_height)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (session_id, ts, obj_name, raw_class, confidence, category,
                  radiation, toxicity, x1, y1, x2, y2, area, fw, fh))
            self.conn.commit()

    def get_detection_history(self, limit=100):
        return self.conn.execute(
            "SELECT * FROM detection_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def get_session_summary(self, session_id):
        return self.conn.execute(
            "SELECT * FROM session_stats WHERE session_id = ?", (session_id,)
        ).fetchone()

    def upsert_session(self, session_id, start_time, **kwargs):
        existing = self.get_session_summary(session_id)
        if not existing:
            with self._lock:
                self.conn.execute(
                    "INSERT OR IGNORE INTO session_stats (session_id, start_time) VALUES (?,?)",
                    (session_id, start_time)
                )
                self.conn.commit()
        if kwargs:
            sets = ", ".join(f"{k} = ?" for k in kwargs)
            vals = list(kwargs.values()) + [session_id]
            with self._lock:
                self.conn.execute(
                    f"UPDATE session_stats SET {sets} WHERE session_id = ?", vals
                )
                self.conn.commit()

    def get_category_distribution(self, session_id=None):
        if session_id:
            rows = self.conn.execute(
                "SELECT category, COUNT(*) as cnt FROM detection_log WHERE session_id=? GROUP BY category",
                (session_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT category, COUNT(*) as cnt FROM detection_log GROUP BY category"
            ).fetchall()
        return {r["category"]: r["cnt"] for r in rows}

    def get_recent_toxicity_timeline(self, session_id, limit=60):
        rows = self.conn.execute(
            "SELECT timestamp, toxicity, radiation_level FROM detection_log "
            "WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
        return list(reversed(rows))

    def close(self):
        self.conn.close()
