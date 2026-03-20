# =============================================================================
# api/disease_db.py — Disease Database with hard-coded mapping and JSON caching
# =============================================================================

import csv
import json
import os
import difflib
from typing import Optional

from utils import normalise_class_name, format_prevention_list, format_cure_list


# ── Hard-coded CSV disease name → folder name mapping ─────────────────────────
# These handle all edge cases (parentheses, commas, underscores) that fuzzy
# matching might get wrong. Every one of the 38 classes is explicitly mapped.

CSV_TO_FOLDER = {
    "Apple scab":                                    "Apple___Apple_scab",
    "Apple Black rot":                               "Apple___Black_rot",
    "Apple Cedar apple rust":                        "Apple___Cedar_apple_rust",
    "Apple healthy":                                 "Apple___healthy",
    "Blueberry healthy":                             "Blueberry___healthy",
    "Cherry Powdery mildew":                         "Cherry_(including_sour)___Powdery_mildew",
    "Cherry healthy":                                "Cherry_(including_sour)___healthy",
    "Corn Cercospora leaf spot Gray leaf spot":      "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn Common rust":                              "Corn_(maize)___Common_rust_",
    "Corn Northern Leaf Blight":                     "Corn_(maize)___Northern_Leaf_Blight",
    "Corn healthy":                                  "Corn_(maize)___healthy",
    "Grape Black rot":                               "Grape___Black_rot",
    "Grape Esca (Black Measles)":                    "Grape___Esca_(Black_Measles)",
    "Grape Leaf blight (Isariopsis Leaf Spot)":      "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape healthy":                                 "Grape___healthy",
    "Orange Haunglongbing (Citrus greening)":        "Orange___Haunglongbing_(Citrus_greening)",
    "Peach Bacterial spot":                          "Peach___Bacterial_spot",
    "Peach healthy":                                 "Peach___healthy",
    "Pepper bell Bacterial spot":                    "Pepper,_bell___Bacterial_spot",
    "Pepper bell healthy":                           "Pepper,_bell___healthy",
    "Potato Early blight":                           "Potato___Early_blight",
    "Potato Late blight":                            "Potato___Late_blight",
    "Potato healthy":                                "Potato___healthy",
    "Raspberry healthy":                             "Raspberry___healthy",
    "Soybean healthy":                               "Soybean___healthy",
    "Squash Powdery mildew":                         "Squash___Powdery_mildew",
    "Strawberry Leaf scorch":                        "Strawberry___Leaf_scorch",
    "Strawberry healthy":                            "Strawberry___healthy",
    "Tomato Bacterial spot":                         "Tomato___Bacterial_spot",
    "Tomato Early blight":                           "Tomato___Early_blight",
    "Tomato Late blight":                            "Tomato___Late_blight",
    "Tomato Leaf Mold":                              "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot":                     "Tomato___Septoria_leaf_spot",
    "Tomato Spider mites Two-spotted spider mite":   "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato Target Spot":                            "Tomato___Target_Spot",
    "Tomato Yellow Leaf Curl Virus":                 "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mosaic virus":                           "Tomato___Tomato_mosaic_virus",
    "Tomato healthy":                                "Tomato___healthy",
}

# Build reverse mapping: folder_name → csv_name
FOLDER_TO_CSV = {v: k for k, v in CSV_TO_FOLDER.items()}


class DiseaseDatabase:
    """
    Loads disease information from a CSV file and provides lookup
    by class folder name using a hard-coded mapping table.

    Falls back to fuzzy matching for any class not in the mapping.
    Caches the parsed data as JSON for fast subsequent loads.

    Usage:
        db = DiseaseDatabase("path/to/csv", "path/to/cache.json")
        info = db.get("Tomato___Early_blight")
        all_diseases = db.all()
    """

    def __init__(self, csv_path: str, cache_json_path: str):
        self.csv_path = csv_path
        self.cache_path = cache_json_path
        self._db: dict = {}              # normalised_csv_name → entry
        self._folder_db: dict = {}       # folder_name → entry
        self._raw_entries: list = []
        self._load()

    def _load(self):
        """Load from cache (if fresh) or parse CSV."""
        if self._try_load_cache():
            return
        self._parse_csv()
        self._save_cache()

    def _try_load_cache(self) -> bool:
        """Attempt to load from JSON cache. Returns True on success."""
        if not os.path.isfile(self.cache_path):
            return False
        if os.path.isfile(self.csv_path):
            csv_mtime = os.path.getmtime(self.csv_path)
            cache_mtime = os.path.getmtime(self.cache_path)
            if csv_mtime > cache_mtime:
                return False

        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._db = data.get("lookup", {})
            self._folder_db = data.get("folder_lookup", {})
            self._raw_entries = data.get("entries", [])
            print(f"[DiseaseDB] Loaded {len(self._raw_entries)} entries from cache")
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def _parse_csv(self):
        """Parse the CSV file and build lookup dictionaries."""
        if not os.path.isfile(self.csv_path):
            print(f"[WARNING] Disease CSV not found: {self.csv_path}")
            return

        self._db = {}
        self._folder_db = {}
        self._raw_entries = []

        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Disease_Name", "").strip()
                if not name:
                    continue

                entry = {
                    "disease_name": name,
                    "description": row.get("Disease_Description", "").strip(),
                    "prevention": format_prevention_list(
                        row.get("Prevention_Methods", "")),
                    "cure": format_cure_list(row.get("Cure_Methods", "")),
                }
                self._raw_entries.append(entry)

                # Index by normalised CSV name
                key = normalise_class_name(name)
                self._db[key] = entry

                # Index by folder name using hard-coded mapping
                folder = CSV_TO_FOLDER.get(name)
                if folder:
                    self._folder_db[folder] = entry

        print(f"[DiseaseDB] Parsed {len(self._raw_entries)} entries from CSV")
        print(f"[DiseaseDB] Hard-coded folder mappings: {len(self._folder_db)}/38")

    def _save_cache(self):
        """Write the parsed data to JSON cache."""
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump({
                "lookup": self._db,
                "folder_lookup": self._folder_db,
                "entries": self._raw_entries,
            }, f, indent=2, ensure_ascii=False)
        print(f"[DiseaseDB] Cached to {self.cache_path}")

    def get(self, class_folder_name: str) -> dict:
        """
        Look up disease info by class folder name.

        Matching strategy (in priority order):
            1. Hard-coded folder → CSV mapping (guaranteed correct for all 38)
            2. Exact match after normalisation
            3. Substring containment
            4. difflib.get_close_matches fallback
            5. Graceful fallback for healthy/unknown classes
        """
        # 1. Hard-coded folder mapping (most reliable)
        if class_folder_name in self._folder_db:
            return self._folder_db[class_folder_name]

        normalised = normalise_class_name(class_folder_name)

        # 2. Exact match on normalised name
        if normalised in self._db:
            return self._db[normalised]

        all_keys = list(self._db.keys())

        # 3. Substring containment
        for key in all_keys:
            if normalised in key or key in normalised:
                return self._db[key]

        # 4. difflib fuzzy match
        matches = difflib.get_close_matches(normalised, all_keys, n=1, cutoff=0.5)
        if matches:
            return self._db[matches[0]]

        # 5. Healthy fallback
        if "healthy" in normalised:
            plant = normalised.split("healthy")[0].strip()
            return {
                "disease_name": f"{plant.title()} Healthy",
                "description": "Plant appears healthy with no disease symptoms.",
                "prevention": ["Maintain good sanitation and balanced nutrition."],
                "cure": ["No treatment needed."],
            }

        # Unknown
        return {
            "disease_name": class_folder_name.replace("___", " ").replace("_", " "),
            "description": "No information available for this condition.",
            "prevention": [],
            "cure": [],
        }

    def all(self) -> list:
        """Return all disease records."""
        return list(self._raw_entries)

    def search(self, query: str) -> list:
        """
        Search for diseases matching a query string.
        Returns matching disease records.
        """
        query_norm = normalise_class_name(query)
        results = []
        for entry in self._raw_entries:
            name_norm = normalise_class_name(entry["disease_name"])
            if query_norm in name_norm or name_norm in query_norm:
                results.append(entry)

        # If no substring matches, try fuzzy
        if not results:
            all_names = [normalise_class_name(e["disease_name"])
                         for e in self._raw_entries]
            matches = difflib.get_close_matches(query_norm, all_names,
                                                 n=5, cutoff=0.4)
            for match in matches:
                for entry in self._raw_entries:
                    if normalise_class_name(entry["disease_name"]) == match:
                        results.append(entry)
                        break

        return results
