import pandas as pd
import numpy as np
import re
from datetime import datetime
from joblib import load


class HousePricePredictor:
    def __init__(self, scaler_path, model_path):
        self.scaler = load(scaler_path)
        self.model = load(model_path)
        self.feature_names = list(getattr(self.scaler, "feature_names_in_", []))
        if not self.feature_names:
            raise RuntimeError("Scaler missing feature names. Re-export after fitting on DataFrame.")

        self.landmarks = {
            "centre": (51.5074, -0.1278),
            "city": (51.5155, -0.0922),
            "canary_wharf": (51.5054, -0.0235),
            "heathrow": (51.4700, -0.4543),
            "chelsea": (51.4869, -0.1700),
            "mayfair": (51.5116, -0.1478),
            "knightsbridge": (51.4991, -0.1644),
        }

    def haversine_km(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        return 6371.0 * 2 * np.arcsin(np.sqrt(a))

    def _set_num(self, row, name, value):
        if name in row.index and value is not None:
            row[name] = float(value)

    def _set_one_hot(self, row, prefix, value):
        if value is None:
            return
        col = f"{prefix}_{value}"
        if col in row.index:
            row[col] = 1.0

    def predict(self, bathrooms, bedrooms, floor_area_sqm, living_rooms, tenure_years,
                property_type, energy_rating, postcode, sale_year, latitude, longitude):

        now = datetime.now()
        total_rooms = bedrooms + living_rooms
        property_age = now.year - sale_year

        m_area = re.match(r"^([A-Za-z]{1,2})", postcode)
        postcode_area = m_area.group(1) if m_area else None

        m_dist = re.match(r"^[A-Za-z]{1,2}(\d{1,2})", postcode)
        postcode_district = m_dist.group(1) if m_dist else None

        row = pd.Series(0.0, index=self.feature_names, dtype="float64")

        self._set_num(row, "bathrooms", bathrooms)
        self._set_num(row, "bedrooms", bedrooms)
        self._set_num(row, "floorAreaSqM", floor_area_sqm)
        self._set_num(row, "livingRooms", living_rooms)
        self._set_num(row, "tenure", tenure_years)
        self._set_num(row, "total_rooms", total_rooms)
        self._set_num(row, "property_age", property_age)
        self._set_num(row, "latitude", latitude)
        self._set_num(row, "longitude", longitude)

        for name, (la, lo) in self.landmarks.items():
            col = f"dist_to_{name}"
            if col in row.index:
                row[col] = self.haversine_km(latitude, longitude, la, lo)

        self._set_one_hot(row, "propertyType", property_type)
        self._set_one_hot(row, "currentEnergyRating", energy_rating)
        self._set_one_hot(row, "postcode_area", postcode_area)
        self._set_one_hot(row, "postcode_district", postcode_district)

        X = pd.DataFrame([row.values], columns=self.feature_names)
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)
        return float(prediction[0])
