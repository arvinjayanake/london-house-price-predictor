import json
from flask import Flask, render_template, request, jsonify
from predictor import HousePricePredictor
import requests
from flask import url_for

app = Flask(__name__)

# Load once at startup
predictor = HousePricePredictor("scaler.pkl", "stacked_model.pkl")

PROPERTY_TYPES = [
    "Detached House",
    "Semi-Detached House",
    "Terraced House",
    "Flat",
    "Maisonette",
    "Bungalow",
]
ENERGY_RATINGS = list("ABCDEFG")

def validate(form):
    errors = {}
    data = {}

    # Simple helpers
    def as_float(name, label, min_v=None, max_v=None):
        val = request.form.get(name, "").strip()
        if val == "":
            errors[name] = f"{label} is required."
            return None
        try:
            x = float(val)
        except ValueError:
            errors[name] = f"{label} must be a number."
            return None
        if min_v is not None and x < min_v:
            errors[name] = f"{label} must be ≥ {min_v}."
        if max_v is not None and x > max_v:
            errors[name] = f"{label} must be ≤ {max_v}."
        return x

    def as_int(name, label, min_v=None, max_v=None):
        val = request.form.get(name, "").strip()
        if val == "":
            errors[name] = f"{label} is required."
            return None
        try:
            x = int(val)
        except ValueError:
            errors[name] = f"{label} must be an integer."
            return None
        if min_v is not None and x < min_v:
            errors[name] = f"{label} must be ≥ {min_v}."
        if max_v is not None and x > max_v:
            errors[name] = f"{label} must be ≤ {max_v}."
        return x

    def as_choice(name, label, choices):
        val = request.form.get(name, "").strip()
        if val not in choices:
            errors[name] = f"Select a valid {label}."
        return val

    def as_postcode(name, label):
        import re
        val = request.form.get(name, "").strip().upper()
        if not val:
            errors[name] = f"{label} is required."
            return None

        # postal code check (accepts formats like E1 3AD, SW1A 1AA, etc.)
        if not re.match(r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2}$", val):
            errors[name] = "Enter a valid UK postcode (e.g., E1 3AD)."
        return val

    # Collect + validate
    data["bathrooms"]   = as_int("bathrooms", "Bathrooms", 0, 10)
    data["bedrooms"]    = as_int("bedrooms", "Bedrooms", 0, 15)
    data["floor_area"]  = as_float("floor_area", "Floor area (sqm)", 5, 2000)
    data["living_rooms"]= as_int("living_rooms", "Living rooms", 0, 10)
    data["tenure"]      = as_int("tenure", "Tenure (years)", 0, 999)
    data["property_type"]= as_choice("property_type", "Property type", PROPERTY_TYPES)
    data["energy_rating"]= as_choice("energy_rating", "Energy rating", ENERGY_RATINGS)
    data["postcode"]    = as_postcode("postcode", "Postcode")
    data["sale_year"]   = as_int("sale_year", "Sale year", 1900, 2025)
    data["latitude"]    = as_float("latitude", "Latitude", 51.2, 51.8)     # London-ish bounds
    data["longitude"]   = as_float("longitude", "Longitude", -0.6, 0.3)   # London-ish bounds

    return data, {k: v for k, v in errors.items() if v}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_values, errors = validate(request.form)

        if not errors:
            # Build JSON payload for the API
            payload = {
                "bathrooms": form_values["bathrooms"],
                "bedrooms": form_values["bedrooms"],
                "floor_area_sqm": form_values["floor_area"],
                "living_rooms": form_values["living_rooms"],
                "tenure_years": form_values["tenure"],
                "property_type": form_values["property_type"],
                "energy_rating": form_values["energy_rating"],
                "postcode": form_values["postcode"],
                "sale_year": form_values["sale_year"],
                "latitude": form_values["latitude"],
                "longitude": form_values["longitude"],
            }

            try:
                # Call internal API (absolute URL required)
                api_url = url_for("api_predict", _external=True)
                resp = requests.post(api_url, json=payload, timeout=10)
                resp.raise_for_status()
                price = resp.json().get("prediction", None)

                print("Price:", price)

                if price is None:
                    errors = {"__all__": "Prediction API returned no price."}
                    price_text = None
                else:
                    price_text = f"{price:,.2f}"

            except requests.RequestException as e:
                errors = {"__all__": f"Prediction API error: {e}"}
                price_text = None

            return render_template(
                "index.html",
                result=price_text,
                values=form_values,
                errors=errors if price_text is None else {},
                PROPERTY_TYPES=PROPERTY_TYPES,
                ENERGY_RATINGS=ENERGY_RATINGS,
            )

        # Validation errors — re-render form
        return render_template(
            "index.html",
            result=None,
            values=request.form,
            errors=errors,
            PROPERTY_TYPES=PROPERTY_TYPES,
            ENERGY_RATINGS=ENERGY_RATINGS,
        )

    # GET
    return render_template(
        "index.html",
        result=None,
        values={},
        errors={},
        PROPERTY_TYPES=PROPERTY_TYPES,
        ENERGY_RATINGS=ENERGY_RATINGS,
    )

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True, silent=False)

        print("API << ", data)

        PROPERTY_TYPES = [
            "Detached House", "Semi-Detached House", "Terraced House",
            "Flat", "Maisonette", "Bungalow"
        ]
        ENERGY_RATINGS = list("ABCDEFG")

        bathrooms       = _as_number(data, "bathrooms", int, 0, 10)
        bedrooms        = _as_number(data, "bedrooms", int, 0, 15)
        floor_area_sqm  = _as_number(data, "floor_area_sqm", float, 5, 2000)
        living_rooms    = _as_number(data, "living_rooms", int, 0, 10)
        tenure_years    = _as_number(data, "tenure_years", int, 0, 999)
        property_type   = _as_choice(data, "property_type", PROPERTY_TYPES)
        energy_rating   = _as_choice(data, "energy_rating", ENERGY_RATINGS)
        postcode        = _as_postcode(data, "postcode")
        sale_year       = _as_number(data, "sale_year", int, 1900, 2100)
        latitude        = _as_number(data, "latitude", float,  -90,  90)
        longitude       = _as_number(data, "longitude", float, -180, 180)

        price = predictor.predict(
            bathrooms=bathrooms,
            bedrooms=bedrooms,
            floor_area_sqm=floor_area_sqm,
            living_rooms=living_rooms,
            tenure_years=tenure_years,
            property_type=property_type,
            energy_rating=energy_rating,
            postcode=postcode,
            sale_year=sale_year,
            latitude=latitude,
            longitude=longitude,
        )
        return jsonify({"ok": True, "prediction": price}), 200

    except ValueError as ve:
        return jsonify({"ok": False, "error": str(ve)}), 400
    except Exception as e:
        # Avoid leaking internals; log e if you have logging
        return jsonify({"ok": False, "error": "Internal error"}), 500

def _as_postcode(d, key):
    import re
    val = d.get(key, "")
    if not isinstance(val, str) or not re.match(r"^[A-Za-z]{1,2}\d{1,2}[A-Za-z]?\s*\d[A-Za-z]{2}$", val.strip(), re.I):
        raise ValueError(f"{key} must be a valid UK postcode like 'E1 3AD'")
    return val.strip().upper()

def _as_choice(d, key, choices):
    val = d.get(key)
    if val not in choices:
        raise ValueError(f"{key} must be one of {choices}")
    return val

def _as_number(d, key, kind=float, min_v=None, max_v=None, required=True):
    if key not in d or d[key] in ("", None):
        if required:
            raise ValueError(f"{key} is required")
        return None
    try:
        val = kind(d[key])
    except Exception:
        raise ValueError(f"{key} must be {kind.__name__}")
    if min_v is not None and val < min_v:
        raise ValueError(f"{key} must be ≥ {min_v}")
    if max_v is not None and val > max_v:
        raise ValueError(f"{key} must be ≤ {max_v}")
    return val

if __name__ == "__main__":
    app.run(debug=True)
