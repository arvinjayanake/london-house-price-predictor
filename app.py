from flask import Flask, render_template, request
from predictor import HousePricePredictor

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
    """Server-side validation. Returns (data_dict, errors_dict)."""
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
            price = predictor.predict(
                bathrooms=form_values["bathrooms"],
                bedrooms=form_values["bedrooms"],
                floor_area_sqm=form_values["floor_area"],
                living_rooms=form_values["living_rooms"],
                tenure_years=form_values["tenure"],
                property_type=form_values["property_type"],
                energy_rating=form_values["energy_rating"],
                postcode=form_values["postcode"],
                sale_year=form_values["sale_year"],
                latitude=form_values["latitude"],
                longitude=form_values["longitude"],
            )
            return render_template(
                "index.html",
                result=f"{price:,.2f}",
                values=form_values,
                errors={},
                PROPERTY_TYPES=PROPERTY_TYPES,
                ENERGY_RATINGS=ENERGY_RATINGS,
            )
        else:
            # Show errors + keep entered values
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

if __name__ == "__main__":
    app.run(debug=True)
