# London House Price Predictor

A simple Flask web app and API that predicts London house prices using a stacked machine‑learning model. The model and scaler are pre‑packaged (`stacked_model.pkl`, `scaler.pkl`) for immediate use with a clean HTML form and basic styling.

> Competition context: built for the **Kaggle: London House Price Prediction – Advanced Techniques** challenge.

---
## Screenshot of the Solution

Below is a screenshot showing the **London House Price Predictor** web interface in action.

![London House Price Predictor Screenshot](https://raw.githubusercontent.com/arvinjayanake/london-house-price-predictor/refs/heads/main/static/london_house_price_predictor_screenshot.png)

> The image above shows the form filled with sample values and the predicted price displayed after clicking **Predict**.

## Features

- Predicts house prices for London properties
- Stacked ensemble model (trees + linear blender)
- Web UI for manual input (HTML + CSS)
- JSON API endpoint for programmatic access
- Ready‑to‑run with included `*.pkl` artefacts

---

## Tech Stack

- Python, Flask
- scikit‑learn (stacking and preprocessing)
- (Model trained externally with tree‑based learners)

---

## Repository Structure
```text
├─ app.py
├─ predictor.py
├─ templates/
│ └─ index.html
├─ static/
│ └─ styles.css
├─ scaler.pkl
└─ stacked_model.pkl
```
---
## Features

- **Web UI**: Enter property details through a user-friendly form and get predictions instantly.
- **REST API**: Send JSON payloads to `/api/predict` for programmatic access.
- **Machine Learning Backend**:  
  - Stacking Regressor using XGBoost, LightGBM, and CatBoost with a Linear Regression meta-learner.
  - Feature scaling with Scikit-Learn’s `StandardScaler`.
- **Feature Engineering**: Includes room totals, property age, location-based distance metrics, and postcode feature extraction.
- **Validation**: Ensures all inputs are valid before making predictions.
---
## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/arvinjayanake/london-house-price-predictor.git
cd london-house-price-predictor
```
### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```
### 3. Install dependencies
```bash
pip install flask pandas numpy scikit-learn joblib xgboost lightgbm catboost
```
### 4. Run Application
```bash
python app.py
```
#### By default:
###### Web UI: http://127.0.0.1:5000/
###### API Endpoint: http://127.0.0.1:5000/api/predict

----

## Web Interface

Open your browser and go to **http://127.0.0.1:5000/**.

### Steps
1. Fill in the form fields listed below.
2. Click **Predict**.
3. The page shows the **estimated price in GBP**. If any input is invalid, the page shows an error message to fix.

### Form Fields (what to enter)

| Field                | Type    | Example     | Notes (typical validation)                          |
|---------------------|---------|-------------|-----------------------------------------------------|
| Bathrooms           | Integer | `2`         | 0–10                                               |
| Bedrooms            | Integer | `3`         | 0–15                                               |
| Floor area (sqm)    | Float   | `85.5`      | 5–2000                                             |
| Living rooms        | Integer | `1`         | 0–10                                               |
| Tenure (years)      | Integer | `99`        | 0–999                                              |
| Property type       | Select  | `Flat`      | One of: Detached House, Semi-Detached House, Terraced House, Flat, Maisonette, Bungalow |
| Energy rating       | Select  | `C`         | One of: A, B, C, D, E, F, G                        |
| Postcode            | Text    | `E1 3AD`    | Valid UK postcode format                           |
| Sale year           | Integer | `2024`      | 1900–current year                                  |
| Latitude            | Float   | `51.515`    | London bounds ~ 51.2–51.8                          |
| Longitude           | Float   | `-0.072`    | London bounds ~ −0.6–0.3                           |

> Tip: Use a real London postcode with matching latitude/longitude for best results.

### What happens after submit
- The app validates your inputs.
- It converts the form to model features, applies the saved **scaler**, and runs the **stacked model**.
- It then renders a card with the **predicted price** (e.g., `£523,411.27`).

### Common issues
- **Field out of range** → adjust the number to the allowed range.
- **Postcode format** → enter a valid UK format like `E1 3AD`.
- **Blank fields** → all fields are required for a stable prediction.
---
## API Usage

### Endpoint
```bash
POST /api/predict
Content-Type: application/json
```
### Example Request (JSON)
```json
{
  "bathrooms": 2,
  "bedrooms": 3,
  "floor_area_sqm": 120.5,
  "living_rooms": 1,
  "tenure_years": 99,
  "property_type": "Detached House",
  "energy_rating": "B",
  "postcode": "E1 3AD",
  "sale_year": 2024,
  "latitude": 51.515,
  "longitude": -0.072
}
```
### Example Response (200 OK)
```json
{
  "ok": true,
  "prediction": 725000.50
}
```
### Error Responses
400 Bad Request (validation error)
```json
{
  "ok": false,
  "error": "sale_year must be between 1900 and the current year"
}
```
### Error Responses
500 Internal Server Error (unexpected server error)
```json
{
  "ok": false,
  "error": "Internal error"
}
```
HTTP Status Codes

```200``` – success, returns prediction

```400``` – invalid or missing input

```500``` – server-side error during prediction

Notes

All fields are required. Use valid ranges and allowed values.

Content-Type must be application/json.