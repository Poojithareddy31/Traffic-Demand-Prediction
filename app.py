import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file

app = Flask(__name__)


class StackingEnsembleRegressor:
    def __init__(self, lgb_model, xgb_model, meta_model):
        self.lgb_model  = lgb_model
        self.xgb_model  = xgb_model
        self.meta_model = meta_model

    def predict(self, X):
        lgb_preds = self.lgb_model.predict(X).reshape(-1, 1)
        xgb_preds = self.xgb_model.predict(X).reshape(-1, 1)
        return self.meta_model.predict(np.hstack([lgb_preds, xgb_preds]))


PREP_PATH     = "preprocessing_artifacts.pkl"
LGB_PATH      = "best_lgb_model.pkl"
XGB_PATH      = "best_xgb_model.pkl"
META_PATH     = "best_meta_model.pkl"
ENSEMBLE_PATH = "best_stacking_ensemble.pkl"

models_loaded  = False
scaler         = None
feature_names  = []
cat_cols       = []
dummy_columns  = []
num_cols       = []
ensemble_model = None

try:
    if os.path.exists(PREP_PATH):
        prep          = joblib.load(PREP_PATH)
        scaler        = prep["scaler"]
        feature_names = prep["feature_names"]
        cat_cols      = prep["cat_cols"]
        dummy_columns = prep["dummy_columns"]
        num_cols      = prep["num_cols"]
    if os.path.exists(ENSEMBLE_PATH):
        ensemble_model = joblib.load(ENSEMBLE_PATH)
    models_loaded = True
    print("[OK] Models loaded successfully.")
except Exception as e:
    print(f"[ERROR] Model loading failed: {e}")

DATA_PATH    = "datasets/traffic_demand_dataset.csv"
history_df   = None
segment_info = {}

if os.path.exists(DATA_PATH):
    try:
        full_df = pd.read_csv(DATA_PATH)
        full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
        full_df = full_df.sort_values(["road_segment_id", "timestamp"])
        history_df = full_df.groupby("road_segment_id").tail(100).copy()

        for _, row in full_df.drop_duplicates("road_segment_id").iterrows():
            segment_info[row["road_segment_id"]] = {
                "road_segment_id":   row["road_segment_id"],
                "road_type":         row["road_type"],
                "number_of_lanes":   int(row["number_of_lanes"]),
                "speed_limit":       int(row["speed_limit"]),
                "latitude":          float(row["latitude"]),
                "longitude":         float(row["longitude"]),
                "nearby_intersections": int(row["nearby_intersections"]),
                "nearby_poi_density":   float(row["nearby_poi_density"]),
                "traffic_signals":   int(row["traffic_signals"]),
                "historical_avg_speed": float(row["historical_avg_speed"]),
                "spatial_density":   float(row["spatial_density"]),
            }
        print(f"[OK] Segment history cached ({len(history_df)} records).")
    except Exception as e:
        print(f"[ERROR] Dataset loading failed: {e}")


def build_inference_row(data, meta):
    ts     = pd.to_datetime(data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    seg_id = data.get("road_segment_id", "SEG_001")

    row = {
        "road_segment_id":      seg_id,
        "road_type":            meta["road_type"],
        "number_of_lanes":      meta["number_of_lanes"],
        "speed_limit":          meta["speed_limit"],
        "latitude":             meta["latitude"],
        "longitude":            meta["longitude"],
        "nearby_intersections": meta["nearby_intersections"],
        "nearby_poi_density":   meta["nearby_poi_density"],
        "traffic_signals":      meta["traffic_signals"],
        "historical_avg_speed": meta["historical_avg_speed"],
        "spatial_density":      meta["spatial_density"],
        "hour":         ts.hour,
        "minute":       ts.minute,
        "day_of_week":  ts.dayofweek,
        "day":          ts.day,
        "month":        ts.month,
        "year":         ts.year,
        "is_weekend":   1 if ts.dayofweek >= 5 else 0,
        "week_of_year": int(ts.isocalendar()[1]),
        "hour_sin":     np.sin(2 * np.pi * ts.hour / 24),
        "hour_cos":     np.cos(2 * np.pi * ts.hour / 24),
        "dow_sin":      np.sin(2 * np.pi * ts.dayofweek / 7),
        "dow_cos":      np.cos(2 * np.pi * ts.dayofweek / 7),
        "month_sin":    np.sin(2 * np.pi * ts.month / 12),
        "month_cos":    np.cos(2 * np.pi * ts.month / 12),
        "weather_condition": data.get("weather_condition", "Clear"),
        "temperature":  float(data.get("temperature", 298.15)),
        "humidity":     float(data.get("humidity", 50.0)),
        "rainfall":     float(data.get("rainfall", 0.0)),
        "wind_speed":   float(data.get("wind_speed", 5.0)),
        "visibility":   float(data.get("visibility", 10000.0)),
        "event_holiday":  data.get("event_holiday", "None"),
        "special_event":  int(data.get("special_event", 0)),
    }

    row["peak_hour_indicator"] = 1 if ts.hour in [7, 8, 16, 17, 18] else 0
    row["rush_hour_indicator"] = 1 if ts.hour in [6, 7, 8, 9, 15, 16, 17, 18, 19] else 0
    row["event_impact_score"]  = float(row["special_event"] * 0.5 + (row["event_holiday"] != "None") * 0.3)

    severity = {"Clear": 0.0, "Clouds": 0.1, "Drizzle": 0.3, "Rain": 0.5,
                "Fog": 0.6, "Snow": 0.8, "Thunderstorm": 1.0}
    row["weather_impact_score"] = float(
        severity.get(row["weather_condition"], 0.0)
        + row["rainfall"] * 0.02
        + (10000 - row["visibility"]) / 10000 * 0.2
    )

    row["temp_hour"] = row["temperature"] * row["hour"]
    row["rain_wind"] = row["rainfall"] * row["wind_speed"]

    seg_history = history_df[history_df["road_segment_id"] == seg_id].sort_values("timestamp")

    if len(seg_history) > 0:
        demand_series = seg_history["traffic_demand"]
        row["lag_1"]  = demand_series.iloc[-1]
        row["lag_2"]  = demand_series.iloc[-2]  if len(demand_series) >= 2  else row["lag_1"]
        row["lag_3"]  = demand_series.iloc[-3]  if len(demand_series) >= 3  else row["lag_1"]
        row["lag_4"]  = demand_series.iloc[-4]  if len(demand_series) >= 4  else row["lag_1"]
        row["lag_6"]  = demand_series.iloc[-6]  if len(demand_series) >= 6  else row["lag_1"]
        row["lag_12"] = demand_series.iloc[-12] if len(demand_series) >= 12 else row["lag_1"]
        for w in [3, 6, 12, 24]:
            window = demand_series.tail(w)
            row[f"rolling_mean_{w}"] = window.mean()
            row[f"rolling_std_{w}"]  = window.std() if len(window) > 1 else 0.0
        row["ema_traffic"] = demand_series.ewm(span=12, adjust=False).mean().iloc[-1]
    else:
        for lag in [1, 2, 3, 4, 6, 12]:
            row[f"lag_{lag}"] = 1000.0
        for w in [3, 6, 12, 24]:
            row[f"rolling_mean_{w}"] = 1000.0
            row[f"rolling_std_{w}"]  = 0.0
        row["ema_traffic"] = 1000.0

    return row


def prepare_model_input(row):
    row_df    = pd.DataFrame([row])
    row_enc   = pd.get_dummies(row_df, columns=cat_cols, drop_first=True)
    model_in  = pd.DataFrame(0, index=[0], columns=feature_names)
    for col in row_enc.columns:
        if col in model_in.columns:
            model_in[col] = row_enc[col]
    model_in[num_cols] = scaler.transform(model_in[num_cols])
    return model_in


def congestion_info(demand, lanes):
    capacity = lanes * 1200
    idx = min(round(float(demand / capacity), 3), 1.0)
    if idx <= 0.30:   level = "Low"
    elif idx <= 0.60: level = "Moderate"
    elif idx <= 0.85: level = "High"
    else:             level = "Severe"
    return idx, level


def accident_risk(weather_impact, congestion_idx):
    score = weather_impact * 0.5 + congestion_idx * 0.5
    if score < 0.25:   return "Low"
    elif score < 0.50: return "Medium"
    elif score < 0.75: return "High"
    else:              return "Critical"


def route_recommendations(cong_level, demand):
    base_routes = [
        {"route": "Route A (Direct)",           "distance_km": 12.4, "base_time_min": 15, "factor": 1.0 + (demand / 1000) * 1.5},
        {"route": "Route B (Bypass Expressway)","distance_km": 18.2, "base_time_min": 18, "factor": 1.0 + (demand / 1000) * 0.3},
        {"route": "Route C (Local Streets)",    "distance_km": 10.1, "base_time_min": 22, "factor": 1.0 + (demand / 1000) * 0.8},
    ]
    results = []
    for r in base_routes:
        is_rec = (
            (cong_level in ["High", "Severe"] and r["route"] == "Route B (Bypass Expressway)")
            or (cong_level not in ["High", "Severe"] and r["route"] == "Route A (Direct)")
        )
        results.append({
            "route":               r["route"],
            "distance_km":         r["distance_km"],
            "estimated_time_min":  round(r["base_time_min"] * r["factor"], 1),
            "status":              "Recommended" if is_rec else "Alternative",
        })
    return sorted(results, key=lambda x: x["status"] == "Alternative")


@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/segments", methods=["GET"])
def get_segments():
    return jsonify(list(segment_info.values()))


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":         "healthy",
        "models_loaded":  models_loaded,
        "records_cached": len(history_df) if history_df is not None else 0,
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded."}), 500

    data   = request.get_json() or {}
    seg_id = data.get("road_segment_id", "SEG_001")

    if seg_id not in segment_info:
        return jsonify({"error": f"Unknown segment: {seg_id}"}), 400

    meta = segment_info[seg_id]

    try:
        row        = build_inference_row(data, meta)
        model_in   = prepare_model_input(row)
        pred       = max(int(round(ensemble_model.predict(model_in.values)[0])), 0)
        cong_idx, cong_level = congestion_info(pred, meta["number_of_lanes"])
        risk       = accident_risk(row["weather_impact_score"], cong_idx)
        routes     = route_recommendations(cong_level, pred)

        global history_df
        if history_df is not None:
            new_row = pd.DataFrame([{
                "timestamp":      pd.to_datetime(data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))),
                "road_segment_id": seg_id,
                "traffic_demand":  pred,
            }])
            history_df = pd.concat([history_df, new_row], ignore_index=True).sort_values("timestamp")
            history_df = history_df.groupby("road_segment_id").tail(100).copy()

        return jsonify({
            "road_segment_id":         seg_id,
            "road_type":               meta["road_type"],
            "predicted_traffic_demand": pred,
            "congestion_index":        cong_idx,
            "congestion_level":        cong_level,
            "accident_risk_level":     risk,
            "weather_impact_score":    round(row["weather_impact_score"], 2),
            "event_impact_score":      round(row["event_impact_score"], 2),
            "recommended_routes":      routes,
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {e}"}), 500


@app.route("/api/batch_predict", methods=["POST"])
def batch_predict():
    if not models_loaded:
        return jsonify({"error": "Models not loaded."}), 500

    batch  = (request.get_json() or {}).get("predictions", [])
    results = []

    for item in batch:
        seg_id = item.get("road_segment_id", "SEG_001")
        if seg_id not in segment_info:
            continue
        try:
            meta     = segment_info[seg_id]
            row      = build_inference_row(item, meta)
            model_in = prepare_model_input(row)
            pred     = max(int(round(ensemble_model.predict(model_in.values)[0])), 0)
            cong_idx, cong_level = congestion_info(pred, meta["number_of_lanes"])
            results.append({
                "road_segment_id":         seg_id,
                "road_type":               meta["road_type"],
                "predicted_traffic_demand": pred,
                "congestion_index":        cong_idx,
                "congestion_level":        cong_level,
                "accident_risk_level":     accident_risk(row["weather_impact_score"], cong_idx),
            })
        except Exception:
            continue

    return jsonify({"predictions": results})


@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    stats = []
    for seg_id, meta in segment_info.items():
        sh = history_df[history_df["road_segment_id"] == seg_id]
        avg = int(sh["traffic_demand"].mean()) if len(sh) > 0 else 1200
        mx  = int(sh["traffic_demand"].max())  if len(sh) > 0 else 2500
        stats.append({
            "road_segment_id": seg_id,
            "road_type":       meta["road_type"],
            "average_demand":  avg,
            "peak_demand":     mx,
            "congestion_rate": round(avg / (meta["number_of_lanes"] * 1200), 2),
        })
    return jsonify({
        "segments":               stats,
        "overall_average_demand": int(np.mean([s["average_demand"] for s in stats])),
        "top_congested_segment":  max(stats, key=lambda x: x["congestion_rate"])["road_segment_id"],
    })


@app.route("/api/download_report", methods=["GET"])
def download_report():
    pdf = "Traffic_Demand_Prediction_Report.pdf"
    if os.path.exists(pdf):
        return send_file(pdf, as_attachment=True)
    return jsonify({"error": "Report PDF not found. Run generate_pdf_report.py first."}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
