from pathlib import Path
import json
import sqlite3
from datetime import datetime
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "fake_profile_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
DB_PATH = ROOT / "data" / "predictions.db"

app = Flask(__name__)
app.secret_key = "change-this-local-demo-secret"
FEATURES = [
    "account_age_days", "followers", "following", "posts",
    "has_profile_photo", "has_bio", "verified",
    "username_digit_ratio", "posts_per_month"
]

def db_connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    DB_PATH.parent.mkdir(exist_ok=True)
    with db_connect() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                username TEXT NOT NULL,
                prediction TEXT NOT NULL,
                risk_score INTEGER NOT NULL,
                account_age_days INTEGER NOT NULL,
                followers INTEGER NOT NULL,
                following INTEGER NOT NULL,
                posts INTEGER NOT NULL,
                reasons TEXT NOT NULL
            )
        """)

def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model not found. Run generate_dataset.py and train_model.py first.")
    return joblib.load(MODEL_PATH)

def make_reasons(d):
    reasons = []
    ratio = d["following"] / max(d["followers"], 1)
    if ratio > 8:
        reasons.append("Following count is much higher than follower count.")
    if d["account_age_days"] < 30:
        reasons.append("Account age is very low.")
    if d["posts"] < 5:
        reasons.append("Very few posts are available.")
    if d["has_profile_photo"] == 0:
        reasons.append("No profile photo was indicated.")
    if d["has_bio"] == 0:
        reasons.append("No profile biography was indicated.")
    if d["username_digit_ratio"] > 0.35:
        reasons.append("Username contains a relatively high proportion of digits.")
    if not reasons:
        reasons.append("No simple rule-based warning stood out; review the profile context.")
    return reasons

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    if request.method == "POST":
        try:
            username = request.form.get("username", "").strip()[:80] or "unnamed_profile"
            data = {
                "account_age_days": int(request.form["account_age_days"]),
                "followers": int(request.form["followers"]),
                "following": int(request.form["following"]),
                "posts": int(request.form["posts"]),
                "has_profile_photo": int(request.form["has_profile_photo"]),
                "has_bio": int(request.form["has_bio"]),
                "verified": int(request.form["verified"]),
                "username_digit_ratio": float(request.form["username_digit_ratio"]),
            }
            if any(data[k] < 0 for k in ["account_age_days", "followers", "following", "posts"]):
                raise ValueError("Age, followers, following, and posts cannot be negative.")
            if not 0 <= data["username_digit_ratio"] <= 1:
                raise ValueError("Username digit ratio must be between 0 and 1.")
            data["posts_per_month"] = round(data["posts"] / max(data["account_age_days"] / 30.44, 1), 3)
            bundle = get_model()
            X = pd.DataFrame([[data[c] for c in bundle["features"]]], columns=bundle["features"])
            model = bundle["model"]
            probability = float(model.predict_proba(X)[0][1])
            prediction = "Potentially fake" if probability >= 0.5 else "Likely genuine"
            risk_score = round(probability * 100)
            reasons = make_reasons(data)
            result = {
                "username": username, "prediction": prediction,
                "risk_score": risk_score, "reasons": reasons,
                "probability": probability
            }
            with db_connect() as con:
                con.execute("""
                    INSERT INTO predictions
                    (created_at, username, prediction, risk_score, account_age_days, followers, following, posts, reasons)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), username,
                      prediction, risk_score, data["account_age_days"], data["followers"],
                      data["following"], data["posts"], " | ".join(reasons)))
        except FileNotFoundError as e:
            flash(str(e), "error")
        except (ValueError, KeyError) as e:
            flash(f"Please check the entered values. {e}", "error")
        except Exception as e:
            flash(f"Could not analyze this profile: {e}", "error")
    with db_connect() as con:
        history = con.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT 8").fetchall()
        summary = con.execute("""
            SELECT COUNT(*) AS total,
            SUM(CASE WHEN prediction='Potentially fake' THEN 1 ELSE 0 END) AS flagged
            FROM predictions
        """).fetchone()
    return render_template("index.html", result=result, metrics=metrics, history=history, summary=summary)

@app.route("/clear-history", methods=["POST"])
def clear_history():
    with db_connect() as con:
        con.execute("DELETE FROM predictions")
    flash("Local prediction history cleared.", "success")
    return redirect(url_for("index"))

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
