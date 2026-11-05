"""Patient Health Prediction app: Flask + SQLite + external AI API (Google Gemini API)."""
import os
import re
import sqlite3
import time
from datetime import date, datetime

import requests
from flask import Flask, flash, g, redirect, render_template, request, url_for

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-me")

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "patients.db"))
API_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}$")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z .'-]*$")

# Plausible physiological ranges, used to catch typos (not clinical thresholds).
RANGES = {
    "glucose": ("Glucose", 20, 800, "mg/dL"),
    "haemoglobin": ("Haemoglobin", 3, 25, "g/dL"),
    "cholesterol": ("Cholesterol", 50, 600, "mg/dL"),
}


# ---------- database ----------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.execute(
            """CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                dob TEXT NOT NULL,            -- stored as dd/mm/yyyy
                email TEXT NOT NULL UNIQUE,
                glucose REAL NOT NULL,
                haemoglobin REAL NOT NULL,
                cholesterol REAL NOT NULL,
                remarks TEXT
            )"""
        )


# ---------- validation ----------
def parse_dob(raw):
    """Accept dd/mm/yyyy (also dd.mm.yyyy or dd-mm-yyyy); return normalised dd/mm/yyyy."""
    raw = raw.strip()
    m = re.fullmatch(r"(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})", raw)
    if not m:
        raise ValueError("Date of birth must be in dd/mm/yyyy format.")
    d, mo, y = map(int, m.groups())
    try:
        parsed = date(y, mo, d)  # rejects impossible dates such as 31/11
    except ValueError:
        raise ValueError(f"Date of birth {raw} is not a real calendar date.")
    if parsed > date.today():
        raise ValueError("Date of birth cannot be in the future.")
    if parsed.year < 1900:
        raise ValueError("Date of birth is too far in the past.")
    return parsed.strftime("%d/%m/%Y")


def validate(form, exclude_id=None):
    """Return (clean_values, errors)."""
    errors, clean = [], {}

    name = " ".join(form.get("full_name", "").split())
    if not name:
        errors.append("Full name is required.")
    elif not NAME_RE.match(name):
        errors.append("Full name may contain only letters, spaces, apostrophes, dots and hyphens.")
    clean["full_name"] = name

    try:
        clean["dob"] = parse_dob(form.get("dob", ""))
    except ValueError as e:
        errors.append(str(e))

    email = form.get("email", "").strip()
    if not EMAIL_RE.match(email):
        errors.append(f"'{email}' is not a valid email address (expected name@domain.tld).")
    else:
        row = get_db().execute(
            "SELECT id FROM patients WHERE lower(email)=lower(?) AND id IS NOT ?", (email, exclude_id)
        ).fetchone()
        if row:
            errors.append(f"A record with email {email} already exists (duplicate).")
    clean["email"] = email

    for key, (label, lo, hi, unit) in RANGES.items():
        raw = form.get(key, "").strip()
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{label} must be a number.")
            continue
        if not lo <= val <= hi:
            errors.append(f"{label} {val:g} is outside the plausible range {lo}-{hi} {unit}.")
        clean[key] = val
    return clean, errors


# ---------- AI integration ----------
def age_from_dob(dob):
    born = datetime.strptime(dob, "%d/%m/%Y").date()
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def predict_remarks(rec):
    """Ask the AI API for a short risk assessment. Returns a remarks string (never raises)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return "AI prediction unavailable: GEMINI_API_KEY not set."
    prompt = (
        "You are a health-screening assistant. Based only on these blood test results, name the most "
        "likely health condition or disease risk (e.g. diabetes risk, anaemia, high cholesterol/"
        "cardiovascular risk, or 'no significant risk'). Reply in at most 30 words, starting with the "
        "risk level (Low/Moderate/High). Plain text only, no markdown or asterisks. "
        "This is a screening aid, not a diagnosis.\n"
        f"Age: {age_from_dob(rec['dob'])}\n"
        f"Fasting glucose: {rec['glucose']:g} mg/dL\n"
        f"Haemoglobin: {rec['haemoglobin']:g} g/dL\n"
        f"Total cholesterol: {rec['cholesterol']:g} mg/dL"
    )
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"maxOutputTokens": 1024, "temperature": 0.2},
    }
    models = [API_MODEL] + ([FALLBACK_MODEL] if FALLBACK_MODEL and FALLBACK_MODEL != API_MODEL else [])
    last = "unknown error"
    for model in models:
        url = API_URL_TEMPLATE.format(model=model)
        for attempt in range(3):  # retry temporary overload (503) / rate limit (429)
            try:
                resp = requests.post(
                    url,
                    headers={"x-goog-api-key": key, "content-type": "application/json"},
                    json=body,
                    timeout=30,
                )
            except requests.RequestException as e:
                print(f"[AI] {model} request error: {e}")
                last = type(e).__name__
                time.sleep(2 * (attempt + 1))
                continue
            if resp.ok:
                try:
                    text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                    return text.replace("*", "").strip()
                except Exception:
                    print(f"[AI] {model} unexpected response: {resp.text[:300]}")
                    return "AI prediction failed (unexpected response). Use Retry."
            print(f"[AI] {model} HTTP {resp.status_code}: {resp.text[:300]}")
            last = f"HTTP {resp.status_code}"
            if resp.status_code not in (429, 500, 503):
                break  # permanent error (bad key / wrong model): do not retry this model
            time.sleep(2 * (attempt + 1))
    return f"AI prediction failed ({last}). Use Retry."


# ---------- routes ----------
@app.route("/")
def index():
    rows = get_db().execute("SELECT * FROM patients ORDER BY id").fetchall()
    return render_template("index.html", rows=rows, edit=None, values={})


@app.route("/add", methods=["POST"])
def add():
    clean, errors = validate(request.form)
    if errors:
        for e in errors:
            flash(e, "danger")
        rows = get_db().execute("SELECT * FROM patients ORDER BY id").fetchall()
        return render_template("index.html", rows=rows, edit=None, values=request.form), 400
    clean["remarks"] = predict_remarks(clean)
    db = get_db()
    try:
        db.execute(
            "INSERT INTO patients (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks) "
            "VALUES (:full_name, :dob, :email, :glucose, :haemoglobin, :cholesterol, :remarks)",
            clean,
        )
        db.commit()
    except sqlite3.IntegrityError:
        # Same email saved by a parallel request (e.g. double click) while the AI call was running.
        db.rollback()
        flash(f"A record with email {clean['email']} already exists (duplicate).", "danger")
        return redirect(url_for("index"))
    flash(f"Record for {clean['full_name']} created.", "success")
    return redirect(url_for("index"))


@app.route("/edit/<int:pid>", methods=["GET", "POST"])
def edit(pid):
    db = get_db()
    row = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    if row is None:
        flash("Record not found.", "danger")
        return redirect(url_for("index"))
    if request.method == "GET":
        rows = db.execute("SELECT * FROM patients ORDER BY id").fetchall()
        return render_template("index.html", rows=rows, edit=row, values=dict(row))
    clean, errors = validate(request.form, exclude_id=pid)
    if errors:
        for e in errors:
            flash(e, "danger")
        rows = db.execute("SELECT * FROM patients ORDER BY id").fetchall()
        return render_template("index.html", rows=rows, edit=row, values=request.form), 400
    changed = any(clean[k] != row[k] for k in ("dob", "glucose", "haemoglobin", "cholesterol"))
    clean["remarks"] = predict_remarks(clean) if changed else row["remarks"]
    clean["id"] = pid
    try:
        db.execute(
            "UPDATE patients SET full_name=:full_name, dob=:dob, email=:email, glucose=:glucose, "
            "haemoglobin=:haemoglobin, cholesterol=:cholesterol, remarks=:remarks WHERE id=:id",
            clean,
        )
        db.commit()
    except sqlite3.IntegrityError:
        db.rollback()
        flash(f"A record with email {clean['email']} already exists (duplicate).", "danger")
        return redirect(url_for("index"))
    flash("Record updated.", "success")
    return redirect(url_for("index"))


@app.route("/retry/<int:pid>", methods=["POST"])
def retry(pid):
    db = get_db()
    row = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    if row:
        db.execute("UPDATE patients SET remarks=? WHERE id=?", (predict_remarks(dict(row)), pid))
        db.commit()
        flash("Prediction refreshed.", "success")
    return redirect(url_for("index"))


@app.route("/delete/<int:pid>", methods=["POST"])
def delete(pid):
    db = get_db()
    db.execute("DELETE FROM patients WHERE id=?", (pid,))
    db.commit()
    flash("Record deleted.", "success")
    return redirect(url_for("index"))


init_db()

if __name__ == "__main__":
    app.run(debug=False)