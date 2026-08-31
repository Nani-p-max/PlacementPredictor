import sys, os

# Allow sibling imports (load_data, preprocess, etc.)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, render_template, jsonify

import load_data    as ld
import preprocess   as pp
import eda          as ed
import linear_model as lm
import logistic_model as logm

# ─────────────────────────────────────────────────────────────────────────────
# Flask app — templates/ and static/ sit one level above backend/
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE, "..", "templates"),
    static_folder=os.path.join(BASE, "..", "static"),
    static_url_path="/static",
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────
def _render(active, **kwargs):
    return render_template("index.html", active=active, **kwargs)


def _safe(fn, active, **extra):
    try:
        data = fn()
        return _render(active, data=data, error=None, **extra)
    except Exception as exc:
        return _render(active, data=None, error=str(exc), **extra)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return _render("home")


@app.route("/data-loading")
def data_loading():
    return _safe(ld.get_data_summary, "data-loading")


@app.route("/preprocessing")
def preprocessing():
    return _safe(pp.get_preprocessing_results, "preprocessing")


@app.route("/eda")
def eda():
    return _safe(ed.get_eda_results, "eda")


@app.route("/linear-regression")
def linear_regression():
    return _safe(lm.get_linear_regression_results, "linear-regression")


@app.route("/logistic-regression")
def logistic_regression():
    return _safe(logm.get_logistic_regression_results, "logistic-regression")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
