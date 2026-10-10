"""Iris Flower Classification — an interactive Streamlit dashboard.

Three pages, reachable from the sidebar:

    Overview    what the Iris dataset is and what the model does
    Data        exploratory summary of the 150-sample dataset
    Predict     measure a new flower and classify it with the saved model

The trained model is loaded from ``model.pkl`` next to this file. It is never
retrained and never overwritten — see the "Model loading" section below.

Run with:

    streamlit run app.py
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from textwrap import dedent
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.datasets import load_iris

# ---------------------------------------------------------------------------
# Design tokens
#
# One palette, used by both the CSS and the Plotly figures, so the charts and
# the page chrome can never drift apart. The three species colors were checked
# for colour-vision separation and contrast against the paper surface.
# ---------------------------------------------------------------------------

PAPER = "#FAFAF5"        # page surface
SURFACE = "#FFFFFF"      # card / chart surface
INK = "#16241E"          # primary text
INK_2 = "#4A5A52"        # secondary text
MUTED = "#7C8A83"        # axis labels, captions
RULE = "#E2E5DA"         # hairline gridline
RULE_STRONG = "#C9CFC0"  # axis / baseline

SPECIES_COLORS = {
    "Iris setosa": "#5B4B9E",
    "Iris versicolor": "#C77E2E",
    "Iris virginica": "#12876A",
}
SPECIES_ORDER = list(SPECIES_COLORS)

# Diverging pair for the correlation heatmap: two opposing hues with a neutral
# middle. A sequential ramp would be wrong here — correlations have a sign.
DIVERGING_LOW = "#C2410C"
DIVERGING_HIGH = "#12876A"
DIVERGING_MID = "#F1EFE9"

FONT_SANS = 'ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif'
FONT_MONO = 'ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, monospace'

# ---------------------------------------------------------------------------
# Dataset constants
#
# These mirror sklearn's Iris dataset exactly. They are defined as module
# constants so the prediction page can build a correctly ordered input frame
# without depending on anything the UI does.
# ---------------------------------------------------------------------------

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# The saved model was trained on an integer-encoded target built from
# load_iris(), so its class labels are the dataset's own 0/1/2 indices.
# verify_class_mapping() below checks this rather than trusting it blindly.
CANONICAL_SPECIES = {0: "Iris setosa", 1: "Iris versicolor", 2: "Iris virginica"}

# Slider ranges and step, taken from the real min/max of the dataset
# (sepal length 4.3-7.9, sepal width 2.0-4.4, petal length 1.0-6.9,
# petal width 0.1-2.5) with a little room on either side.
INPUT_SPECS = [
    ("sepal length (cm)", "Sepal length", 4.0, 8.0, 5.1, "cm"),
    ("sepal width (cm)", "Sepal width", 1.5, 5.0, 3.5, "cm"),
    ("petal length (cm)", "Petal length", 0.5, 7.5, 1.4, "cm"),
    ("petal width (cm)", "Petal width", 0.0, 3.0, 0.2, "cm"),
]

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"


# ---------------------------------------------------------------------------
# Model loading
#
# Kept apart from the UI: these functions only ever return a model or raise.
# They never train a replacement, because silently substituting a model the
# user did not ship would make every prediction on the page a lie.
# ---------------------------------------------------------------------------


class ModelLoadError(Exception):
    """Raised when model.pkl is missing, unreadable, or not a usable model."""


@st.cache_resource(show_spinner=False)
def load_model(path_str: str) -> Any:
    """Load and cache the trained model.

    ``path_str`` is a string (not a Path) so Streamlit can hash it as a cache
    key. Tries joblib first — it reads both joblib dumps and plain pickle
    files — then falls back to the stdlib pickle module.
    """
    path = Path(path_str)

    if not path.is_file():
        raise ModelLoadError(
            f"No model file found at {path}. Place the trained `model.pkl` "
            "in the same folder as `app.py` and reload the page."
        )

    loaded = None
    joblib_error = None
    try:
        loaded = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure is fatal here
        joblib_error = exc

    if loaded is None:
        try:
            with path.open("rb") as handle:
                loaded = pickle.load(handle)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"`model.pkl` exists but could not be read. joblib reported: "
                f"{joblib_error}. pickle reported: {exc}. The file may be "
                "corrupt, or it may have been saved with an incompatible "
                "version of scikit-learn."
            ) from exc

    if not hasattr(loaded, "predict"):
        raise ModelLoadError(
            "`model.pkl` loaded, but the object inside it has no `predict()` "
            "method, so it is not a fitted scikit-learn estimator. Re-export "
            "the model with `joblib.dump(model, 'model.pkl')`."
        )

    return loaded


def describe_model(model: Any) -> tuple[str, str]:
    """Return (estimator name, short parameter summary) for display."""
    name = type(model).__name__
    n_estimators = getattr(model, "n_estimators", None)
    if isinstance(n_estimators, int):
        return name, f"{n_estimators} trees"
    return name, "fitted estimator"


def verify_input_columns(model: Any) -> tuple[list[str], str | None]:
    """Return the feature names and order the model expects, plus an optional note.

    Estimators fitted on a pandas DataFrame expose ``feature_names_in_``; this
    one was fitted on a plain numpy array, so it does not. In that case we fall
    back to the dataset's documented order and say so.
    """
    names = list(getattr(model, "feature_names_in_", []) or [])
    if not names:
        return FEATURE_COLUMNS, None

    if sorted(names) == sorted(FEATURE_COLUMNS):
        return names, None

    return FEATURE_COLUMNS, (
        "The model reports feature names that differ from the standard Iris "
        "columns; the four standard columns are being sent in dataset order."
    )


def verify_class_mapping(model: Any) -> tuple[dict[Any, str], str | None]:
    """Map the model's raw class labels to species names.

    Uses ``classes_`` when the labels are the dataset's 0/1/2 indices. For any
    other encoding we fall back to positional order and warn, because the
    numbers themselves would then be meaningless to us.
    """
    classes = list(getattr(model, "classes_", []))

    if classes and all(label in CANONICAL_SPECIES for label in classes):
        return {label: CANONICAL_SPECIES[label] for label in classes}, None

    if len(classes) == 3:
        by_position = {label: name for label, name in zip(classes, SPECIES_ORDER)}
        return by_position, (
            "The model encodes classes as "
            f"{classes}, which are not the dataset's 0/1/2 indices. Species "
            "names are mapped by class order instead, so treat them as a best "
            "guess rather than a certainty."
        )

    return {}, (
        f"The model reports {len(classes)} classes, but the Iris dataset has "
        "three. Predictions are shown as raw class labels."
    )


# ---------------------------------------------------------------------------
# Dataset access
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    """The standard Iris dataset as a tidy DataFrame with readable columns."""
    bunch = load_iris()
    frame = pd.DataFrame(bunch.data, columns=FEATURE_COLUMNS)
    frame["species"] = [CANONICAL_SPECIES[int(i)] for i in bunch.target]
    return frame


def species_means(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean measurement per species — the 'typical flower' for each class."""
    return frame.groupby("species", sort=False)[FEATURE_COLUMNS].mean()


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------


def style_figure(fig: go.Figure, height: int = 380, show_legend: bool = True) -> go.Figure:
    """Apply the shared chart styling: palette, recessive axes, quiet grid."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_SANS, size=13, color=INK_2),
        colorway=list(SPECIES_COLORS.values()),
        margin=dict(l=8, r=8, t=56, b=8),
        bargap=0.28,
        hoverlabel=dict(
            bgcolor=SURFACE,
            bordercolor=RULE_STRONG,
            font=dict(family=FONT_SANS, size=12, color=INK),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=12, color=INK_2),
            # Plotly reverses legend order for stacked traces; pin it so the
            # legend always reads setosa, versicolor, virginica on every chart.
            traceorder="normal",
        ),
        title=dict(font=dict(size=15, color=INK), x=0, xanchor="left", y=0.97),
        showlegend=show_legend,
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=RULE_STRONG,
        linewidth=1,
        ticks="outside",
        ticklen=4,
        tickcolor=RULE_STRONG,
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=INK_2, size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=RULE,
        gridwidth=1,
        zeroline=False,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=MUTED, size=12),
        title_font=dict(color=INK_2, size=12),
    )
    return fig


def show_chart(fig: go.Figure, key: str) -> None:
    """Render a Plotly figure at full width with the figure's own theme."""
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,  # keep our own template rather than Streamlit's
        config={"displaylogo": False, "displayModeBar": False},
        key=key,
    )


# ---------------------------------------------------------------------------
# Page chrome
# ---------------------------------------------------------------------------

# A hand-drawn iris mark for the hero. Kept on one logical line (no newlines)
# because the hero is rendered as Markdown, where any indented line would be
# read as a code block. Must go through st.markdown, not st.html — st.html
# sanitises its input and strips inline SVG.
HERO_IRIS = (
    '<svg class="hero-iris" viewBox="0 0 120 120" aria-hidden="true" focusable="false">'
    '<g fill="none" stroke="#FFFFFF" stroke-width="1.4" stroke-linecap="round">'
    '<path d="M60 104 C58 84 56 70 60 56"/>'
    '<path d="M60 58 C60 40 50 26 44 18 C40 30 42 46 50 56 Z"/>'
    '<path d="M60 58 C60 40 70 26 76 18 C80 30 78 46 70 56 Z"/>'
    '<path d="M60 62 C74 54 88 50 98 50 C92 62 78 70 66 70 Z"/>'
    '<path d="M60 62 C46 54 32 50 22 50 C28 62 42 70 54 70 Z"/>'
    '<path d="M60 56 C56 46 56 38 60 30 C64 38 64 46 60 56 Z"/>'
    "</g></svg>"
)

# Note for anyone editing this stylesheet: never write a less-than sign inside
# it, comments included. The HTML parser ends a style element at the first
# closing-tag sequence it sees, and a stray angle bracket can form one early --
# silently discarding every rule after it, with no error shown anywhere.
CSS = """
<style>
:root {
  --paper: #FAFAF5;
  --surface: #FFFFFF;
  --ink: #16241E;
  --ink-2: #4A5A52;
  --muted: #7C8A83;
  --rule: #E2E5DA;
  --rule-strong: #C9CFC0;
  --setosa: #5B4B9E;
  --versicolor: #C77E2E;
  --virginica: #12876A;
  --evergreen: #16332A;
  --radius: 14px;
  --font-sans: ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  --font-mono: ui-monospace, "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
}

/* --- surfaces ---------------------------------------------------------- */
.stApp { background: var(--paper); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
  background: #F2F3EB;
  border-right: 1px solid var(--rule);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }

/* --- typography -------------------------------------------------------- */
/* Icons and code keep their own faces; everything else inherits the UI sans. */
.stApp :where(p, span, div, label, li, input, button, th, td,
              h1, h2, h3, h4, h5, h6):not([data-testid="stIconMaterial"]):not([class*="material-symbols"]) {
  font-family: var(--font-sans);
}
.stApp :where(code, pre, kbd) { font-family: var(--font-mono); }
[data-testid="stIconMaterial"],
[class*="material-symbols"] { font-family: "Material Symbols Rounded" !important; }

.stApp h1, .stApp h2, .stApp h3, .stApp h4 { color: var(--ink); letter-spacing: -0.015em; }
/* Wrapped in :where() so this default carries zero specificity — otherwise it
   would outrank the component classes below (the hero sits on a dark panel and
   needs its own light text). Any class-based rule now beats it. */
:where(.stApp [data-testid="stMarkdownContainer"] p) { color: var(--ink-2); line-height: 1.65; }
/* Button labels are paragraphs inside a markdown container. The rule above sets
   an explicit colour on them, which beats the white the button inherits - so
   they are put back on `inherit` here. (:where() lowers specificity against
   competing rules, but an explicit value still wins over an inherited one.) */
.stApp [data-testid="stButton"] [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stDownloadButton"] [data-testid="stMarkdownContainer"] p {
  color: inherit;
}
.stApp [data-testid="stMarkdownContainer"] a { color: #0E5C49; }

/* --- hero -------------------------------------------------------------- */
.hero {
  position: relative;
  background: var(--evergreen);
  border-radius: var(--radius);
  padding: 42px 40px 38px;
  overflow: hidden;
  margin-bottom: 10px;
}
.hero::after {
  content: "";
  position: absolute;
  inset: 0;
  background-image: repeating-linear-gradient(
    90deg, rgba(255,255,255,0.05) 0 1px, transparent 1px 76px
  );
  pointer-events: none;
}
.hero-iris {
  position: absolute;
  right: 28px;
  bottom: -16px;
  width: 200px;
  height: 200px;
  opacity: 0.2;
}
.hero-kicker {
  position: relative;
  margin: 0 0 14px;
  font-size: 13px;
  color: #A9C4B8;
  font-variant-numeric: tabular-nums;
}
.hero h1 {
  position: relative;
  margin: 0 0 16px;
  font-size: 46px;
  line-height: 1.08;
  font-weight: 600;
  letter-spacing: -0.03em;
  color: #FBFCF8;
  max-width: 17ch;
}
.hero-lede {
  position: relative;
  margin: 0;
  max-width: 62ch;
  font-size: 16px;
  line-height: 1.7;
  color: #C6D6CC;
}
.hero-key {
  position: relative;
  display: flex;
  flex-wrap: wrap;
  gap: 8px 22px;
  margin: 26px 0 0;
  padding: 0;
  list-style: none;
}
.hero-key li {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 13.5px;
  color: #DCE7DF;
}
.hero-key .dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  flex: none;
  box-shadow: 0 0 0 1px rgba(255,255,255,0.28);
}

/* --- section headings -------------------------------------------------- */
.sec { margin: 44px 0 14px; }
.sec h2 {
  margin: 0;
  font-size: 23px;
  font-weight: 600;
  color: var(--ink);
}
.sec p {
  margin: 7px 0 0;
  max-width: 68ch;
  font-size: 14.5px;
  color: var(--ink-2);
  line-height: 1.6;
}
.sub { margin: 26px 0 8px; }
.sub h3 { margin: 0; font-size: 16px; font-weight: 600; color: var(--ink); }

/* --- cards ------------------------------------------------------------- */
.grid { display: grid; gap: 14px; }
.grid.four { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
@media (max-width: 1100px) {
  .grid.four { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .grid.three { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .grid.four, .grid.three { grid-template-columns: minmax(0, 1fr); }
  .hero { padding: 30px 22px; }
  .hero h1 { font-size: 31px; }
  .hero-iris { display: none; }
}

.card {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: var(--radius);
  padding: 18px 18px 16px;
  height: 100%;
}
.card .card-label { font-size: 14px; font-weight: 600; color: var(--ink); }
.card .card-value {
  margin: 10px 0 0;
  font-size: 26px;
  font-weight: 600;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
  letter-spacing: -0.02em;
}
.card .card-value .unit { font-size: 14px; font-weight: 500; color: var(--muted); margin-left: 3px; }
.card .card-note { margin: 9px 0 0; font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; }

.species-card { border-top: 3px solid var(--rule-strong); }
.species-card .species-name {
  display: flex;
  align-items: center;
  gap: 9px;
  font-size: 15px;
  font-weight: 600;
  color: var(--ink);
}
.species-card .dot { width: 11px; height: 11px; border-radius: 50%; flex: none; }
.species-card .species-metric {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  padding: 7px 0;
  border-bottom: 1px solid var(--rule);
  font-size: 13px;
}
.species-card .species-metric:last-child { border-bottom: 0; padding-bottom: 0; }
.species-card .species-metric span:first-child { color: var(--ink-2); }
.species-card .species-metric span:last-child { color: var(--ink); font-variant-numeric: tabular-nums; }
.species-card .species-head { margin-bottom: 12px; }

/* --- numbered steps (a real sequence) ---------------------------------- */
.steps { display: grid; gap: 14px; grid-template-columns: repeat(3, minmax(0, 1fr)); }
@media (max-width: 900px) { .steps { grid-template-columns: minmax(0, 1fr); } }
.step {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: var(--radius);
  padding: 18px;
}
.step .step-no {
  font-size: 12px;
  color: var(--muted);
  font-variant-numeric: tabular-nums;
  letter-spacing: 0.02em;
}
.step .step-title { margin: 8px 0 0; font-size: 15px; font-weight: 600; color: var(--ink); }
.step .step-body { margin: 7px 0 0; font-size: 13.5px; color: var(--ink-2); line-height: 1.6; }

/* --- chips ------------------------------------------------------------- */
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
  border: 1px solid var(--rule-strong);
  border-radius: 999px;
  padding: 6px 14px;
  font-size: 13px;
  color: var(--ink-2);
  background: var(--surface);
}

/* --- result panel ------------------------------------------------------ */
.result {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-left: 4px solid var(--rule-strong);
  border-radius: var(--radius);
  padding: 24px 26px;
}
.result .result-kicker { font-size: 12.5px; color: var(--muted); }
.result .result-name {
  margin: 9px 0 0;
  font-size: 34px;
  font-weight: 600;
  letter-spacing: -0.025em;
  color: var(--ink);
}
.result .result-note { margin: 12px 0 0; max-width: 62ch; font-size: 14px; color: var(--ink-2); line-height: 1.65; }
.result .result-meta { margin: 16px 0 0; font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; }

/* --- sidebar brand + nav ---------------------------------------------- */
.brand { padding: 2px 0 18px; border-bottom: 1px solid var(--rule); margin-bottom: 14px; }
.brand .brand-name { font-size: 16px; font-weight: 600; color: var(--ink); letter-spacing: -0.015em; }
.brand .brand-sub { margin-top: 5px; font-size: 12.5px; color: var(--muted); }
.brand .brand-status {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-top: 11px;
  font-size: 12px;
  color: var(--ink-2);
  font-family: var(--font-mono);
}
.brand .brand-status .pip { width: 7px; height: 7px; border-radius: 50%; flex: none; }
[data-testid="stSidebar"] [role="radiogroup"] label {
  padding: 8px 10px;
  border-radius: 9px;
  margin-bottom: 2px;
  transition: background 120ms ease;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(22,36,30,0.05); }
.side-note { margin-top: 18px; padding-top: 15px; border-top: 1px solid var(--rule); }
.side-note p { font-size: 12px; color: var(--muted); line-height: 1.6; margin: 0 0 8px; }

/* --- native widget polish --------------------------------------------- */
.stApp [data-testid="stMetric"] {
  background: var(--surface);
  border: 1px solid var(--rule);
  border-radius: var(--radius);
  padding: 16px 18px;
}
.stApp [data-testid="stMetricLabel"] p { font-size: 13px; color: var(--ink-2); }
.stApp [data-testid="stMetricValue"] {
  font-size: 30px;
  font-weight: 600;
  color: var(--ink);
  letter-spacing: -0.02em;
}
.stApp [data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface);
  border-radius: var(--radius);
}
.stApp [data-testid="stDataFrame"] { border: 1px solid var(--rule); border-radius: 10px; }
.stApp .stButton > button[kind="primary"] {
  background: var(--evergreen);
  border: 1px solid var(--evergreen);
  border-radius: 10px;
  padding: 0.5rem 1.3rem;
  font-weight: 500;
}
.stApp .stButton > button[kind="primary"]:hover {
  background: #1D4536;
  border-color: #1D4536;
}
.stApp .stButton > button[kind="secondary"] {
  border-radius: 10px;
  border: 1px solid var(--rule-strong);
  background: var(--surface);
}
.stApp .stButton > button[kind="secondary"]:hover { border-color: var(--ink-2); color: var(--ink); }
.stApp .stButton > button:focus-visible {
  outline: 2px solid #0E5C49;
  outline-offset: 2px;
}
.stApp .stNumberInput input { font-variant-numeric: tabular-nums; border-radius: 9px; }
.stApp .stSelectbox div[data-baseweb="select"] > div { border-radius: 9px; }
.stApp [data-testid="stExpander"] details {
  border: 1px solid var(--rule);
  border-radius: var(--radius);
  background: var(--surface);
}
.footnote { margin: 34px 0 8px; padding-top: 16px; border-top: 1px solid var(--rule); }
.footnote p { font-size: 12px; color: var(--muted); line-height: 1.65; margin: 0; }

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
}
</style>
"""


def _assert_stylesheet_is_parseable() -> None:
    """Fail loudly if an angle bracket crept into the stylesheet.

    A less-than sign anywhere inside a style element -- a comment is enough --
    can make the HTML parser close the element early. The browser then discards
    every rule after that point and renders the app unstyled, without raising
    anything on the server. Checking here turns that silent failure into a
    visible one.
    """
    body = CSS.split("<style>", 1)[1].rsplit("</style>", 1)[0]
    offenders = [
        f"line {number}: {line.strip()}"
        for number, line in enumerate(body.splitlines(), 1)
        if "<" in line
    ]
    if offenders:
        raise ValueError(
            "The stylesheet contains an angle bracket, which truncates it in the "
            "browser. Remove it (see the note above CSS):\n  " + "\n  ".join(offenders)
        )


def section(title: str, lede: str | None = None) -> None:
    """Render a page section heading with an optional one-line explanation."""
    body = f"<div class='sec'><h2>{title}</h2>"
    if lede:
        body += f"<p>{lede}</p>"
    st.html(body + "</div>")


def subsection(title: str) -> None:
    st.html(f"<div class='sub'><h3>{title}</h3></div>")


def footnote(text: str) -> None:
    st.html(f"<div class='footnote'><p>{text}</p></div>")


def feature_card(label: str, value: str, unit: str, note: str) -> str:
    return (
        "<div class='card'>"
        f"<div class='card-label'>{label}</div>"
        f"<div class='card-value'>{value}<span class='unit'>{unit}</span></div>"
        f"<div class='card-note'>{note}</div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Page 1 — Overview
# ---------------------------------------------------------------------------


def page_overview() -> None:
    # Rendered through st.markdown rather than st.html: st.html sanitises the
    # markup and drops inline SVG, which the iris mark below needs.
    # The string is deliberately flush-left — indented lines would be parsed
    # as a Markdown code block.
    st.markdown(
        dedent(
            f"""\
            <section class="hero">
              {HERO_IRIS}
              <p class="hero-kicker">150 specimens &nbsp;|&nbsp; 4 measurements &nbsp;|&nbsp; 3 species</p>
              <h1>Iris Flower Classification</h1>
              <p class="hero-lede">
                A trained scikit-learn model reads four measurements from an iris
                flower — the lengths and widths of its sepal and petal — and names
                the species. Enter a flower's measurements and see the model's
                answer, with the dataset it learned from alongside it.
              </p>
              <ul class="hero-key">
                <li><span class="dot" style="background:{SPECIES_COLORS['Iris setosa']}"></span>Iris setosa</li>
                <li><span class="dot" style="background:{SPECIES_COLORS['Iris versicolor']}"></span>Iris versicolor</li>
                <li><span class="dot" style="background:{SPECIES_COLORS['Iris virginica']}"></span>Iris virginica</li>
              </ul>
            </section>
            """
        ),
        unsafe_allow_html=True,
    )

    section(
        "What the model reads",
        "Four measurements per flower, in centimetres. These are the only inputs "
        "the model receives — everything else is the species it predicts.",
    )

    frame = get_dataset()
    cards = []
    for column, label, _, _, _, unit in INPUT_SPECS:
        low, high = frame[column].min(), frame[column].max()
        cards.append(
            feature_card(label, f"{frame[column].mean():.1f}", unit, f"Dataset range {low:.1f} – {high:.1f} {unit}")
        )
    st.html(f"<div class='grid four'>{''.join(cards)}</div>")

    section(
        "The three species",
        "Sepal and petal proportions separate the species — the petals in "
        "particular. Values below are each species' average across the dataset.",
    )

    means = species_means(frame)
    species_cards = []
    for species in SPECIES_ORDER:
        rows = "".join(
            f"<div class='species-metric'><span>{label.replace(' (cm)', '').capitalize()}</span>"
            f"<span>{means.loc[species, column]:.2f} cm</span></div>"
            for column, label, *_ in INPUT_SPECS
        )
        species_cards.append(
            f"<div class='card species-card' style='border-top-color:{SPECIES_COLORS[species]}'>"
            f"<div class='species-head'><div class='species-name'>"
            f"<span class='dot' style='background:{SPECIES_COLORS[species]}'></span>{species}</div></div>"
            f"{rows}</div>"
        )
    st.html(f"<div class='grid three'>{''.join(species_cards)}</div>")

    section("How a prediction is made")

    st.html(
        """
        <div class="steps">
          <div class="step">
            <div class="step-no">Step 1</div>
            <div class="step-title">Measure the flower</div>
            <div class="step-body">Four values in centimetres: sepal length and
            width, petal length and width.</div>
          </div>
          <div class="step">
            <div class="step-no">Step 2</div>
            <div class="step-title">The model compares them</div>
            <div class="step-body">A random forest of decision trees splits on
            those measurements against the boundaries it learned from 150
            labelled flowers.</div>
          </div>
          <div class="step">
            <div class="step-no">Step 3</div>
            <div class="step-title">A species comes back</div>
            <div class="step-body">Trees vote, and the majority species wins.
            The share of votes becomes the probability shown on the result.</div>
          </div>
        </div>
        """
    )

    st.write("")
    left, _ = st.columns([1, 2])
    with left:
        if st.button("Test a new sample", type="primary", width="stretch"):
            st.session_state.page = "Predict"
            st.rerun()

    section(
        "Built with",
        "A small, standard scientific Python stack — no custom dependencies.",
    )
    st.html(
        "<div class='chips'>"
        + "".join(
            f"<span class='chip'>{name}</span>"
            for name in ["Python", "Streamlit", "pandas", "scikit-learn", "Plotly", "NumPy", "joblib"]
        )
        + "</div>"
    )

    footnote(
        "The model file <code>model.pkl</code> ships with this project and is "
        "loaded as-is. This app never trains a model."
    )


# ---------------------------------------------------------------------------
# Page 2 — Data summary
# ---------------------------------------------------------------------------


def page_data() -> None:
    frame = get_dataset()
    bunch = load_iris()

    section(
        "Data summary",
        "The Iris dataset as it ships with scikit-learn: 150 flowers collected "
        "by Edgar Anderson and published by R. A. Fisher in 1936. Balanced, "
        "complete, and small enough to inspect by eye.",
    )

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Samples", f"{len(frame):,}", border=True)
    metric_b.metric("Measurements per flower", f"{len(FEATURE_COLUMNS)}", border=True)
    metric_c.metric("Species", f"{frame['species'].nunique()}", border=True)

    subsection("Features and classes")
    info_left, info_right = st.columns(2)
    with info_left:
        st.dataframe(
            pd.DataFrame(
                {
                    "Feature": FEATURE_COLUMNS,
                    "Unit": ["cm"] * len(FEATURE_COLUMNS),
                    "Minimum": [f"{frame[c].min():.1f}" for c in FEATURE_COLUMNS],
                    "Maximum": [f"{frame[c].max():.1f}" for c in FEATURE_COLUMNS],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    with info_right:
        st.dataframe(
            pd.DataFrame(
                {
                    "Encoded label": list(range(len(bunch.target_names))),
                    "Species": [CANONICAL_SPECIES[i] for i in range(len(bunch.target_names))],
                    "Samples": frame["species"].value_counts().reindex(SPECIES_ORDER).tolist(),
                }
            ),
            hide_index=True,
            width="stretch",
        )

    section("First 10 rows", "Every row is one flower, with its species alongside its measurements.")
    st.dataframe(
        frame.head(10),
        hide_index=True,
        width="stretch",
        column_config={
            column: st.column_config.NumberColumn(column, format="%.1f") for column in FEATURE_COLUMNS
        },
    )

    section(
        "Descriptive statistics",
        "Summary for all four measurement features across all 150 flowers.",
    )
    stats = frame[FEATURE_COLUMNS].describe().T
    stats = stats.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std. dev.",
            "min": "Min",
            "25%": "25th pct",
            "50%": "Median",
            "75%": "75th pct",
            "max": "Max",
        }
    )
    st.dataframe(
        stats,
        width="stretch",
        column_config={
            label: st.column_config.NumberColumn(label, format="%.2f")
            for label in stats.columns
        },
    )

    section("Distributions and relationships")

    # 1. Species distribution -------------------------------------------------
    subsection("Samples per species")
    counts = frame["species"].value_counts().reindex(SPECIES_ORDER)
    bar = go.Figure(
        go.Bar(
            x=counts.index,
            y=counts.to_numpy(),
            marker=dict(
                color=[SPECIES_COLORS[s] for s in counts.index],
                line=dict(color=SURFACE, width=2),
            ),
            text=counts.to_numpy(),
            textposition="outside",
            textfont=dict(color=INK_2, size=12),
            hovertemplate="<b>%{x}</b><br>%{y} flowers<extra></extra>",
            width=0.5,
        )
    )
    bar.update_layout(title="Each species contributes 50 flowers", showlegend=False)
    bar.update_yaxes(title="Flowers", range=[0, 58])
    show_chart(style_figure(bar, height=330, show_legend=False), key="chart_species_bar")

    # 2. Feature distributions ------------------------------------------------
    subsection("Feature distributions")
    histogram_feature = st.selectbox(
        "Measurement to plot",
        options=FEATURE_COLUMNS,
        format_func=lambda c: c.replace(" (cm)", "").capitalize() + " (cm)",
        key="hist_feature",
    )
    edges = np.linspace(frame[histogram_feature].min(), frame[histogram_feature].max(), 19)
    histogram = go.Figure()
    for species in SPECIES_ORDER:
        subset = frame.loc[frame["species"] == species, histogram_feature]
        histogram.add_trace(
            go.Histogram(
                x=subset,
                name=species,
                xbins=dict(start=edges[0], end=edges[-1], size=edges[1] - edges[0]),
                marker=dict(
                    color=SPECIES_COLORS[species],
                    line=dict(color=SURFACE, width=1),
                ),
                hovertemplate=(
                    f"<b>{species}</b><br>{histogram_feature}: %{{x}}"
                    "<br>flowers: %{y}<extra></extra>"
                ),
            )
        )
    histogram.update_layout(barmode="stack", title=f"Where the {histogram_feature} values fall")
    histogram.update_xaxes(title=f"{histogram_feature.replace(' (cm)', '').capitalize()} (cm)")
    histogram.update_yaxes(title="Flowers")
    show_chart(style_figure(histogram, height=400), key="chart_histogram")

    # 3. Feature relationships ------------------------------------------------
    subsection("Two measurements against each other")
    axis_left, axis_right = st.columns(2)
    with axis_left:
        x_feature = st.selectbox(
            "Horizontal axis",
            options=FEATURE_COLUMNS,
            index=2,
            format_func=lambda c: c.replace(" (cm)", "").capitalize() + " (cm)",
            key="scatter_x",
        )
    with axis_right:
        y_feature = st.selectbox(
            "Vertical axis",
            options=FEATURE_COLUMNS,
            index=3,
            format_func=lambda c: c.replace(" (cm)", "").capitalize() + " (cm)",
            key="scatter_y",
        )

    scatter = go.Figure()
    for species in SPECIES_ORDER:
        subset = frame[frame["species"] == species]
        scatter.add_trace(
            go.Scatter(
                x=subset[x_feature],
                y=subset[y_feature],
                mode="markers",
                name=species,
                marker=dict(
                    size=9,
                    color=SPECIES_COLORS[species],
                    opacity=0.85,
                    line=dict(color=SURFACE, width=1.5),
                ),
                hovertemplate=(
                    f"<b>{species}</b><br>"
                    f"{x_feature}: %{{x:.1f}} cm<br>{y_feature}: %{{y:.1f}} cm<extra></extra>"
                ),
            )
        )
    scatter.update_layout(title=f"{x_feature} vs {y_feature}")
    scatter.update_xaxes(title=f"{x_feature.replace(' (cm)', '').capitalize()} (cm)")
    scatter.update_yaxes(title=f"{y_feature.replace(' (cm)', '').capitalize()} (cm)")
    show_chart(style_figure(scatter, height=440), key="chart_scatter")

    # 4. Correlation heatmap --------------------------------------------------
    subsection("Correlation between measurements")
    st.caption(
        "Petals and sepals move together strongly; sepal width runs slightly "
        "against the rest."
    )
    correlations = frame[FEATURE_COLUMNS].corr()
    heatmap = go.Figure(
        go.Heatmap(
            z=correlations.to_numpy(),
            x=FEATURE_COLUMNS,
            y=FEATURE_COLUMNS,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.0, DIVERGING_LOW],
                [0.5, DIVERGING_MID],
                [1.0, DIVERGING_HIGH],
            ],
            text=[[f"{value:.2f}" for value in row] for row in correlations.to_numpy()],
            texttemplate="%{text}",
            textfont=dict(size=12, family=FONT_MONO),
            hovertemplate="%{y}<br>%{x}<br>r = %{z:.2f}<extra></extra>",
            colorbar=dict(
                title=dict(text="r", font=dict(size=12, color=INK_2)),
                thickness=10,
                len=0.8,
                tickfont=dict(size=11, color=MUTED),
                outlinewidth=0,
            ),
            xgap=2,
            ygap=2,
        )
    )
    heatmap.update_layout(title="Pearson correlation, all 150 flowers", height=430)
    heatmap = style_figure(heatmap, height=430, show_legend=False)
    heatmap.update_xaxes(showgrid=False, ticks="", linecolor="rgba(0,0,0,0)")
    heatmap.update_yaxes(showgrid=False, ticks="", linecolor="rgba(0,0,0,0)", autorange="reversed")
    show_chart(heatmap, key="chart_heatmap")

    # 5. Pair plot ------------------------------------------------------------
    subsection("Every pair of measurements")
    with st.expander("Show the pairwise view", expanded=False):
        pair_plot = px.scatter_matrix(
            frame,
            dimensions=FEATURE_COLUMNS,
            color="species",
            color_discrete_map=SPECIES_COLORS,
            category_orders={"species": SPECIES_ORDER},
            height=760,
            labels={column: column.replace(" (cm)", "") for column in FEATURE_COLUMNS},
        )
        pair_plot.update_traces(
            marker=dict(size=5, opacity=0.8, line=dict(width=0)),
            diagonal_visible=False,
        )
        pair_plot.update_layout(
            title="Pairwise relationships, coloured by species",
            dragmode="select",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        )
        pair_plot = style_figure(pair_plot, height=760)
        pair_plot.update_xaxes(showgrid=False, ticks="outside", ticklen=3)
        pair_plot.update_yaxes(showgrid=False, ticks="outside", ticklen=3)
        show_chart(pair_plot, key="chart_pairplot")

    footnote(
        "Every figure on this page is drawn from "
        "<code>sklearn.datasets.load_iris()</code> — the original 150-flower "
        "dataset, not the trained model."
    )


# ---------------------------------------------------------------------------
# Page 3 — Test a new sample
# ---------------------------------------------------------------------------


def page_predict() -> None:
    section(
        "Test a new sample",
        "Enter four measurements for a flower. Nothing runs until you press "
        "Predict species.",
    )

    if "form_revision" not in st.session_state:
        st.session_state.form_revision = 0
    revision = st.session_state.form_revision

    with st.container(border=True):
        subsection("Measurements")
        left, right = st.columns(2, gap="large")
        values: dict[str, float] = {}

        for index, (column, label, low, high, default, unit) in enumerate(INPUT_SPECS):
            target = left if index % 2 == 0 else right
            with target:
                values[column] = st.number_input(
                    f"{label} ({unit})",
                    min_value=float(low),
                    max_value=float(high),
                    value=float(default),
                    step=0.1,
                    format="%.1f",
                    key=f"input_{index}_{revision}",
                )

        st.write("")
        action_left, action_right, _ = st.columns([1, 1, 2])
        with action_left:
            predict_clicked = st.button("Predict species", type="primary", width="stretch")
        with action_right:
            another_clicked = st.button("Test another flower", width="stretch")

    if another_clicked:
        st.session_state.form_revision = revision + 1
        st.session_state.pop("prediction", None)
        st.rerun()

    model = None
    model_error = None
    try:
        model = load_model(str(MODEL_PATH))
    except ModelLoadError as exc:
        model_error = str(exc)

    if model_error:
        st.error(f"**The model could not be loaded.** {model_error}", icon=":material/error:")
        footnote(
            "Predictions are unavailable until <code>model.pkl</code> is "
            "readable. This app will not train a substitute model, because a "
            "stand-in would not be the model this project ships."
        )
        return

    feature_names, feature_note = verify_input_columns(model)
    class_map, class_note = verify_class_mapping(model)

    if feature_note:
        st.warning(feature_note, icon=":material/warning:")
    if class_note:
        st.warning(class_note, icon=":material/warning:")

    if predict_clicked:
        try:
            row = pd.DataFrame([{name: values[name] for name in feature_names}])[feature_names]

            # The input frame carries proper column names (models fitted on a
            # DataFrame need them for ordering). This model was fitted on a
            # bare array, so scikit-learn warns that the names are unexpected.
            # That is already reported in the UI above, so the warning is
            # silenced here rather than left to spam the server console.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
                predicted = model.predict(row)[0]
                probabilities = None
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(row)[0]

            st.session_state.prediction = {
                "revision": revision,
                "values": dict(values),
                "raw_label": predicted,
                "species": class_map.get(predicted, str(predicted)),
                "probabilities": None if probabilities is None else list(probabilities),
                "classes": list(getattr(model, "classes_", [])),
            }
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            st.session_state.pop("prediction", None)
            st.error(
                "**The prediction failed.** The model raised: "
                f"`{type(exc).__name__}: {exc}`. Check that the four inputs are "
                "within the ranges the model was trained on.",
                icon=":material/error:",
            )

    prediction = st.session_state.get("prediction")
    if not prediction or prediction["revision"] != revision:
        if prediction:
            st.caption("Inputs were reset — predict again to see a result.")
        return

    render_result(prediction, model, class_map)

    if prediction["values"] != values:
        st.caption("The measurements above have changed since this prediction was made.")


def render_result(prediction: dict[str, Any], model: Any, class_map: dict[Any, str]) -> None:
    """The results section: what the model said, and how it compares."""
    species = prediction["species"]
    color = SPECIES_COLORS.get(species, RULE_STRONG)
    estimator_name, estimator_note = describe_model(model)

    section("Result")

    st.html(
        f"""
        <div class="result" style="border-left-color:{color}">
          <div class="result-kicker">Predicted species</div>
          <div class="result-name">{species}</div>
          <p class="result-note">
            The model placed this flower in <strong>{species}</strong> — its
            measurements fall closest to the {species} flowers it was trained
            on. Colour it has been given throughout this dashboard:
            <span style="display:inline-block;width:10px;height:10px;border-radius:50%;
            background:{color};vertical-align:middle;margin:0 2px;"></span>
          </p>
          <div class="result-meta">{estimator_name} &middot; {estimator_note}</div>
        </div>
        """
    )

    frame = get_dataset()
    left, right = st.columns([1, 1], gap="large")

    with left:
        subsection("Submitted measurements")
        submitted = pd.DataFrame(
            {
                "Measurement": [label + " (cm)" for _, label, *_ in INPUT_SPECS],
                "Your flower": [prediction["values"][column] for column, *_ in INPUT_SPECS],
                "Dataset mean": [
                    frame[column].mean() for column, *_ in INPUT_SPECS
                ],
            }
        )
        st.dataframe(
            submitted,
            hide_index=True,
            width="stretch",
            column_config={
                "Your flower": st.column_config.NumberColumn("Your flower", format="%.1f"),
                "Dataset mean": st.column_config.NumberColumn("Dataset mean", format="%.2f"),
            },
        )

    with right:
        subsection("Class probabilities")
        probabilities = prediction["probabilities"]
        if probabilities is None:
            st.info(
                "This model does not expose class probabilities, so none are "
                "shown. Only the predicted species is available.",
                icon=":material/info:",
            )
        else:
            classes = prediction["classes"]
            labels = [class_map.get(label, str(label)) for label in classes]
            order = np.argsort(probabilities)[::-1]
            probability_figure = go.Figure(
                go.Bar(
                    x=[probabilities[i] for i in order],
                    y=[labels[i] for i in order],
                    orientation="h",
                    marker=dict(
                        color=[SPECIES_COLORS.get(labels[i], RULE_STRONG) for i in order],
                        line=dict(color=SURFACE, width=2),
                    ),
                    text=[f"{probabilities[i] * 100:.1f}%" for i in order],
                    textposition="outside",
                    textfont=dict(color=INK_2, size=12),
                    hovertemplate="<b>%{y}</b><br>%{x:.1%}<extra></extra>",
                )
            )
            probability_figure.update_layout(
                title="Share of the forest's votes", showlegend=False, bargap=0.4
            )
            probability_figure.update_xaxes(
                title="Probability", tickformat=".0%", range=[0, 1.18], showgrid=True
            )
            probability_figure.update_yaxes(showgrid=False, autorange="reversed")
            show_chart(style_figure(probability_figure, height=260, show_legend=False), key="chart_proba")

    subsection("Against typical flowers")
    st.caption(
        "Your measurements beside the average of each species in the dataset. "
        "Species averages come from the 150 flowers the model learned from."
    )

    means = species_means(frame)
    comparison = go.Figure()
    comparison.add_trace(
        go.Bar(
            x=FEATURE_COLUMNS,
            y=[prediction["values"][column] for column in FEATURE_COLUMNS],
            name="Your flower",
            marker=dict(color=INK, line=dict(color=SURFACE, width=2)),
            hovertemplate="<b>Your flower</b><br>%{x}: %{y:.1f} cm<extra></extra>",
        )
    )
    # One group per species, in the fixed order, so the species colours here
    # match the result card above and the charts on the data page.
    for species_name in SPECIES_ORDER:
        comparison.add_trace(
            go.Bar(
                x=FEATURE_COLUMNS,
                y=[means.loc[species_name, column] for column in FEATURE_COLUMNS],
                name=species_name,
                marker=dict(color=SPECIES_COLORS[species_name], line=dict(color=SURFACE, width=2)),
                hovertemplate=(
                    f"<b>{species_name}</b><br>%{{x}}: %{{y:.2f}} cm (mean)<extra></extra>"
                ),
            )
        )
    comparison.update_layout(
        barmode="group",
        title="Your flower against the species averages",
        bargap=0.3,
        bargroupgap=0.08,
    )
    comparison.update_xaxes(title="", tickangle=0)
    comparison.update_yaxes(title="Centimetres")
    show_chart(style_figure(comparison, height=430), key="chart_comparison")

    footnote(
        "These predictions are estimates produced by a trained machine learning "
        "model (<code>model.pkl</code>). They describe how closely a flower "
        "resembles the examples the model has seen — not a botanical "
        "identification. No model is perfect, and flowers near a species "
        "boundary can be classified either way."
    )


# ---------------------------------------------------------------------------
# Sidebar and entry point
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Data summary": page_data,
    "Test a new sample": page_predict,
}


def sidebar() -> str:
    with st.sidebar:
        model_found = MODEL_PATH.is_file()
        pip_color = SPECIES_COLORS["Iris virginica"] if model_found else "#C2410C"
        st.html(
            f"""
            <div class="brand">
              <div class="brand-name">Iris Flower Classification</div>
              <div class="brand-sub">Measurements in, species out</div>
              <div class="brand-status">
                <span class="pip" style="background:{pip_color}"></span>
                model.pkl {"found" if model_found else "not found"}
              </div>
            </div>
            """
        )

        st.session_state.setdefault("page", "Overview")
        choice = st.radio(
            "Page",
            options=list(PAGES),
            key="page",
            label_visibility="collapsed",
        )

        st.html(
            """
            <div class="side-note">
              <p>Dataset: <code>sklearn.datasets.load_iris()</code>, 150 flowers.</p>
              <p>Predictions use the saved <code>model.pkl</code> exactly as it is.</p>
            </div>
            """
        )
    return choice


def main() -> None:
    st.set_page_config(
        page_title="Iris Flower Classification",
        page_icon=":material/local_florist:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # The stylesheet must go through st.markdown, not st.html: st.html runs its
    # input through a sanitiser that strips <style> and inline <svg>. Passing it
    # to st.html silently drops the entire design system, leaving the app on
    # Streamlit's default look with no error anywhere.
    _assert_stylesheet_is_parseable()
    st.markdown(CSS, unsafe_allow_html=True)

    _, body, _ = st.columns([1, 22, 1])
    with body:
        PAGES[sidebar()]()


if __name__ == "__main__":
    main()
