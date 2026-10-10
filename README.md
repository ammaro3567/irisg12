# Iris Flower Classification

An interactive Streamlit dashboard for a trained Iris classification model. It
has three pages:

| Page | What it does |
| --- | --- |
| **Overview** | Introduces the dataset, the four measurements, the three species, and how a prediction is made. |
| **Data summary** | Explores the 150-flower dataset: preview, descriptive statistics, and five interactive Plotly charts. |
| **Test a new sample** | Takes four measurements, runs the saved model, and shows the predicted species and class probabilities. |

## Purpose

The Iris dataset is a classic: 150 iris flowers, 50 from each of three species,
each described by four measurements in centimetres. Because the species are not
equally separable — *setosa* stands apart, while *versicolor* and *virginica*
overlap — it is a good small problem for showing how a classifier behaves and
where it becomes uncertain.

This app makes a trained model usable without touching the code: enter a
flower's measurements, get a species, and see how that flower compares to the
typical flower of each species.

## Requirements

- **Python 3.10 or newer** (tested on 3.11).
- The packages in `requirements.txt`.

## Install

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Streamlit prints a local URL, normally <http://localhost:8501>. Open it in a
browser. Stop the server with `Ctrl+C`.

## Where `model.pkl` goes

Put `model.pkl` **in the same folder as `app.py`**. The app resolves the path
relative to its own file, so it works no matter which directory you launch it
from:

```text
iris-app/
├── app.py
├── model.pkl          <-- here
├── requirements.txt
└── README.md
```

If the file is missing, unreadable, or does not contain a fitted estimator, the
prediction page shows a clear error. **The app never trains a replacement
model** — a stand-in would not be the model this project ships, so every
prediction shown would be misleading.

## How prediction works

1. **The model is loaded once.** `load_model()` tries `joblib.load()`, then
   falls back to the standard-library `pickle` module, and finally checks the
   loaded object actually has a `predict()` method. It is wrapped in
   `@st.cache_resource`, so the file is read once per server session rather
   than on every interaction. Model loading lives in its own functions, apart
   from the UI code.

2. **The model's own expectations are read, not assumed.** Before predicting,
   the app inspects the estimator:

   - `feature_names_in_` — if present, the input DataFrame is built with those
     names in that order. This model was fitted on a plain NumPy array, so it
     has no such attribute, and the app falls back to the dataset's documented
     column order, telling you it has done so.
   - `classes_` — if the labels are the dataset's `0/1/2` indices, they are
     mapped to `Iris setosa`, `Iris versicolor`, and `Iris virginica`. Any
     other encoding is mapped by class order **and flagged in the UI**, because
     the numbers would then not mean what they appear to mean.
   - `predict_proba()` — used when available, to show class probabilities as
     percentages.

3. **Nothing runs until you press *Predict species***. The inputs sit in a
   DataFrame with one row and four columns, that row goes to `model.predict()`,
   and the resulting label is looked up in the mapping from step 2.

4. **The result** shows the predicted species, your four measurements next to
   the dataset mean, the class probabilities (if the model provides them), and
   a grouped bar chart comparing your flower against the average flower of each
   species.

5. **Test another flower** resets the four inputs to their defaults.

## Notes and limitations

- Every figure on the data page is drawn from
  `sklearn.datasets.load_iris()` — the original dataset, which is *not* the
  same thing as the trained model in `model.pkl`.
- Predictions are estimates from a statistical model, not a botanical
  identification. Flowers near a species boundary can legitimately be
  classified either way.
- The app reports no accuracy or performance figures, because it does not
  evaluate the model — inventing such numbers would be worse than omitting
  them.
- If the model in `model.pkl` was saved with a much older scikit-learn version,
  unpickling may fail. The app reports this rather than hiding it; re-export
  the model with a matching version if it happens.

## Design

A restrained botanical palette carries through the charts and the page chrome
from a single set of tokens at the top of `app.py`:

| Role | Colour |
| --- | --- |
| *Iris setosa* | `#5B4B9E` violet |
| *Iris versicolor* | `#C77E2E` amber |
| *Iris virginica* | `#12876A` teal |

These three were checked for colour-vision separation and for contrast against
the page surface, so the species stay distinguishable in the scatter plots and
stacked histograms without relying on the legend alone.
