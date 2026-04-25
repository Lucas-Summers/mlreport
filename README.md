# mlreport

`mlreport` is a Python library for generating useful model evaluation reports for scikit-learn models.

It supports exports to:

- HTML
- PDF
- JSON
- Markdown
- TXT

It also supports comparison reports built from multiple model reports.

## Installation

Clone the repository:

```bash
git clone https://github.com/Lucas-Summers/mlreport.git
cd mlreport
```

Create and activate a virtual environment, then install in editable mode:

```bash
git clone
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## API flow

### 1) Create a report

```python
from mlreport import Report

report = Report(
    model,  # fitted sklearn or custom model
    title="Model Report",
    author="Your Name",
    description="Optional description",
    theme="light",  # or "dark"
    cmap="viridis", # or any matplotlib colormap
    model_type=None, # optional: "classification" or "regression"
    model_params=None # optional dict of hyperparameters
)
```

When `model_type` is provided, it is used to choose the report type. Otherwise,
scikit-learn model types are detected automatically. When `model_params` is
provided, those parameters are shown in the report. Otherwise, scikit-learn
models use `get_params()`.

For custom models that do not implement `get_params()`, pass both
`model_type` and `model_params`:

```python
report = Report(
    custom_model,
    model_type="classification",
    model_params={"threshold": 0.4},
)
```

**Note:** only regression and classification reports are currently supported.

### 2) Add evaluation data (choose one path)

Train/test splits:

```python
report.add_split("train", X_train, y_train, y_pred_train)
report.add_split("test", X_test, y_test, y_pred_test)
```

OR cross-validation:

```python
report.add_crossval(X, y, cv=cv)          # y_pred computed via cross_val_predict
# or
report.add_crossval(X, y, y_pred_cv)      # provide your own OOF predictions
```

When `cv` is provided, fold-level metric summaries are shown. `cv` can be a
splitter object or an integer fold count. For custom models, pass precomputed
out-of-fold predictions. If you also want fold-level summaries, provide either
the splitter used to create those predictions or one fold id per row:

```python
report.add_crossval(X, y, y_pred_cv, cv=cv)
report.add_crossval(X, y, y_pred_cv, fold_ids=fold_ids)
```

### 3) Optionally add hyperparameter search results

```python
report.add_search(search_cv)  # fitted search object with cv_results_
```

### 4) Build metrics and plots

```python
report.build(
    exclude_metrics=[],  # optional metric IDs to skip
    exclude_plots=[],    # optional plot IDs to skip
)
```

**Tip:** To discover valid metric and plot IDs for `exclude_metrics` and `exclude_plots`, run:

```python
report.available_metrics()
report.available_plots()
```

### 5) Export outputs

```python
report.to_txt()
report.to_html("file.html")
report.to_md("file.md")
report.to_json("file.json")
report.to_pdf("file.pdf")
```

Themes are now stylesheet-based. The built-in `light` and `dark` themes select
different CSS files while sharing the same report template.

## Model comparison

Build the individual reports first, then compare them:

```python
from mlreport import ComparisonReport, Report

report_a = (
    Report(model_a, title="Baseline", theme="light")
    .add_split("test", X_test, y_test)
    .build()
)

report_b = (
    Report(model_b, title="Candidate", theme="light")
    .add_split("test", X_test, y_test)
    .build()
)

comparison = ComparisonReport(
    reports=[report_a, report_b],
    title="Model Comparison",
    split="test",
    theme="light",
).build()

comparison.to_txt()
comparison.to_html("comparison.html")
comparison.to_md("comparison.md")
comparison.to_json("comparison.json")
comparison.to_pdf("comparison.pdf")
```

`reports[0]` is treated as the baseline model when displaying deltas.
