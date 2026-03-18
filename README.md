# mlreport

`mlreport` is a Python library for generating useful model evaluation reports for scikit-learn models.

It supports exports to:

- HTML
- PDF
- JSON
- Markdown

## Requirements

- Python 3.10+

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
    model,  # fitted sklearn model
    title="Model Report",
    author="Your Name",
    description="Optional description",
    theme="light",  # or "dark"
    cmap="viridis" # or any matplotlib colormap
)
```

**Note:** only regression and classification models are currently supported.

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
report.add_crossval(X, y, y_pred_cv, cv)  # provide your own OOF predictions
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
report.summary()
report.to_html("file.html")
report.to_md("file.md")
report.to_json("file.json")
report.to_pdf("file.pdf")
```
