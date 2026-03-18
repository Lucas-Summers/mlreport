from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from mlreport import Report

X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

model = LogisticRegression()

splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

report = Report(
    model,
    title="Classification Report (Cross-Validation)",
    author="Lucas Summers",
    description="CSC 466-02",
    theme="light",
)
report.add_crossval(X, y, cv=splitter)  # type: ignore
report.build().to_html("reports/report.html").to_pdf("reports/report.pdf").to_md(
    "reports/report.md"
).to_json("reports/report.json")
