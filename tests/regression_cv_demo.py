from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from mlreport import Report

X, y = make_regression(n_samples=1000, n_features=5, noise=20, random_state=42)  # type: ignore

model = LinearRegression()

splitter = KFold(n_splits=5, shuffle=True, random_state=42)

report = Report(
    model,
    title="Regression Report (Cross-Validation)",
    author="Lucas Summers",
    description="CSC 466-02",
    theme="light",
)
report.add_crossval(X, y, cv=splitter)  # type: ignore
report.build().to_html("reports/report.html").to_pdf("reports/report.pdf").to_md(
    "reports/report.md"
).to_json("reports/report.json")
