from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlreport import ComparisonReport, Report

X, y = make_classification(
    n_samples=1000,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    n_classes=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

logreg = LogisticRegression(max_iter=1000)
tree = DecisionTreeClassifier(max_depth=5, random_state=42)

logreg.fit(X_train, y_train)
tree.fit(X_train, y_train)

baseline_report = Report(
    logreg,
    title="Logistic Regression",
    author="Lucas Summers",
    description="Two-model comparison demo",
    theme="light",
)
baseline_report.add_split("test", X_test, y_test).build()  # type: ignore

candidate_report = Report(
    tree,
    title="Decision Tree",
    author="Lucas Summers",
    description="Two-model comparison demo",
    theme="light",
)
candidate_report.add_split("test", X_test, y_test).build()  # type: ignore

comparison = ComparisonReport(
    reports=[baseline_report, candidate_report],
    title="Comparison Report (2 Models)",
    author="Lucas Summers",
    description="Baseline first, candidate second",
    split="test",
    theme="light",
)

comparison.build().to_html("reports/report.html").to_pdf("reports/report.pdf").to_md(
    "reports/report.md"
).to_json("reports/report.json").summary()
