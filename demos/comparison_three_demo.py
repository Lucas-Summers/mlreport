from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlreport import ComparisonReport, Report

X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=6,
    n_redundant=2,
    n_classes=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline_model = LogisticRegression(max_iter=1000)
tree_model = LogisticRegression(C=0.5, max_iter=1000)
forest_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42,
)

baseline_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

baseline_report = Report(
    baseline_model,
    title="Logistic Regression",
    author="Lucas Summers",
    description="Three-model comparison demo",
    theme="light",
)
baseline_report.add_split("test", X_test, y_test).build()  # type: ignore

tree_report = Report(
    tree_model,
    title="Logistic Regression Variant",
    author="Lucas Summers",
    description="Three-model comparison demo",
    theme="light",
)
tree_report.add_split("test", X_test, y_test).build()  # type: ignore

forest_report = Report(
    forest_model,
    title="Random Forest",
    author="Lucas Summers",
    description="Three-model comparison demo",
    theme="light",
)
forest_report.add_split("test", X_test, y_test).build()  # type: ignore

comparison = ComparisonReport(
    reports=[baseline_report, tree_report, forest_report],
    title="Comparison Report (3 Models)",
    author="Lucas Summers",
    description="The first report is treated as the baseline",
    split="test",
    theme="light",
)

comparison.build().to_html("reports/report.html").to_pdf("reports/report.pdf").to_md(
    "reports/report.md"
).to_json("reports/report.json").summary()
