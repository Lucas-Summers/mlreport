from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from mlreport import Report

X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

report = Report(
    model,
    title="Classification Report",
    author="Lucas Summers",
    description="CSC 466-02",
    theme="light",
)
report.add_split("train", X_train, y_train, y_pred_train)  # type: ignore
report.add_split("test", X_test, y_test, y_pred_test)  # type: ignore
report.build().to_html("reports/report.html").to_pdf("reports/report.pdf").to_md(
    "reports/report.md"
).to_json("reports/report.json").summary()
