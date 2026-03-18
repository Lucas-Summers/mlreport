from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from mlreport import Report

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

search = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid={
        "n_neighbors": [3, 5, 7, 9, 11],
        "metric": ["euclidean", "manhattan"],
    },
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

report = Report(
    best_model,
    title="Classification Report (Hyperparameter Search)",
    author="Lucas Summers",
    description="KNN tuning demo",
    theme="light",
)

report.add_split("train", X_train, y_train, y_pred_train)  # type: ignore
report.add_split("test", X_test, y_test, y_pred_test)  # type: ignore
report.add_search(search)

report.build().to_html("reports/report.html").to_pdf("reports/report.pdf").to_md(
    "reports/report.md"
).to_json("reports/report.json")
