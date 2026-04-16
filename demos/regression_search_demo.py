from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

from mlreport import Report

X, y = make_regression(  # type: ignore
    n_samples=1000,
    n_features=10,
    n_informative=8,
    noise=20,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid={
        "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        "bootstrap": [True, False],
    },
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

report = Report(
    best_model,
    title="Regression Report (Categorical x Categorical Search)",
    author="Lucas Summers",
    description="CSC 466-02",
    theme="light",
)

report.add_split("train", X_train, y_train, y_pred_train)  # type: ignore
report.add_split("test", X_test, y_test, y_pred_test)  # type: ignore
report.add_search(search)

report.build().to_html("reports/report.html").to_pdf("reports/report.pdf").to_md(
    "reports/report.md"
).to_json("reports/report.json").summary()
