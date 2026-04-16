# Comparison Report (3 Models)

Lucas Summers

The first report is treated as the baseline

## Models

| Model | Description | Type | Data | Tuned | Params | Baseline |
|-------|-------------|------|------|-------|--------|----------|
| LogisticRegression | Three-model comparison demo | Classification | Train/Test | None | 14 | Yes |
| DecisionTreeClassifier | Three-model comparison demo | Classification | Train/Test | None | 13 | No |
| RandomForestClassifier | Three-model comparison demo | Classification | Train/Test | None | 19 | No |


## Metrics

| Metric | LogisticRegression | DecisionTreeClassifier | RandomForestClassifier | Best |
|--------|-------|-------|-------|------|
| Accuracy | 0.8200 | 0.8450 (+0.0250) | 0.8500 (+0.0300) | RandomForestClassifier |
| F1 Score (Macro) | 0.8195 | 0.8441 (+0.0246) | 0.8496 (+0.0301) | RandomForestClassifier |
| F1 Score (Weighted) | 0.8195 | 0.8441 (+0.0246) | 0.8496 (+0.0301) | RandomForestClassifier |
| Precision (Macro) | 0.8232 | 0.8529 (+0.0297) | 0.8535 (+0.0303) | RandomForestClassifier |
| Precision (Weighted) | 0.8232 | 0.8529 (+0.0297) | 0.8535 (+0.0303) | RandomForestClassifier |
| Recall (Macro) | 0.8200 | 0.8450 (+0.0250) | 0.8500 (+0.0300) | RandomForestClassifier |
| Recall (Weighted) | 0.8200 | 0.8450 (+0.0250) | 0.8500 (+0.0300) | RandomForestClassifier |



## Visualizations

### Confusion Matrix

#### LogisticRegression

![Confusion Matrix - LogisticRegression](reports/images/comparison_confusion_matrix_logisticregression.png)

#### DecisionTreeClassifier

![Confusion Matrix - DecisionTreeClassifier](reports/images/comparison_confusion_matrix_decisiontreeclassifier.png)

#### RandomForestClassifier

![Confusion Matrix - RandomForestClassifier](reports/images/comparison_confusion_matrix_randomforestclassifier.png)


### Per-Class Metrics

#### LogisticRegression

![Per-Class Metrics - LogisticRegression](reports/images/comparison_per_class_metrics_logisticregression.png)

#### DecisionTreeClassifier

![Per-Class Metrics - DecisionTreeClassifier](reports/images/comparison_per_class_metrics_decisiontreeclassifier.png)

#### RandomForestClassifier

![Per-Class Metrics - RandomForestClassifier](reports/images/comparison_per_class_metrics_randomforestclassifier.png)





---

*Generated at 2026-04-16 15:46 by [mlreport](https://github.com/Lucas-Summers/mlreport)*