# {% if meta.title %}{{ meta.title }}{% else %}Report - {{ model.name }}{% endif %}

{% if meta.author %}{{ meta.author }}{% endif %}

{% if meta.description %}{{ meta.description }}{% endif %}

## Model Overview

| Parameter | Value |
|-----------|-------|
| Name | {{ model.name }} |
| Type | {{ model.type }} |
| Sklearn | {{ model.version }} |

## Hyperparameters

| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
{% set tuned_params = tuning.summary.best_params if tuning.summary else {} %}
{% set params = model.params.items() | list %}
{% for i in range(0, params | length, 2) -%}
| {% if params[i][0] in tuned_params %}**{{ params[i][0] }}**{% else %}{{ params[i][0] }}{% endif %} | {% if params[i][0] in tuned_params %}**{{ params[i][1] }}**{% else %}{{ params[i][1] }}{% endif %} | {% if i + 1 < params | length %}{% if params[i + 1][0] in tuned_params %}**{{ params[i + 1][0] }}**{% else %}{{ params[i + 1][0] }}{% endif %}{% endif %} | {% if i + 1 < params | length %}{% if params[i + 1][0] in tuned_params %}**{{ params[i + 1][1] }}**{% else %}{{ params[i + 1][1] }}{% endif %}{% endif %} |
{% endfor %}

{% if tuning.summary %}
### Tuning Summary

| Property | Value |
|----------|-------|
| Method | {{ tuning.summary.method }} |
| Metric | {{ tuning.summary.metric }} |
| CV Folds | {% if tuning.summary.cv_folds is not none %}{{ tuning.summary.cv_folds }}{% else %}None{% endif %} |
| Candidates | {{ tuning.summary.n_candidates }} |
| Best Score | {% if tuning.summary.best_score is not none %}{{ "%.4f"|format(tuning.summary.best_score) }}{% else %}None{% endif %} |

{% if tuning.plots %}
### Tuning Plots

{% for plot_id, plot_data in tuning.plots.items() -%}
#### {{ plot_data.name }}

![{{ plot_data.name }}]({{ plot_data.path }})

{% endfor %}
{% endif %}
{% endif %}

## Dataset Overview

| Property | Value |
|----------|-------|
| Features | {{ data.features }} |
| Total Observations | {{ data.total }} |
| CV Folds | {% if data.cv_folds is not none %}{{ data.cv_folds }}{% else %}None{% endif %} |
{% for split_name, count in data.splits.items() -%}
{% if split_name != 'cv' -%}
| {{ split_name | capitalize }} | {{ count }} ({{ "%.1f"|format(count / data.total * 100) }}%) |
{% endif -%}
{% endfor %}

{% if data.class_distribution %}
## Class Distribution

{% set first_split = data.splits.keys() | list | first %}
| Class | {% for split_name in data.splits.keys() %}{{ split_name | capitalize }} | {% endfor %}Overall % |
|------|{% for split_name in data.splits.keys() %}-------|{% endfor %}-----------|
{% for class_label in data.class_distribution[first_split].keys() -%}
| {{ class_label }} | {% for split_name in data.splits.keys() %}{{ data.class_distribution[split_name][class_label] }} | {% endfor %}{{ "%.1f"|format(data.class_percentages[class_label]) }}% |
{% endfor %}
{% endif %}

{% if data.cv_folds is not none %}
## Metrics (Cross-Validation)

| Metric | Mean | Std Dev | Min | Max |
|--------|------|---------|-----|-----|
{% for metric_id, metric_data in metrics.items() -%}
{% set summary = metric_data["values"]["cv"] %}
| {{ metric_data.name }} | {{ "%.4f"|format(summary["mean"]) }} | {{ "%.4f"|format(summary["std"]) }} | {{ "%.4f"|format(summary["min"]) }} | {{ "%.4f"|format(summary["max"]) }} |
{% endfor %}
{% else %}
## Metrics (Train/Test Split)
{% set split_names = metrics[metrics.keys()|list|first]["values"].keys()|reject("equalto", "per_class")|list %}

| Metric |{% for split_name in split_names %} {{ split_name | capitalize }} |{% endfor %}{% if 'train' in split_names and 'test' in split_names %} Gap |{% endif %}
|--------|{% for split_name in split_names %}-------|{% endfor %}{% if 'train' in split_names and 'test' in split_names %}-------|{% endif %}
{% for metric_id, metric_data in metrics.items() -%}
| {{ metric_data.name }} |{% for split_name in split_names %} {{ "%.4f"|format(metric_data["values"][split_name]) }} |{% endfor %}{% if 'train' in split_names and 'test' in split_names %} {{ "%+.4f"|format(metric_data["values"]["train"] - metric_data["values"]["test"]) }} |{% endif %}
{% endfor %}

{% if 'train' in split_names and 'test' in split_names %}
*Gap = Train - Test. Large positive values may indicate overfitting.*
{% endif %}
{% endif %}

{% if "precision_macro" in metrics and "per_class" in metrics["precision_macro"]["values"] %}
## Metrics (Per-Class)

{% if data.cv_folds is not none %}
{% set p = metrics["precision_macro"]["values"]["per_class"]["cv"] %}
{% set r = metrics["recall_macro"]["values"]["per_class"]["cv"] %}
{% set f = metrics["f1_macro"]["values"]["per_class"]["cv"] %}
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
{% for class_label in p.keys() -%}
| {{ class_label }} | {{ "%.4f"|format(p[class_label]) }} | {{ "%.4f"|format(r[class_label]) }} | {{ "%.4f"|format(f[class_label]) }} |
{% endfor %}
{% else %}
{% for split_name in split_names %}
### {{ split_name | capitalize }}

{% set p = metrics["precision_macro"]["values"]["per_class"][split_name] %}
{% set r = metrics["recall_macro"]["values"]["per_class"][split_name] %}
{% set f = metrics["f1_macro"]["values"]["per_class"][split_name] %}
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
{% for class_label in p.keys() -%}
| {{ class_label }} | {{ "%.4f"|format(p[class_label]) }} | {{ "%.4f"|format(r[class_label]) }} | {{ "%.4f"|format(f[class_label]) }} |
{% endfor %}

{% endfor %}
{% endif %}
{% endif %}

## Visualizations

{% for plot_id, plot_data in plots.items() -%}
### {{ plot_data.name }}

![{{ plot_data.name }}]({{ plot_data.path }})

{% endfor %}

---

*Generated at {{ meta.generated_at }} by [mlreport](https://github.com/Lucas-Summers/mlreport)*
