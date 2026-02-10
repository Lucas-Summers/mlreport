# {% if meta.title %}{{ meta.title }}{% else %}Report - {{ model.name }}{% endif %}

{% if meta.author %}{{ meta.author }}{% endif %}

{% if meta.description %}{{ meta.description }}{% endif %}

## Model

| Parameter | Value |
|-----------|-------|
| Name | {{ model.name }} |
| Type | {{ model.type }} |
| Sklearn | {{ model.version }} |

## Hyperparameters

| Parameter | Value |
|-----------|-------|
{% for key, value in model.params.items() -%}
| {{ key }} | {{ value }} |
{% endfor %}

## Data

| Property | Value |
|----------|-------|
| Features | {{ data.features }} |
| Total Observations | {{ data.total }} |
{% for split_name, count in data.splits.items() -%}
| {{ split_name | capitalize }} | {{ count }} ({{ "%.1f"|format(count / data.total * 100) }}%) |
{% endfor %}

## Metrics
{% set split_names = metrics[metrics.keys()|list|first]["values"].keys()|list %}

| Metric |{% for split_name in split_names %} {{ split_name | capitalize }} |{% endfor %}{% if 'train' in split_names and 'test' in split_names %} Gap |{% endif %}

|--------|{% for split_name in split_names %}-------|{% endfor %}{% if 'train' in split_names and 'test' in split_names %}-------|{% endif %}

{% for metric_id, metric_data in metrics.items() -%}
| {{ metric_data.name }} |{% for split_name in split_names %} {{ "%.4f"|format(metric_data["values"][split_name]) }} |{% endfor %}{% if 'train' in split_names and 'test' in split_names %} {{ "%+.4f"|format(metric_data["values"]["train"] - metric_data["values"]["test"]) }} |{% endif %}

{% endfor %}

{% if 'train' in split_names and 'test' in split_names %}
*Gap = Train - Test. Large positive values may indicate overfitting.*
{% endif %}

## Plots

{% for plot_id, plot_data in plots.items() -%}
### {{ plot_data.name }}

![{{ plot_data.name }}]({{ plot_data.path }})

{% endfor %}

---

*Generated at {{ meta.generated_at }} by [mlreport](https://github.com/Lucas-Summers/mlreport)*
