# {% if meta.title %}{{ meta.title }}{% else %}Comparison Report{% endif %}

{% if meta.author %}{{ meta.author }}{% endif %}

{% if meta.description %}{{ meta.description }}{% endif %}

## Models

| Model | Description | Type | Data | Tuned | Params | Baseline |
|-------|-------------|------|------|-------|--------|----------|
{% for model in models -%}
| {{ model.key }} | {% if model.description %}{{ model.description }}{% endif %} | {{ model.type }} | {{ model.data_label }} | {{ model.tuned_label }} | {{ model.param_count }} | {% if model.is_baseline %}Yes{% else %}No{% endif %} |
{% endfor %}

## Metrics

| Metric |{% for model in models %} {{ model.key }} |{% endfor %} Best |
|--------|{% for model in models %}-------|{% endfor %}------|
{% for metric in metrics -%}
| {{ metric.metric_name }} |{% for model in models %}{% set value = metric["values"][model.key] %}{% if model.is_baseline %} {{ "%.4f"|format(value) }} |{% else %} {{ "%.4f"|format(value) }} ({{ "%+.4f"|format(metric["deltas"][model.key]) }}) |{% endif %}{% endfor %} {{ metric.best_key }} |
{% endfor %}

{% if plots %}
## Visualizations

{% for plot in plots -%}
### {{ plot.name }}

{% for card in plot.cards -%}
#### {{ card.model_key }}

![{{ plot.name }} - {{ card.model_key }}]({{ card.path }})

{% endfor %}
{% endfor %}
{% endif %}

---

*Generated at {{ meta.generated_at }} by [mlreport](https://github.com/Lucas-Summers/mlreport)*
