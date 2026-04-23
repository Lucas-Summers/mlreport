# {% if meta.title %}{{ meta.title }}{% else %}Comparison Report{% endif %}


{% if meta.author %}{{ meta.author }}{% endif %}


{% if meta.description %}{{ meta.description }}{% endif %}


---

## Models

| Model | Description | Type | Data | Params | Tuned | Baseline |
|-------|-------------|------|------|--------|-------|----------|
{% for model in models -%}
| {{ model.title_name }} | {% if model.description %}{{ model.description }}{% endif %} | {{ model.type }} | {{ model.data_label }} | {{ model.param_count }} | {{ model.tuned_label }} | {% if model.is_baseline %}Yes{% else %}No{% endif %} |
{% endfor %}


## Metrics

| Metric |{% for model in models %} Model {{ model.index + 1 }} |{% endfor %} Best |
|--------|{% for model in models %}-------|{% endfor %}------|
{% for metric in metrics -%}
| {{ metric.metric_name }} |{% for model in models %}{% set value = metric["values"][model.key] %}{% if model.is_baseline %} {{ "%.4f"|format(value) }} |{% else %} {{ "%.4f"|format(value) }} ({{ "%+.4f"|format(metric["deltas"][model.key]) }}) |{% endif %}{% endfor %} Model {{ metric.best_index }} |
{% endfor %}

{% if comparison.mixed_splits %}
Metrics may be drawn from different evaluation splits across models.
{% endif %}

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
