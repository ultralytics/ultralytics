{% macro plan_badges(plans) -%}

<p class="platform-plan-badges" aria-label="Available plans">
  <span>Available on</span>
  {% for plan in plans %}<strong class="platform-plan-badge platform-plan-badge--{{ plan | lower }}">{{ plan }}</strong>{% endfor %}
</p>
{%- endmacro %}
