import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape
import os

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "ultralytics", "solutions", "templates")
TEMPLATE_FILE = "similarity-search.html"

@pytest.mark.parametrize("payload", [
    '"><img src=x onerror=alert(1)>',  # exact exploit: breaks out of attribute
    '" onmouseover="alert(1)',           # boundary: event handler injection
    "normal search query",              # valid input: should pass through safely
])
def test_similarity_search_template_escapes_user_input(payload):
    """Invariant: User-supplied query input must always be HTML-escaped in the rendered template,
    preventing injection of event handlers or breaking out of attribute context."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "htm", "xml"]),
    )
    template = env.get_template(TEMPLATE_FILE)

    # Simulate what the template receives from request.form['query']
    rendered = template.render(request=type("R", (), {"form": {"query": payload}})())

    # The raw payload must not appear unescaped in the output
    assert payload not in rendered, (
        f"Unescaped payload found in rendered output: {payload!r}"
    )
    # Specifically, dangerous characters must be escaped
    if "<" in payload or ">" in payload:
        assert "<" not in rendered.split('value=')[1].split('"')[1] if 'value=' in rendered else True
    if '"' in payload:
        # A literal unescaped quote inside an attribute value would break the attribute boundary
        attr_value = rendered.split('value="')[1].split('"')[0] if 'value="' in rendered else ""
        assert '"' not in attr_value, (
            f"Unescaped double-quote found inside attribute value for payload: {payload!r}"
        )