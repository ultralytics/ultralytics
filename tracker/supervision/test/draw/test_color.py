from contextlib import ExitStack as DoesNotRaise
from typing import Optional

import pytest

from supervision.draw.color import Color


@pytest.mark.parametrize(
    "color_hex, expected_result, exception",
    [
        ("fff", Color.white(), DoesNotRaise()),
        ("#fff", Color.white(), DoesNotRaise()),
        ("ffffff", Color.white(), DoesNotRaise()),
        ("#ffffff", Color.white(), DoesNotRaise()),
        ("f00", Color.red(), DoesNotRaise()),
        ("0f0", Color.green(), DoesNotRaise()),
        ("00f", Color.blue(), DoesNotRaise()),
        ("#808000", Color(r=128, g=128, b=0), DoesNotRaise()),
        ("", None, pytest.raises(ValueError)),
        ("00", None, pytest.raises(ValueError)),
        ("0000", None, pytest.raises(ValueError)),
        ("0000000", None, pytest.raises(ValueError)),
        ("ffg", None, pytest.raises(ValueError)),
    ],
)
def test_color_from_hex(
    color_hex, expected_result: Optional[Color], exception: Exception
) -> None:
    with exception:
        result = Color.from_hex(color_hex=color_hex)
        assert result == expected_result
