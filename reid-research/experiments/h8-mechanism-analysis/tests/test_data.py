"""Unit tests for the h8 Market wrapper. Schema-only — does not need real Market data."""
import pytest

import data as D


def test_record_has_required_fields():
    rec = D.MarketRecord(
        image_id="0001_c1s1_001051_00",
        split="query",
        pid=1,
        camid=0,
        img_path="/tmp/0001_c1s1_001051_00.jpg",
    )
    assert rec.image_id == "0001_c1s1_001051_00"
    assert rec.split == "query"
    assert rec.pid == 1
    assert rec.camid == 0


def test_record_split_must_be_query_or_gallery():
    with pytest.raises(ValueError):
        D.MarketRecord(
            image_id="x", split="train", pid=1, camid=0, img_path="/tmp/x.jpg"
        )
