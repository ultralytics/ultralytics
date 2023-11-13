from enum import Enum
from typing import Dict, Tuple

BASE_VIDEO_URL = "https://media.roboflow.com/supervision/video-examples/"


class VideoAssets(Enum):
    """
    Each member of this enum represents a video asset. The value associated with each
    member is the filename of the video.

    | Enum Member            | Video Filename             | Video URL                                                                             |
    |------------------------|----------------------------|---------------------------------------------------------------------------------------|
    | `VEHICLES`             | `vehicles.mp4`             | [Link](https://media.roboflow.com/supervision/video-examples/vehicles.mp4)            |
    | `MILK_BOTTLING_PLANT`  | `milk-bottling-plant.mp4`  | [Link](https://media.roboflow.com/supervision/video-examples/milk-bottling-plant.mp4) |
    | `VEHICLES_2`           | `vehicles-2.mp4`           | [Link](https://media.roboflow.com/supervision/video-examples/vehicles-2.mp4)          |
    | `GROCERY_STORE`        | `grocery-store.mp4`        | [Link](https://media.roboflow.com/supervision/video-examples/grocery-store.mp4)       |
    | `SUBWAY`               | `subway.mp4`               | [Link](https://media.roboflow.com/supervision/video-examples/subway.mp4)              |
    | `MARKET_SQUARE`        | `market-square.mp4`        | [Link](https://media.roboflow.com/supervision/video-examples/market-square.mp4)       |
    | `PEOPLE_WALKING`       | `people-walking.mp4`       | [Link](https://media.roboflow.com/supervision/video-examples/people-walking.mp4)      |
    """

    VEHICLES = "vehicles.mp4"
    MILK_BOTTLING_PLANT = "milk-bottling-plant.mp4"
    VEHICLES_2 = "vehicles-2.mp4"
    GROCERY_STORE = "grocery-store.mp4"
    SUBWAY = "subway.mp4"
    MARKET_SQUARE = "market-square.mp4"
    PEOPLE_WALKING = "people-walking.mp4"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))


VIDEO_ASSETS: Dict[str, Tuple[str, str]] = {
    VideoAssets.VEHICLES.value: (
        f"{BASE_VIDEO_URL}{VideoAssets.VEHICLES.value}",
        "8155ff4e4de08cfa25f39de96483f918",
    ),
    VideoAssets.VEHICLES_2.value: (
        f"{BASE_VIDEO_URL}{VideoAssets.VEHICLES_2.value}",
        "830af6fba21ffbf14867a7fea595937b",
    ),
    VideoAssets.MILK_BOTTLING_PLANT.value: (
        f"{BASE_VIDEO_URL}{VideoAssets.MILK_BOTTLING_PLANT.value}",
        "9e8fb6e883f842a38b3d34267290bdc7",
    ),
    VideoAssets.GROCERY_STORE.value: (
        f"{BASE_VIDEO_URL}{VideoAssets.GROCERY_STORE.value}",
        "11402e7b861c1980527d3d74cbe3b366",
    ),
    VideoAssets.SUBWAY.value: (
        f"{BASE_VIDEO_URL}{VideoAssets.SUBWAY.value}",
        "453475750691fb23c56a0cffef089194",
    ),
    VideoAssets.MARKET_SQUARE.value: (
        f"{BASE_VIDEO_URL}{VideoAssets.MARKET_SQUARE.value}",
        "859179bf4a21f80a8baabfdb2ed716dc",
    ),
    VideoAssets.PEOPLE_WALKING.value: (
        f"{BASE_VIDEO_URL}{VideoAssets.PEOPLE_WALKING.value}",
        "0574c053c8686c3f1dc0aa3743e45cb9",
    ),
}
