# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

import tlc

from ultralytics.models.yolo.model import YOLO

from ultralytics.nn.tasks import ClassificationModel, DetectionModel
from ultralytics.utils.tlc.classify import TLCClassificationTrainer, TLCClassificationValidator
from ultralytics.utils.tlc.detect import TLCDetectionTrainer, TLCDetectionValidator
from ultralytics.utils.tlc.settings import Settings
from ultralytics.utils.tlc.utils import check_tlc_version, reduce_embeddings

from typing import Iterable

class TLCYOLO(YOLO):
    """ YOLO (You Only Look Once) object detection model with 3LC integration. """

    def __init__(self, *args, **kwargs):
        """ Initialize YOLO model with 3LC integration. Checks that the installed version of 3LC is compatible. """
        
        check_tlc_version()

        super().__init__(*args, **kwargs)

    @property
    def task_map(self):
        """ Map head to 3LC model, trainer, validator, and predictor classes. """
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": TLCDetectionTrainer,
                "validator": TLCDetectionValidator,
            },
            "classify": {
                "model": ClassificationModel,
                "trainer": TLCClassificationTrainer,
                "validator": TLCClassificationValidator,
            },
        }
    
    def collect(
        self,
        data: str | None = None,
        splits: Iterable[str] | None = None,
        tables: dict[str, str | tlc.Url | tlc.Table] | None = None,
        settings: Settings | None = None,
        **kwargs
    ) -> dict[str, dict[str, float]]:
        """ Perform calls to model.val() to collect metrics on a set of splits, all under one tlc.Run.
        If enabled, embeddings are reduced at the end of validation.
         
        :param data: Path to a YOLO or 3LC YAML file. If provided, splits must also be provided.
        :param splits: List of splits to collect metrics for. If provided, data must also be provided.
        :param tables: Dictionary of splits to tables to collect metrics for. Mutually exclusive with data and splits.
        :param settings: 3LC settings to use for collecting metrics. If None, default settings are used.
        :param kwargs: Additional keyword arguments are forwarded as model.val(**kwargs).
        :return: Dictionary of split names to results returned by model.val().
        """
        # Verify only data+splits or tables are provided
        if not ( (data and splits ) or tables):
            raise ValueError("Either data and splits or tables must be provided to collect.")

        if settings is None:
            settings = Settings()

        results_dict = {}
        # Call val for each split or table
        if data and splits:
            for split in splits:
                results_dict[split] = self.val(data=data, split=split, settings=settings, **kwargs)
        elif tables:
            for split in tables:
                results_dict[split] = self.val(table=tables[split], settings=settings, **kwargs)

        # Reduce embeddings
        if settings and settings.image_embeddings_dim > 0:
            # TODO: Allow user to pass in preferred foreign_table_url

            reduce_embeddings(
                tlc.active_run(),
                method=settings.image_embeddings_reducer,
                n_components=settings.image_embeddings_dim,
            )

        return results_dict
        
