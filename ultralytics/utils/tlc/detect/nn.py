from ultralytics.nn.tasks import DetectionModel

# For backwards compatibility to be able to import TLCDetectionModel from ultralytics.utils.tlc.detect.nn,
# this is needed to load weights trained with previous commits of the integration.
TLCDetectionModel = DetectionModel