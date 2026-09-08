# COCO8 evaluation ground truth

`coco8-annotations.json` contains independent official COCO train2017 annotations
for the existing COCO8 and COCO8-pose validation images: 36, 42, 49, 61, 110, 113,
136 and 151. These annotations are not derived from model predictions.

Source: [official COCO 2017 annotations](https://images.cocodataset.org/annotations/annotations_trainval2017.zip).
The source archive SHA-256 is `113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268`.

Reproduce by filtering `instances_train2017.json` to those eight image IDs,
retaining its `info`, `licenses`, `categories`, matching images and annotations.
For each retained annotation ID also present in `person_keypoints_train2017.json`,
copy its original `keypoints` and `num_keypoints` fields. No box, polygon, area,
crowd flag or keypoint coordinates are generated or changed. The result has eight
images and 66 annotations. The tests select each task's validation images and
map YOLO's contiguous class IDs back to official COCO category IDs.

The fixture retains COCO's source and license metadata. Complete validation-set
accuracy and timing evidence is recorded separately in the PR benchmark report.
