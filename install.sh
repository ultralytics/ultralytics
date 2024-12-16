set -e
set -x

conda create -n yoloe python=3.10 -y
conda activate yoloe

pip install -r requirements.txt

# Generate segmentation data
# python tools/generate_sam_masks.py --img-path ../datasets/Objects365v1/images/train --json-path ../datasets/Objects365v1/annotations/objects365_train.json --batch
# python tools/generate_sam_masks.py --img-path ../datasets/flickr/full_images/ --json-path ../datasets/flickr/annotations/final_flickr_separateGT_train.json
# python tools/generate_sam_masks.py --img-path ../datasets/mixed_grounding/gqa/images --json-path ../datasets/mixed_grounding/annotations/final_mixed_train_no_coco.json

# Generate data
# python tools/generate_objects365v1.py

# Generate grounding segmentation cache
# python tools/generate_grounding_cache.py --img-path ../datasets/flickr/full_images/ --json-path ../datasets/flickr/annotations/final_flickr_separateGT_train_segm.json
# python tools/generate_grounding_cache.py --img-path ../datasets/mixed_grounding/gqa/images --json-path ../datasets/mixed_grounding/annotations/final_mixed_train_no_coco_segm.json

# Verify data
# python tools/verify_objects365.py
# python tools/verify_lvis.py

# Generate train label embeddings
# python tools/generate_label_embedding.py
# python tools/generate_global_neg_cat.py
# python tools/generate_lvis_visual_prompt_data.py