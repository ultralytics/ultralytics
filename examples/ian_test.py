import torch
import multiprocessing as mp
from ultralytics import YOLO, SAM, FastSAM
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import os
from pathlib import Path
from PIL import Image
import numpy as np
from ultralytics.utils.instance import Instances
import matplotlib.pyplot as plt
from ultralytics.utils.ops import resample_segments

this_fpath = os.path.split(Path(__file__).absolute())[:-1]

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)[1:2]
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def test():
    image = np.array(Image.open(
        os.path.join(*(*this_fpath, 'photo_22.jpg'))
    ))
    sam = sam_model_registry["vit_b"](checkpoint=os.path.join(*(*this_fpath, '../sam_vit_b_01ec64.pth'))).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=0.9)
    masks = mask_generator.generate(
        image
    )
    print(masks)
    print(masks[0]['segmentation'].shape)
    #plt.figure(figsize=(10,8))
    #plt.imshow(image)
    #show_anns(masks)
    #plt.axis('off')
    #plt.show() 
    
    sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)[1:2]
    bboxes = np.array([mask['bbox'] for mask in sorted_masks], dtype=np.float32)
    segments = [np.argwhere(mask['segmentation']) for mask in sorted_masks]
    #segments = np.stack(resample_segments(segments, n=1000))
    instances = Instances(bboxes, segments=segments, normalized=False)
    h, w = image.shape[:2]
    print(h, w)
    instances.convert_bbox(format="xyxy")
    #instances.denormalize(w, h)
    print(instances.segments)
    plt.figure(figsize=(10,8))
    plt.imshow(image)

    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0
    m = instances.segments[0].astype(int)
    print(m)
    color_mask = np.concatenate([[1.0, 1.0, 1.0], [.35]])
    row_indices = m[:, 0]
    col_indices = m[:, 1]
    img[row_indices, col_indices] = color_mask
    ax.imshow(img)

    plt.axis('off')
    plt.show() 

def main():
    # Load a pretrained YOLO model
    model = YOLO("yolov8n.yaml")  # build a new model from YAML
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")  # build from YAML and transfer weights

    # Train the model for 50 epochs
    # NOTE: All augmentations other than affine and Copy Past with Auto Segmentation are disabled
    results = model.train(
        data=os.path.join(*(*this_fpath, "../personal_dataset/data.yaml")), 
        epochs=50, imgsz=640,
        hsv_h = 0.00,
        hsv_s=0.0, 
        hsv_v=0.0, 
        #degrees=0.0, 
        #translate=0.0, 
        #scale=0.0, 
        #shear=0.0, 
        #perspective=0.0, 
        flipud=0.0, 
        fliplr=0.0, 
        bgr=0.0, 
        mosaic=0.0,
        mixup=0.0, 
        copy_paste=0.0, 
        copy_paste_sam=0.1, 
        cpsam_imdir = 'bg_photos',
        #auto_augment="", 
        erasing=0.0, 
        crop_fraction=1.0,
    )

if __name__ == "__main__":
    # Spawn multiprocessing required to attach SAM to cuda within dataloader
    torch.multiprocessing.set_start_method('spawn')
    main()