---
comments: true
description: MobileSAM aims to make the recent Segment Anything Model (SAM) lightweight for mobile applications. It keeps has exactly the same functionality as the original SAM but is significantly faster, which makes MobileSAM compatible with CPU-only edge devices, like mobile phones.
keywords: MobileSAM, Faster Segment Anything, Segment Anything, Segment Anything Model, SAM, Meta SAM, image segmentation, promptable segmentation, zero-shot performance, SA-1B dataset, advanced architecture, auto-annotation, Ultralytics, pre-trained models, SAM base, SAM large, instance segmentation, computer vision, AI, artificial intelligence, machine learning, data annotation, segmentation masks, detection model, YOLO detection model, bibtex, Meta AI
---

<p float="center">
  <img src="assets/logo2.png?raw=true" width="99.1%" />
</p>

# Faster Segment Anything (MobileSAM)
:pushpin: MobileSAM paper is available at [ResearchGate](https://www.researchgate.net/publication/371851844_Faster_Segment_Anything_Towards_Lightweight_SAM_for_Mobile_Applications) and [arXiv](https://arxiv.org/pdf/2306.14289.pdf). The latest version will first appear on [ResearchGate](https://arxiv.org/pdf/2306.14289.pdf), since it takes time for arXiv to update the content.

:pushpin: **A demo of MobileSAM** running on **CPU** is open at [demo link](https://huggingface.co/spaces/dhkim2810/MobileSAM). On a Mac i5 CPU, it takes around 3s. On the hugging face demo, the interface and inferior CPUs make it slower but still works fine. 

:grapes: Regarding Segment Anything, there is a trend to replace the original SAM with our MobileSAM in numerous prohects, like [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling), [SegmentAnythingin3D](https://github.com/Jumpat/SegmentAnythingin3D), etc.


:star: **How is MobileSAM trained?** MobileSAM is trained on a single GPU with 100k datasets (1% of the original images) for less than a day. The training code will be available soon.

:star: **How to Adapt from SAM to MobileSAM?** Since MobileSAM keeps exactly the same pipeline as the original SAM, we inherit pre-processing, post-processing, and all other interfaces from the original SAM. Therefore, by assuming everything is exactly the same except for a smaller image encoder, those who use the original SAM for their projects can **adapt to MobileSAM with almost zero effort**.
 
:star: **MobileSAM performs on par with the original SAM (at least visually)** and keeps exactly the same pipeline as the original SAM except for a change on the image encoder. Specifically, we replace the original heavyweight ViT-H encoder (632M) with a much smaller Tiny-ViT (5M). On a single GPU, MobileSAM runs around 12ms per image: 8ms on the image encoder and 4ms on the mask decoder. 

* The comparison of ViT-based image encoder is summarzed as follows: 

    Image Encoder                                      | Original SAM | MobileSAM 
    :-----------------------------------------:|:---------|:-----:
    Paramters      |  611M   | 5M
    Speed      |  452ms  | 8ms

* Original SAM and MobileSAM have exactly the same prompt-guided mask decoder: 

    Mask Decoder                                      | Original SAM | MobileSAM 
    :-----------------------------------------:|:---------|:-----:
    Paramters      |  3.876M   | 3.876M
    Speed      |  4ms  | 4ms

* The comparison of the whole pipeline is summarized as follows:

    Whole Pipeline (Enc+Dec)                                      | Original SAM | MobileSAM 
    :-----------------------------------------:|:---------|:-----:
    Paramters      |  615M   | 9.66M
    Speed      |  456ms  | 12ms

:star: **Original SAM and MobileSAM with a point as the prompt.**  

<p float="left">
  <img src="assets/mask_point.jpg?raw=true" width="99.1%" />
</p>

:star: **Original SAM and MobileSAM with a box as the prompt.** 
<p float="left">
  <img src="assets/mask_box.jpg?raw=true" width="99.1%" />
</p>

:muscle: With superior performance, MobileSAM is arunnd 5 times smaller and 7 times faster than the current FastSAM. See [MobileSAM project](https://github.com/ChaoningZhang/MobileSAM) for more details. 

## Testing MobileSAM in Ultralytics
Following the original SAM, we provide a simple testing method in Ultralytics that includes modes for Point prompt and Box prompt. 

### Point Prompt
```python
from ultralytics import MobileSAM

model = MobileSAM('mobile_sam.pt')
model.predict_point('picture1.jpg',[[400, 400]],[1])
```
### Box Prompt
```python
from ultralytics import MobileSAM

model = MobileSAM('mobile_sam.pt')
model.predict_box('picture1.jpg',[190, 70, 460, 280])
```

### BibTex of our MobileSAM
If you use MobileSAM in your research, please use the following BibTeX entry. :mega: Thank you!

```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```
