# Black Re-ID: A Head-shoulder Descriptor for the Challenging Problem of Person Re-Identification

## Training

To train a model, run

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <config.yml>
```

## Evaluation

To evaluate the model in test set, run similarly:

```bash
CUDA_VISIBLE_DEVICES=gpus python train_net.py --config-file <configs.yaml> --eval-only MODEL.WEIGHTS model.pth
```

## Experimental Results

### Market1501 dataset

| Method | Pretrained | Rank@1 | mAP |
| :---: | :---: | :---: |:---: | 
| ResNet50 | ImageNet | 93.3% | 84.6% | 
| MGN | ImageNet | 95.7% | 86.9% | 
| HAA (ResNet50) | ImageNet | 95% | 87.1% | 
| HAA (MGN) | ImageNet | 95.8% | 89.5% | 

### DukeMTMC dataset

| Method | Pretrained | Rank@1 | mAP | 
| :---: | :---: | :---: |:---: | 
| ResNet50 | ImageNet | 86.2% | 75.3% | 
| MGN | ImageNet | 88.7% | 78.4% | 
| HAA (ResNet50) | ImageNet | 87.7% | 75.7% | 
| HAA (MGN) | ImageNet | 89% | 80.4% | 

### Black-reid black group

| Method | Pretrained | Rank@1 | mAP | 
| :---: | :---: | :---: |:---: | 
| ResNet50 | ImageNet | 80.9% | 70.8% | 
| MGN | ImageNet | 86.7% | 79.1% | 
| HAA (ResNet50) | ImageNet | 86.7% | 79% | 
| HAA (MGN) | ImageNet | 91.0%  | 83.8% | 

### White-reid white group

| Method | Pretrained | Rank@1 | mAP | 
| :---: | :---: | :---: |:---: | 
| ResNet50 | ImageNet | 89.5% | 75.8% | 
| MGN | ImageNet | 94.3% | 85.8% | 
| HAA (ResNet50) | ImageNet | 93.5% | 84.4% | 
| HSE (MGN) | ImageNet | 95.3%  | 88.1% | 

