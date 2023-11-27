# Fastreid Model Deployment

The `gen_wts.py` script convert a fastreid model to [.wts format](https://github.com/wang-xinyu/tensorrtx/blob/master/tutorials/getting_started.md#the-wts-content-format) file, then it will be used in [FastRT](https://github.com/JDAI-CV/fast-reid/blob/master/projects/FastRT) directly. 

### Convert Environment

* Same as fastreid.
    
### How to Generate

This is a general example for converting fastreid to TensorRT model. We use `FastRT` to build the model with nvidia TensorRT APIs.

In this part you need to convert the pytorch model to '.wts' file using `gen_wts.py` follow instructions below.

1. Run command line below to generate the '.wts' file from pytorch model
   
   It's similar to how you use fastreid.
    ```bash
    python projects/FastRT/tools/gen_wts.py --config-file='config/you/use/in/fastreid/xxx.yml' \
    --verify --show_model --wts_path='outputs/trt_model_file/xxx.wts' \
    MODEL.WEIGHTS '/path/to/checkpoint_file/model_best.pth' MODEL.DEVICE "cuda:0"
    ```

    then you can check the TensorRT model weights `outputs/trt_model_file/xxx.wts`.

3. Copy the `outputs/trt_model_file/xxx.wts` to [FastRT](https://github.com/JDAI-CV/fast-reid/blob/master/projects/FastRT)


### More convert examples

+ Ex1. `sbs_R50-ibn`
    - [x] resnet50, ibn, non-local, gempoolp
    ```bash
    python projects/FastRT/tools/gen_wts.py --config-file='configs/DukeMTMC/sbs_R50-ibn.yml' \
    --verify --show_model --wts_path='outputs/trt_model_file/sbs_R50-ibn.wts' \
    MODEL.WEIGHTS '/path/to/checkpoint_file/model_best.pth' MODEL.DEVICE "cuda:0"
    ```
    
+ Ex2. `sbs_R50`
    - [x] resnet50, gempoolp   
    ```bash
    python projects/FastRT/tools/gen_wts.py --config-file='configs/DukeMTMC/sbs_R50.yml' \
    --verify --show_model --wts_path='outputs/trt_model_file/sbs_R50.wts' \
    MODEL.WEIGHTS '/path/to/checkpoint_file/model_best.pth' MODEL.DEVICE "cuda:0"
    ``` 
    
* Ex3. `sbs_r34_distill`
    - [x] train-alone distill-r34 (hint: distill-resnet is slightly different from resnet34), gempoolp
    ```bash
    python projects/FastRT/tools/gen_wts.py --config-file='projects/FastDistill/configs/sbs_r34.yml' \
    --verify --show_model --wts_path='outputs/to/trt_model_file/sbs_r34_distill.wts' \
    MODEL.WEIGHTS '/path/to/checkpoint_file/model_best.pth' MODEL.DEVICE "cuda:0"
    ```

* Ex4.`kd-r34-r101_ibn`
    - [x] teacher model(r101_ibn), student model(distill-r34). the one for deploying is student model, gempoolp
    ```bash
    python projects/FastRT/tools/gen_wts.py --config-file='projects/FastDistill/configs/kd-sbs_r101ibn-sbs_r34.yml' \
    --verify --show_model --wts_path='outputs/to/trt_model_file/kd_r34_distill.wts' \
    MODEL.WEIGHTS '/path/to/checkpoint_file/model_best.pth' MODEL.DEVICE "cuda:0"
    ```

## Acknowledgements

Thanks to [tensorrtx](https://github.com/wang-xinyu/tensorrtx) for demonstrating the usage of trt network definition APIs.

