# Model Deployment

This directory contains:

1. The scripts that convert a fastreid model to Caffe/ONNX/TRT format.

2. The exmpales that load a R50 baseline model in Caffe/ONNX/TRT and run inference.

## Tutorial

### Caffe Convert

<details>
<summary>step-to-step pipeline for caffe convert</summary>

This is a tiny example for converting fastreid-baseline in `meta_arch` to Caffe model, if you want to convert more complex architecture, you need to customize more things.

1. Run `caffe_export.py` to get the converted Caffe model,

    ```bash
    python tools/deploy/caffe_export.py --config-file configs/market1501/bagtricks_R50/config.yml --name baseline_R50 --output caffe_R50_model --opts MODEL.WEIGHTS logs/market1501/bagtricks_R50/model_final.pth
    ```

    then you can check the Caffe model and prototxt in `./caffe_R50_model`.

2. Change `prototxt` following next three steps:

   1) Modify `MaxPooling` in `baseline_R50.prototxt` and delete `ceil_mode: false`.
   
   2) Add `avg_pooling` in `baseline_R50.prototxt`

        ```prototxt
        layer {
            name: "avgpool1"
            type: "Pooling"
            bottom: "relu_blob49"
            top: "avgpool_blob1"
            pooling_param {
                pool: AVE
                global_pooling: true
            }
        }
        ```

   2) Change the last layer `top` name to `output`

        ```prototxt
        layer {
            name: "bn_scale54"
            type: "Scale"
            bottom: "batch_norm_blob54"
            top: "output" # bn_norm_blob54
            scale_param {
                bias_term: true
            }
        }
        ```

3. (optional) You can open [Netscope](https://ethereon.github.io/netscope/quickstart.html), then enter you network `prototxt` to visualize the network.

4. Run `caffe_inference.py` to save Caffe model features with input images

   ```bash
    python caffe_inference.py --model-def outputs/caffe_model/baseline_R50.prototxt \
    --model-weights outputs/caffe_model/baseline_R50.caffemodel \
    --input test_data/*.jpg --output caffe_output
   ```

6. Run `demo/demo.py` to get fastreid model features with the same input images, then verify that Caffe and PyTorch are computing the same value for the network.

    ```python
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-6)
    ```

</details>

### ONNX Convert

<details>
<summary>step-to-step pipeline for onnx convert</summary>

This is a tiny example for converting fastreid-baseline in `meta_arch` to ONNX model. ONNX supports most operators in pytorch as far as I know and if some operators are not supported by ONNX, you need to customize these.

1. Run `onnx_export.py` to get the converted ONNX model,

    ```bash
    python onnx_export.py --config-file root-path/bagtricks_R50/config.yml --name baseline_R50 --output outputs/onnx_model --opts MODEL.WEIGHTS root-path/logs/market1501/bagtricks_R50/model_final.pth
    ```

    then you can check the ONNX model in `outputs/onnx_model`.

2. (optional) You can use [Netron](https://github.com/lutzroeder/netron) to visualize the network.

3. Run `onnx_inference.py` to save ONNX model features with input images

   ```bash
    python onnx_inference.py --model-path outputs/onnx_model/baseline_R50.onnx \
    --input test_data/*.jpg --output onnx_output
   ```

4. Run `demo/demo.py` to get fastreid model features with the same input images, then verify that ONNX Runtime and PyTorch are computing the same value for the network.

    ```python
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-6)
    ```

</details>

### TensorRT Convert

<details>
<summary>step-to-step pipeline for trt convert</summary>

This is a tiny example for converting fastreid-baseline in `meta_arch` to TRT model.

First you need to convert the pytorch model to ONNX format following [ONNX Convert](https://github.com/JDAI-CV/fast-reid#fastreid), and you need to remember your `output` name. Then you can convert ONNX model to TensorRT following instructions below.

1. Run command line below to get the converted TRT model from ONNX model,

    ```bash
    python trt_export.py --name baseline_R50 --output outputs/trt_model \
    --mode fp32 --batch-size 8 --height 256 --width 128 \
    --onnx-model outputs/onnx_model/baseline.onnx 
    ```

    then you can check the TRT model in `outputs/trt_model`.

2. Run `trt_inference.py` to save TRT model features with input images

   ```bash
    python3 trt_inference.py --model-path outputs/trt_model/baseline.engine \
    --input test_data/*.jpg --batch-size 8 --height 256 --width 128 --output trt_output 
   ```

3. Run `demo/demo.py` to get fastreid model features with the same input images, then verify that TensorRT and PyTorch are computing the same value for the network.

    ```python
    np.testing.assert_allclose(torch_out, trt_out, rtol=1e-3, atol=1e-6)
    ```

Notice: The int8 mode in tensorRT runtime is not supported now and there are some bugs in calibrator. Need help!

</details>

## Acknowledgements

Thank to [CPFLAME](https://github.com/CPFLAME), [gcong18](https://github.com/gcong18), [YuxiangJohn](https://github.com/YuxiangJohn) and [wiggin66](https://github.com/wiggin66) at JDAI Model Acceleration Group for help in PyTorch model converting.
