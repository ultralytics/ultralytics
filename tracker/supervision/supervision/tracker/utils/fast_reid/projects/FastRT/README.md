# C++ FastReID-TensorRT


Implementation of reid model with TensorRT network definition APIs to build the whole network. 

So we don't use any parsers here.

### How to Run

1. Generate '.wts' file from pytorch with `model_best.pth`

   See [How_to_Generate.md](tools/How_to_Generate.md)

2. Config your model
   
   See [Tensorrt Model Config](#ConfigSection)
   
3. (Optional) Build <a name="step3"></a>`third party` libs

   See [Build third_party section](#third_party)
   
4. Build <a name="step4"></a>`fastrt` execute file
   
   ``` 
   mkdir build
   cd build
   cmake -DBUILD_FASTRT_ENGINE=ON \
         -DBUILD_DEMO=ON \
         -DUSE_CNUMPY=ON ..
   make
   ```

5. Run <a name="step5"></a>`fastrt`
   
   put `model_best.wts` into `FastRT/`

   ``` 
   ./demo/fastrt -s  // serialize model & save as 'xxx.engine' file
   ```

   ``` 
   ./demo/fastrt -d  // deserialize 'xxx.engine' file and run inference
   ```
   
6. Verify the output with pytorch

7. (Optional) Once you verify the result, you can set FP16 for speed up
   ``` 
   mkdir build
   cd build
   cmake -DBUILD_FASTRT_ENGINE=ON \
         -DBUILD_DEMO=ON \
         -DBUILD_FP16=ON ..
   make
   ```
   
   then go to [step 5](#step5) 

8. (Optional) You can use INT8 quantization for speed up

   prepare CALIBRATE DATASET and set the path via cmake. (The path must end with /)

   ``` 
   mkdir build
   cd build
   cmake -DBUILD_FASTRT_ENGINE=ON \
         -DBUILD_DEMO=ON \
         -DBUILD_INT8=ON \
         -DINT8_CALIBRATE_DATASET_PATH="/data/Market-1501-v15.09.15/bounding_box_test/" ..
   make
   ```
   then go to [step 5](#step5)

9. (Optional) Build tensorrt model as shared libs

   ``` 
   mkdir build
   cd build
   cmake -DBUILD_FASTRT_ENGINE=ON \
         -DBUILD_DEMO=OFF \
         -DBUILD_FP16=ON ..
   make
   make install
   ```
   You should find libs in `FastRT/libs/FastRTEngine/`
   
   Now build your application execute file
   ``` 
   cmake -DBUILD_FASTRT_ENGINE=OFF -DBUILD_DEMO=ON ..
   make
   ```

   then go to [step 5](#step5)
   
10. (Optional) Build tensorrt model with python interface, then you can use FastRT model in python.

    ``` 
    mkdir build
    cd build
    cmake -DBUILD_FASTRT_ENGINE=ON \
        -DBUILD_DEMO=ON \
        -DBUILD_PYTHON_INTERFACE=ON ..
    make
    ```
    
    You should get a so file `FastRT/build/pybind_interface/ReID.cpython-37m-x86_64-linux-gnu.so`. 
   
    Then go to [step 5](#step5) to create engine file.

    After that you can import this so file in python, and deserialize engine file to infer in python. 

    You can find use example in `pybind_interface/test.py` and `pybind_interface/market_benchmark.py`.
    
    ``` 
    from PATH_TO_SO_FILE import ReID
    model = ReID(GPU_ID)
    model.build(PATH_TO_YOUR_ENGINEFILE)
    numpy_feature = np.array([model.infer(CV2_FRAME)])
    ```
    
    * `pybind_interface/test.py` use `pybind_interface/docker/trt7cu100/Dockerfile` (without pytorch installed)
    * `pybind_interface/market_benchmark.py` use `pybind_interface/docker/trt7cu102_torch160/Dockerfile` (with pytorch installed)
    
### <a name="ConfigSection"></a>`Tensorrt Model Config`

Edit `FastRT/demo/inference.cpp`, according to your model config

The config is related to [How_to_Generate.md](tools/How_to_Generate.md)

+ Ex1. `sbs_R50-ibn`
```
static const std::string WEIGHTS_PATH = "../sbs_R50-ibn.wts"; 
static const std::string ENGINE_PATH = "./sbs_R50-ibn.engine";

static const int MAX_BATCH_SIZE = 4;
static const int INPUT_H = 384;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 2048;
static const int DEVICE_ID = 0;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r50; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = true; 
static const bool WITH_NL = true;
static const int EMBEDDING_DIM = 0; 
```

+ Ex2. `sbs_R50`
```
static const std::string WEIGHTS_PATH = "../sbs_R50.wts";
static const std::string ENGINE_PATH = "./sbs_R50.engine"; 

static const int MAX_BATCH_SIZE = 4;
static const int INPUT_H = 384;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 2048;
static const int DEVICE_ID = 0;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r50; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = false; 
static const bool WITH_NL = true;
static const int EMBEDDING_DIM = 0; 
```

+ Ex3. `sbs_r34_distill`
```
static const std::string WEIGHTS_PATH = "../sbs_r34_distill.wts"; 
static const std::string ENGINE_PATH = "./sbs_r34_distill.engine";

static const int MAX_BATCH_SIZE = 4;
static const int INPUT_H = 384;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 512;
static const int DEVICE_ID = 0;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r34_distill; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = false; 
static const bool WITH_NL = false;
static const int EMBEDDING_DIM = 0; 
```

+ Ex4.`kd-r34-r101_ibn`
```
static const std::string WEIGHTS_PATH = "../kd_r34_distill.wts"; 
static const std::string ENGINE_PATH = "./kd_r34_distill.engine"; 

static const int MAX_BATCH_SIZE = 4;
static const int INPUT_H = 384;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 512;
static const int DEVICE_ID = 0;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r34_distill; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = false; 
static const bool WITH_NL = false;
static const int EMBEDDING_DIM = 0; 
```


+ Ex5.`kd-r18-r101_ibn`
```
static const std::string WEIGHTS_PATH = "../kd-r18-r101_ibn.wts"; 
static const std::string ENGINE_PATH = "./kd_r18_distill.engine"; 

static const int MAX_BATCH_SIZE = 16;
static const int INPUT_H = 384;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 512;
static const int DEVICE_ID = 1;

static const FastreidBackboneType BACKBONE = FastreidBackboneType::r18_distill; 
static const FastreidHeadType HEAD = FastreidHeadType::EmbeddingHead;
static const FastreidPoolingType HEAD_POOLING = FastreidPoolingType::gempoolP;
static const int LAST_STRIDE = 1;
static const bool WITH_IBNA = true; 
static const bool WITH_NL = false;
static const int EMBEDDING_DIM = 0; 
```

### Supported conversion

*  Backbone: resnet50, resnet34, distill-resnet50, distill-resnet34, distill-resnet18
*  Heads: embedding_head
*  Plugin layers: ibn, non-local
*  Pooling layers: maxpool, avgpool, GeneralizedMeanPooling, GeneralizedMeanPoolingP

### Benchmark

| Model | Engine | Batch size | Image size | Embd | Time |
|:-:|:-:|:-:|:-:|:-:|:-:|
| Vanilla R34 | Python/Pytorch1.6 fp32 | 1 | 256x128 | 512 | 6.49ms | 
| Vanilla R34 | Python/Pytorch1.6 fp32 | 4 | 256x128 | 512 | 7.16ms | 
| Vanilla R34 | C++/trt7 fp32 | 1 | 256x128 | 512 | 2.34ms | 
| Vanilla R34 | C++/trt7 fp32 | 4 | 256x128 | 512 | 3.99ms | 
| Vanilla R34 | C++/trt7 fp16 | 1 | 256x128 | 512 | 1.83ms | 
| Vanilla R34 | C++/trt7 fp16 | 4 | 256x128 | 512 | 2.38ms | 
| Distill R34 | Python/Pytorch1.6 fp32 | 1 | 256x128 | 512 | 5.68ms | 
| Distill R34 | Python/Pytorch1.6 fp32 | 4 | 256x128 | 512 | 6.26ms | 
| Distill R34 | C++/trt7 fp32 | 1 | 256x128 | 512 | 2.36ms | 
| Distill R34 | C++/trt7 fp32 | 4 | 256x128 | 512 | 4.05ms | 
| Distill R34 | C++/trt7 fp16 | 1 | 256x128 | 512 | 1.86ms | 
| Distill R34 | C++/trt7 fp16 | 4 | 256x128 | 512 | 2.68ms | 
| R50-NL-IBN | Python/Pytorch1.6 fp32 | 1 | 256x128 | 2048 | 14.86ms | 
| R50-NL-IBN | Python/Pytorch1.6 fp32 | 4 | 256x128 | 2048 | 15.14ms | 
| R50-NL-IBN | C++/trt7 fp32 | 1 | 256x128 | 2048 | 4.67ms | 
| R50-NL-IBN | C++/trt7 fp32 | 4 | 256x128 | 2048 | 6.15ms | 
| R50-NL-IBN | C++/trt7 fp16 | 1 | 256x128 | 2048 | 2.87ms | 
| R50-NL-IBN | C++/trt7 fp16 | 4 | 256x128 | 2048 | 3.81ms | 

* Time: preprocessing(normalization) + inference (100 times average) 
* GPU: GTX 2080 TI

### Test Environment

1. fastreid v1.0.0 / 2080TI / Ubuntu18.04 / Nvidia driver 435 / cuda10.0 / cudnn7.6.5 / trt7.0.0 / nvinfer7.0.0 / opencv3.2

2. fastreid v1.0.0 / 2080TI / Ubuntu18.04 / Nvidia driver 450 / cuda10.2 / cudnn7.6.5 / trt7.0.0 / nvinfer7.0.0 / opencv3.2

### Installation

* Set up with Docker

   for cuda10.0

   ```
   cd docker/trt7cu100
   sudo docker build -t trt7:cuda100 .
   sudo docker run --gpus all -it --name fastrt -v /home/YOURID/workspace:/workspace -d trt7:cuda100
   // then put the repo into `/home/YOURID/workspace/` before you getin container
   ```

   for cuda10.2

   ```
   cd docker/trt7cu102
   sudo docker build -t trt7:cuda102 .
   sudo docker run --gpus all -it --name fastrt -v /home/YOURID/workspace:/workspace -d trt7:cuda102 
   // then put the repo into `/home/YOURID/workspace/` before you getin container
   ```

* [Installation reference](https://github.com/wang-xinyu/tensorrtx/blob/master/tutorials/install.md)

### Build <a name="third_party"></a> third party

* for read/write numpy

   ```
   cd third_party/cnpy
   cmake -DCMAKE_INSTALL_PREFIX=../../libs/cnpy -DENABLE_STATIC=OFF . && make -j4 && make install
   ```