# Three threads inference

##### Three threads inference in Yolov8 with Opencv DNN C++ loading Onnx, and Python.

C++ or Python, it all have three thread to cowork. each of them works independently. Like customer and producer.

1. the first thread reads video frame;

2. the second thread predict the frame;

3. the third thread plot the prediction result to the frame and save it to video;

Reading video thread is a producer, predicting the frame thread is a customer. Then predicting the frame thread will give a prediction result, now it is a producer, plotting and saving thread is a customer. To write these function, we need a queue which is safe between different threads. Python have a Queue which is thread-safe in queue package. C++ doesn't provide such thread-safe queue, so I write a thread-safe queue by myself. Reference is the book [C++ Concurrency in Action PRACTICAL MULTITHREADING](https://www.manning.com/books/c-plus-plus-concurrency-in-action), [Queue](https://docs.python.org/3.12/library/queue.html#queue-objects) class in queue package in Python standard packages, and the book's codes [https://github.com/ZouJiu1/multithread_Cplusplus](https://github.com/ZouJiu1/multithread_Cplusplus).

## C++

the program can run in window10 or ubuntu Linux system

#### Exporting YOLOv8 Onnx

To export yolov8n.onnx model:

```
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format=r"onnx", batch = 2, simplify=True) # batch=1, if your onnx input batch = 1
```

#### 1. Window10

1\) according to [opencv get-started](https://opencv.org/get-started/), download the window release, like [windows.exe](https://github.com/opencv/opencv/releases/download/4.9.0/opencv-4.9.0-windows.exe). and install it.

next you need add some path to global variable, setting->system->about->high system set->environment variable->sys variable->Path->create. like below:

```bash []
C:\ruanjian\opencv\opencv\build\x64\vc16\lib
C:\ruanjian\opencv\opencv\build\include
C:\ruanjian\opencv\opencv\build\x64\vc16\bin
```

2\) open the Microsoft Visual studio 2022 community, file->create->create project from existing codes, choose the visual C++ --> YOLOv8-Threads-Cpp-Python-Video-> console application. Now you import the project.

3\) Microsoft Visual studio--> project --> YOLOv8-Threads-Cpp-Python-Video and property --> VC++ directory

```bash []
IncludePath: C:\ruanjian\opencv\opencv\build\include
LibraryPath: C:\ruanjian\opencv\opencv\build\x64\vc16\lib
```

Microsoft Visual studio--> project --> YOLOv8-Threads-Cpp-Python-Video and property --> linker --> input --> AdditionalDependencies

```
opencv_world*.lib
```

Microsoft Visual studio--> project --> YOLOv8-Threads-Cpp-Python-Video and property --> common --> C++ language standard --> C++20. CXX_STANDARD should larger than or equal C++17.

you should choose the Release not Debug, because opencv_world\*.lib don't support debug. If you want to debug, you should compile the opencv from source and choose BUILD_opencv_world=False, disable world lib. Then you will get so many static \*.lib file, add them to VS library path, which can be used to debug.

## Run

after you compile and generate the codes, you can run the program, you need do some modifications like directory path in multi_thread_read_predict_write.cpp. The batch size of the exported Onnx should be same with the batch_size in multi_thread_read_predict_write.cpp.

if you have cuda which can be used in opencv, you should build opencv with cuda, and set GPU=true.

```
GPU
batch_size
projectBasePath
Onnx_path
video_path
```

demo video can download from

[https://motchallenge.net/vis/MOT16-09](https://motchallenge.net/vis/MOT16-09)

[https://motchallenge.net/vis/MOT16-06](https://motchallenge.net/vis/MOT16-06)

or

[https://www.alipan.com/s/p9MQVG1wrCr](https://www.alipan.com/s/p9MQVG1wrCr)

#### 2. Ubuntu

1). Install the opencv

```bash
git clone https://github.com/opencv/opencv.git
cd opencv
git clone https://github.com/opencv/opencv_contrib.git
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_FFMPEG=ON -DWITH_OPENCL=ON -DWITH_CUDA=ON  -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ..
make -j10
make install
vim ~/.bashrc
# add below to .bashrc tail
...
export PATH=/usr/local/lib:/usr/local/include/opencv4:$PATH
export LIB_PATH=/usr/local/lib:$LIB_PATH
export LD_LIB_PATH=/usr/local/lib:$LD_LIB_PATH
...
source ~/.bashrc
```

2\) compile and run

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd examples/YOLOv8-Threads-Cpp-Python-Video

mkdir build
cd build
cmake ..
make
./YOLOv8ThreadsCppPythonVideo
```

if you program can not run with .mp4 input video in Linux system, you can use .avi input video, .avi input video can run normally. Also you should modify codes.

```C++ []
string output_path = video_path.substr(0, video_path.size() - 4) + "_output.avi";
const int fourcc = VideoWriter::fourcc('X', 'V', 'I', 'D');  // for avi
// cv2.VideoWriter_fourcc('I', '4', '2', '0') // for avi
// cv2.VideoWriter_fourcc('P', 'I', 'M', 'I') // for avi
```

## Python

win or linux all can run it.

> pip install ultralytics

```Python []
python multi_thread_read_predict_write.py
```

before running, you need do some modifications in multi_thread_read_predict_write.cpp like above Window10.

## thread-safe queue

I write the thread_safe_Queue.hpp, a thread-safe queue, can set the max size of the queue. If the queue size equal to the max size, it will wait until the queue size smaller than the max size. If the queue is not empty, it can get. If the queue is not full, it can put. Also the queue have front, empty, full, qsize, join, task_done functions.

## Reference

[https://opencv.org/](https://opencv.org/)

[C++ Concurrency in Action PRACTICAL MULTITHREADING](https://www.manning.com/books/c-plus-plus-concurrency-in-action)

[Queue](https://docs.python.org/3.12/library/queue.html#queue-objects)
