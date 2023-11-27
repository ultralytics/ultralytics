# cuda10.2
FROM darrenhsieh1717/trt7-cu102-cv34:pybind

RUN pip install torch==1.6.0 torchvision==0.7.0

RUN pip install opencv-python tensorboard cython yacs termcolor scikit-learn tabulate gdown gpustat ipdb h5py fs faiss-gpu

RUN git clone https://github.com/NVIDIA/apex && \
    cd apex && \
    python3 setup.py install
