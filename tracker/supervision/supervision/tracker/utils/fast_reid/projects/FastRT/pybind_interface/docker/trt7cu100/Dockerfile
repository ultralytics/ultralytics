# cuda10.0
FROM fineyu/tensorrt7:0.0.1

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    cmake \
    wget \
    python3.7-dev python3-pip 

RUN add-apt-repository -y ppa:timsc/opencv-3.4 && \
    apt-get update && \
    apt-get install -y \
    libopencv-dev \
    libopencv-dnn-dev \
    libopencv-shape3.4-dbg && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py --force-reinstall && \
    rm get-pip.py

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1 && \
    update-alternatives --set python3 /usr/bin/python3.7

RUN pip install pytest opencv-python 

RUN cd /usr/local/src && \
    wget https://github.com/pybind/pybind11/archive/v2.2.3.tar.gz && \
    tar xvf v2.2.3.tar.gz && \
    cd pybind11-2.2.3 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j12 && \
    make install && \
    cd ../.. && \
    rm -rf pybind11-2.2.3 && \
    rm -rf v2.2.3.tar.gz
