# cuda10.2
FROM nvcr.io/nvidia/tensorrt:20.03-py3

RUN apt-get update && apt-get dist-upgrade -y && \
    apt-get install -y \
    software-properties-common \
    build-essential \
    cmake \
    git \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \ 	
    libdc1394-22-dev libgl1-mesa-glx && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    
RUN mkdir opencv34 && cd opencv34 && \
    git clone -b 3.4 https://github.com/opencv/opencv && \
    git clone -b 3.4 https://github.com/opencv/opencv_contrib && \
    mkdir build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/opencv \
    -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_opencv_xfeatures2d=OFF \
    -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules ../opencv && \
    make -j12 && \
    make install && \
    ldconfig && \
    cd ../.. \
    && rm -rf opencv34
