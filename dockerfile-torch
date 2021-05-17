FROM nvcr.io/nvidia/pytorch:20.12-py3
LABEL maintainer="Brad Larson"

RUN echo 'alias py=python' >> ~/.bashrc
RUN apt-get update

# RUN apt-get install -y libsm6 libxext6 ffmpeg # required by opencv-python==4.4.0.42
RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y && \
    apt-get install -y libgl1-mesa-glx && \
    apt-get install -y wget

# RUN apt-get install --no-install-recommends lsb-release wget less udev sudo -y

# Setup the ZED SDK https://download.stereolabs.com/zedsdk/3.4/cu111/ubuntu20
RUN zedpackage=ZED_SDK_Ubuntu20_cuda11.1_v3.4.2.run && \
    wget -q -O ZED_SDK_Ubuntu20_cuda11.1_v3.4.2.run https://download.stereolabs.com/zedsdk/3.4/cu111/ubuntu20 && \
    chmod +x ZED_SDK_Ubuntu20_cuda11.1_v3.4.2.run && \
    ./ZED_SDK_Ubuntu20_cuda11.1_v3.4.2.run -- silent runtime_only && \
    rm ZED_SDK_Ubuntu20_cuda11.1_v3.4.2.run


RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
        opencv-python==4.5.1.48 \
        minio==7.0.2 \
        tqdm==4.56.0 \
        natsort==7.0.1 \
        ptvsd \
        debugpy \
        path \
        matplotlib\
        torch \
        torchvision \
        tensorboard \
        tensorboardX \
        scipy \
        scikit-image \
        apex \
        wget \
        configparser

# Tutorial dependencies
RUN pip3 --no-cache-dir install \
        cython \
        pycocotools

# ZED Python API https://stackoverflow.com/a/63457606/7036639
RUN wget download.stereolabs.com/zedsdk/pyzed -O /usr/local/zed/get_python_api.py && \
    python3 /usr/local/zed/get_python_api.py && \
    rm *.whl ; rm -rf /var/lib/apt/lists/*

RUN echo 'alias py=python' >> ~/.bashrc

#COPY segment /app/segment/
#COPY utils /app/utils/

RUN git clone https://github.com/NVIDIA/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
# RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /app
ENV LANG C.UTF-8
# port 6006 exposes tensorboard
EXPOSE 6006 
# port 3000 exposes debugger
EXPOSE 3000

# Launch training
#ENTRYPOINT ["python", "segment/train.py"]
RUN ["/bin/bash"]
