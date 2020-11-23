#FROM tensorflow/tensorflow:2.3.1-gpu
FROM tensorflow/tensorflow:2.3.0-gpu
LABEL maintainer="Brad Larson"

RUN echo 'alias py=python' >> ~/.bashrc

RUN apt-get update
RUN apt-get install -y libsm6 libxext6 ffmpeg # required by opencv-python==4.4.0.42
RUN apt-get install unzip

# numpy==1.18.5 tensorflow-gpu 2.3.0 requires numpy<1.19.0,>=1.16.0 https://github.com/tensorflow/models/issues/9200
# scipy==1.4.1 tensorflow-gpu 2.3.0 requires scipy==1.4.1
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
        opencv-python==4.4.0.42 \
        #numpy==1.18.5 \ 
        #scipy==1.4.1 \ 
        matplotlib==3.3.1\
        minio==6.0.0 \
        tqdm==4.48.2 \
        natsort==7.0.1 \
        ptvsd==4.3.2 \
        tifffile==2020.5.5 \
        xmltodict==0.12.0 \
        tensorflow-model-optimization==0.4.1 \
        tfds-nightly \
        onnxruntime==1.5.2\
        keras2onnx==1.7.0 \
        tf2onnx==1.7.1\
        tensorflow-addons==0.11.2 \
        flask==1.1.2 \
        pycocotools==2.0.1 \
        shutils==0.1.0

RUN pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_x86_64.whl

WORKDIR /app
ENV LANG C.UTF-8

# jupyter port
EXPOSE 8888 
# tensorboard port
EXPOSE 6006
# debugger port
EXPOSE 3000 
# flask port
EXPOSE 5000 

# Launch shell script
RUN ["/bin/bash"]