#FROM tensorflow/tensorflow:2.3.1-gpu
FROM tensorflow/tensorflow:2.3.0-gpu
LABEL maintainer="Brad Larson"

RUN echo 'alias py=python' >> ~/.bashrc

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx # for opencv-python
RUN apt-get install unzip

RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
g        opencv-python==4.4.0.42 \
        matplotlib==3.3.1\
        minio==6.0.2 \
        tqdm==4.56.0 \
        natsort==7.0.1 \
        ptvsd==4.3.2 \
        tfds-nightly \
        tensorflow-addons==0.11.2 \
        flask==1.1.2 \
        pycocotools==2.0.1 \
        shutils==0.1.0 \
        tf2onnx==1.8.3

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