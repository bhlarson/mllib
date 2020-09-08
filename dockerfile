FROM tensorflow/tensorflow:2.2.0-gpu
LABEL maintainer="Brad Larson"

RUN echo 'alias py=python' >> ~/.bashrc

RUN apt-get install -y libsm6 libxext6 libxrender-dev graphviz
RUN pip3 install --upgrade pip
RUN pip3 --no-cache-dir install \
        opencv-python==4.2.0.34 \
        scipy==1.4.1 \
        numpy==1.18.2 \
        Pillow==7.1.1 \
        minio==5.0.10 \
        natsort==7.0.1 \
        ptvsd==4.3.2 \
        matplotlib==3.1.2\
        tifffile==2020.5.5 \
        xmltodict==0.12.0 \
        #tensorflow_datasets==3.1.0 \
        tfds-nightly \
        tqdm==4.46.0 \
        pydot==1.4.1 \
        graphviz==0.14 \
        tensorflow-addons==0.10.0 \
        flask==1.1.2 \
        pycocotools==2.0.1 \
        shutils

WORKDIR /app
ENV LANG C.UTF-8

# jupyter port
EXPOSE 8888 
# tensorboard port
EXPOSE 8008
# debugger port
EXPOSE 3000 
# flask port
EXPOSE 5000 

# Launch shell script
RUN ["/bin/bash"]