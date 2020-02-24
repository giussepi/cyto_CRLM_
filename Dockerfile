FROM ubuntu:18.04


# Installing dependencies and repositories

RUN apt-get update
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update
RUN add-apt-repository ppa:xorg-edgers/ppa
RUN apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-dev
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y cmake git pkg-config ffmpeg curl gcc g++ wget unzip zlib1g-dev libwebp-dev \
                    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
                    libgtk2.0-dev libavcodec-dev libavformat-dev libswscale-dev uuid-dev python3-pip

RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install libjasper1 libjasper-dev


# Installing nvidia repositories and drivers

RUN DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:graphics-drivers/ppa
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y update
RUN DEBIAN_FRONTEND=noninteractive DEBIAN_FRONTEND=noninteractive apt-get -y install nvidia-settings nvidia-driver-440


# Installing cuda

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.1.243

ENV CUDA_PKG_VERSION 10-1=$CUDA_VERSION-1

# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
cuda-compat-10-1 && \
ln -s cuda-10.1 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64


# Installing lubcudnn7

ENV CUDNN_VERSION 7.6.4.38

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda10.1 \
libcudnn7-dev=$CUDNN_VERSION-1+cuda10.1 \
&& \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


# nvidia-container-runtime

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
# ENV NVIDIA_REQUIRE_CUDA "cuda>=10.1 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411, driver>=440"


# Installing CRLM model

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && apt-get install -y git

RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openslide-tools python-openslide python3-tk

RUN mkdir -p /CRLM

COPY . /CRLM/

RUN pip3 install -r /CRLM/requirements.txt

RUN mv /CRLM/settings.py.docker_image /CRLM/settings.py

ENTRYPOINT ["python3", "/CRLM/cytomine_add_annotations.py"]