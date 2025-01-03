ARG UBUNTU_VERSION=20.04

ARG CUDA=11.8
# https://qiita.com/yutake27/items/3a3a44ea8185887eff1c
# https://hub.docker.com/r/nvidia/cuda/tags?page=&page_size=&ordering=&name=11.8%20
# https://vasteelab.com/2018/08/07/2018-08-07-141446/
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
#https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html#cudnn-versions-linux
ARG CUDNN=9.1.1.*-1
ARG CUDNN_MAJOR_VERSION=8
ARG LIB_DIR_PREFIX=x86_64
ARG LIBNVINFER=7.2.2-1
ARG LIBNVINFER_MAJOR_VERSION=7

# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# Needed for string substitution
SHELL ["/bin/bash", "-c"]
# Pick up some TF dependencies

# libgl1-mesa-dev is need for use OpenCV
# https://cocoinit23.com/docker-opencv-importerror-libgl-so-1-cannot-open-shared-object-file/
# https://docs.nvidia.com/deeplearning/cudnn/latest/installation/linux.html
# https://zenn.dev/pon_pokapoka/articles/nvidia_cuda_install

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt update && apt install -y --no-install-recommends \
        build-essential \
        cuda-command-line-tools-${CUDA/./-} \
        libcublas-${CUDA/./-} \
        cuda-nvrtc-${CUDA/./-} \
        libcufft-${CUDA/./-} \
        libcurand-${CUDA/./-} \
        libcusolver-${CUDA/./-} \
        libcusparse-${CUDA/./-} \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libzmq3-dev \
        pkg-config \
        software-properties-common \
        unzip \
        git \
        vim \
        eog \
        libgl1-mesa-dev

RUN apt update && apt install -y --no-install-recommends wget git cmake gedit

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

###############################################################################################################
# start install python
# https://www.linuxcapable.com/install-python-3-8-on-ubuntu-linux/

RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt update
RUN apt install -y python3.8
RUN apt install -y python3.8-dbg python3.8-dev python3.8-distutils \
                   python3.8-lib2to3 python3.8-tk


# Set alias
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc
RUN . ~/.bashrc

RUN apt autoremove -y

# end install python
###############################################################################################################

###############################################################################################################
# setup pip
# https://www.linuxcapable.com/install-python-3-8-on-ubuntu-linux/

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py
RUN python3.8 -m pip install --upgrade pip

# end setup pip
###############################################################################################################

###############################################################################################################
# Install pytorch and onnx

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir onnx==1.8.1

#COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

RUN apt autoremove -y 
EXPOSE 8888

###############################################################################################################
# Install YOLOX
# https://circleken.net/2021/12/post76/#%E3%82%A8%E3%83%A9%E3%83%BC%E4%BE%8B

RUN midir programs
WORKDIR /programs
RUN git clone https://github.com/Megvii-BaseDetection/YOLOX.git
WORKDIR /programs/YOLOX
RUN pip install -r requirements.txt
RUN pip install -v -e .

###############################################################################################################
# set X windows and GL to bashrc
# https://stackoverflow.com/questions/66497147/cant-run-opengl-on-wsl2#:~ \ 
# :text=To%20solve%20this%2C%20do%20the%20following%3A%20In%20the, \ 
# your%20bashrc%2Fzshrc%20file%20if%20you%20have%20added%20it.

RUN echo 'export DISPLAY=:0.0' >> ~/.bashrc 
RUN echo 'export LIBGL_ALWAYS_INDIRECT=0' >> ~/.bashrc
RUN echo 'export PYTHONPATH="${PYTHONPATH}:/programs/YOLOX/"' >> ~/.bashrc
RUN source ~/.bashrc

