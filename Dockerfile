ARG UBUNTU_VERSION=20.04

ARG ARCH=
ARG CUDA=11.8
# https://qiita.com/yutake27/items/3a3a44ea8185887eff1c
# https://hub.docker.com/r/nvidia/cuda/tags?page=&page_size=&ordering=&name=11.8%20
# https://vasteelab.com/2018/08/07/2018-08-07-141446/
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
# ARCH and CUDA are specified again because the FROM directive resets ARGs
# (but their default value is retained if set previously)
ARG ARCH
ARG CUDA
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



# Link the libcuda stub to the location where tensorflow is searching for it and reconfigure
# dynamic linker run-time bindings
#RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
#    && echo "/usr/local/cuda/lib64/stubs" > /etc/ld.so.conf.d/z-cuda-stubs.conf \
#    && ldconfig

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

######################################################################
# start install python
# https://notes.nakurei.com/post/build-python3-environment-with-docker-ubuntu/
######################################################################

ARG python_version="3.12.3"

# Set locale
RUN apt update -y \
    && apt install -y --no-install-recommends \
    language-pack-en
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime
RUN apt update -y \
    && apt install -y --no-install-recommends \
    tzdata

# Install packages
# Ref: https://github.com/pyenv/pyenv/wiki#suggested-build-environment
RUN apt update -y \
    && apt install -y --no-install-recommends \
    git \
    wget \
    xz-utils \
    build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
    libgdbm-dev libdb-dev uuid-dev

# Install Python
RUN cd /usr/local/src \
    && wget --no-check-certificate https://www.python.org/ftp/python/${python_version}/Python-${python_version}.tar.xz \
    && tar -Jxvf Python-${python_version}.tar.xz \
    && cd Python-${python_version} \
    && ./configure --with-ensurepip \
    && make \
    && make install

# Set alias
RUN echo 'alias python=python3' >> ~/.bashrc
RUN echo 'alias pip=pip3' >> ~/.bashrc
RUN . ~/.bashrc

RUN apt autoremove -y

######################################################################
# end install python
######################################################################

######################################################################
# setup pip
# https://qiita.com/Zombie_PG/items/62f4b792ac541c04ee40
######################################################################

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

RUN python3 -m pip --no-cache-dir install --upgrade \
    "pip<20.3" \
    setuptools

######################################################################
# end setup pip
######################################################################

# Options:
#   tensorflow
#   tensorflow-gpu
#   tf-nightly
#   tf-nightly-gpu
# Set --build-arg TF_PACKAGE_VERSION=1.11.0rc0 to install a specific version.
# Installs the latest version by default.
#ARG TF_PACKAGE=tensorflow
#ARG TF_PACKAGE_VERSION=

#RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

#COPY bashrc /etc/bash.bashrc
RUN chmod a+rwx /etc/bash.bashrc

#RUN python3 -m pip install --no-cache-dir matplotlib
# Pin ipykernel and nbformat; see https://github.com/ipython/ipykernel/issues/422
# Pin jedi; see https://github.com/ipython/ipython/issues/12740
#RUN python3 -m pip install --no-cache-dir jupyter_http_over_ws ipykernel==5.1.1 nbformat==4.4.0 jedi==0.17.2
#RUN jupyter serverextension enable --py jupyter_http_over_ws

#RUN mkdir -p /tf/tensorflow-tutorials && chmod -R a+rwx /tf/
#RUN mkdir /.local && chmod a+rwx /.local
RUN apt update && apt install -y --no-install-recommends wget git
RUN apt autoremove -y && apt remove -y wget
EXPOSE 8888


######################################################################
# set X windows and GL to bashrc
# https://stackoverflow.com/questions/66497147/cant-run-opengl-on-wsl2#:~:text=To%20solve%20this%2C%20do%20the%20following%3A%20In%20the,your%20bashrc%2Fzshrc%20file%20if%20you%20have%20added%20it.
######################################################################

RUN echo 'export DISPLAY=:0.0' >> ~/.bashrc \
    echo 'export LIBGL_ALWAYS_INDIRECT=0' >> ~/.bashrc

