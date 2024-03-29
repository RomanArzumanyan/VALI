FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

ENV NVIDIA_DRIVER_CAPABILITIES=compute,video

RUN apt update
RUN apt-get -y install \
    software-properties-common

RUN type apt-add-repository
RUN add-apt-repository ppa:ubuntuhandbook1/ffmpeg6

RUN apt-get update && DEBIAN_FRONTEND=noninteractive \
    apt-get -y install \
    ffmpeg \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswresample-dev \
    libavutil-dev \
    libswscale-dev \
    wget \
    build-essential \
    ninja-build \
    cmake \
    git \
    python3 \
    python3-pip \
    python-is-python3 \
    pkg-config

ARG PIP_INSTALL_EXTRAS=""

RUN mkdir /src
WORKDIR /src
COPY src src
COPY tests tests
COPY extern extern
COPY setup.py setup.py
COPY pyproject.toml pyproject.toml
COPY CMakeLists.txt CMakeLists.txt
COPY common.cmake common.cmake
COPY FindFFmpeg.cmake FindFFmpeg.cmake
RUN python3 -m pip install --no-cache-dir setuptools wheel
RUN python3 -m pip install --no-cache-dir .[$PIP_INSTALL_EXTRAS]
RUN python3 -m pip install pydantic

RUN mkdir /cwd
WORKDIR /cwd

ENTRYPOINT ["/bin/bash"]