# VALI

VALI is a video analytics and processing project for python. VALI is a spin off from NVIDIA's VPF.
It’s set of C++ libraries and Python bindings which provides full HW acceleration for video processing tasks such as decoding, encoding, transcoding and GPU-accelerated color space and pixel format conversions.

VALI also supports DLPack and can share memory with all the modules which supports DLPack (e. g. share decoded surfaces with torch).

## Documentation
https://romanarzumanyan.github.io/VALI

## Prerequisites
VALI works on Linux(tested on Ubuntu 22.04) and Windows

- NVIDIA display driver: 525.xx.xx or above
- CUDA Toolkit 11.2 or above 
  - CUDA toolkit has driver bundled with it e.g. CUDA Toolkit 12.0 has driver `530.xx.xx`. During installation of CUDA toolkit you could choose to install or skip installation of the bundled driver. Please choose the appropriate option.
- FFMPEG
  - VALI will download FFMPEG build from https://github.com/BtbN/FFmpeg-Builds
  - Make sure you don't have system FFMPEG packages installed. Otherwise, CMake will use them instead. You may run ```apt remove libavfilter-dev libavformat-dev libavcodec-dev libswresample-dev libavutil-dev``` on Ubuntu to remove them. Alternatively, build in Docker.

- Python 3 and above
- Install a C++ toolchain either via Visual Studio or Tools for Visual Studio.
  - Recommended version is Visual Studio 2017 and above
(Windows only)

## Samples and best practices
VALI unit tests are written in a way to illustrate the API usage. One may follow them as samples.

### Linux
Ubuntu 22.04 is recommended.

#### Install dependencies
```bash
apt install -y \          
          wget \
          build-essential \
          git
```

##### Install CUDA Toolkit (if not already present)
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda
# Ensure nvcc to your $PATH (most commonly already done by the CUDA installation)
export PATH=/usr/local/cuda/bin:$PATH
```

##### Install VALI
```bash
# Update git submodules
git submodule update --init --recursive
pip3 install .
```

To check whether VALI is correctly installed run the following Python script
```python
import PyNvCodec
```
If using Docker via [Nvidia Container Runtime](https://developer.nvidia.com/nvidia-container-runtime),
please make sure to enable the `video` driver capability: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#driver-capabilities via
the `NVIDIA_DRIVER_CAPABILITIES` environment variable in the container or the `--gpus` command line parameter (e.g.
`docker run -it --rm --gpus 'all,"capabilities=compute,utility,video"' nvidia/cuda:12.1.0-base-ubuntu22.04`).

### Windows

- Install a C++ toolchain either via Visual Studio or Tools for Visual Studio (https://visualstudio.microsoft.com/downloads/)
- Install the CUDA Toolkit: https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64
```pwsh
# Update git submodules
git submodule update --init --recursive
pip install .
```
To check whether VALI is correctly installed run the following Python script
```python
cuda_path = os.environ["CUDA_PATH"]
os.add_dll_directory(os.path.join(cuda_path, "bin"))
import PyNvCodec
```
## Docker

For convenience, we provide a Docker images located at `docker` that you can use to easily install all dependencies 
([docker](https://docs.docker.com/engine/install/ubuntu/) and [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
are required)


```bash
DOCKER_BUILDKIT=1 sudo docker build \
  --tag vali-gpu \
  --file docker/Dockerfile \
  --build-arg PIP_INSTALL_EXTRAS=torch .

docker run -it --rm --gpus=all vali-gpu
```

`PIP_INSTALL_EXTRAS` can be any subset listed under `project.optional-dependencies` in [pyproject.toml](pyproject.toml).

## Offline documentation

A documentation for VALI can be generated from this repository:
```bash
pip install . # install VALI
pip install sphinx  # install documentation tool sphinx
cd docs
make html
```
You can then open `_build/html/index.html` with your browser.

## Community Support
Please use project's Discussions page for that.

[![OnPullRequest](https://github.com/RomanArzumanyan/VALI/actions/workflows/onPullRequest.yml/badge.svg)](https://github.com/RomanArzumanyan/VALI/actions/workflows/onPullRequest.yml)
