"""

"""


import sys
import os

from pkg_resources import VersionConflict, require

try:
    require("setuptools>=42")
except VersionConflict:
    print("Error: version of setuptools is too old (<42)!")
    sys.exit(1)


if __name__ == "__main__":
    import skbuild

    skbuild.setup(
        name="PyNvCodec",
        version="3.2.1",
        description="Video Processing Library with full NVENC/NVDEC hardware acceleration",
        author="Roman Arzumanyan",
        license="Apache 2.0",
        install_requires=["numpy"],
        extras_require={
            "dev": ["pycuda", "pyopengl", "torch", "torchvision", "opencv-python", "onnx", "tensorrt",],
            "samples": ["pycuda", "pyopengl", "torch", "torchvision", "opencv-python", "onnx", "tensorrt", "tqdm"],
            "tests": ["pycuda", "pyopengl", "torch", "torchvision", "opencv-python", "pydantic"],
            "torch": ["torch", "torchvision"],
            "tensorrt": ["torch", "torchvision"],
        },
        dependency_links=[
            "https://pypi.ngc.nvidia.com"
        ],
        packages=["PyNvCodec"],
        package_data={"PyNvCodec": ["__init__.pyi"]},
        package_dir={"": "src"},
        cmake_install_dir="src",
        # cmake_args=["-DCMAKE_BUILD_TYPE=Debug"]
    )
