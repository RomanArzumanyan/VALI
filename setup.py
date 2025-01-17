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
        name="python_vali",
        version="4.2.6-1",
        description="Video Processing Library with full NVENC/NVDEC hardware acceleration",
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        author="Roman Arzumanyan",
        license="Apache 2.0",
        install_requires=["numpy"],
        extras_require={
            "dev": [
                "torch",
                "torchvision",
                "onnx",
                "tensorrt",],
            "samples": [
                "torch",
                "torchvision",
                "onnx",
                "tensorrt",
                "pynvml"],
            "tests": [
                "torch",
                "torchvision",
                "parameterized",
                "pydantic",
                "pynvml",
                "unittest",
                "json",
                "logging",
                "random",
                "typing",
                "math",
                "sys",
                "os"],
            "torch": [
                "torch",
                "torchvision"],
            "tensorrt": [
                "torch",
                "torchvision"],
        },
        dependency_links=[
            "https://pypi.ngc.nvidia.com"
        ],
        packages=["python_vali"],
        package_data={"python_vali": ["__init__.pyi"]},
        package_dir={"": "src"},
        cmake_install_dir="src",
        # cmake_args=["-DTRACK_TOKEN_ALLOCATIONS=ON"],
        # cmake_args=["-DCMAKE_BUILD_TYPE=Debug"]
    )
