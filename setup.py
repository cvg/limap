import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

# Cores used for building the project
N_CORES = 8

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath(ext.name))
        )

        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        # cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DVERSION_INFO={self.distribution.get_version()}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", ".", f"--parallel {N_CORES}"] + build_args,
            cwd=self.build_temp,
        )


# Read requirements from requirements.txt
def parse_requirements_for_limap(filename):
    requirements = []
    with open(filename) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip()
            and not line.startswith("#")
            and not line.startswith(".")
        ]
    requirements.extend(
        [
            "pytlsd @ git+https://github.com/iago-suarez/pytlsd.git@21381ca4c5c8da297ef44ce07dda115075c5a648#egg=pytlsd",
            "hloc @ git+https://github.com/cvg/Hierarchical-Localization.git@abb252080282e31147db6291206ca102c43353f7#egg=hloc",
            "deeplsd @ git+https://github.com/cvg/DeepLSD.git@17c764595b17f619e6f78c5f9fc18f1f970ea579#egg=deeplsd",
            "gluestick @ git+https://github.com/cvg/GlueStick.git@40d71d5f4adc7f4fccae4cd7675a49daee3e873e#egg=gluestick",
        ]
    )
    return requirements


# The information here can also be placed in setup.cfg - better separation of
# logic and declaration, and simpler if you include description/version in
# one file.
setup(
    name="limap",
    version="1.0.0",
    packages=find_packages("limap"),
    package_data={"limap": ["third-party/JLinkage/lib/libJLinkage.so"]},
    python_requires=">=3.8, < 3.13",
    install_requires=parse_requirements_for_limap("requirements.txt"),
    author="Shaohui Liu",
    author_email="b1ueber2y@gmail.com",
    description="A toolbox for mapping and localization with line features",
    long_description="",
    ext_modules=[CMakeExtension("_limap")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
)
