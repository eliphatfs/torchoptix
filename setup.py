import os
import sys
import setuptools
from setuptools.command.build_ext import build_ext
import glob
import platform
import subprocess
from wheel.bdist_wheel import bdist_wheel


IS_WINDOWS = sys.platform == 'win32'
SUBPROCESS_DECODE_ARGS = ('oem',) if IS_WINDOWS else ()


def _find_cuda_home():
    """Find the CUDA install path."""
    # Guess #1
    cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
    if cuda_home is None:
        # Guess #2
        try:
            which = 'where' if IS_WINDOWS else 'which'
            with open(os.devnull, 'w') as devnull:
                nvcc = subprocess.check_output([which, 'nvcc'],
                                               stderr=devnull).decode(*SUBPROCESS_DECODE_ARGS).rstrip('\r\n')
                cuda_home = os.path.dirname(os.path.dirname(nvcc))
        except Exception:
            # Guess #3
            if IS_WINDOWS:
                cuda_homes = glob.glob(
                    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v*.*')
                if len(cuda_homes) == 0:
                    cuda_home = ''
                else:
                    cuda_home = cuda_homes[0]
            else:
                cuda_home = '/usr/local/cuda'
            if not os.path.exists(cuda_home):
                cuda_home = None
    return cuda_home


class bdist_wheel_abi3(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            # on CPython, our wheels are abi3 and compatible back to 3.6
            return python, "abi3", plat

        return python, abi, plat


with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()


cuda_home = _find_cuda_home()
cc_args = []
cl_args = []
libraries = ['cuda']
library_paths = []
if platform.system() == 'Windows':
    cc_args += ['/DEBUG', '/Z7', '/std:c++17']
    cl_args += ['/DEBUG']
    libraries += ['Advapi32']
    library_paths += [os.path.join(cuda_home, "lib", "x64")]
else:
    cc_args += ['-D__FUNCTION__=""', '-std=c++17', '-fno-crossjumping']
    lib_dir = 'lib64'
    if (not os.path.exists(os.path.join(cuda_home, lib_dir)) and
            os.path.exists(os.path.join(cuda_home, 'lib'))):
        # 64-bit CUDA may be installed in 'lib' (see e.g. gh-16955)
        # Note that it's also possible both don't exist (see
        # _find_cuda_home) - in that case we stay with 'lib64'.
        lib_dir = 'lib'
    library_paths += [os.path.join(cuda_home, lib_dir, "stubs")]


class build_ext_subclass(build_ext):
    def build_extensions(self):
        original_compile = self.compiler._compile
        def new_compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
            if src.endswith('.c'):
                extra_postargs = [s for s in extra_postargs if "c++17" not in s]
            return original_compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        self.compiler._compile = new_compile
        try:
            build_ext.build_extensions(self)
        finally:
            del self.compiler._compile


setuptools.setup(
    name="torchoptix",
    version="0.0.1",
    author="eliphatfs",
    author_email="shiruoxi61@gmail.com",
    description="Modular python bindings for OptiX.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eliphatfs/torchoptix",
    include_package_data=True,
    # cmdclass={"build_ext": build_ext_subclass, "bdist_wheel": bdist_wheel_abi3},
    cmdclass={"build_ext": build_ext_subclass},
    ext_modules=[setuptools.Extension(
        "torchoptix",
        glob.glob("csrc/**/*.c", recursive=True)
        + glob.glob("csrc/**/*.cpp", recursive=True),
        # define_macros=[("NO_COMBINE", "1"), ("MIR_INTERP_TRACE", "1")],
        include_dirs=["include", os.path.join(cuda_home, "include")],
        extra_compile_args=cc_args,
        extra_link_args=cl_args,
        libraries=libraries,
        library_dirs=library_paths,
        py_limited_api=True
    )],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License"
    ],
    options={
        'bdist_wheel': {
            'py_limited_api': 'cp38'
        }
    },
    python_requires='~=3.8'
)
