from setuptools import setup, Extension

import pybind11

setup(
    name="hexworld",
    version="1.0.0",
    package_dir = {'': 'src'},
    packages=['hexworld'],
    ext_modules=[
        Extension(
            "hexworld._hexworld",
            ['src/cpp/hexworld.cc'],
            include_dirs=[pybind11.get_include()],
            extra_compile_args=['-O0', '-Wall', '-std=c++1z'],
            language='c++'
        ),
    ],
)

