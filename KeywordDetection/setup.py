from setuptools import setup

setup(
    name='kwdetect',
    version='0.3.0',
    description='Simple keyword detection using tensorflow',
    author='Christopher Prohm',
    author_email='mail@cprohm.de',
    license='MIT',
    packages=["kwdetect"],
    install_requires=[
        'click',
        'numba',
        'numpy',
        'pandas',
        'pysoundfile',
        'python_speech_features',
        'scipy',
        'sounddevice',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
    ],
)
