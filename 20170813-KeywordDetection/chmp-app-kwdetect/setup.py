from setuptools import setup

setup(
    name='chmp-app-kwdetect',
    version='0.4.0',
    description='Simple keyword detection using tensorflow',
    author='Christopher Prohm',
    author_email='mail@cprohm.de',
    license='MIT',
    packages=["chmp.app.kwdetect"],
    package_dir={'': 'src'},
    install_requires=[
        'click',
        'numba',
        'numpy',
        'pandas',
        'pysoundfile',
        'python_speech_features',
        'scipy',
        'sounddevice',
        'janus',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
    ],
)
