from setuptools import setup, find_packages

setup(
    name='chmp',
    packages=[
        'chmp.{}'.format(package)
        for package in find_packages('src/chmp')
    ],
    package_dir={'': 'src'},
    test_requires=['pytest'],
)
