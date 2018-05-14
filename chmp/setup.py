from setuptools import setup, PEP420PackageFinder

setup(
    name='chmp',
    version='0.2.0',
    packages=PEP420PackageFinder.find('src'),
    package_dir={'': 'src'},
    tests_require=['pytest'],
)
