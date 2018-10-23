from setuptools import setup, PEP420PackageFinder


setup(
    name='ipytest-demo',
    version='0.0.0',
    packages=PEP420PackageFinder.find("src"),
    package_dir={'': 'src'},
)
