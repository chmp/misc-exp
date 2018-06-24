from setuptools import setup, PEP420PackageFinder

setup(
    name='vegawidget',
    author='Christopher Prohm',
    version='0.1.0',
    packages=PEP420PackageFinder.find('src'),
    package_dir={'': 'src'},
    install_requires=['ipywidgets', 'vega'],
    tests_require=['pytest'],
    include_package_data=True,
    data_files=[
        ('share/jupyter/nbextensions/vegawidget', ['src/vegawidget/static/index.js']),
        ('etc/jupyter/nbconfig/notebook.d', ['vegawidget.json']),
    ],
)