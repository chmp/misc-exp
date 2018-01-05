from setuptools import setup

setup(
    name='chmp-app-off_policy',
    version='0.1.0',
    description='...',
    author='Christopher Prohm',
    author_email='mail@cprohm.de',
    license='MIT',
    packages=["chmp.app.off_policy"],
    package_dir={'': 'src'},
    install_requires=[
        'chmp',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
