from setuptools import setup, PEP420PackageFinder


setup(
    name="chmp",
    description="Support code for machine learning / data science experiments",
    author="Christopher Prohm",
    long_description=open("Readme.pypi.md").read(),
    long_description_content_type="text/markdown",
    packages=PEP420PackageFinder.find("src"),
    package_dir={"": "src"},
    tests_require=["pytest"],
    use_scm_version={"root": "..", "relative_to": __file__},
    url="https://github.com/chmp/misc-exp",
    setup_requires=["setuptools_scm"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
