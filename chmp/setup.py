from setuptools import setup, PEP420PackageFinder

setup(
    name="chmp",
    packages=PEP420PackageFinder.find("src"),
    package_dir={"": "src"},
    tests_require=["pytest"],
    use_scm_version={"root": "..", "relative_to": __file__},
    setup_requires=["setuptools_scm"],
)
