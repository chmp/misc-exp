import os
import pathlib
import shlex

from invoke import task

files_to_format = ["chmp/src", "tasks.py", "chmp/setup.py"]

inventories = [
    "http://daft-pgm.org",
    "https://matplotlib.org",
    "http://www.numpy.org",
    "https://pandas.pydata.org",
    "https://docs.python.org/3",
    "https://pytorch.org/docs/stable",
]

directories_to_test = ["chmp", "20170813-KeywordDetection/chmp-app-kwdetect"]

notebook_directories_to_test = [
    "20181026-TestingInJupyter/notebooks",
]

notebooks_to_test = [
    str(nb)
    for p in notebook_directories_to_test
    for nb in pathlib.Path(p).glob('*.ipynb')
]


@task
def precommit(c):
    format(c)
    docs(c)
    test(c)


@task
def notebook_integration(c):
    if notebooks_to_test:
        run(c, "pytest", "--nbval-lax", *notebooks_to_test)


@task
def test(c):
    run(c, "pytest", *directories_to_test)


@task
def docs(c):
    run(
        c,
        *["python", "-m", "chmp.tools", "mddocs"],
        *(part for inventory in inventories for part in ["--inventory", inventory]),
        *["chmp/docs/src", "chmp/docs"],
    )


@task
def format(c):
    run(c, "black", *files_to_format)


@task
def release(c, yes=False):
    import packaging.version

    with c.cd("chmp"):
        run(c, "python", "setup.py", "bdist_wheel")

    latest_package = max(
        (
            package
            for package in os.listdir("chmp/dist")
            if not package.startswith(".") and package.endswith(".whl")
        ),
        key=packaging.version.parse,
    )

    if not yes:
        answer = input(f"upload {latest_package} [yN] ")
        if answer != "y":
            print("stop")
            return

    with c.cd("chmp/dist"):
        run(c, "twine", "upload", latest_package)


def run(c, *args, **kwargs):
    args = [shlex.quote(arg) for arg in args]
    args = " ".join(args)
    return c.run(args, **kwargs)
