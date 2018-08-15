import shlex

from invoke import task

files_to_format = ["chmp/src", "tasks.py", "chmp/setup.py"]

inventories = [
    "http://daft-pgm.org",
    "https://matplotlib.org",
    "http://www.numpy.org",
    "https://pandas.pydata.org",
    "https://docs.python.org/3",
]

directories_to_test = ["chmp"]


@task
def precommit(c):
    format(c)
    docs(c)
    test(c)


@task
def test(c):
    run(c, "py.test", *directories_to_test)


@task
def docs(c):
    run(
        c,
        "python",
        "-m",
        "chmp.tools",
        "mddocs",
        *(part for inventory in inventories for part in ["--inventory", inventory]),
        "chmp/docs/src",
        "chmp/docs",
    )


@task
def format(c):
    run(c, "black", *files_to_format)


def run(c, *args, **kwargs):
    args = [shlex.quote(arg) for arg in args]
    args = " ".join(args)
    return c.run(args, **kwargs)
