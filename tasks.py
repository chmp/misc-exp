import hashlib
import json
import os
import pathlib
import shlex

import nbformat
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

notebook_directories_to_test = ["20181026-TestingInJupyter/notebooks"]

notebooks_to_test = [
    str(nb)
    for p in notebook_directories_to_test
    for nb in pathlib.Path(p).glob("*.ipynb")
    if not nb.name.startswith("wip")
]

notebooks_to_test = notebooks_to_test + [
    "BuildingBlocks/Bishop_Notes_01.ipynb",
    "BuildingBlocks/Bishop_Notes_02.ipynb",
    "BuildingBlocks/Bishop_Notes_03.ipynb",
    "BuildingBlocks/Bishop_Notes_04.ipynb",
    "BuildingBlocks/Bishop_Notes_05.ipynb",
    "BuildingBlocks/Bishop_Notes_06.ipynb",
    "BuildingBlocks/Bishop_Notes_13.ipynb",
    "BuildingBlocks/tech_TorchModels.ipynb",
    "20180107-Causality/BlogPost.ipynb",
    # TODO: fix the notebook "20180107-Causality/Index.ipynb"
    # "20180107-Causality/Index.ipynb",
    "20180107-Causality/Notes.ipynb",
]

notebooks_no_static_check = {"20181026-TestingInJupyter/notebooks/IPyTestIntro.ipynb"}


@task
def precommit(c):
    format(c)
    docs(c)
    static_checks(c)
    test(c)


@task()
def precommit_full(c):
    """Run all precommit tasks and integration tests"""
    precommit(c)
    integration(c)


@task
def integration(c):
    if notebooks_to_test:
        run(c, "pytest", "--nbval-lax", *notebooks_to_test)


@task
def test(c):
    run(c, "pytest", *directories_to_test)


@task()
def static_checks(c):
    # export all notebooks as scripts
    pathlib.Path("tmp").mkdir(exist_ok=True)

    export_paths = []
    for nb in notebooks_to_test:
        if nb in notebooks_no_static_check:
            continue

        mod_path = os.path.join(
            "tmp",
            (
                str_sha1(os.path.dirname(nb))
                + "-"
                + (os.path.basename(nb).rstrip(".ipynb") + ".py")
            ),
        )
        export_notebook_as_module(nb, mod_path)
        export_paths.append(mod_path)

    print("run static checks")
    c.run(
        "mypy --ignore-missing-imports chmp/src/chmp/**/*.py " + " ".join(export_paths)
    )
    c.run("pyflakes chmp/src/chmp/**/*.py " + " ".join(export_paths))

    print("done")


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


def export_notebook_as_module(notebook_path, module_path):
    with open(notebook_path, "rt") as fobj:
        nb = nbformat.read(fobj, as_version=4)

    content = []

    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue

        for line in cell["source"].splitlines():
            line = line.rstrip()
            if line.startswith("%") or line.startswith("!"):
                line = "# " + line

            content += [line]

        content += [""]

    with open(module_path, "wt") as fobj:
        for line in content:
            fobj.write(line + "\n")


def str_sha1(obj):
    s = json.dumps(obj, indent=None, sort_keys=True, separators=(",", ":"))
    s = s.encode("utf8")
    return hashlib.sha1(s).hexdigest()
