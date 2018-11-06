from invoke import task


@task()
def precommit(c):
    format(c)
    test(c)


@task()
def format(c):
    c.run("black src tests setup.py tasks.py")


@task()
def test(c):
    c.run("pytest tests")
    c.run("pytest --nbval-lax notebooks/*.ipynb")
