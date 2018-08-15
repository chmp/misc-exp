from invoke import task


@task
def precommit(c):
    format(c)
    docs(c)
    test(c)


@task
def test(c):
    c.run('py.test chmp')


@task
def docs(c):
    c.run("""
        python -m chmp.tools mddocs \
            --inventory http://daft-pgm.org \
            --inventory https://matplotlib.org \
            --inventory http://www.numpy.org \
            --inventory https://pandas.pydata.org \
            --inventory https://docs.python.org/3 \
            chmp/docs/src chmp/docs
    """.strip())


@task
def format(c):
    c.run("black chmp/src")
