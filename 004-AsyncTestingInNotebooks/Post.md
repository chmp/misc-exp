Title: asyncio testing inside notebooks
Date: 2019-07-28
Author: Christopher Prohm


Recently I started to work on a project mixing lots of network IO and concurrent tasks. 
Due to these requirements [asyncio](https://docs.python.org/3/library/asyncio.html) seemed 
like a prefect fit. However this project was also my first big foray into the async world 
of python and I needed a bit of time to learn the ropes. 
As mentioned in a [previous post](https://cprohm.de/article/notebooks-and-modules.html), I 
do love to familiarize myself with new technology by experimenting in a notebook and only 
then to refactor the result into python packages.

In this blog post, I would like to show how to combine asyncio and pytest inside notebooks. 
IPython has had an [excellent async support](https://blog.jupyter.org/ipython-7-0-async-repl-a35ce050f7f7) 
for quite some while now. However, running async tests inside IPython requires a bit of 
care. Below I describe what changes in [`ipytest`](github.com/chmp/ipytest) were necessary.

But first, I would like to explore a bit how IPython's async support helps you to directly
call async code in the notebook. For example, the following piece of code will directly work 
with `IPython>=7.0.0`.

```python
import asyncio
await asyncio.sleep(0.1)
```

To support concurrent execution, asyncio uses a coordination layer called the event loop. 
At every `await` expression control is transferred back from the user's code to the event 
loop. It may then run a different asynchronous function or wait for external events. (For 
a more thorough introduction, see for example [this great PyCon talk](https://www.youtube.com/watch?v=iG6fr81xHKA)
by Miguel Grinberg). Event loops are created per thread, i.e., each thread can execute 
its own collection of async functions.  One detail that will become important is that 
IPython creates a default event loop for the main thread that is used to execute any top 
level async code.

The integration of asynchronous code into [`pytest`](https://docs.pytest.org) is also pretty 
straightforward thanks to the excellent [`pytest-asyncio`](https://github.com/pytest-dev/pytest-asyncio)
plugin. First install the packages via `pip install pytest pytest-asyncio`. Then, writing tests for asyncio 
code is as simple as:

```python
import pytest 

@pytest.mark.asyncio
async def test_some_asyncio_code():
    actual = await do_something()
    assert actual == 42
    

async def do_something():
    await asyncio.sleep(0.1)
    return 42
```

However, things get tricky when trying to run these tests inside notebooks. The reason is 
that `pytest-asyncio` tries to creates its own event loop, which then conflicts with the 
main-thread event loop of `IPython`. Luckily, there is a simple solution: execute the tests
in a separate thread. This thread will not have a pre-existing event loop and 
`pytest-asyncio` is free to create its own. 

Since version *`0.7.0`*, [`ipytest`](github.com/chmp/ipytest) supports exactly this behavior
by passing `run_in_thread=True` to `config`. First install the most recent version via 
`pip install -U ipytest`. Then configure `ipytest`:

```python
# import ipytest
import ipytest

# expose the notebook name
__file__ = 'Post.ipynb'

ipytest.config(
    rewrite_asserts=True, addopts=['-qq'], 
    
    # run in a separate thread
    run_in_thread=True,
)
```

And finally run tests via:

```python
ipytest.run()
```

And that's all that needs to be done to run asyncio tests inside jupyter notebooks. If 
you have any feedback or comments, feel free to reach out to me on twitter 
[@c_prohm](https://twitter.com/@c_prohm) or post an issue on [github](https://github.com/chmp/ipytest).
