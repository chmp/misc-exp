import asyncio


async def branch(*aws):
    """Run multiple coroutines in parallel with exception / cancellation support.

    Usage::

        await branch(
            func1(...),
            func2(...),
            ...
        )

    If any function raises an exception or is cancelled, all other functions
    are cancelled as well.
    """
    tasks = [asyncio.ensure_future(obj) for obj in aws]

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    except asyncio.CancelledError:
        # ensure all tasks are cancelled
        for task in tasks:
            task.cancel()

        raise

    else:
        for task in pending:
            task.cancel()

        # reraise any exceptions
        for task in done:
            task.result()
