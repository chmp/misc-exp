import asyncio

import tensorflow as tf

from chmp.app.seqdec.sequencing import run
from chmp.app.kwdetect.aio import detect


def main():
    raise NotImplementedError()


def execute(sequence, keyword_model, *, loop=None, debug=True):
    if loop is None:
        loop = asyncio.get_event_loop()

    if debug:
        loop.set_debug(True)

    q = asyncio.Queue()

    fg_task = asyncio.ensure_future(run_sequence(q, sequence, loop=loop))
    bg_task = asyncio.ensure_future(wait_for_commands(q, keyword_model), loop=loop)

    # make sure the bg task is canceled
    fg_task.add_done_callback(lambda *_: bg_task.cancel())

    events, _ = loop.run_until_complete(asyncio.gather(fg_task, bg_task, loop=loop, return_exceptions=True))
    return events


async def run_sequence(q, sequence, *, loop):
    async def next_command(*, timeout):
        try:
            return await asyncio.wait_for(q.get(), timeout=timeout, loop=loop)

        except asyncio.TimeoutError:
            return None

    async def say(message):
        proc = await asyncio.create_subprocess_exec('say', message)
        await proc.wait()

    # wait for the start token
    await q.get()
    return await run(
        sequence,
        say=say, next_command=next_command, time=loop.time, wait_command_time=10,
    )


async def wait_for_commands(q, model):
    graph = tf.Graph()

    with tf.Session(graph=graph) as session:
        tf_model = model.restore(session)

        async for command in detect(
                model=tf_model,
                session=session,
                start_token=None,
                sample_target='../KeywordDetection/data/',
        ):
            await q.put(command)
