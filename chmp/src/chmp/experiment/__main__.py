import argparse
import logging
import os.path
import threading
import time

_logger = logging.getLogger(__name__)

__usage__ = '''
# Experiments

Start in one terminal:

    python -m chmp.experiment watch ./experiments | xargs -n1 -I{} bash {} --atomic
    
'''


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    watch_parser = subparsers.add_parser('watch')
    watch_parser.set_defaults(cmd=watch)
    watch_parser.add_argument('path')

    usage_parser = subparsers.add_parser('usage')
    usage_parser.set_defaults(cmd=usage)

    args = parser.parse_args()
    args.cmd(args)


def usage(args):
    print(__usage__.strip())
    raise SystemExit(1)


def watch(args):
    from watchdog.observers import Observer

    emitter = StartHandler()

    observer = Observer()
    observer.schedule(emitter, args.path, recursive=True)
    observer.start()

    for dirpath, dirnames, filenames in os.walk(args.path):
        for p in filenames:
            emitter.backfill(os.path.join(dirpath, p))

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


class StartHandler:
    def __init__(self):
        self.lock = threading.Lock()
        self.emitted = set()

    def backfill(self, path):
        if os.path.basename(path) != 'start':
            return

        self.emit(path)

    def emit(self, path):
        abspath = os.path.abspath(path)
        script_path = os.path.join(os.path.dirname(path), 'run.sh')
        script_path = os.path.abspath(script_path)

        with self.lock:
            if abspath in self.emitted:
                return

            self.emitted.add(abspath)
            print(script_path, flush=True)

    def dispatch(self, event):
        if event.event_type != 'created' or event.is_directory or os.path.basename(event.src_path) != 'start':
            return

        self.emit(event.src_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
