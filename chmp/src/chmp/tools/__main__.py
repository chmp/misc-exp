import logging
import sys

import click


@click.group()
def main():
    """Command line tools"""
    pass


@main.command()
@click.argument('src')
@click.argument('dst')
@click.option('-f', 'continue_on_error', is_flag=True)
def mddocs(src, dst, continue_on_error):
    """Render a subset of sphinx commands to markdown"""
    from chmp.tools.mddocs import transform_directories

    if continue_on_error:
        print('continue on errors', file=sys.stderr)

    print('translate', src, '->', dst, file=sys.stderr)
    transform_directories(src, dst, continue_on_error=continue_on_error)
    print('done', file=sys.stderr)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
