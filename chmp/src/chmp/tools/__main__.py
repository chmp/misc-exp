import sys

import click


@click.group()
def main():
    """Command line tools"""
    pass


@main.command()
@click.argument('src')
@click.argument('dst')
def mddocs(src, dst):
    """Render a subset of sphinx commands to markdown"""
    from chmp.tools.mddocs import transform_directories

    print('translate', src, '->', dst, file=sys.stderr)
    transform_directories(src, dst)


if __name__ == "__main__":
    main()
