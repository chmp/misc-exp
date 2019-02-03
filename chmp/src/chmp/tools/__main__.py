import bz2
import json
import logging
import os.path
import pathlib
import sys

import click


@click.group()
def main():
    """Command line tools"""
    pass


@main.command()
@click.argument("src")
@click.argument("dst")
@click.option("-f", "continue_on_error", is_flag=True)
@click.option("--inventory", multiple=True)
def mddocs(src, dst, continue_on_error, inventory=()):
    """Render a subset of sphinx commands to markdown"""
    from chmp.tools.mddocs import transform_directories

    if continue_on_error:
        print("continue on errors", file=sys.stderr)

    inventory = open_inventory(inventory)

    print("translate", src, "->", dst, file=sys.stderr)
    transform_directories(
        src, dst, continue_on_error=continue_on_error, inventory=inventory
    )
    print("done", file=sys.stderr)


def open_inventory(inventory, cache_file=".inventory.json.bz2"):
    from chmp.tools.mddocs import load_inventory

    if not inventory:
        return {}

    if os.path.exists(cache_file):
        with bz2.open(cache_file, "rt") as fobj:
            cached_inventory = json.load(fobj)

        print("use cached inventory from", cache_file, file=sys.stderr)
        if cached_inventory["uris"] == list(inventory):
            return cached_inventory["inventory"]

    print("load inventory from", inventory, file=sys.stderr)
    inventory = load_inventory(inventory)

    print("write inventory cache", cache_file, file=sys.stderr)
    with bz2.open(cache_file, "wt") as fobj:
        json.dump(inventory, fobj, indent=2, sort_keys=True)

    return inventory["inventory"]


@main.command()
@click.argument("source-path", type=click.Path(exists=True))
@click.argument("target-path", type=click.Path(exists=True))
@click.option("-y", "yes", is_flag=True, default=False)
def paper(source_path, target_path, yes=False):
    """Sort papers into a common normalized directory structure."""
    from chmp.tools.papers import sort_arxiv_papers, sort_non_arxiv_papers

    non_arxiv_source_path = pathlib.Path(source_path) / "PapersToSort"

    print(f":: this script will perform the following actions:")
    print(f":: - sort arxiv papers from {source_path!s} to {target_path!s}")
    print(
        f":: - sort non arxiv papers from {non_arxiv_source_path!s} to {target_path!s}"
    )

    if not yes and input(":: continue? [yN] ") != "y":
        return

    print()
    print(":: sort arxiv papers")
    sort_arxiv_papers(source_path, target_path)

    print(":: sort non arxiv papers")
    sort_non_arxiv_papers(non_arxiv_source_path, target_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
