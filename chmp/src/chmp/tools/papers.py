"""Helper to sort papers.

...

"""

import json
import pathlib
import re
import shutil
import subprocess

import requests


def sort_arxiv_papers(source_path, target_path):
    source_path = pathlib.Path(source_path)
    target_path = pathlib.Path(target_path)

    arxiv_papers = [p for p in source_path.glob("*.pdf") if is_arxiv_paper(p)]
    existing_arxiv_papers = []

    for idx, p in enumerate(arxiv_papers):
        meta_file_name = "meta/{}.json".format(p.stem)

        if (target_path / meta_file_name).exists():
            print("skip", p)
            existing_arxiv_papers += [p]
            continue

        print("handle", p, idx, "/", len(arxiv_papers))
        meta = fetch_meta_data(p)
        meta = parse_meta_data(meta)

        normalized_title = meta["Title"]
        normalized_title = re.sub("[^\w ]", "", normalized_title)
        normalized_title = normalized_title.lower()
        normalized_title = "_".join(re.split("\s+", normalized_title))

        normalized_file_name = "{}_{}.pdf".format(p.stem, normalized_title)

        if (target_path / normalized_file_name).exists():
            print("target file exists", target_path / normalized_file_name)
            continue

        shutil.move(p, target_path / normalized_file_name)

        with (target_path / meta_file_name).open("wt") as fobj:
            json.dump(meta, fobj, indent=2, sort_keys=True)

    return existing_arxiv_papers


def sort_non_arxiv_papers(source_path, target_path):
    source_path = pathlib.Path(source_path).resolve()
    target_path = pathlib.Path(target_path).resolve()

    print(source_path)
    candidate_papers = list(source_path.glob("*.pdf"))

    for p in candidate_papers:
        s = shasum(p)

        normalized_file_name = re.sub("[^\w -]", "", p.stem)
        normalized_file_name = normalized_file_name.lower()
        normalized_file_name = normalized_file_name.replace(" ", "_")
        normalized_file_name = normalized_file_name.replace("-", "_")
        normalized_file_name = f"{s[:9]}_{normalized_file_name}.pdf"

        target_fname = target_path / normalized_file_name

        if target_fname.exists():
            print("skip", p, "->", target_fname)
            continue

        print(p, "->", target_fname)
        shutil.move(p, target_fname)


def shasum(p):
    p = subprocess.run(["shasum", p], stdout=subprocess.PIPE, check=True)
    s = p.stdout.decode()
    s, = s.splitlines()
    s, *_ = s.partition(" ")
    return s


def is_arxiv_paper(p):
    return arxiv_path_pattern.match(p.name) is not None


def fetch_meta_data(p):
    url = "https://arxiv.org/abs/{}?fmt=txt".format(p.stem)
    return requests.get(url).content.decode()


def parse_meta_data(desc):
    """Parse the meta data as retrieved from arxiv.org."""
    data = desc.split(r"\\")

    header = data[1].strip()
    abstract = data[2].strip()

    res = {}
    for line in header.splitlines():
        if not line:
            continue

        key, _, val = line.partition(":")
        res[key.strip()] = val.strip()

    res["abstract"] = abstract
    return res


arxiv_path_pattern = re.compile(
    """
    ^
    \d{4}
    \.
    \d*
    (v\d+)?
    \.pdf
    $
""",
    re.VERBOSE,
)
