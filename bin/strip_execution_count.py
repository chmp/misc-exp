import argparse
import os
import os.path

import nbformat


def main():
    args = parser.parse_args()
    backup_fname = f'bak.{args.fname}'

    with open(args.fname, 'rt', encoding='utf8') as fobj:
        data = nbformat.read(fobj,  nbformat.NO_CONVERT)

    for cell in data.get('cells', []):
        if 'execution_count' in cell:
            cell['execution_count'] = None

    if os.path.exists(backup_fname):
        print('cannot create backup file, please delete the old one')
        raise SystemExit(1)

    # make a temporary backup
    os.rename(args.fname, backup_fname)

    with open(args.fname, 'wt', encoding='utf8') as fobj:
        nbformat.write(data, fobj,  nbformat.NO_CONVERT)


parser = argparse.ArgumentParser()
parser.add_argument('fname')


if __name__ == "__main__":
    main()

