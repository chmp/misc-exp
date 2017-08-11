import argparse
import nbformat


def main():
    args = parser.parse_args()

    with open(args.fname, 'r') as fobj:
        data = nbformat.read(fobj,  nbformat.NO_CONVERT)

    for cell in data.get('cells', []):
        if 'execution_count' in cell:
            cell['execution_count'] = None

    with open(args.fname, 'w') as fobj:
        nbformat.write(data, fobj,  nbformat.NO_CONVERT)


parser = argparse.ArgumentParser()
parser.add_argument('fname')


if __name__ == "__main__":
    main()

