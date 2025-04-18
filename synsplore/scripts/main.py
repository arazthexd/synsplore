import click

from synsplore.scripts.prepare import prepare
from synsplore.scripts.sample import sample
from synsplore.scripts.featurize import featurize

@click.group()
def cli():
    """
    Synsplore CLI: A command-line interface for synsplore.
    """
    pass

def main():
    cli.add_command(prepare)
    cli.add_command(sample)
    cli.add_command(featurize)
    cli()

if __name__ == '__main__':
    main()