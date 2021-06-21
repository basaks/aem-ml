"""Console script for aem."""
import sys
import click
from aem import __version__
from aem.logger import configure_logging, aemlogger


@click.group()
@click.version_option(version=__version__)
@click.option("-v", "--verbosity",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]),
              default="INFO", help="Level of logging")
def main(verbosity: str) -> int:
    """Train a model and use it to make predictions."""
    configure_logging(verbosity)
    return 0


@main.command()
@click.option("--config", type=click.Path(exists=True), required=True,
              help="The model configuration file")
def train(config: str) -> None:
    """Train a model specified by a config file."""
    aemlogger.info(f"Training Model using config {config}")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
