"""Console script for aem."""
import sys
import click
from pathlib import Path
import geopandas as gpd
from aem import __version__
from aem.config import Config
from aem.logger import configure_logging, aemlogger as log


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
    log.info(f"Training Model using config {config}")
    conf = Config(config)


    import IPython; IPython.embed(); import sys; sys.exit()


def load_data(conf):
    log.info("Reading covariates...")
    log.info("reading interp data...")

    all_interp_data = gpd.GeoDataFrame.from_file(conf.interp_data)

    log.info("reading covariates ...")
    original_aem_data = gpd.GeoDataFrame.from_file(conf.aem_data)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
