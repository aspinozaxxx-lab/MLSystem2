import click
import sys
from .download import download
from loguru import logger

@click.group()
@click.option('--log_level', default=20,
              help='debug - 10, info - 20, warn - 30, error - 40')
def cli(log_level):
    # set logging
    #logging.basicConfig(level=log_level, format='[%(levelname)s] - %(asctime)s - %(name)s - %(message)s')
    logger.remove()
    logger.add(sys.stderr, format="{level} {time} {message}", level=log_level)
    pass


cli.add_command(download)