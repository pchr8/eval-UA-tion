import pdb
import sys
import traceback
import argparse

from pathlib import Path

import logging
from rich.logging import RichHandler

logging.basicConfig(
    #  level="NOTSET",
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            #  show_path=False
            #  rich_tracebacks=True,
        )
    ],
)
logger = logging.getLogger(__package__)

import rich
from rich import inspect, print

from typing import List, Tuple, Optional, Dict, Union

from lmentry_static.consts import (
    templates_path,
    config_path,
    default_output_path,
    CONFIG_FN,
)
from lmentry_static.generate_dataset import run


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser()
    #  parser.add_argument(
    #      "--input",
    #      "-i",
    #      help="Input dir with templates path (%(default)s)",
    #      type=Path,
    #      default=templates_path,
    #  )
    parser.add_argument(
        "--config",
        "-c",
        help="Config file for evaluation(%(default)s)",
        type=Path,
        default=config_path,
    )
    parser.add_argument(
        "--output",
        "-o",
        default=default_output_path,
        help="Output for the dataset (%(default)s)",
        type=Path,
    )
    parser.add_argument(
        "--format",
        "-f",
        help="Output format,  one of (%(default)s)",
        type=str,
        default="json",
    )
    parser.add_argument("--pdb", "-P", help="Run PDB on exception", action="store_true")
    parser.add_argument(
        "-q",
        help="Output only warnings",
        action="store_const",
        dest="loglevel",
        const=logging.WARN,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Output more details",
        action="store_const",
        dest="loglevel",
        const=logging.DEBUG,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger.setLevel(args.loglevel if args.loglevel else logging.INFO)

    logger.debug(args)

    try:
        run(args)
    except Exception as e:
        if args.pdb:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            logger.exception(e)


if __name__ == "__main__":
    main()
