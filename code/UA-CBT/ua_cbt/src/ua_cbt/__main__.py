import argparse
import logging
import sys, traceback, pdb

from pathlib import Path

from rich import print
from rich.logging import RichHandler

logging.basicConfig(
    #  level="NOTSET",
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            show_path=True,
            rich_tracebacks=True,
        )
    ],
)

from ua_cbt.iteration import run
from ua_cbt.consts import DEFAULT_INPUT_CSV_FILE, DEFAULT_OUTPUT_DIR#, DEFAULT_DATA_YAML_FILE

#  logging.basicConfig()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input",
        "-i", 
        help="Input CSV file with stories (%(default)s)",
        type=Path,
        default=DEFAULT_INPUT_CSV_FILE

    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output for the dataset (%(default)s)",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--limit",
        "-l", 
        help="How many stories to parse (%(default)s)",
        type=int,
        default=None
    )
    #  parser.add_argument(
    #      "--data_yaml",
    #      "-d",
    #      help="YAML file with strings and data(%(default)s)",
    #      type=Path,
    #      default=DEFAULT_DATA_YAML_FILE
    #  )
    parser.add_argument(
        "-splits",
        "-s", 
        #  help="Share of first n-1 splits (%(default)s)",
        help="Share of all splits except train (%(default)s)",
        type=float,
        nargs="+", 
        default=None
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
    logger = logging.getLogger(__package__)
    logger.setLevel(args.loglevel if args.loglevel else logging.INFO)
    #  breakpoint()
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
