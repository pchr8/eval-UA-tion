import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import BuilderConfig, Dataset
from omegaconf import OmegaConf
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(show_time=False, rich_tracebacks=True),
    ],
)


# TODO ugly, but..
from lmentry_static.data.words import WSP, build_vocabulary

# This runs the thing that makes data.words.WORDS/SENTS more interesting
build_vocabulary()

from typing import Optional

from rich import inspect, print

from lmentry_static.consts import (
    CONFIG_FN,
    KEY_SYSTEM_PROMPT,
    TASK_HF_OUTPUT_FORMAT,
    TASK_OUTPUT_CSV_FORMAT,
    TASK_OUTPUT_FN_FORMAT,
    config_path,
    templates_path,
)
from lmentry_static.tasks.are_all_words_same_cat import DoAllWordsBelongToCatTask
from lmentry_static.tasks.data_structures import TaskDataset, TaskDatasetInstance
from lmentry_static.tasks.generic_task import Task
from lmentry_static.tasks.letters_of_the_word import LOWTask
from lmentry_static.tasks.which_word_alph import WordsAlphabetOrderTask
from lmentry_static.tasks.which_word_is_longer import WordLengthComparisonTask
from lmentry_static.tasks.which_word_wrong_cat import WhichWordWrongCatTask
from lmentry_static.tasks.words_in_sentence import WISTask
from lmentry_static.writers import OutputFormat, write_task

TASK_TYPES = [LOWTask, WISTask]
TASK_TYPES = [LOWTask]
TASK_TYPES = [WordsAlphabetOrderTask]
TASK_TYPES = [WordLengthComparisonTask, LOWTask, WISTask, WordsAlphabetOrderTask]
TASK_TYPES = [WhichWordWrongCatTask]
TASK_TYPES = [
    WordLengthComparisonTask,
    LOWTask,
    WISTask,
    WordsAlphabetOrderTask,
    WhichWordWrongCatTask,
    DoAllWordsBelongToCatTask,
]

logger = logging.getLogger(__package__)
b = breakpoint


class LMentryDatasetGenerator:
    """
    Iterates through the existing tasks and generates a dataset
    from each of them.

    Uses an optional config file with parameters related
    to task generation for specific tasks.

    TODO - do HF dataset with diff configs a la GLUE
    TODO support comma-separated output formats?
    """

    def __init__(
        self,
        config_or_config_file: Path | str | OmegaConf,
        tasks: Optional[list[Task]] = None,
        output_format: Optional[OutputFormat | str] = OutputFormat.CSV,
        output_dir: Optional[str | Path] = None,
        add_system_prompts: Optional[bool] = True,
        train_test_splits: tuple[float, float] = [0.7, 0.3],
    ):
        """

        Args:
            config_or_config_file (Path | str | OmegaConf): config_or_config_file
            tasks (Optional[list[Task]]): list of task classes to generate datasets for
            output_format (Optional[OutputFormat | str]): format to use. CSV, JSON, HF
            output_dir: output directory, will use tmpdir if none
        """
        self.cfg = self._get_config(config_or_config_file=config_or_config_file)

        #  logger.info(f"Initialized LMentryDatasetGenerator with config:")
        #  logger.debug("Config: \n" + OmegaConf.to_yaml(self.cfg))
        print(OmegaConf.to_yaml(self.cfg))

        self.output_dir = self._setup_paths(output_dir=output_dir)

        self.output_format = OutputFormat(
            output_format.upper() if isinstance(output_format, str) else output_format
        )

        self.train_test_splits = train_test_splits
        self.train_test_splits = self.cfg.cfg.train_test_split

        self.tasks = tasks if tasks else TASK_TYPES
        self.add_system_prompts = add_system_prompts
        logger.info(f"{', '.join([x.__name__ for x in self.tasks])}")

    def run(self):
        """Generate datasets for all chosen tasks."""

        t = self.generate_tasks(
            self.cfg,
            tasks=self.tasks,
            output_format=self.output_format,
            output_dir=self.output_dir,
            add_system_prompts=self.add_system_prompts,
            train_test_splits=self.train_test_splits,
        )
        return t

    @staticmethod
    def generate_tasks(
        cfg: OmegaConf, tasks: list[Task], output_dir: Path, **kwargs
    ) -> list[TaskDataset]:
        """Generate datasets from all tasks in `tasks`, using
        config params in `cfg` if present.

        In `cfg`, each task key (named after the class) can have tow dicts:
        - init_params: passed to class during init. E.g. to choose other spacy model
        - call_params: passed to task generation method. E.g. instances limit

        Sample config:
            ```
            CONF = OmegaConf.create(
                {
                    "LOWTask": {
                        "call_params": {
                            "lim": 40,
                            #  "words": WORDS,
                        },
                    },
                    "WISTask": {
                        "init_params": {"model_name": "uk_core_news_sm"},
                        "call_params": {
                            "lim": 200,
                            #  "haystacks": SENTENCES,
                            "abstand": 4,
                        },
                    },
                    "WordLengthComparisonTask": {
                        "call_params": {
                            "lim": 20,
                            #  "words": WORDS,
                        },
                    },
                }
            )
            ```

        Args:
            cfg (OmegaConf): cfg
            tasks (list[Task]): tasks
            output_dir (Path): output_dir
            kwargs: will be passed to write_task

        Returns:
            list[SingleTask]: list of generated task datasets, the ones written to disk
        """

        created_tasks = list()
        main_config = cfg.get("cfg")
        tasks_from_cfg = main_config.tasks if main_config else list()
        tasks_to_run = [tn for tn in tasks if tn.__name__ in tasks_from_cfg]
        if tasks_to_run:
            logger.info(
                f"Tasks to run from config: {[x.__name__ for x in tasks_to_run]}"
            )
        else:
            logger.info(f"No task limits from config, running everything")
        for tn in tasks:
            if tn not in tasks_to_run and tasks_to_run:
                logger.debug(f"Skipping {tn} because it's not in config.cfg.tasks")
                continue
            logger.debug(f"Doing task {tn.__name__}")
            init_params = dict()
            call_params = dict()

            # Get the config for the specific task type we're doing
            task_cfg = cfg.get(tn.__name__, dict())
            if task_cfg:
                # params for initializing task
                init_params = task_cfg.get("init_params", dict())
                # params for generating instances for task
                call_params = task_cfg.get("call_params", dict())

            t = tn(**init_params)
            task = t.gen_single_task(**call_params)
            logger.info(
                f"Generated {tn.__name__} task with {len(task.instances)} instances."
            )

            write_task(
                task=task,
                output_dir=output_dir,
                **kwargs,
                #  output_format=OutputFormat.CSV,
            )

            created_tasks.append(task)
        config_save_path = output_dir / CONFIG_FN
        logger.debug(f"Saved config to {str(config_save_path)}")
        OmegaConf.save(cfg, f=config_save_path)
        logger.debug(f"Done!")
        return created_tasks

    @staticmethod
    def _setup_paths(output_dir: Path = None) -> Path:
        """Make given path writable or use tempdir."""
        output_dir = (
            Path(TemporaryDirectory(prefix="gen_lmentry_").name)
            if not output_dir
            else Path(output_dir)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = output_dir.resolve().absolute()
        logger.info(f"Output dir set up: {str(output_dir)}")
        return output_dir

    @staticmethod
    def _get_config(config_or_config_file: Path | str | OmegaConf) -> OmegaConf:
        """Reads from file or returns given OmegaConf"""
        if isinstance(config_or_config_file, OmegaConf):
            cfg = config_or_config_file
        else:
            path = Path(config_or_config_file)
            if path.exists():
                cfg = OmegaConf.load(path)
            else:
                raise ValueError
        return cfg


def run(args):
    # TODO argparse for choosing a config file, and remove the current hardcoded path
    #  CFG_FILE = config_path.parent / "config_full.yaml"
    #  CFG_FILE = config_path
    dg = LMentryDatasetGenerator(
        config_or_config_file=args.config,
        output_dir=args.output,
        output_format=args.format,
    )
    t = dg.run()
    if args.pdb:
        b()


if __name__ == "__main__":
    run()
