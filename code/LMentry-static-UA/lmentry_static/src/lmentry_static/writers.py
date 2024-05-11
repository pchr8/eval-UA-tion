from pathlib import Path

import json
from omegaconf import OmegaConf
from tempfile import TemporaryDirectory
from datasets import Dataset, BuilderConfig

from typing import Optional

import logging

logger = logging.getLogger(__name__)


from enum import Enum


from lmentry_static.consts import (
    templates_path,
    config_path,
    CONFIG_FN,
    TASK_OUTPUT_FN_FORMAT,
    TASK_OUTPUT_CSV_FORMAT,
    TASK_HF_OUTPUT_FORMAT,
    TASK_OUTPUT_JSONL_FORMAT,
    JSON_DUMP_KWARGS,
    KEY_SYSTEM_PROMPT,
)
from lmentry_static.tasks.data_structures import TaskDatasetInstance, TaskDataset


class OutputFormat(str, Enum):
    JSON = "JSON"
    OPENAI_JSONL = "JSONL"
    HF = "HF"
    CSV = "CSV"


def write_task(
    task: TaskDataset,
    output_dir: Path,
    output_format: Optional[OutputFormat | str] = OutputFormat.CSV,
    add_system_prompts: bool = False,
    train_test_splits: Optional[tuple[float, float]] = None,
):
    """Serialize task to output_dir in a specific format.

    Output directory is the one that contains all tasks, a file/dir name
    will be chosen based on task and output format.

    JSON will be deserializable back into tasks through SingleTask.from_json_file()

    Args:
        task (SingleTask): task
        output_dir (Path): output_dir
        output_format (Optional[OutputFormat | str]): output_format
        train_test_splits if not None, passed to ds.train_test_split()
    """
    if train_test_splits and (output_format is not OutputFormat.HF):
        #  logger.error(f"Splits {splits} SUPPORTED ONLY IN HF OUTPUT FORMAT!")
        raise ValueError(
            f"Splits {train_test_splits} SUPPORTED ONLY IN HF OUTPUT FORMAT!"
        )

    if output_format is OutputFormat.JSON:
        # NB will add system_prompts in the same place as in the dataclass
        # (so one can directly read it into the dataclass as well)
        task_output_path = output_dir / TASK_OUTPUT_FN_FORMAT.format(task.name)
        # dataclass wizard
        task.to_json_file(task_output_path, **JSON_DUMP_KWARGS)
        logger.info(f"+ {task_output_path}")

    if output_format is OutputFormat.HF:
        # TODO multi-config HF dataset containing all tasks of all types
        task_output_path = output_dir / TASK_HF_OUTPUT_FORMAT.format(task.name)
        ds = _task_to_hf_dataset(task=task, add_system_prompts=add_system_prompts)
        if train_test_splits:
            logger.info(f"Applying train_test_splits {train_test_splits}")
            #  ds.train_test_split(train_test_split[0], train_test_split[1])
            ds = ds.train_test_split(*train_test_splits)
        ds.save_to_disk(task_output_path)
        logger.info(f"+ {task_output_path}")

    if output_format is OutputFormat.CSV:
        task_output_path = output_dir / TASK_OUTPUT_CSV_FORMAT.format(task.name)
        ds = _task_to_hf_dataset(task=task, add_system_prompts=add_system_prompts)
        ds.to_csv(task_output_path)
        logger.info(f"+ {task_output_path}")

    if output_format is OutputFormat.OPENAI_JSONL:
        # jsonl format a la OpenAI eval input: https://github.com/openai/evals/blob/main/evals/registry/data/README.md
        task_output_path = output_dir / TASK_OUTPUT_JSONL_FORMAT.format(task.name)
        res = _task_to_jsonl_dataset(task=task, path=task_output_path)
        logger.info(f"+ {task_output_path}")
        #  ds = LMentryDatasetGenerator._task_to_hf_dataset(task=task)
        #  ds.to_csv(task_output_path)
        #  logger.info(f"+ {task_output_path}")


def _task_to_hf_dataset(
    task: TaskDataset,
    add_system_prompts: bool = False,
) -> Dataset:
    """Generates a HF dataset from the task."""
    # TODO - datasetInfo and stuff?
    gen = (
        task._system_prompts_row_generator if add_system_prompts else task.row_generator
    )
    ds = Dataset.from_generator(gen)
    return ds


def _task_to_jsonl_dataset(task: TaskDataset, path: Optional[Path] = None) -> list[str]:
    res = task.write_as_jsonl_ds(
        path=path, flatten=True, add_system_prompts_key=KEY_SYSTEM_PROMPT
    )
