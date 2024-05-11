""" This file will contain the I/O and iterating-through-text-containing-things logic. """

import pdb
import sys
import traceback
import argparse

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any

import logging

import spacy
import pymorphy2

from spacy.lang.uk.examples import sentences
from spacy.matcher import Matcher
from dataclasses import dataclass
from collections import defaultdict
from collections import Counter

from omegaconf import OmegaConf
from dataclass_wizard import JSONWizard

import pandas as pd
import numpy as np

from ua_cbt.epub import read_epub
from ua_cbt.ml import CBTTaskMaker
from ua_cbt.data_structures import GeneratedTasks, SingleTask

from ua_cbt.consts import MODEL_NAME  # , EPUB_FILE, OUTPUT_DIR  # , LIMIT
from ua_cbt.consts import FILE_NAME_SPLIT_PAT, SPLIT_NAMES
from ua_cbt.consts import YAML_DATA

from ua_cbt.writing import to_csv, to_jsonl, read_csv

from rich import inspect
from rich import print
from rich.panel import Panel
from rich.console import Console

logging.basicConfig()
logger = logging.getLogger(__name__)

b = breakpoint


def process_tales(
    tales: list[tuple[int, str]],
    min_size_chars: int = 2000,
    lim: Optional[int] = None,
    metadata: Optional[dict] = None,
    save_path: Path = Path("/tmp/cbt"),
    **kwargs,
) -> GeneratedTasks:
    gt = GeneratedTasks(tasks=list())

    cbt_maker = CBTTaskMaker(**kwargs)

    global_number = 0
    num_failures = 0
    for i, (tale_id, t, tale_md) in enumerate(tales):
        tasks, doc_md = cbt_maker.process_txt(t)
        # TODO ugly remove after I finish pymorphy2/3 tess
        num_failures+=doc_md['num_failures']
        if not tasks:
            logger.error(
                f"Skipping tale {i} starting with {t[:30]} as no CBT task was created"
            )
            continue
        for ta in tasks:
            #  ta.md["text_id"] = i
            # For train/test splitting by-text this is important, not 'additional' anymore
            # NB we'll still have the tasks split by tales as sub-lists
            # but given how important this information is...
            ta.story_id = tale_id
            ta.md["global_number"] = global_number
            ta.md["tale_id_labelstudio"] = tale_id
            ta.md.update(tale_md)
            ta.md.update(doc_md)
            global_number += 1
            #  b()
        gt.tasks.append(tasks)
        #  gt.tasks.extend(tasks)
    logger.info(f"Used {len(gt.tasks)}/{len(tales)} stories.")

    #  gt.source = path.name
    gt.other_md = dict() if not metadata else metadata
    gt.other_md["limit"] = lim
    gt.other_md["spacy_model"] = cbt_maker.model_name
    #  gt.other_md["source"] = cbt_maker.model_name
    gt.other_md["min_size_chars"] = min_size_chars
    gt.other_md["min_size_chars"] = min_size_chars
    gt.other_md["_db_num_failures"] = num_failures
    gt.other_md["cbt_settings"] = cbt_maker._dump_settings()
    # TODO additional settings like whether distractors are used etc.

    #  save_tasks(gt=gt, save_dir=save_path)
    #  b()
    gt.other_md['stats']= do_stats(gt)
    return gt


def _find_split_sizes(split_sizes: list[float]) -> list[float]:
    """Given all split sizes except train, calculate the last one (train).

    E.g. 0.2, 0.3 -> 0.2, 0.3, 0.7
    """
    if isinstance(split_sizes, int) or isinstance(split_sizes, float):
        split_sizes = [split_sizes]
    if not split_sizes or split_sizes in ([1], [1.0], [0]):
        return [1.0]

    actual_split_sizes = list()

    last_split_size = 1.0
    for s in split_sizes:
        actual_split_sizes.append(s)
        last_split_size -= s
    if last_split_size <= 0 or not split_sizes:
        raise ValueError(
            f"Please provide split sizes of all splits except the last one, e.g. 0.7, 0.1 -> 0.7 0.1 0.2. {split_sizes=}"
        )
    #  actual_split_sizes.append(last_split_size)
    actual_split_sizes.insert(0, last_split_size)
    return actual_split_sizes


def split_tales_by_story(
    stories: list[list[Any]],
    split_sizes: list[float] = (0.7,),
) -> list[GeneratedTasks]:
    # TODO: could have I just used a pandas groupby instead of all this?..
    #  stories = tales.tasks
    num_stories = len(stories)
    num_task_instances = sum([len(x) for x in stories])
    actual_split_sizes = _find_split_sizes(split_sizes)

    if num_stories < len(actual_split_sizes):
        raise ValueError(
            f"Can't split {num_stories} into {len(actual_split_sizes)} splits."
        )

    logger.debug(f"Stories will be split as {[x for x in actual_split_sizes]}")
    splits = list()

    for ss in actual_split_sizes:
        num_stories_in_split = int(np.round(num_stories * ss))
        stories_in_split = stories[:num_stories_in_split]
        if len(stories_in_split) < 1.0:
            # todo later maybe 'make sure at least one story is in split'
            raise ValueError(
                f"Can't create split of size {ss} ({num_stories_in_split} stories) out of {num_stories} stories."
            )
        splits.append(stories_in_split)
        stories = stories[num_stories_in_split:]
    #  b()
    logger.info(
        f"{num_stories} stories (in total {num_task_instances} tasks) split into {len(splits)} of {'/'.join([str(len(x)) for x in splits])} stories ({'/'.join([f'{x:.2f}' for x in actual_split_sizes])})"
    )
    return splits

def do_stats(gt: GeneratedTasks):
    stats_type = Counter()
    for s in gt.tasks:
        for t in s:
            stats_type[t.task_type]+=1
    print(stats_type)
    return dict(stats_type)


def split_gentasks(
    gt: GeneratedTasks, split_sizes: list[float] = (0.7,)
) -> list[GeneratedTasks]:
    """Split tasks in gt by story into split_sizes.

    We expect gt.tasks to be a list of lists of SingleTasks, with the external
    list containing sub-lists of the same source (=same story).

    Args:
        gt (GeneratedTasks): gt
        split_sizes (list[float]): split_sizes

    Returns:
        list[GeneratedTasks]:
    """
    split_stories = split_tales_by_story(stories=gt.tasks, split_sizes=split_sizes)
    res = list()
    for s in split_stories:
        sg = GeneratedTasks(tasks=s, other_md=gt.other_md)
        res.append(sg)

    # sanity_check
    stories_by_split = list()
    for gt in res:
        for task_list in gt.tasks:
            story_ids = {t.story_id for t in task_list}
            stories_by_split.append(story_ids)
    len_all_story_ids = sum([len(x) for x in stories_by_split])
    #  b()
    # TODO finish sanity check;
    #   add writing to separate files
    # TODO - or do I just keep the same data structure
    return res


def save_tasks(
    gt: GeneratedTasks, save_dir: Path = Path("/tmp/cbt"), file_name: str = "tasks"
):
    # TODO - create folder etc. a la up-crawler
    csv_fn = save_dir / (file_name + ".csv")
    json_fn = save_dir / (file_name + ".json")
    jsonl_fn = save_dir / (file_name + ".jsonl")
    yaml_data_fn = save_dir / (file_name + ".data.yaml")
    jsonl_valid_fn = save_dir / (file_name + ".labelstudio.json")
    save_dir.mkdir(parents=True, exist_ok=True)
    gt.save(json_fn)
    logger.info(f"\t+ Written {str(json_fn)}")
    to_csv(gt, path=csv_fn)
    logger.info(f"\t+ Written {str(csv_fn)}")
    #  to_jsonl(gt, path=jsonl_fn)
    to_jsonl(gt, path=jsonl_fn)
    logger.info(f"\t+ Written {str(jsonl_fn)}")
    to_jsonl(gt, path=jsonl_valid_fn, for_labelstudio=True)
    logger.info(f"\t+ Written {str(jsonl_valid_fn)}")
    OmegaConf.save(YAML_DATA, yaml_data_fn)
    logger.info(f"\t+ Written {str(yaml_data_fn)}")


def run(args):
    #  LIMIT = 20
    #  LIMIT = 2
    #  tales, md = read_epub(path=EPUB_FILE)
    #  b()
    input_csv = args.input
    limit = args.limit
    output_dir = args.output
    splits = args.splits

    #  yaml_file_location = args.data_yaml
    #  data_content = OmegaConf.load(yaml_file_location)

    # tales is a list of (tale_str, id, dict-with-md)
    tales, md = read_csv(path=input_csv, lim=limit)

    tasks = process_tales(
        tales,
        lim=limit,
        metadata=md,
        #  words_and_data = data_content,
        #  data_yaml = yaml_file_location,
        #  n_context_sents=10,
        #  n_question_sents=4,
        question_sents_share=0.35,
        save_path=output_dir,
    )
    splits = split_gentasks(tasks, split_sizes=splits)
    for i, s in enumerate(splits):
        #  split_name = f"{i}-{len(s.tasks)}"
        split_name = (
            f"{SPLIT_NAMES[i]}"
            if i < len(SPLIT_NAMES)
            else f"{FILE_NAME_SPLIT_PAT.format(i)}"
        )
        logger.info(f"Split {i} ({split_name}) has ({len(s.tasks)} stories)")
        save_tasks(
            gt=s, save_dir=output_dir, file_name=FILE_NAME_SPLIT_PAT.format(split_name)
        )

    #  b()

    s = tasks.tasks[0][0]
    #  b()
