from pathlib import Path

from typing import List, Tuple, Optional, Dict, Union, Any
from typing import NamedTuple

from dataclasses import dataclass, field
from collections import defaultdict
from collections import Counter

from flatten_dict import flatten
import pandas as pd

import json
from dataclass_wizard import JSONWizard

import logging

logger = logging.getLogger(__name__)

from ua_cbt.data_structures import GeneratedTasks, ReplacementOptionTypes
from ua_cbt.consts import REPLACEMENT_TOKEN_SYMBOL
from ua_cbt.consts import (
    CSV_STORIES_KEY,
    CSV_USABLE_KEY,
    CSV_TODO_KEY,
    CSV_MD_TARGET_KEY,
    CSV_LS_STORY_ID,
)

b = breakpoint

# TODO rename file into io.py or reading_writing.py or sth


COLUMNS_TO_KEEP_IN_MD = [
    "BAD_ENDING",
    "NUM_WORDS",
    "N_MAIN_CHARACTERS",
    "N_MINOR_CHARACTERS",
    "READING_LEVEL",
    "Unnamed: 0",
    "model",
    "temperature",
    "md_errors_fixed_using_model",
]


def to_csv(task: GeneratedTasks, path: Path) -> str:
    list_dicts = list()
    for t in task.tasks:
        dicts = [st.to_dict() for st in t]
        fd = [flatten(d, reducer="underscore", keep_empty_types=(dict,)) for d in dicts]
        list_dicts.extend(fd)
    df = pd.DataFrame(list_dicts)
    df.to_csv(path, index=False)
    return df


def options_to_options_str(options: list[str]) -> list[str]:
    letters = "АБВГДЕЄЖЗІЇКЛМНОПРСТУФХЦЧШЩЬЮЯ"
    res_l = list()
    for i, o in enumerate(options):
        res_l.append(f"{letters[i]}: {o}")
    #  res_str = "&lt;br&gt;-\n".join(res_l)
    res_str = "\n".join(res_l)
    return res_str


def _format_story_for_labelstudio(st):
    di = dict()
    options_html = list()
    options_html_adv = list()
    for i, o in enumerate(st.options):
        add_metadata = list()
        if i in st.md["opts_correct_answer_idx"]:
            add_metadata.append(("+", "darkgreen"))
        #  if i in st.md['opts_replacement_option_from_text_idx']:
        #      add_metadata.append(("O", "black"))
        if i in st.md["opts_distractor_external_idx"]:
            add_metadata.append(("E", "chocolate"))
        if i in st.md["opts_baseline_most_frequent_idx"]:
            add_metadata.append(("F", "blue"))
        if i in st.md["opts_distractor_most_frequent_any_gender_idx"]:
            add_metadata.append(("F", "red"))
        amd_html = ""

        def amd_to_html(amd):
            return f"<font color={amd[1]}>{amd[0]}</font>"

        amd_html = "&nbsp;" + "-".join(
            [amd_to_html(x) for x in add_metadata]
        )
        #  for amd in add_metadata:
        el_html = f"{o}" + "&nbsp;<b><sub>" + amd_html + "</sub></b>"
        el = {"value": o, "html": o + ""}
        el_adv = {"value": o, "html": el_html}
        options_html.append(el)
        options_html_adv.append(el_adv)
    di["LS_options"] = options_html
    di["LS_options_adv"] = options_html_adv

    # For human parsing if ever needed
    di["LS_options_str"] = options_to_options_str(st.options)
    # Bold gap
    di["LS_question_html"] = st.question.replace(
        REPLACEMENT_TOKEN_SYMBOL, f"<b>&rArr;{REPLACEMENT_TOKEN_SYMBOL}&lArr;</b>"
    ).replace("\n", "<br>")
    return di


def to_jsonl(
    task: GeneratedTasks,
    path: Path,
    for_labelstudio: bool = False,
    gap_is_bold: bool = True,
) -> str:
    list_dicts = list()
    for story_tasks in task.tasks:
        #  b()
        #  for task_instance in story_tasks:
        dicts = list()
        for st in story_tasks:
            di = st.to_dict()
            if for_labelstudio:
                di_ls_opts = _format_story_for_labelstudio(st)
                di.update(di_ls_opts)
                #  options_html = list()
                #  for i,o in enumerate(st.options):
                #      add_metadata = list()
                #      #  if i in st.md['opts_correct_answer_idx']:
                #      #      add_metadata.append(("T", "grey"))
                #      el = {"value": o, "html": o + ""}
                #      options_html.append(el)
                #  di['LS_options'] = options_html
                #  # For human parsing if ever needed
                #  di["LS_options_str"] = options_to_options_str(st.options)
                #  # Bold gap
                #  di["LS_question_html"] = st.question.replace(
                #      REPLACEMENT_TOKEN_SYMBOL, f"<b>&rArr;{REPLACEMENT_TOKEN_SYMBOL}&lArr;</b>"
                #  ).replace("\n", "<br>")
            dicts.append(di)
        #
        #
        #  dicts = [st.to_dict() for st in story_tasks]
        #  if for_labelstudio:
        #      for di in dicts:
        #          # For parsing by label-studio
        #          options_html = list()
        #          for i,o in enumerate(di["options"]):
        #              if i in di[gt]
        #              el = {"value": o, "html": o + ""}
        #              options_html.append(el)
        #          #  di["options_forls"] = [
        #          #      {"value": o, "html": o + ""} for o in di["options"]
        #          #  ]
        #          # For human parsing if ever needed
        #          di["options_show_str"] = options_to_options_str(di["options"])
        #          # Bold gap
        #          di["question_html"] = di["question"].replace(
        #              REPLACEMENT_TOKEN_SYMBOL, f"<b>&rArr;{REPLACEMENT_TOKEN_SYMBOL}&lArr;</b>"
        #          )

        fd = [flatten(d, reducer="underscore", keep_empty_types=(dict,)) for d in dicts]
        ls_dicts = [{"data": d} for d in fd] if for_labelstudio else fd
        list_dicts.extend(ls_dicts)

    #  b()
    if for_labelstudio:
        txt = json.dumps(list_dicts)
    else:
        txt = "\n".join([json.dumps(x) for x in list_dicts])
    path.write_text(txt, encoding="utf-8")
    #  breakpoint()
    return path


def get_usable(df: pd.DataFrame) -> pd.DataFrame:
    """Returns subset of df that is known-usable."""
    logger.info(f"Loaded {len(df)} rows")
    df_small = df[(df[CSV_USABLE_KEY] == "usable") & (df[CSV_TODO_KEY] != "todo")]
    logger.info(f"{len(df_small)}/{len(df)} stories marked as usable")
    df_small = df_small[~df_small[CSV_STORIES_KEY].isna()]
    logger.info(f"Removed empty stories, final size  {len(df_small)}.")
    #  b()
    return df_small


def read_csv(
    path: Path,  # = CSV_FILE,
    csv_stories_key=CSV_STORIES_KEY,
    csv_md_target_key=CSV_MD_TARGET_KEY,
    csv_story_id=CSV_LS_STORY_ID,
    columns_to_keep_in_md: list[str] = COLUMNS_TO_KEEP_IN_MD,
    #  csv_usable_key=CSV_USABLE_KEY,
    min_size_chars: int = 2000,
    lim: Optional[int] = None,
    tale_metadata_prefix="st_",
) -> tuple[list[tuple[str, int, dict]], dict]:
    #  df_orig = pd.read_csv(path, index_col=0)
    # TODO fix index colums
    df_orig = pd.read_csv(path)  # , index_col="Unnamed: 0")
    df = get_usable(df_orig)

    #  if csv_usable_key:
    #      df = df[df[CSV_USABLE_KEY].isin([1, "1"])]
    #      logger.info(f"Of them {len(df)} usable")
    tales = df[:lim]
    #  other_cols = list(tales.columns)
    other_cols = columns_to_keep_in_md
    #  other_cols.remove(csv_stories_key)
    tales[csv_md_target_key] = (
        tales[other_cols]
        .add_prefix(tale_metadata_prefix)
        .apply(lambda x: x.to_dict(), axis=1)
    )
    tales = tales[[csv_stories_key, csv_md_target_key, csv_story_id]]

    # TODO ugly bit — rewrite it happen above during parsing of df
    # Intent: if there are two annotations for fixed story we get a dictionary here 
    #   we make it a str.
    tales_str_raw = list(tales[csv_stories_key])
    tales_str = list()
    for i, itale in enumerate(tales_str_raw):
        if itale.startswith("{\"text"):
            #  b()
            jdict = eval(itale)['text']
            # TODO ugly 
            differing_tales = [x for x in jdict if x!=df.iloc[i].generated_story]
            if len(differing_tales)>1:
                logger.error(f"More than 1 edited tale given for tale {tales.iloc[i].storyId}")
            else:
                tales_str.append(differing_tales[0])
        else:
            tales_str.append(itale)

    #  tales_sq_ids = list(range(len(tales)))
    tales_ids = list(tales[csv_story_id])
    tales_md = list(tales[csv_md_target_key])
    #  tales = tales[~tales.isna()]

    # I here refers to the index in the original dataframe
    zipped_tales = list(zip(tales_ids, tales_str, tales_md))
    #  tales = [(i, t) for i, t in tales.items()]

    logger.info(f"Loaded {len(tales)} stories.")

    metadata = {"filename": str(path)}
    return zipped_tales, metadata
