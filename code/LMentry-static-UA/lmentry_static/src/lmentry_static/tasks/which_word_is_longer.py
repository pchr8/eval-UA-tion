import re

import spacy
import logging

from enum import Enum
from typing import Optional, Iterable
from overrides import override
from collections import Counter

#  from lmentry_static.tasks.templating_utils import pattern_from_caps_words
from lmentry_static.tasks.task_type_compare_two_things import TaskTypeCompareTwoThings

from lmentry_static.data.words import WSP

logging.basicConfig()
logger = logging.getLogger(__name__)


b = breakpoint


class WordLengthComparisonTask(TaskTypeCompareTwoThings):
    """Which word is longer/shorter?

    TODO add task type 'are words the same length?'
    TODO finish refactoring like the rest
    """

    def __init__(
        self,
    ):
        self.name = "WordLengthComparison"
        super().__init__()

    @override
    def gen_gold(self, t1: str, t2: str, kind: str) -> Optional[str]:
        if len(t1) == len(t2):
            return
        if kind == "more":
            res = t1 if len(t1) > len(t2) else t2
        else:
            res = t1 if len(t1) < len(t2) else t2
        return res

    @override
    #  def get_meta(self, t1, t2, **kwargs) -> dict:
    def get_meta(self, **kwargs) -> dict:
        t1 = kwargs.get("t1").lower()
        t2 = kwargs.get("t2").lower()
        answer = kwargs.get("answer").lower()
        # Add metadata parent TaskTypeCompareTwoThings  might wish to add
        meta = super().get_meta(t1=t1, t2=t2, answer=answer)
        #  if t1 in WSP.mddict:
        #      meta.update(WSP.mddict[t1])
        #  if t2 in WSP.mddict:

        # They SHOULD be there, bug with capitalization fixed.
        # For HF/CSV they have to have the same fields so skipping not an option.
        meta["t1_meta"] = WSP.mddict[t1]
        meta["t2_meta"] = WSP.mddict[t2]
        return meta


def run():

    from rich import print, inspect

    w = WordLengthComparisonTask()
    t1 = "зозуля"
    t2 = "Галя"

    task = w.gen_single_task()
    print(task)
    b()


if __name__ == "__main__":
    run()
