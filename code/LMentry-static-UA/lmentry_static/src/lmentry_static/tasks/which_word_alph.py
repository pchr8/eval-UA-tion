import re

import spacy
import logging

from enum import Enum
from typing import Optional, Iterable
from overrides import override
from collections import Counter

from lmentry_static.tasks.task_type_compare_two_things import TaskTypeCompareTwoThings
from lmentry_static.tasks.data_structures import TaskTemplates
from lmentry_static.data.words import WSP

import locale


logging.basicConfig()
logger = logging.getLogger(__name__)


b = breakpoint

def locale_stuff():
    locale.setlocale(locale.LC_COLLATE, "uk_UA.UTF-8")
    # TODO test and stuff
    if sorted("лї",key=locale.strxfrm) != ['ї', 'л']:
        raise ValueError(f"Could'nt set Ukrainian locale for correct alphabetic order, TODO.")

class WordsAlphabetOrderTask(TaskTypeCompareTwoThings):
    """
    - which word is first alphabet order
    - TODO later yes/no are both words in alphabetic order or not?
    """

    def __init__(
        self,
    ):
        self.name = "WordsAlphabetOrder"
        locale_stuff()
        super().__init__()
        #  r = TaskTemplates.create_from_str_list(self._LIST, self._get_templates_path())

    # Used only for template generation
    _LIST = [
        "Яке слово далі по алфавітному порядку: '{t1}' чи '{t2}'?",
        "Яке слово перше по алфавітному порядку: '{t1}' чи '{t2}'?",
        "Яке слово стоїть ближче до початку алфавіту: '{t1}' чи '{t2}'?",
        "Серед '{t1}' та '{t2}', яке слово розташоване ближче до кінця алфавіту?",
        "Серед '{t1}' і '{t2}', яке слово знаходиться ближче до літери A в алфавіті?",
        "Серед '{t1}' і '{t2}', яке слово знаходиться ближче до літери Я в алфавіті?",
        # ChatGPT used wrong відмінок внизу:
        #  "Визначте, яке з цих слів '{t1}' або '{t2}' знаходиться далі по алфавіті?",
    ]

    @override
    def gen_gold(self, t1: str, t2: str, kind: str) -> str:
        # Use Ukrainian alphabet order
        s = sorted([t1, t2], key=locale.strxfrm)
        if kind == "more":
            return s[1]
        else:
            return s[0]

    @override
    #  def get_meta(self, t1, t2, **kwargs) -> dict:
    def get_meta(self, **kwargs) -> dict:
        """Adds number of shared leading characters to metadata.

        Case insensitive.
        
        E.g. 'bob' and 'Bill' have 1, 'bob' and 'mike' have 0.

        Args:
            kwargs: Expects t1 and t2 to be strings.

        Returns:
            dict:
        """
        t1=kwargs.get('t1').lower()
        t2=kwargs.get('t2').lower()
        answer=kwargs.get('answer').lower()
        # Add metadata parent TaskTypeCompareTwoThings  might wish to add
        meta = super().get_meta(t1=t1, t2=t2, answer=answer)

        num_shared_characters = 0
        for i in range(min(len(t1), len(t2))):
            if t1[i] == t2[i]:
                num_shared_characters += 1
            else:
                break
        meta["num_shared_characters"] = num_shared_characters

        #  if t1 in WSP.mddict:
        meta['t1_meta'] = WSP.mddict[t1]
        #  if t2 in WSP.mddict:
        meta['t2_meta'] = WSP.mddict[t2]

        return meta

def run():

    from rich import print, inspect

    w = WordsAlphabetOrderTask()
    t1 = "зозуля"
    t2 = "Галя"
    t2 = "Зоя"
    words=[t1, t2]

    task = w.gen_single_task(words=words)
    print(task)
    #  b()


if __name__ == "__main__":
    run()
