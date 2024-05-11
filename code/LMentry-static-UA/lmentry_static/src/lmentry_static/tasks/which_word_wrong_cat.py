import re

import spacy
import logging

from enum import Enum
from typing import Optional, Iterable
from overrides import override
from collections import Counter

#  from lmentry_static.tasks.templating_utils import pattern_from_caps_words
from lmentry_static.tasks.task_type_word_categories import TaskTypeWordCategories

from lmentry_static.tasks.data_structures import TaskDatasetInstance

from random import choice, choices

from rich import print

logging.basicConfig()
logger = logging.getLogger(__name__)


b = breakpoint


class WhichWordWrongCatTask(TaskTypeWordCategories):
    def __init__(self, num_words_in_cat=5, **kwargs):
        self.name = "WhichWordWrongCatTask"
        #  self.num_words_in_cat = num_words_in_cat
        super().__init__(**kwargs)

    @override
    def gen_gold(
        self,
        #  words_list: list[str],
        other_word: str = None,
        #  cat_name_main: str = None,
        #  cat_name_other: str = None,
        **kwargs
    ) -> Optional[str]:
        return other_word

    def gen_instance_str_from_template(
        self, words_list: list[str], template_str: str
    ) -> Optional[str]:
        pretty_words_list = ", ".join(words_list)
        res = template_str.template.format(words_list=pretty_words_list)
        return res

    def get_meta(self, a=1, **kwargs):
        d = super().get_meta(**kwargs)
        # some additional metadata here maybe
        # e.g. we strictly speaking have no need for cat_name** anywhere in this from_simple
        # but this is easiest
        #  d['ccc']=a
        return d

    def gen_task_instances(
        self,
        main_cat_words: list[str],
        other_cat_words: list[str],
        cat_name_main: str,
        cat_name_other: str,
        num_words_in_cat: int = None,
        num_diff_instances_from_same_cat=5,
    ) -> list[TaskDatasetInstance]:
        instances = list()

        main_cat_words = choices(main_cat_words, k=num_words_in_cat - 1) if num_words_in_cat else main_cat_words

        other_words = choices(other_cat_words, k=num_diff_instances_from_same_cat)
        #  other_word = choice(other_cat_words)
        for other_word in other_words:
            answer = self.gen_gold(
                #  words_list=main_cat_words,
                other_word=other_word,
                #  cat_name_main=cat_name_main,
                #  cat_name_other=cat_name_other,
            )

            words_list = main_cat_words+[other_word]

            # Get templates for this specific kind
            templates = self.templates()

            for i, t in enumerate(templates):
                # For each template, generate an instance and a reversed instances

                # TODO meta for "correct answer closer to beginning of question"
                new_template = self.gen_instance_str_from_template(
                    words_list=words_list, template_str=t
                )
                # tuples for (im)mutability,  TODO there must be a pythonic way to do this better
                meta = self.get_meta(
                    all_options=tuple(words_list),
                    label=words_list.index(answer),
                    main_cat_words=tuple(main_cat_words),
                    other_word=other_word,
                    cat_name_main=cat_name_main,
                    cat_name_other=cat_name_other,
                )#.copy()
                ti = TaskDatasetInstance.create_from_template(
                    template=t,
                    question=new_template,
                    correct_answer=answer,
                    additional_metadata=meta,
                )
                instances.append(ti)
        #  b()
        return instances


def run():

    #  from rich import print, inspect

    #  w = WhichWordWrongCatTask()
    #  t1 = "зозуля"
    #  t2 = "Галя"
    #
    #  task = w.gen_single_task()
    #  print(task)
    b()


if __name__ == "__main__":
    run()
