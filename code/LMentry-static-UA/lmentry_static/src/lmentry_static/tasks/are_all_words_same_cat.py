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

from random import choice, sample

from rich import print

logging.basicConfig()
logger = logging.getLogger(__name__)


b = breakpoint


class DoAllWordsBelongToCatTask(TaskTypeWordCategories):
    def __init__(self, num_words_in_cat=5, **kwargs):
        self.name = "DoAllWordsBelongToCatTask"
        #  self.num_words_in_cat = num_words_in_cat
        super().__init__(**kwargs)

    @override
    def gen_gold(
        self,
        #  words_list: list[str],
        #  other_word: str = None,
        #  cat_name_main: str = None,
        #  cat_name_other: str = None,
        #  res: bool,
        **kwargs
    ) -> bool:
        # Unused for this task
        pass

    def gen_instance_str_from_template(
        self, words_list: list[str], cat_name: str, template_str: str
    ) -> Optional[str]:
        pretty_words_list = ", ".join(words_list)
        res = template_str.template.format(
            words_list=pretty_words_list, cat_name=cat_name
        )
        return res

    def get_meta(self, **kwargs):
        d = super().get_meta(**kwargs)
        # some additional metadata here maybe
        # e.g. we strictly speaking have no need for cat_name** anywhere in this from_simple
        # but this is easiest
        #  d['ccc']=a
        return d#.copy()

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
        templates = self.templates()

        # we do it twice, once they belong, once they don't

        main_cat_words_small = (
            sample(main_cat_words, k=num_words_in_cat)
            if num_words_in_cat
            else main_cat_words
        )

        other_words = sample(other_cat_words, k=num_diff_instances_from_same_cat)
        for other_word in other_words:
            #  answer = self.gen_gold(
            #      #  words_list=main_cat_words,
            #      other_word=other_word,
            #      #  cat_name_main=cat_name_main,
            #      #  cat_name_other=cat_name_other,
            #  )

            words_list_no = main_cat_words_small[:-1] + [other_word]
            words_list_yes = main_cat_words_small

            # Get templates for this specific kind
            for i, t in enumerate(templates):
                # For each template, generate an instance and a reversed instances
                new_template_yes = self.gen_instance_str_from_template(
                    words_list=words_list_yes, cat_name=cat_name_main, template_str=t
                )
                new_template_no = self.gen_instance_str_from_template(
                    words_list=words_list_no, cat_name=cat_name_main, template_str=t
                )

                meta_yes = self.get_meta(
                    all_options=tuple(words_list_yes),
                    main_cat_words=tuple(main_cat_words_small),
                    #  other_word=other_word,
                    cat_name_main=cat_name_main,
                    #  cat_name_other=cat_name_other,
                )
                meta_no = self.get_meta(
                    all_options=tuple(words_list_no),
                    main_cat_words=tuple(main_cat_words_small[:-1]),
                    other_word=other_word,
                    cat_name_main=cat_name_main,
                    cat_name_other=cat_name_other,
                )
                ti_yes = TaskDatasetInstance.create_from_template(
                    template=t,
                    question=new_template_yes,
                    correct_answer=True,
                    additional_metadata=meta_yes,
                )
                ti_no = TaskDatasetInstance.create_from_template(
                    template=t,
                    question=new_template_no,
                    correct_answer=False,
                    additional_metadata=meta_no,
                )
                instances.append(ti_yes)
                instances.append(ti_no)

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
