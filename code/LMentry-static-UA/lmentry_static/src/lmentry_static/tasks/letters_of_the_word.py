from num2words import num2words
import re

from enum import Enum
from typing import Optional, Iterable
from overrides import override
from collections import Counter

from ukr_numbers.nums import Numbers
from lmentry_static.tasks.task_type_n_in_m import TaskTypeNinM

from lmentry_static.tasks.data_structures import TaskDatasetInstance, TaskDataset
from lmentry_static.tasks.data_structures import Template, TaskTemplates

from lmentry_static.data.words import WSP

b = breakpoint


class LOWTask(TaskTypeNinM):
    """Tasks about the first/second/nth/last letter of the word."""

    def __init__(self):
        self.name = "LOWTask"
        self.ukr_numbers = Numbers(graceful_failure=False, negative_one_is_last=True)
        super().__init__()

    @override
    def gen_gold(self, haystack: str, needle: int) -> str:
        return self.nth_letter_of_the_word(word=haystack, n=needle)

    @staticmethod
    def nth_letter_of_the_word(word: str, n: int) -> str:
        # !!! one-indexed!!!
        if n == -1:
            # 'last'
            return word[n]
        if n < 1:
            raise ValueError(f"1-indexed and positive numbers only!")
        res = word[n - 1]
        return res

    @override
    def gen_instance_template(
        self, template_str: str, haystack: str, needle: int
    ) -> Optional[str]:
        """Fill `template` with needle/haystack and return the template question."""
        word = haystack
        n = needle
        ukr_nums = self.ukr_numbers

        nums, new_t = self.pattern_from_caps_words(template_str)
        correct_num_words = list()
        for num_desc in nums:
            # may be None if -1 and 'один'
            correctly_inflected_n = ukr_nums.convert_to_auto(n, num_desc)
            correct_num_words.append(correctly_inflected_n)

        # don't add to list if any of the words aren't inflectable
        if not any([n is None for n in correct_num_words]):
            task_from_t = new_t.format(*correct_num_words, WORD=word)
            return task_from_t
        # None if not inflectable
        return

    @override
    def get_meta_from_haystack(self, haystack) -> dict:
        # TODO more robust numbers based on real statistics
        m = dict()
        l = len(haystack)
        if l <= 3:
            m["word_length"] = "short"
        elif l <= 6:
            m["word_length"] = "mid"
        else:
            m["word_length"] = "long"

        # Add any metadata the word thingy hsa
        if haystack in WSP.mddict:
            m.update(WSP.mddict[haystack])
        return m

    @override
    def gen_single_task(
        self, haystacks: list[str] = WSP.words, lim: Optional[int] = None, abstand: int = 4
    ) -> TaskDataset:
        return super().gen_single_task(haystacks=haystacks, lim=lim, abstand=abstand)


def run():
    from rich import print, inspect
    #  from lmentry_static.data.words import WORDS

    #  print(LOWTask.nth_letter_of_the_word("перемога", 1))
    #  print(LOWTask.nth_letter_of_the_word("перемога", -1))
    low = LOWTask()
    #  gt = low.generate_task("завтра", n=-1)
    #  print(gt)
    #  r= low.gen_task_instances(haystack="слово",needle=1)
    #  haystacks=['слово', 'один']
    haystacks = WSP.words
    rr = low.gen_single_task(haystacks=haystacks, abstand=2)
    print(rr)

    b()


if __name__ == "__main__":
    run()
