import re

import spacy
import logging

from spacy.matcher import Matcher
from spacy.tokens import Span, Doc, Token
from num2words import num2words

from enum import Enum
from typing import Optional, Iterable
from overrides import override
from collections import Counter

from lmentry_static.tasks.data_structures import Template, TaskTemplates
from lmentry_static.tasks.data_structures import TaskDataset
from lmentry_static.tasks.task_type_n_in_m import TaskTypeNinM
from ukr_numbers.nums import Numbers

from lmentry_static.data.words import WSP

logging.basicConfig()
logger = logging.getLogger(__name__)

b = breakpoint

MODEL_NAME = "uk_core_news_sm"


class WISTask(TaskTypeNinM):
    """Tasks about the first/second/nth/last word in a sentence"""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
    ):
        self.name = "WISTask"
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
        self.ukr_numbers = Numbers(graceful_failure=False, negative_one_is_last=True)
        # TODO add to init in parent
        super().__init__()

    """
    TEMPLATES = [
        'Яке ПЕРШЕ слово y реченні "{SENT}"?',
        'Яке слово в реченні "{SENT}" ПЕРШЕ?',
        'В реченні "{SENT}" на ПЕРШОМУ місці знаходиться слово ...',
        'В реченні "{SENT}" під номером ОДИН знаходиться слово ...',
    ]
    TASK_TEMPLATES = TaskTemplates.create_from_str_list(template_str=TEMPLATES)
    TASK_TEMPLATES = TaskTemplates(
        templates=[
            Template(
                template='Яке ПЕРШЕ слово y реченні "{SENT}"?',
                additional_metadata={"type": "ordinal"},
            ), 
            Template(
                template='Яке слово в реченні "{SENT}" ПЕРШЕ?',
                additional_metadata={"type": "ordinal"},
            ), 
            Template(
                template='В реченні "{SENT}" на ПЕРШОМУ місці знаходиться слово ...',
                additional_metadata={"type": "ordinal"},
            ), 
            Template(
                template='В реченні "{SENT}" під номером ОДИН знаходиться слово ...',
                additional_metadata={"type": "ordinal"},
            ), 
        ]
    )

    # TODO rewrite this fn to get a static dict-like as input, MAYBE A YAML
    TASK_TEMPLATES.add_numbers()
    """

    @override
    def gen_gold(self, haystack: str, needle: int) -> Optional[str]:
        doc = self.parse_text(haystack)
        return self.nth_word_in_sentence(sent=doc, n=needle)

    def parse_text(self, texts: str | list[str]) -> Doc | list[Doc]:
        if isinstance(texts, list):
            return [self.nlp(t) for t in texts]

        return self.nlp(texts)

    def nth_word_in_sentence(self, sent: Doc, n: int) -> Optional[str]:
        """Find nth word in sentence.
        -1 means 'last'.

        None if out of bounds of the shorter alpha_tokens

        Args:
            sent (Doc): sent
            n (int): n

        Returns:
            str:
        """
        # !!! one-indexed!!!
        #  alpha_tokens = [t for t in sent if t.is_alpha]
        # Prev. version failed for words like "Словʼянське" that are not alpha!
        alpha_tokens = [t for t in sent if not t.is_punct]
        if not alpha_tokens:
            raise ValueError(f"Found no alpha tokens in {sent=}")
        if n > len(alpha_tokens) - 1:
            return None

        if n == -1:
            # 'last'
            return str(alpha_tokens[-1])

        if n < 1:
            raise ValueError(f"1-indexed and positive numbers only!")
        res = str(alpha_tokens[n - 1])
        return res

    @override
    def gen_instance_template(
        self, template_str: str, haystack: str, needle: int
    ) -> Optional[str]:
        """Fill `template` with needle/haystack and return the template question."""
        sentence = haystack
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
            task_from_t = new_t.format(*correct_num_words, SENT=sentence)
            return task_from_t
        return

    @override
    def gen_single_task(
        self,
        haystacks: list[str] = WSP.sents,
        lim: Optional[int] = None,
        abstand: int = 4,
    )->TaskDataset:
        haystacks = self.parse_text(haystacks)
        return super().gen_single_task(haystacks=haystacks, lim=lim, abstand=abstand)


def run():
    from rich import print, inspect

    w = WISTask()
    #  haystack=w.parse_text(sentences[:5])
    #  r = w.gen_single_task(haystacks=haystack, abstand=4, lim=20)
    r = w.gen_single_task(abstand=4, lim=20)
    print(r)
    print(len(r.instances))
    b()
    #  SENT = sentences[1]
    #  sent = w.parse_text(SENT)
    #  print(f"Sent: {sent}")
    #  res = w.nth_word_in_sentence(n=2, sent=sent)
    #  res = w.nth_word_in_sentence(n=-1, sent=sent)
    #  print(res)
    #  print(LOWTask.nth_letter_of_the_word("перемога", 1))
    #  print(LOWTask.nth_letter_of_the_word("перемога", -1))
    #  low = LOWTask()
    #  gt = low.generate_task("завтра", n=-1)
    #  print(gt)
    pass


if __name__ == "__main__":
    run()
