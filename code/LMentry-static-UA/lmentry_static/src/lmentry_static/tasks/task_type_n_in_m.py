import re

from abc import ABC, abstractmethod

from enum import Enum

from spacy.tokens import Doc

from lmentry_static.tasks.generic_task import Task

from lmentry_static.tasks.data_structures import TaskDatasetInstance, TaskDataset
from lmentry_static.tasks.data_structures import Template, TaskTemplates

#  from lmentry_static.data.words import WORDS
from lmentry_static.consts import KEY_SYSTEM_PROMPT


from typing import Optional, Iterable, Any
from collections import Counter


b = breakpoint


class TaskTypeNinM(Task):
    """Abstract class for task types like 'what's the 4th word in sentence'
    or 'what's the eight letter in word'

    Provides:

        - pattern_from_caps_words() that creates a template+ for
            an input text based on the CAPITALIZED words therein.
    TODO document all this much better
    """

    def __init__(self):
        # read templates from YAML
        super().__init__()

    @abstractmethod
    def gen_gold(self, haystack: str, needle: int) -> Optional[Any]:
        """Method that returns the 'correct answer' for task instance"""
        pass

    @abstractmethod
    def gen_instance_template(
        self, template_str: str, haystack: str, needle: int
    ) -> Optional[str]:
        """Generate a single question from a
        single template for a specific needle/haystack pair."""
        pass

    @staticmethod
    def pattern_from_caps_words(text: str) -> tuple[list[str], str]:
        """Find words written in ALL CAPS longer than one letter.

        In: 'Це те ПЕРШЕ місце яке я взяв, УРА'
        Out: [ПЕРШЕ, УРА], 'Це те {} місце яке я взяв, {}'
        """

        # https://regex101.com/r/tDsPmX/1
        pattern = r"(\b(?:[А-ЯІЇЄ]){2,}\b)"

        # Use the findall() function from the re module to find all matches
        #  capitalized_words = re.findall(pattern, text)
        capitalized_words = list(re.finditer(pattern, text))
        replacements = re.sub(pattern, "{}", text)
        res = list()
        for w in capitalized_words:
            #  res.append((w.group(), text.split().index(w.group())))
            res.append(w.group())
        return res, replacements

    def get_meta_from_needle(self, needle) -> dict:
        res = dict(needle=needle)
        return res

    def get_meta_from_haystack(self, haystack) -> dict:
        res = dict(len_haystack=len(haystack))
        return res

    def gen_task_instances(
        self, haystack: str, needle: int
    ) -> list[TaskDatasetInstance]:
        if needle == 0 or needle < -1:
            raise ValueError(
                f"One-indexed and not supporting negative indexes, so {needle} is invalid!"
            )

        answer = self.gen_gold(haystack, needle)

        if not answer:
            return list()

        res = list()
        for i, t in enumerate(self.templates()):
            task_from_t = self.gen_instance_template(
                template_str=t.template, haystack=haystack, needle=needle
            )

            if task_from_t:
                meta = dict()
                meta.update(self.get_meta_from_needle(needle))
                meta.update(self.get_meta_from_haystack(haystack))

                ti = TaskDatasetInstance.create_from_template(
                    template=t,
                    question=task_from_t,
                    correct_answer=answer,
                    additional_metadata=meta,
                )
                res.append(ti)
        return res

    def gen_single_task(
        self, haystacks: list[str], lim: Optional[int] = None, abstand: int = 1
    ) -> TaskDataset:
        instances = list()
        for h in haystacks:
            for num in list(range(1, len(h), abstand)) + [-1]:
                inst = self.gen_task_instances(haystack=h, needle=num)
                #  for x in inst:
                #      x.additional_metadata[KEY_SYSTEM_PROMPT] = self.templates.system_templates
                instances.extend(inst)
            if lim and len(instances) > lim:
                break

        for i,inst in enumerate(instances):
            inst.additional_metadata['id'] = i
        task = TaskDataset.create(name=self.name, instances=instances[:lim], system_prompts=self.templates.system_prompts)
        return task
