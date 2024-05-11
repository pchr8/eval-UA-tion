import re

from abc import ABC, abstractmethod
from overrides import override

from enum import Enum
from typing import Optional, Iterable
from collections import Counter

from lmentry_static.tasks.generic_task import Task
from lmentry_static.tasks.data_structures import TaskDatasetInstance, TaskDataset
from lmentry_static.data.words import WSP
from lmentry_static.consts import KEY_SYSTEM_PROMPT, OPTION_KEY, LABEL_KEY
from itertools import permutations
from random import shuffle


b = breakpoint


class TaskTypeCompareTwoThings(Task):
    """Abstract class for task types like 'what word has more letters, a or b?'
    or 'which number is bigger, a or b?'

    Concepts:
    - "kind" is roughly less/more, bigger/smaller, whatever fits best for the
       specific child task.

       It's needed as a separate parameter everywhere because we need to
       know which of the two  we're looking for when generating ground truth.
       (It also has to be part of templates metadata!)


    Abstract methods:
    - gen_gold(): given t1/t2/kind, return the 'correct answer' for this case

    Overrides:
    - get_meta() adds info about what were t1 and t2
    - gen_single_task() because we always do

    Has:
    - gen_instance_str_from_template() that generates a template string based on a t1 and t2
    - gen_task_instances() that generate all TaskDatasetInstance for a
        single combination of t1, t2, and kind, based on all relevant templates.
    """
    # TODO finish refactoring like the other examples, e.g. words don't belong here

    # 'kind' field in datasets. Can mean shorter/longer, before/after, smaller/bigger
    # TODO replace with '<', '>'?
    KINDS = ["less", "more"]

    def __init__(self):
        super().__init__()

        if not self.templates({"kind": self.KINDS[0]}) and not self.templates(
            {"kind": self.KINDS[1]}
        ):
            raise ValueError(
                f"None of the templates have a valid ({self.KINDS})"
                f" kinds metadata field: {self.templates()}"
            )

    ###
    # ABSTRACT METHODS
    ###

    @abstractmethod
    def gen_gold(self, t1: str, t2: str, kind: str) -> Optional[str]:
        """Method that returns the 'correct answer' for task instance.
        In this case - either t1 or t2.

        Returns None if the 'length'/'size'/whatever-comparison-metric is equal.
        """
        pass

    ###
    # OVERRIDES
    ###

    @override
    #  def get_meta(self, t1:str, t2:str, **kwargs) -> dict:
    def get_meta(self, **kwargs) -> dict:
        """Adds info about what were t1 and t2"""
        t1 = kwargs.get("t1")
        t2 = kwargs.get("t2")
        answer = kwargs.get("answer")
        if not answer:
            # Had a bug where it was none, and the comparison below always returned 1
            raise ValueError(f"No answer provided in get_meta: {t1=}, {t2=}, {kwargs=}")

        #  res = {"t1": t1, "t2": t2}
        res = {OPTION_KEY.format(n=0): t1, OPTION_KEY.format(n=1): t2}
        res[LABEL_KEY]=0 if answer==t1 else 1
        #  print(res)
        #  b()
        return res

    @override
    def gen_single_task(
            self, lim: Optional[int] = None, words: list[str] = WSP.words, do_shuffle=True,  **kwargs
    ) -> TaskDataset:
        """Generates the dataset based on this task.

        Args:
            lim (Optional[int]): how many instances to generate
            words (list[str]): which words/THINGS to generate instances
                from (will do all pair permutations from them)
            kwargs: not used, there to make override happy

        Returns:
            TaskDataset:
        """
        words_permutations_ord = permutations(words, r=2)  # [:lim]

        if do_shuffle:
            words_permutations = list(words_permutations_ord)
            shuffle(words_permutations)
        else:
            words_permutations = words_permutations_ord

        instances = list()
        for t1, t2 in words_permutations:
            for k in self.KINDS:
                inst = self.gen_task_instances(t1=t1, t2=t2, kind=k)
                # TODO - add the system template to each task instance, till I find a better place for it
                #  for x in inst:
                #      x.additional_metadata[KEY_SYSTEM_PROMPT] = self.templates.system_templates
                instances.extend(inst)
            if lim and len(instances) > lim:
                break
        for i,inst in enumerate(instances):
            inst.additional_metadata['id'] = i

        # more than one instance can be created from a single permutation
        task = TaskDataset.create(name=self.name, instances=instances[:lim], system_prompts=self.templates.system_prompts)
        return task

    ###
    # OTHER
    ###

    def gen_instance_str_from_template(
        self, t1: str, t2: str, template_str: str
    ) -> Optional[str]:
        """Creates a string with specific t1/t2 from a template string"""
        res = template_str.template.format(t1=t1, t2=t2)
        return res

    def gen_task_instances(
        self, t1: str, t2: str, kind: str
    ) -> list[TaskDatasetInstance]:
        """Create zero or more dataset rows for a single t1/t2 of a certain
        kind from all relevant templates.

        If t1 is not more or less than t2 an empty list will be returned.

        Args:
            t1 (str): t1
            t2 (str): t2
            kind (str): kind whether we are looking for a larger or smaller t1

        Returns:
            list[TaskDatasetInstance]: can be empty.
        """
        answer = self.gen_gold(t1=t1, t2=t2, kind=kind)

        # If words are equal no task possible
        if answer is None:
            return list()

        # Get templates for this specific kind
        templates = self.templates({"kind": kind})

        res = list()
        for i, t in enumerate(templates):
            # For each template, generate an instance and a reversed instances

            # TODO meta for "correct answer closer to beginning of question"
            new_template = self.gen_instance_str_from_template(
                t1=t1, t2=t2, template_str=t
            )
            meta = self.get_meta(t1=t1, t2=t2, answer=answer)
            meta.update({"reversed": False})
            ti = TaskDatasetInstance.create_from_template(
                template=t,
                question=new_template,
                correct_answer=answer,
                additional_metadata=meta,
            )
            res.append(ti)

            meta = self.get_meta(t1=t2, t2=t1, answer=answer)
            meta.update({"reversed": True})
            new_template = self.gen_instance_str_from_template(
                t1=t2, t2=t1, template_str=t
            )
            ti = TaskDatasetInstance.create_from_template(
                template=t,
                question=new_template,
                correct_answer=answer,
                additional_metadata=meta,
            )
            res.append(ti)
            #  b()
        return res
