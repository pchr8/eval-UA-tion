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
from lmentry_static.consts import data_word_classes_yaml_path
from itertools import permutations

from omegaconf import OmegaConf

b = breakpoint


class TaskTypeWordCategories(Task):
    """Abstract class for task types like 'which word doesn't belong to
    the rest, dog/cat/wolf/potato?'


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

    def __init__(self, path=None):
        self.categories = self.get_categories(path=path)
        super().__init__()

    def get_categories(self, path=None):
        conf = OmegaConf.load(path if path else data_word_classes_yaml_path)
        return conf

    ###
    # ABSTRACT METHODS
    ###

    @abstractmethod
    def gen_gold(
        self,
        **kwargs,
        #  words_list: list[str],
        #  other_word: str,
        #  cat_name_main: str,
        #  cat_name_other: str,
    ) -> Optional[str|bool]:
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
        # TODO
        return kwargs

    @override
    def gen_single_task(
        self, lim: Optional[int] = None, num_words_in_cat: int = None, **kwargs
    ) -> TaskDataset:
        """Generates the dataset based on this task.

        Args:
            lim (Optional[int]): how many instances to generate
            kwargs: not used, there to make override happy

        Returns:
            TaskDataset:
        """
        cats_permutations = permutations(list(self.categories.keys()), r=2)  # [:lim]
        instances = list()
        for c1, c2 in cats_permutations:
            ws1 = self.categories[c1]
            ws2 = self.categories[c2]
            inst = self.gen_task_instances(
                main_cat_words=ws1,
                other_cat_words=ws2,
                cat_name_main=c1,
                cat_name_other=c2,
                num_words_in_cat=num_words_in_cat,
            )
            instances.extend(inst)
            #  if lim and len(instances) > lim:
            # TODO sample or something maybe?
            if lim and len(instances) > lim:
                break
        #  # more than one instance can be created from a single permutation

        for i,inst in enumerate(instances):
            inst.additional_metadata['id'] = i
        task = TaskDataset.create(
            name=self.name,
            instances=instances[:lim],
            system_prompts=self.templates.system_prompts,
        )
        return task

    ###
    # OTHER
    ###

    #
    #  def gen_task_instances(
    #      self, t1: str, t2: str, kind: str
    #  ) -> list[TaskDatasetInstance]:
    #      """Create zero or more dataset rows for a single t1/t2 of a certain
    #      kind from all relevant templates.
    #
    #      If t1 is not more or less than t2 an empty list will be returned.
    #
    #      Args:
    #          t1 (str): t1
    #          t2 (str): t2
    #          kind (str): kind whether we are looking for a larger or smaller t1
    #
    #      Returns:
    #          list[TaskDatasetInstance]: can be empty.
    #      """
    #      answer = self.gen_gold(t1=t1, t2=t2, kind=kind)
    #
    #      # If words are equal no task possible
    #      if answer is None:
    #          return list()
    #
    #      # Get templates for this specific kind
    #      templates = self.templates({"kind": kind})
    #
    #      res = list()
    #      for i, t in enumerate(templates):
    #          # For each template, generate an instance and a reversed instances
    #
    #          # TODO meta for "correct answer closer to beginning of question"
    #          new_template = self.gen_instance_str_from_template(
    #              t1=t1, t2=t2, template_str=t
    #          )
    #          meta = self.get_meta(t1=t1, t2=t2, answer=answer)
    #          meta.update({"reversed": False})
    #          ti = TaskDatasetInstance.create_from_template(
    #              template=t,
    #              question=new_template,
    #              correct_answer=answer,
    #              additional_metadata=meta,
    #          )
    #          res.append(ti)
    #
    #          meta = self.get_meta(t1=t2, t2=t1, answer=answer)
    #          meta.update({"reversed": True})
    #          new_template = self.gen_instance_str_from_template(
    #              t1=t2, t2=t1, template_str=t
    #          )
    #          ti = TaskDatasetInstance.create_from_template(
    #              template=t,
    #              question=new_template,
    #              correct_answer=answer,
    #              additional_metadata=meta,
    #          )
    #          res.append(ti)
    #          #  b()
    #      return res
