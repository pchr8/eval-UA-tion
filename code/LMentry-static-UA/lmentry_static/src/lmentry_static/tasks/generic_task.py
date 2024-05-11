from abc import ABC, abstractmethod
from pathlib import Path

from lmentry_static.tasks.data_structures import TaskDatasetInstance, TaskDataset
from lmentry_static.tasks.data_structures import TaskTemplates, Template

from lmentry_static.consts import templates_path


class Task(ABC):
    """Most generic abstract class for a single Task type.

    Goal: force functions to implement a gen_single_task() interface
        for later automatic task-independent TaskDataset generation.
        It would get as **kwargs an (OmegaConf-like) task config 
        with things like how many instances to generate etc.

    Abstract functions:
        - gen_single_task() that returns a TaskDataset. 

    Contains and runs during init functions for
        - reading task templates from YAML and making them available under self.templates

    Overrid-able fn-s:
        - gen_meta() for generating metadata to add to existing TaskDatasetInstance 
            metadata (e.g. when creating the instance check sentence length
            and add metadata based on it for future analysis)

    TODO: 
        - maybe reproducibility bits like dump the exact source data
            (list of words etc.) used when generating instances
    """

    def __init__(self):
        """Read templates from YAML. Don't forget to super().__init__() in 
        child classes!"""

        self.templates = None
        self.read_templates_from_yaml()

    ###
    # ABSTRACT METHODS
    ###

    @abstractmethod
    def gen_single_task(self, **kwargs) -> TaskDataset:
        """Method that returns a SingleTask instance
        with the entire dataset of questions of the type.
        """
        pass

    #  @abstractmethod
    def get_meta(self, **kwargs) -> dict:
        """Generates dict with additional metadata to add to existing"""
        return dict()

    ###
    # TEMPLATES
    ###

    def _get_templates_yaml_path(self) -> Path:
        """Gets path with YAML containing task templates. """
        path = templates_path / (self.name + ".yaml")
        return path

    def read_templates_from_yaml(self) -> TaskTemplates:
        """ yaml -> TaskTemplates """
        path = self._get_templates_yaml_path()
        templates = TaskTemplates.from_yaml_file(path)
        self.templates = templates
        return templates

