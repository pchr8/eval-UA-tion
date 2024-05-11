from pathlib import Path

from typing import List, Tuple, Optional, Dict, Union, Any
from typing import NamedTuple

from dataclasses import dataclass, field
from collections import defaultdict
from collections import Counter

from enum import Enum

from dataclass_wizard import JSONWizard

from rich import inspect
from rich import print
from rich.panel import Panel
from rich.console import Console

######
# DATA STRUCTURES
######

# TODO 
SpacyMatcherPattern = NamedTuple(
    "SpacyMatherPattern",
    [
        ("name", str), 
        ("pattern", dict[str, Any]),
    ],
)

#  @rich.repr.auto(angular=True)
@dataclass
class SingleTask(JSONWizard):
    """Single self-contained task"""

    context: str
    question: str
    options: list[str]
    answer: str
    # Some story ID to split tasks by story later on
    story_id: Optional[str] = None
    task_type: str = None
    md: Optional[dict] = field(default_factory=dict)

    def __str__(self):
        res = f"CONTEXT:\n{self.context}\nQUESTION:\t{self.question}\nOPTIONS:\t{self.options}\nANSWER:\t{self.answer}\n({self.md})"
        return res

    def __rich_repr__(self):
        #  yield "SINGLE TASK"
        yield "context", self.context
        yield "question", self.question
        yield "options", self.options
        yield "answer", self.answer
        yield "story_id", self.story_id
        yield "md", self.md

    def as_qa(self, with_prompt: bool = False):
        res = self.context + " " + self.question 
        res+= "\n" 
        for i,o in enumerate(self.options, start=1):
            res+=f"{i}. {o}\n"
        if with_prompt:
            res+="\nВідповідь: "
        return res


    __rich_repr__.angular = True


@dataclass
class GeneratedTasks(JSONWizard):
    # TODO decide. Currently sublists are tasks from the same story
    #  tasks: list[SingleTask] | list[list[SingleTask]]
    #  tasks: list[SingleTask]
    tasks: list[list[SingleTask]]
    #  spacy_model: Optional[str] = None

    # book title or whatever
    #  source: Optional[str] = None
    other_md: Optional[dict] = field(default_factory=dict)

    def save(self, path: Path):
        txt = self.to_json(ensure_ascii=False, indent=4)
        path.write_text(txt, encoding="utf-8")

    def __str__(self):
        return f"<{len(self.tasks)} tales / {sum([len(x) for x in self.tasks])} tasks>"

    def __rich_repr__(self):
        yield "TASKS"
        yield "tasks", self.tasks
        if self.spacy_model:
            yield "model", self.spacy_model
        if self.source:
            yield "source", self.source
        if self.other_md:
            yield "metadata", self.other_md

    __rich_repr__.angular = True



@dataclass
class MatchPattern:
    name: str
    pattern: dict
    md: dict = field(default_factory=dict)


class ReplacementOptionTypes(Enum):
    TRUE_ANSWER = "correct_answer"
    NORMAL = "replacement_option_from_text"
    BASELINE_MOSTFREQUENT = "baseline_most_frequent"
    DISTRACTOR_EXTERN = "distractor_external"
    # it's always present as part as baseline_mf in clean_ro[0] by definition of both
    #  DISTRACTOR_MOSTFREQUENT = "distractor_most_frequent"
    DISTRACTOR_MOSTFREQUENT_AG = "distractor_most_frequent_any_gender"
#
#  @dataclass
#  class ReplacementOption:
#      opt_inflected: str = None
#      opt_lemma: str = None
#      opt_type: ReplacementOptionTypes = None
#      md: dict = None


