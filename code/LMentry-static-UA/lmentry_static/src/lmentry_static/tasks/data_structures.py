from uuid import UUID, uuid4
from pathlib import Path

from dataclass_wizard import JSONSerializable, JSONFileWizard, YAMLWizard
from dataclasses import dataclass, field
from typing import Optional, Any

from omegaconf import OmegaConf
import json

from flatten_dict import flatten

CANARY_STRING = "3146ecde-ef2b-448c-9cd7-f72985a50b85"
# TODO integrate canary string to datasets or at least README:
#   https://github.com/google/BIG-bench/blob/main/docs/doc.md#creating-a-task-from-scratch

# TODO rewrite completely simplifying/moving the serialization bits! CSV-first, the rest is easy 


#############
### TEMPLATES
#############


@dataclass
class Template(JSONSerializable):
    """Template string with optional additional metadata about it.
    - Metadata:
        - can be used when filtering templates or later for analysis
        - if template includes spelling errors, you could do
            {'has_errors':True}
    """

    template: str
    additional_metadata: dict = field(default_factory=dict)

    # unique template identifier; regenerated every time template is created
    uuid: UUID = field(default_factory=uuid4)


@dataclass
class TaskTemplates(YAMLWizard, JSONSerializable, JSONFileWizard):
    """A set of templates belonging to a task.

    I/O:
    - pretty_dump_yaml()
    - YAMLWizard adds
        - from_yaml[file]
    """

    templates: list[Template]
    # System template w/ instructions on solving tasks of this type
    # e.g. 'you are solving an entrance exam, answer only with the letter of the correct answer'
    system_prompts: Optional[list[str]] = None

    def __call__(self, filter_dict: Optional[dict] = None) -> list[Template]:
        return self.get_templates(filter_dict=filter_dict)

    def get_templates(self, filter_dict: Optional[dict] = None) -> list[Template]:
        """Return a list of templates whose metadata matches (is a superset)
        of filter_dict.

        Use-case: 'give me all templates you have where the order of arguments
        is reversed.'
        """
        if not filter_dict:
            return self.templates
        filtered = list()
        for t in self.templates:
            if filter_dict.items() <= t.additional_metadata.items():
                filtered.append(t)
        return filtered

    @classmethod
    def create_from_str_list(
        cls, template_str: list[str], yaml_target: Optional[Path | str] = None
    ):
        """create_from_str_list.

        Generate simple templates list based on a list
        of strings.

        Args:
            template_str (list[str]): template_str
            yaml_target: will dump there if provided
        """
        templs = [Template(template=x) for x in template_str]
        templs = cls._add_ns(templs)

        res = cls(templates=templs)

        # add sequential numbers to metadata
        res.make_compliant()

        res.pretty_dump_yaml(
            yaml_target,
        )

        return res

    def pretty_dump_yaml(self, yaml_target: Path | str) -> str:
        """to_yaml() with better standard parameters.

        - Will use unicode characters as-is, so cyrillic stays cyrillic.
        - Keys won't be alphabetically sorted (=template str will be first)

        Args:
            yaml_target (Path | str): If given, will write YAML there

        Returns:
            str: pretty YAML
        """
        params = dict(
            allow_unicode=True,  # write Ukrainian as Ukrainian
            default_flow_style=False,
            sort_keys=False,  # so template is first in the YAML for readability
        )
        self.to_yaml_file(yaml_target, **params)
        return self.to_yaml(**params)

    @staticmethod
    def _add_ns(templs):
        """Adds sequential to template metadata."""
        for i, t in enumerate(templs):
            t.additional_metadata["template_n"] = i
        return templs

    def make_compliant(self):
        """Add template numbers to metadata of each template."""
        self.templates = self._add_ns(self.templates)
        #  for t in self.templates: t.set_uuid()


#############
### TASKS
#############


@dataclass
class TaskDatasetInstance(JSONSerializable):
    """An individual x/y pair w/ metadata.
    Basically future row in a CSV.
    """

    question: str
    correct_answer: str

    # UUID of template used when generating this instance
    template_uuid: Optional[UUID] = None

    # UUID of this instance (one template can generate multiple instances!)
    task_instance_uuid: Optional[UUID] = field(default_factory=uuid4)

    # Metadata attached to this instance. E.g. if our Y is a short word and
    #   we want to preserve that info to analyze whether predictions on
    #   short words are better.
    additional_metadata: Optional[dict] = None

    @classmethod
    def create_from_template(
        cls,
        template: Template,
        question: str,
        correct_answer: str,
        additional_metadata: Optional[dict] = None,
    ):
        """Utility method to create an instance using info from a template.

        Chiefly: adds the template's metadata to the metadata we get in the
        constructor.

        Args:
            template (Template): template
            question (str): question
            correct_answer (str): correct_answer
            additional_metadata (Optional[dict]): additional_metadata
        """

        md_dict = template.additional_metadata.copy()
        md_dict.update(additional_metadata if additional_metadata else dict())
        return cls(
            template_uuid=template.uuid,
            question=question,
            correct_answer=correct_answer,
            additional_metadata=md_dict,
        )


def flatten_dict(d: JSONSerializable) -> dict:
    """Generator that returns instance as flattened
    dict."""
    res_raw = d.to_dict()
    res_flat = flatten(res_raw, reducer="underscore", keep_empty_types=(dict,))
    return res_flat
    #  res.update(self.additional_metadata)
    #
    #  # TODO decide whether to keep dataclass_wizard's camelCase transformation
    #  for k in ["additional_metadata", "additionalMetadata"]:
    #      if k in res:
    #          del res[k]
    #  return res


@dataclass
class TaskDataset(JSONSerializable, JSONFileWizard):
    """All generated test instances of a single task type.

    (E.g. 50 questions about which word is longer.)
    """

    name: str  # name of the task that generated this
    instances: list[TaskDatasetInstance]
    system_prompts: Optional[list[str]] = None

    def _system_prompts_row_generator(self, add_system_prompts_key="system_prompts"):
        # Ugly way to pass a param to a generator needed for HF from_generator
        return self.row_generator(add_system_prompts_key=add_system_prompts_key)

    def row_generator(self, add_system_prompts_key: Optional[str] = None):
        """Generator emitting instances as flattened dicts, one by one.

        TODO document add_system_prompts_key
        """
        for i in self.instances:
            res = flatten_dict(i)
            if add_system_prompts_key:
                res[add_system_prompts_key] = self.system_prompts
            yield res

    # TODO Backporting additions from eval_ua_tion eval script
    # (till I figure out how to do it for real, maybe making it one package)
    def __iter__(self):
        return self.instances.__iter__()

    def __getitem__(self, i):
        return self.instances.__getitem__(i)

    @classmethod
    def create(cls, name, instances, **kwargs):
        # TODO - remove this method
        #  instances = cls._add_uuids_to_instances(instances)
        return cls(name=name, instances=instances, **kwargs)

    def write_as_jsonl_ds(
        self,
        path: Optional[Path] = None,
        flatten=False,
        add_system_prompts_key: Optional[str] = None,
    ) -> str:
        """
        Writes as jsonl file,that is one object per line:
            {"question": "Яке слово має менше літер: \"кіт\" чи \"ліжко\"?", "correctAnswer": "кіт", "templateUuid": "20019ed544b54cf08a2a66de4854a6a4", "taskInstanceUuid": "0789c7f82a854c579b832636a76a24eb", "additionalMetadata": {"kind":"less", "template_n": 3, "t1": "кіт", "t2": "ліжко", "reversed": false}}

        """
        jsonl_rows: str = ""
        #  for r in self.row_generator():
        if flatten:
            rs = [
                json.dumps(x, ensure_ascii=False)
                for x in self.row_generator(
                    add_system_prompts_key=add_system_prompts_key
                )
            ]
            for r in rs:
                jsonl_rows += r
        else:
            for r in self.instances:
                # TODO ..unintuitive
                if add_system_prompts_key:
                    r.additional_metadata[add_system_prompts_key] = self.system_prompts
                row = r.to_json(ensure_ascii=False, indent=4)
                jsonl_rows += row + "\n"
        if path:
            path.write_text(jsonl_rows)
        return jsonl_rows
