import pytest
import lmentry_static
from lmentry_static.tasks.words_in_sentence import WISTask
from lmentry_static.tasks.which_word_is_longer import WordLengthComparisonTask

from lmentry_static.tasks.data_structures import TaskDataset, TaskDatasetInstance
from lmentry_static.tasks.data_structures import Template, TaskTemplates

from lmentry_static.data.words import WORDS, SENTENCES

#  import logging
#  logger = logging.getLogger("test")
b = breakpoint

# TODO rewrite it all to be less dependent on resources / yaml templates


def test_wis_create_instances():
    w = WISTask()
    haystack = w.parse_text(SENTENCES[:5])
    r = w.gen_single_task(haystacks=haystack, abstand=15, lim=5)
    # abstand of 15 -> only first and last token
    # lim of 5 - max 5 sentences

    # remove uuids and untested metadata
    # TODO rewrite all this, it's ugly
    for st in r.instances:
        st.template_uuid = None
        st.task_instance_uuid = None
        md = {"template_n": st.additional_metadata["template_n"]}
        st.additional_metadata = md

    exp = TaskDataset(
        name="WISTask",
        instances=[
            TaskDatasetInstance(
                question='Яке перше слово y реченні "Ніч ' 'на середу буде морозною."?',
                correct_answer="Ніч",
                additional_metadata={"template_n": 0},
            ),
            TaskDatasetInstance(
                question='Яке слово в реченні "Ніч на ' 'середу буде морозною." перше?',
                correct_answer="Ніч",
                additional_metadata={"template_n": 1},
            ),
            TaskDatasetInstance(
                question='В реченні "Ніч на середу буде '
                'морозною." на першому місці '
                "знаходиться слово ...",
                correct_answer="Ніч",
                additional_metadata={"template_n": 2},
            ),
            TaskDatasetInstance(
                question='В реченні "Ніч на середу буде '
                'морозною." під номером один '
                "знаходиться слово ...",
                correct_answer="Ніч",
                additional_metadata={"template_n": 3},
            ),
            TaskDatasetInstance(
                question='Яке останнє слово y реченні "Ніч '
                'на середу буде морозною."?',
                correct_answer="морозною",
                additional_metadata={"template_n": 0},
            ),
        ],
    )
    for e in exp.instances:
        e.task_instance_uuid = None
    assert r == exp


def test_comp_create_instances():
    w = WordLengthComparisonTask()
    t1 = "зозуля"
    t2 = "Галя"
    t3 = "синхрофазотрон"

    task = w.gen_single_task(words=[t1, t2, t3], lim=3)

    # same as above - rewrite, it's ugly
    for st in task.instances:
        st.template_uuid = None
        st.task_instance_uuid = None
        md = {
            "reversed": st.additional_metadata["reversed"],
            "kind": st.additional_metadata["kind"],
        }
        st.additional_metadata = md

    exp = TaskDataset(
        name="WordLengthComparison",
        instances=[
            TaskDatasetInstance(
                question='Яке слово довше: "зозуля" чи ' '"Галя"?',
                correct_answer="зозуля",
                additional_metadata={
                    "kind": "more",
                    "reversed": False,
                },
            ),
            TaskDatasetInstance(
                question='Яке слово довше: "Галя" чи ' '"зозуля"?',
                correct_answer="зозуля",
                additional_metadata={
                    "kind": "more",
                    "reversed": True,
                },
            ),
            TaskDatasetInstance(
                question="Яке слово має більше літер: " '"зозуля" чи "Галя"?',
                correct_answer="зозуля",
                additional_metadata={
                    "kind": "more",
                    "reversed": False,
                },
            ),
        ],
    )
    for e in exp.instances:
        e.task_instance_uuid = None
    assert task == exp


def test_templates():
    t1 = Template(
        template="t1 template",
        additional_metadata={"reversed": True, "something": "else"},
    )
    t2 = Template(
        template="t2 template",
        additional_metadata={"reversed": False, "something": "else"},
    )
    templ = [t1, t2]

    tt = TaskTemplates(templates=templ)
    filtered = tt.get_templates(filter_dict=dict(reversed=True))
    assert filtered == [t1]
    assert tt.get_templates() == templ
    assert tt.get_templates(dict(something="not else")) == list()
    assert tt.get_templates(dict(something="else", reversed=True)) == [t1]
    assert tt.get_templates(dict(unknown_key="something")) == list()
