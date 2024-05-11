import pytest
import lmentry_static
from lmentry_static.tasks.words_in_sentence import (
    WISTask,
)

from spacy.lang.uk.examples import sentences
import logging

logger = logging.getLogger("test")
b = breakpoint

SENT = "Чим кращі книги ти читав, тим гірше спиш."


def test_wis():
    w = WISTask()
    sent = w.parse_text(SENT)

    print(f"Sent: {sent}")
    res = w.nth_word_in_sentence(n=2, sent=sent)
    assert res == "кращі"


@pytest.mark.now
def test_gen_task():
    w = WISTask()
    res = list()
    for t in w.templates():
        r = w.gen_instance_template(template_str=t.template, haystack=SENT, needle=3)
        res.append(r)
    exp = [
        'Яке третє слово y реченні "Чим кращі книги ти читав, тим гірше спиш."?',
        'Яке слово в реченні "Чим кращі книги ти читав, тим гірше спиш." третє?',
        'В реченні "Чим кращі книги ти читав, тим гірше спиш." на третьому місці знаходиться слово ...',
        'В реченні "Чим кращі книги ти читав, тим гірше спиш." під номером три знаходиться слово ...',
    ]
    assert res == exp

    res = list()
    for t in w.templates():
        r = w.gen_instance_template(template_str=t.template, haystack=SENT, needle=1)
        res.append(r)

    exp = [
        'Яке перше слово y реченні "Чим кращі книги ти читав, тим гірше спиш."?',
        'Яке слово в реченні "Чим кращі книги ти читав, тим гірше спиш." перше?',
        'В реченні "Чим кращі книги ти читав, тим гірше спиш." на першому місці знаходиться слово ...',
        'В реченні "Чим кращі книги ти читав, тим гірше спиш." під номером один знаходиться слово ...',
    ]
    assert res == exp

    res = list()
    for t in w.templates():
        r = w.gen_instance_template(template_str=t.template, haystack=SENT, needle=-1)
        res.append(r)
    exp = [
        'Яке останнє слово y реченні "Чим кращі книги ти читав, тим гірше спиш."?',
        'Яке слово в реченні "Чим кращі книги ти читав, тим гірше спиш." останнє?',
        'В реченні "Чим кращі книги ти читав, тим гірше спиш." на останньому місці знаходиться слово ...',
        None,
    ]
    assert res == exp
