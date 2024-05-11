import pytest
from lmentry_static.tasks.letters_of_the_word import LOWTask
from lmentry_static.tasks.task_type_n_in_m import TaskTypeNinM

import logging

logger = logging.getLogger("test")

WORD = "перемога"


def test_low():
    low = LOWTask()
    assert low.nth_letter_of_the_word(WORD, 1) == "п"
    assert low.nth_letter_of_the_word(WORD, -1) == "а"
    assert low.nth_letter_of_the_word(WORD, 4) == "е"


def test_gen_task():
    l = LOWTask()
    res = list()
    for t in l.templates():
        r = l.gen_instance_template(template_str=t.template, haystack=WORD, needle=3)
        res.append(r)
    exp = [
        'Яка третя літера y слові "перемога"?',
        'Яка літера в слові "перемога" третя?',
        'В слові "перемога" на третьому місці знаходиться літера ...',
        'В слові "перемога" під номером три знаходиться літера ...',
    ]
    #  exp = [(i, e) for i, e in enumerate(exp)]
    assert res == exp


@pytest.mark.now
def test_gen_task_last():
    l = LOWTask()
    res = list()
    for t in l.templates():
        r = l.gen_instance_template(template_str=t.template, haystack=WORD, needle=-1)
        res.append(r)

    exp = [
        'Яка остання літера y слові "перемога"?',
        'Яка літера в слові "перемога" остання?',
        'В слові "перемога" на останньому місці знаходиться літера ...',
        None  # can't inflect
    ]
    assert res == exp

    res = list()
    for t in l.templates():
        r = l.gen_instance_template(template_str=t.template, haystack=WORD, needle=1)
        res.append(r)
    exp = [
        'Яка перша літера y слові "перемога"?',
        'Яка літера в слові "перемога" перша?',
        'В слові "перемога" на першому місці знаходиться літера ...',
        'В слові "перемога" під номером один знаходиться літера ...',
    ]
    assert res == exp

    res = list()
    for t in l.templates():
        r = l.gen_instance_template(template_str=t.template, haystack="всеодно", needle=5)
        res.append(r)
    exp = [
        'Яка п\'ята літера y слові "всеодно"?',
        'Яка літера в слові "всеодно" п\'ята?',
        'В слові "всеодно" на п\'ятому місці знаходиться літера ...',
        'В слові "всеодно" під номером п\'ять знаходиться літера ...',
    ]
    assert res == exp


def test_find_caps():
    w = [
        "Моє ПЕРШЕ слово",
        "Воно на ПЕРШОМУ місці",
        "Воно на ПЕРШОМУ місці, бо воно ПЕРШЕ",
        "Воно на ПЕРШОМУ місці, бо воно ПЕРШЕ\nУРА",
        "Воно на ПЕРШОМУ, місці, бо воно ПЕРШЕ\nУРА",
    ]

    r = [
        (["ПЕРШЕ"], "Моє {} слово"),
        (["ПЕРШОМУ"], "Воно на {} місці"),
        (["ПЕРШОМУ", "ПЕРШЕ"], "Воно на {} місці, бо воно {}"),
        (["ПЕРШОМУ", "ПЕРШЕ", "УРА"], "Воно на {} місці, бо воно {}\n{}"),
        (["ПЕРШОМУ", "ПЕРШЕ", "УРА"], "Воно на {}, місці, бо воно {}\n{}"),
    ]

    for i, s in enumerate(w):
        res = TaskTypeNinM.pattern_from_caps_words(s)
        assert res == r[i]
