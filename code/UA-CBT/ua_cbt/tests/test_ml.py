import pytest
import spacy
from ua_cbt.ml import CBTTaskMaker
from ua_cbt.consts import SMALL_STORY

b = breakpoint


MODEL_NAME_DE = "de_core_news_sm"
MODEL_NAME_UA = "uk_core_news_sm"


@pytest.fixture
def nlpde():
    nlp = spacy.load(MODEL_NAME_DE)
    return nlp

@pytest.fixture
def nlp():
    nlp = spacy.load(MODEL_NAME_UA)
    return nlp

@pytest.fixture
def doc_de(nlpde):
    txt = "This is my spacy doc. It has three sentences. This is the third. Fourth, I lied. Fifth. Sixth and a half. Seventh."
    doc = nlpde(txt)
    return doc

@pytest.fixture
def doc(nlp):
    txt = SMALL_STORY
    doc = nlp(txt)
    b()
    return doc

def test_get_context_questions_spans(doc_de):
    c = CBTTaskMaker
    for nc, nq in [
        (3, 1),
        (1, 1),
        (1, 3),
    ]:
        cont, quest = c.get_context_question_spans(doc_de, nc, nq)
        exp_cont = ".".join(str(doc_de).split(".")[0:nc])
        exp_cont += "."

        exp_q = ".".join(str(doc_de).split(".")[nc : nc + nq]).strip()
        exp_q += "."

        assert str(cont) == exp_cont
        assert str(quest) == exp_q

        assert c.get_context_question_spans(doc_de, 4, 5) is None
        # TODO off-by-one errors test


@pytest.mark.now
def test_create_cbt_task(doc):
    c = CBTTaskMaker(n_context_sents=3, n_question_sents=2)
    res = c.create_cbt_task(doc)
    assert False
    #  b()
