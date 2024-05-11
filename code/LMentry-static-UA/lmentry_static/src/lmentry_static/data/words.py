from collections import defaultdict
from itertools import chain
from logging import getLogger
from pathlib import Path

import pandas as pd
import spacy
from spacy.lang.uk.examples import sentences as spacy_sentences

logger = getLogger("__name__")

from typing import Callable

from lmentry_static.consts import resources_path

MODEL_NAME = "uk_core_news_sm"

NUM_ROWS = 50
NUM_ROWS = 10
NUM_ROWS = 5
NUM_ROWS = 100

b = breakpoint

# CSV of this dataset: https://huggingface.co/datasets/shamotskyi/ukr_pravda_2y
UP_CSV_PATH = "/home/sh/uuni/master/data/up_crawls/full_crawls/dataset_up_since_2022_2023-12-13_nocommit.csv"

# WORDS_CSV_PATH = Path(__file__).parent / "sampled_words.csv"
# FEWSHOT_WORDS_CSV_PATH = Path(__file__).parent / "sampled_words_fewshot.csv"
WORDS_CSV_PATH = resources_path / "words" / "sampled_words.csv"
FEWSHOT_WORDS_CSV_PATH = resources_path / "words" / "sampled_words_fewshot.csv"


# Ipm tired
POEM_SENTS = """Переведіть мене через майдан,
Де я співав усіх пісень, що знаю.
Я в тишу увійду і там сконаю.
Переведіть мене через майдан,
Де жінка плаче, та, що був я з нею.
Мину її і навіть не пізнаю.
Переведіть мене через майдан
З жалями й незабутою любов'ю.
Там дужим був і там нікчемним був я.
Переведіть мене через майдан,
Де на тополях виснуть хмари п'яні.
Мій син тепер співає на майдані.
Переведіть мене через майдан.""".replace(
    "\n", " "
).split(
    "."
)
POEM_SENTS = [x.strip() for x in POEM_SENTS]


class UPCSVReader:
    """
    TODO serialize the current output of this and then read from smaller committable files
    """

    TITLE_KEY = "ukr_title"
    TEXT_KEY = "ukr_text"

    FILLER_TEXT = [
        "деталі:",
        "джерело:",
        "нагадаємо:",
        "передісторія:",
        "що передувало:",
    ]

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        nlp=None,
        enabled_filters: list[Callable] = None,
        num_rows=NUM_ROWS,
    ):
        # maybe we have a spacy pipeline setup somewhere
        self.nlp = spacy.load(model_name) if not nlp else nlp
        self.use_col = UPCSVReader.TEXT_KEY
        self.num_rows = num_rows

        self.enabled_filters = (
            enabled_filters
            if enabled_filters
            else [
                UPCSVReader.has_no_quotes,
                UPCSVReader.longer_than,
                UPCSVReader.has_no_up_filler,
                UPCSVReader.has_no_brackets,
            ]
        )

    def parse_up_crawl(
        self, path: str | Path = UP_CSV_PATH, num_rows=NUM_ROWS
    ) -> tuple[list[str], list[str]]:
        path = Path(path).absolute()
        assert path.exists()
        #  df = pd.read_csv(path, usecols=[self.TITLE_KEY, self.TEXT_KEY])
        df = pd.read_csv(path, usecols=[self.use_col])
        # it's either dropna after sample and we get a bit less, or before but it's ~1sec more
        # neither matters
        df_s = df.sample(num_rows).dropna()
        sp_df = df_s[self.use_col].apply(self.nlp)
        sp_dfl = sp_df.apply(lambda x: list(x.sents))
        sp_dfle = sp_dfl.explode()
        ok_sents = self.filter_sentences(
            sp_dfle,
            filters=self.enabled_filters,
        )
        ok_sents = [x.text for x in ok_sents]
        #  for s in ok_sents:
        #      print(s)

        # monster thing that return a series of OK words
        df_words = (
            sp_dfle.apply(lambda x: list(x))
            .apply(
                lambda x: [
                    str(y)
                    for y in x
                    if (
                        y.is_alpha
                        and not y.is_ascii
                        and not y.is_title
                        and not y.is_upper
                        and not y.morph.get("Foreign")
                    )
                ]
            )
            .explode()
            .dropna()
            .drop_duplicates()
            #  .sort_values()
            .sample(frac=1)
        )
        #  b()
        ok_words = list(df_words)
        return ok_sents, ok_words

    @staticmethod
    def filter_sentences(
        sentences: list[spacy.tokens.span.Span], filters: list[Callable]
    ) -> list[spacy.tokens.span.Span]:
        new_sents = list()
        for s in sentences:
            if all([f(s) for f in filters]):
                #  fr = [f(s) for f in filters]
                #  if all(fr):
                new_sents.append(s)
        return new_sents

    @staticmethod
    def has_no_brackets(s) -> bool:
        for token in s:
            if token.is_bracket:
                return False
        return True

    @staticmethod
    def has_no_quotes(s) -> bool:
        for token in s:
            if token.is_quote:
                return False
        return True

    @staticmethod
    def has_no_up_filler(s) -> bool:
        for ft in UPCSVReader.FILLER_TEXT:
            if ft.lower() in s.text.lower():
                return False
        return True

    @staticmethod
    def longer_than(s, min_length_tokens=5) -> bool:
        return len(s) >= min_length_tokens


class WordsAndSentencesProvider:
    # TODO read serialized w/s instead of doing UP/CSV r/w every time
    # TODO maybe in the future metadata
    #   e.g that tells you where the sentence came from
    #   or words by frequency
    #   ... and then that can be analyzed (are politics sentences harder than Shevchenko?)
    def __init__(self, num_sentences=200, is_fewshot: bool = False):
        self.words = None
        self.sents = None
        self.num_sents = num_sentences

        # dict of str -> metadata
        #  self.mddict = dict()
        self.mddict = defaultdict(dict)
        #  self.words_with_md = None
        #  self.sents_with_md = None
        #  self.build(path=WORDS_CSV_PATH if not is_fewshot else FEWSHOT_WORDS_CSV_PATH)

        # A much less idiotic way to do that would be to pass this through the config file
        #   (and get the few-shot words/sentences not from hard-coded things)
        #   yet another TODO for hypothetically later
        self.build(is_fewshot=is_fewshot)
        #  self.build_with_md()
        #  b()

    def from_simple(self):
        WORDS = [
            "кіт",
            "собака",
            "хвороба",
            "ліжко",
            "їжа",
            "синхрофазотрон",
        ]

        SENTENCES = spacy_sentences
        return SENTENCES, WORDS

    def from_up(self):
        pr_reader = UPCSVReader()
        up_sents, up_words = pr_reader.parse_up_crawl()
        return up_sents, up_words

    #  @staticmethod
    def get_words_from_csv(
        self, path: Path = WORDS_CSV_PATH, with_meta: bool = True, shuffle: bool = True
    ) -> list[str, dict] | list[str]:
        """
        Ugly, but:
        - sets self.words_metadata
        - returns either list of words or list of words with metadata
        """
        WORD_KEY = "word"
        MD_KEY = "md"
        df = pd.read_csv(path)
        if shuffle:
            df = df.sample(frac=1)
        md_cols = list(df.columns)
        md_cols.remove(WORD_KEY)
        df[MD_KEY] = df[md_cols].apply(lambda x: x.to_dict(), axis=1)
        df = df[[WORD_KEY, MD_KEY]]

        self.mddict = dict()

        res = list()
        for i, r in df.iterrows():
            self.mddict[r[WORD_KEY]] = r[MD_KEY]
            if with_meta:
                res.append((r[WORD_KEY], r[MD_KEY]))
            else:
                res.append(r[WORD_KEY])
        return res

    #  def build_with_md(self):
    #      self.words_with_md = self.get_words_from_csv()
    #      self.sents_with_md = [(x, dict()) for x in self.sents]

    def build(self, is_fewshot: bool = False, **kwargs):
        up_sents, up_words = self.from_up()
        sp_sents, sp_words = self.from_simple()

        if is_fewshot:
            # I'm sorry
            logger.info(
                f"FEW-SHOT SETTING: Using random words and sentences from a poem..."
            )
            csv_words = self.get_words_from_csv(
                with_meta=False, path=FEWSHOT_WORDS_CSV_PATH
            )
            self.words = csv_words
            self.sents = POEM_SENTS
        else:
            csv_words = self.get_words_from_csv(with_meta=False, **kwargs)
            self.words = csv_words
            #  self.sents = sp_sents
            self.sents = sp_sents + up_sents[: self.num_sents - len(sp_sents)]
            logger.debug(f"Sentences len: {len(self.sents)}")
        logger.info(f"Sentences len: {len(self.sents)}")

        logger.info(
            f"In total, we have {len(self.words)} words and {len(self.sents)} sentences!"
        )
        #  b()

    def get_words(self, noiter=False, lim=None):
        return self.words[:lim] if noiter else iter(self.words)

    def get_sents(self, noiter=False, lim=None):
        return self.sents[:lim] if noiter else iter(self.sents)


# Ugly global variable
WSP = WordsAndSentencesProvider(is_fewshot=False)

# TODO - words of three categories of frequency, and three categories of length.
# frequent-short, frequent-middle, frequent-long; average-short, ...


def build_vocabulary():
    """Runs magic to provide WORDS and SENTENCES to whichever program neeeds them."""

    print("Running getting vocabulary...")
    wsp = WordsAndSentencesProvider()
    WORDS = wsp.get_words(noiter=True)
    SENTENCES = wsp.get_sents(noiter=True)
    #  b()
    return SENTENCES, WORDS


# TODO ugly, do this for real
#  SENTENCES, WORDS = build_vocabulary()


def run():
    build_vocabulary()


if __name__ == "__main__":
    run()
