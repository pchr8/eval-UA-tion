import argparse
import logging
import pdb
import sys
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pyinflect
import pymorphy2
import spacy
from dataclass_wizard import JSONWizard
from flatten_dict import flatten
from omegaconf import OmegaConf
from pymorphy_spacy_disambiguation.disamb import \
    Disambiguator  # , SimilarityWeighting
from rich import inspect, print
from rich.console import Console
from rich.panel import Panel
from spacy.lang.uk.examples import sentences
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span, Token

#  from ua_cbt.consts import PAT_NAMED_ENTITY, PAT_COMMON_NOUN, PAT_PRON_SG
from ua_cbt.consts import (ANOTHER_STORY, MODEL_NAME, N_CONTEXT_SENTENCES,
                           N_QUESTION_SENTENCES, REPLACEMENT_TOKEN_SYMBOL,
                           SMALL_STORY, YAML_DATA)
from ua_cbt.data_structures import (GeneratedTasks, ReplacementOptionTypes,
                                    SingleTask, SpacyMatcherPattern)
from ua_cbt.epub import read_epub
from ua_cbt.utils import _pretty_output_morph
from ua_cbt.writing import to_csv

#  from rich.pretty import pprint

logging.basicConfig()
logger = logging.getLogger(__name__)

b = breakpoint

NOUN = "NOUN"
PROPN = "PROPN"
VERB = "VERB"
ADJF = "ADJF"

# Get (former) consts from YAML words_and_data.yaml
# Ugly but not too ugly, since I consider them actually part of the code, not config
LEMMA_FIXES = YAML_DATA.lemma_fixes
WORD_REPLACEMENTS = YAML_DATA.word_replacements
WORD_BLACKLIST = YAML_DATA.word_blacklist
DISTRACTORS = YAML_DATA.distractors

# Consts for pattern names
P_COMMON_NOUN = "COMMON_NOUN"
P_NAMED_ENTITY = "NAMED_ENTITY"
P_VERB = "VERB"
#  P_VERB_CONV = "P_VERB_CONV"

# Patterns enabled in task instances generation
ENABLED_PATTERNS = [
    P_COMMON_NOUN,
    P_NAMED_ENTITY,
    #  P_VERB,
]


def replace_some_words(text: str, fixes: dict = WORD_REPLACEMENTS) -> str:
    """Hardcore word replacement. Does three usual word shapes (Xxx,XXX,xxx)
    but doesn't care about word margins etc. Don't replace
    'problem' with 'issue' if you want want to fail
    at 'PROBLEMatic'.
    """
    new_fixes = dict()
    for f in fixes:
        # for fmod in [f,f.capitalize(),f.upper()]:
        new_fixes[f] = fixes[f]
        new_fixes[f.capitalize()] = fixes[f].capitalize()
        new_fixes[f.upper()] = fixes[f].upper()

    new_text = text
    for f in new_fixes:
        new_text = new_text.replace(f, new_fixes[f])
    if new_text != text:
        logger.debug(f"Replaced some words in text: {text}->{new_text}")
    return new_text


def get_lemma(token: Token, fixes: dict[str, str] = LEMMA_FIXES) -> str:
    """Fancier spacy lemma_ with some manual fixes for known-wrong lemmas."""
    # TODO - https://spacy.io/api/attributeruler exists and is the way to go, not this!
    lemma = token.lemma_
    lemma_fixed = fixes.get(lemma.lower(), lemma)
    if lemma_fixed != lemma:
        logger.debug(f"Fixed {token} lemma {lemma}->{lemma_fixed}")
    return lemma_fixed


@dataclass
class MatchPatterns:
    """Contains all the matcher patterns used in the task."""

    # (pattern_name, pattern)
    patterns: list[tuple[str, dict[str, str | dict]]]

    def _to_spacypatterns(self, subset: list[str] = None) -> list[SpacyMatcherPattern]:
        """Convert to list of SpacyMatcherPattern"""
        all_ps = self.patterns
        if subset:
            all_ps = [x for x in all_ps if x in subset]
        patterns = [SpacyMatcherPattern(name=x[0], pattern=x[1]) for x in all_ps]
        return patterns

    def pnames(self) -> list[str]:
        """Return names of patterns present"""
        return [x[0] for x in self.patterns]

    def add_to_matcher(self, matcher) -> None:
        """Add patterns to spacy matcher"""
        for p in self._to_spacypatterns():
            matcher.add(p.name, [[p.pattern]])


@dataclass
class MatchesHelper:
    """Class containing the helper functions dealing with spacy matchES in the form of Spans."""

    matches: list[str]
    nlp: spacy.pipeline

    def get(self, matches: Optional[list[Span]] = None) -> list[Span]:
        if matches is None:
            matches = self.matches
        return matches

    def test(self, matches=None):
        matches = self.get(matches)

        not_ok = list()
        for m in matches:
            if len(m) != 1:
                not_ok.append(m)
        if not_ok:
            logger.error(f"Matches >1 token not supported: {not_ok}")
            return not_ok

    def get_lemmas_dict(self, matches: Optional[list[str]] = None) -> dict[str, Span]:
        matches = self.get(matches)

        lemmas = dict()
        for m in matches:
            # lemmas[m[0].lemma_] = m
            lemmas[get_lemma(m[0])] = m
        return lemmas

    def as_type_dict(
        self, matches=None, as_tokens: bool = False
    ) -> dict[str, list[str]]:
        """Return matches as dict match_type->list"""
        matches = self.get(matches)

        ret = dict()
        for t in self.get_matches_types():
            mot = self.get_type(match_type=t, matches=matches)
            if as_tokens:
                mot = [m[0] for m in mot]
            ret[t] = mot
        return ret

    def get_matches_types(self, matches=None) -> list[str]:
        types = set()
        matches = self.get(matches)
        for m in matches:
            types.add(self.get_match_type(m))
        return types

    def get_type(
        self, match_type: str, matches=None, as_tokens: bool = False
    ) -> list[Span]:
        matches = self.get(matches)
        matches_of_type = list()
        for m in matches:
            if self.get_match_type(m) == match_type:
                matches_of_type.append(m if not as_tokens else m)
        return matches_of_type

    def get_match_type(self, match: Span) -> str:
        match_name = self.nlp.vocab.strings[match.label]
        return match_name

    def _clean_up_inanimate_matches(
        self,
        anim_match_name: str,
        inan_match_name: str,
        additional_anim_lemmas: Optional[list[str]] = None,
    ) -> None:
        """
        Fix wrong anim/inanim detection of matches.

        We assume anim_match_name matches are correctly labeled as animate,
        and if any inan_match_name matches have the same lemma as the anim_match_name
        ones their label gets changed to be anim_match_name.

        Goal: many animate tokens get detected as inan but not the other way around,
            here we use lemmas to find inanimate false positives if their lemma
            matches to animate lemmas.
        """
        anim_matches = self.get_type(anim_match_name)

        if not anim_matches:
            logger.info(
                f"Couldn't clean anim/inan matches because no anim matches presen."
            )
            return

        anim_lemmas = [get_lemma(m[0]) for m in anim_matches]

        if additional_anim_lemmas:
            anim_lemmas.extend(additional_anim_lemmas)

        anim_label = anim_matches[0].label

        num_rewritten: int = 0
        for m in self.get_type(inan_match_name):
            if get_lemma(m[0]) in anim_lemmas:
                num_rewritten+=1
                m.label = anim_label
                #  logger.error(f"Rewrote animacy {m[0]}")

        #  logger.error(f"Rewrote animacy of {num_rewritten} tokens ({len(anim_lemmas)} anim lemmas)")
        #  b()


class CBTTaskMaker:
    """Creates a list of CBT tasks from a str."""

    # Names of matcher patterns for animate and inanimate patterns for later filtering.
    # TODO don't make them lists if not needed
    ANIM_PATTERNS_NAMES = [P_NAMED_ENTITY]
    INAN_PATTERNS_NAMES = [P_COMMON_NOUN]

    MATCHER_PATTERNS_ALL = [
        (
            P_NAMED_ENTITY,
            {
                # See also: ent_type=="PER" and `NameType=Sur`, pos=='PROPN'
                "POS": {"IN": ["NOUN", "PROPN"]},
                "MORPH": {
                    "IS_SUPERSET": [
                        #  "Number=Sing",
                        #  "Case=Nom",
                        "Animacy=Anim",
                    ]
                },
            },
        ),
        (
            P_COMMON_NOUN,
            {
                #  "TAG": "NOUN",
                "POS": "NOUN",
                "MORPH": {
                    "IS_SUPERSET": [
                        #  "Number=Sing",
                        #  "Case=Nom",
                        "Animacy=Inan",
                    ]
                },
            },
        ),
        (
            P_VERB,
            {
                #  "TAG": "NOUN",
                "POS": "VERB",
                "MORPH": {
                    "REGEX": r"VerbForm=(Inf|Fin)",
                },
                #  "MORPH": {
                #      "IS_SUPERSET": [
                #          "VerbForm=Inf",
                #      ],
                #  },
            },
        ),
    ]

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        n_context_sents: int = N_CONTEXT_SENTENCES,
        n_question_sents: int = N_QUESTION_SENTENCES,
        replacement_token_symbol: str = REPLACEMENT_TOKEN_SYMBOL,
        #  num_occs_in_prev_text_for_gap: int = 1,  # number of times a lemma has to occur to be considered for ___
        num_occs_in_prev_text_for_gap: dict = dict(
            default=2, P_COMMON_NOUN=4
        ),
        question_sents_share: float = -1,
        enabled_patterns: list[str] = ENABLED_PATTERNS,
        distractors: Optional[dict[str, dict[str, dict[str, str]]]] = DISTRACTORS,
        use_pymorphy3_analyzer_experimental: Optional = False,
    ):
        """

        Args:
            model_name (str): spacy model_name
            n_context_sents (int): number of context sentences
            n_question_sents (int): number of sentences in question span
                if both > total sentences, text is not used
                if both < total sentences, the rest of the story is thrown away
                if one is -1, it's inferred from the other
                if both are -1, both get ignored and question_sents_share is used
            question_sents_share (float): % of sentences that should become
                the question span (context span then is inferred).
                If set, overrides teh n_.._sents options above
            replacement_token_symbol (str): what is used inside gaps, e.g. ____
            num_occs_in_prev_text_for_gap (int): for a token to be considered
                for becoming a gap, it has to occur in the text before it
                at least this many times. (=don't gap a token we see for the first
                time.)
            enabled_patterns: which spacy matcher patterns are enabled (=become gaps),
                e.g. ['VERB'] means instances will be created only from verbs
        """
        self.model_name = model_name

        # Both can be -1 - then we look at .._share.
        # If one is -1 it means "whatever is left after the other"
        self.n_context_sents = n_context_sents
        self.n_question_sents = n_question_sents

        # If provided will override the n_..
        self.question_sents_share = question_sents_share

        self.replacement_token_symbol = replacement_token_symbol

        self.num_occs_in_prev_text_for_gap = num_occs_in_prev_text_for_gap

        self.nlp = spacy.load(model_name)

        self.mpatterns = MatchPatterns(
            patterns=[x for x in self.MATCHER_PATTERNS_ALL if x[0] in enabled_patterns]
        )

        logger.info(
            f"Enabled patterns {enabled_patterns} out of the {len(self.MATCHER_PATTERNS_ALL)} defined."
        )

        self.matcher: Matcher = None
        self._create_matcher()

        self.use_pymorphy3_analyzer_experimental = use_pymorphy3_analyzer_experimental

        if use_pymorphy3_analyzer_experimental:
            # more support for more Ukrainian words
            logger.info(f"Will be using experimental pymorphy3 analyzer")
            import pymorphy3

            pya = pymorphy3.MorphAnalyzer(lang="uk")
            self.disamb = Disambiguator(pymorphy_analyzer=pya)
        else:
            self.disamb = Disambiguator()

        self.distractors = OmegaConf.to_container(distractors)
        if distractors:
            logger.info(f"Enabled following distractors: {distractors}")

        #  self.metadata = self.dump_settings()
        self.metadata = dict()

        logger.info(
            f"Using settings {self.model_name=} {self.n_context_sents=} {self.n_question_sents=} {self.question_sents_share=},  {self.num_occs_in_prev_text_for_gap=}"
        )
        print(
            #  f"Using settings {self.model_name=} {self.n_context_sents=} {self.n_question_sents=}"
            f"Using settings {self.model_name=} {self.n_context_sents=} {self.n_question_sents=} {self.question_sents_share=},  {self.num_occs_in_prev_text_for_gap=}"
        )

    def _dump_settings(self) -> dict:
        md_dict = self.__dict__
        del md_dict["mpatterns"]
        #  del md_dict['metadata']
        del md_dict["disamb"]
        del md_dict["matcher"]
        del md_dict["nlp"]
        return md_dict

    def _create_matcher(self) -> None:
        """Creates spacy Matcher with patterns"""
        matcher = Matcher(self.nlp.vocab)
        self.mpatterns.add_to_matcher(matcher)
        self.matcher = matcher
        logger.debug(f"{self.matcher=}")

    def process_txt(self, txt: str) -> Optional[list[SingleTask]]:
        """txt -> CBT tasks"""
        txt_rep = replace_some_words(txt)
        doc = self.nlp(txt_rep)
        tasks, doc_md = self.create_cbt_task(doc=doc)
        return tasks, doc_md

    # def _quick_show_question(self,tasks):
    #     for t in tasks:
    #         logger.info(f"{cand.sent}\n\t{cand} -> {replacement_options_agreed}")

    @staticmethod
    def _calculate_n_sents(
        n_sents: int,
        n_context_sents: int,
        n_question_sents: int,
        question_sents_share: float,
    ) -> tuple[int, int]:
        """
        Calculate the number of context/question sents based on total.

        - question_sents_share if not -1 will be the % of text that will become
            question sents.
        - The other two if passed explicitly will be used as-is
            - If one is -1 then it'll become whatever is left after the other
        """
        if question_sents_share != -1:
            assert n_sents > 1

            assert 0 < question_sents_share <= 1
            # We take the % of the questions, except one sentence.
            n_question_sents = round(n_sents * question_sents_share)
            if n_question_sents == n_sents:
                n_question_sents -= 1
            elif n_question_sents == 0:
                n_question_sents += 1

            n_context_sents = n_sents - n_question_sents
            logger.info(
                f"{question_sents_share=} => {n_context_sents}/{n_question_sents} context/question sentences."
            )
        else:
            if n_context_sents == -1 and n_question_sents == -1:
                raise ValueError(f"Can't have both equal to -1")

            if n_context_sents == -1:
                n_context_sents = n_sents - n_question_sents
            elif n_question_sents == -1:
                n_question_sents = n_sents - n_context_sents
                logger.info(
                    f"For story {n_context_sents=}/{n_question_sents=} led to {n_context_sents}/{n_question_sents} c/q sentences."
                )
            else:
                # We have an explicit N of c/q sents
                logger.info(
                    f"{n_context_sents=}/{n_question_sents=} provided explicitly"
                )

                if n_context_sents + n_question_sents > n_sents:
                    # Document is too small for this
                    return None

        return n_context_sents, n_question_sents

    @staticmethod
    def get_context_question_spans(
        doc: Doc,
        n_context_sents: int = 5,
        n_question_sents: int = 5,
        question_sents_share: float = -1,
    ) -> Optional[tuple[Span, Span]]:
        """From a Document get two Spans:
        - a context span (long chunk of unchanged sentences)
        - a question span (much shorter span, and the one where
            the cloze/masking/replacements will be happening)

        All three params can be -1, for the first two -1 means
        "whatever is left from the other".

        If question_sents_share is not -1 it takes precedence and it's the % of
        the story that will be questions.


        Args:
            doc (Doc): spacy document
            n_context_sents (int): n_sents
            n_question_sents (int): q_sent_window_size
            question_sents_share (float): how many % of the story are question sentences

        Returns:
            tuple[Span, Span]:
        """
        sents = list(doc.sents)

        c_q_sents_len = CBTTaskMaker._calculate_n_sents(
            n_sents=len(sents),
            n_context_sents=n_context_sents,
            n_question_sents=n_question_sents,
            question_sents_share=question_sents_share,
        )

        #  if len(sents) < n_context_sents + n_question_sents + 1:
        #  if len(sents) < n_context_sents + n_question_sents:
        if c_q_sents_len is None:
            logger.warning(
                f"Doc too small and has {len(sents)}<{n_context_sents}"
                "+{q_sent_window_size} sentences."
            )
            return None
        #  b()

        n_context_sents, n_question_sents = c_q_sents_len

        # Context sents are provided unchanged, question sents are the ones where
        #   words will be masked
        context_sents = sents[:n_context_sents]
        question_sents = sents[n_context_sents : n_context_sents + n_question_sents]

        # both as token start/end token indexes
        context_sents_ids_span = (context_sents[0].start, context_sents[-1].end)
        question_sents_ids_span = (
            sents[n_context_sents].start,
            sents[n_context_sents + n_question_sents - 1].end,
        )
        # both as spacy spans
        context_sents_span = doc[context_sents_ids_span[0] : context_sents_ids_span[1]]
        question_sents_span = doc[
            question_sents_ids_span[0] : question_sents_ids_span[1]
        ]
        #  b()
        return context_sents_span, question_sents_span

    def create_cbt_task(
        self, doc: Doc, take_word_options_from_entire_doc: bool = True
    ) -> Optional[list[SingleTask]]:
        """create_cbt_task.
        Args:
            doc (Doc): doc
            take_word_options_from_entire_doc (bool): if True, multi-choice options
                will be taken from the entire document, otherwise only from context
                span.
                Practically:
                    - if many options are needed, the span may be too small.
                    - but if all doc is used, the task may contain never seen before words
                        which may or may not be a clue.

        Returns:
            Optional[list[SingleTask]]:
        """
        doc_md = dict()

        # Generate two spans that follow other in the text. The one
        #   given as context, and the one where masking/questions will happen
        doc_matches_h = self._get_matches(doc)

        anim_ms = doc_matches_h.get_type(self.ANIM_PATTERNS_NAMES[0])
        anim_lemmas = list(doc_matches_h.get_lemmas_dict(matches=anim_ms).keys())

        #  b()
        c_q_spans = self.get_context_question_spans(
            doc=doc,
            n_context_sents=self.n_context_sents,
            n_question_sents=self.n_question_sents,
            question_sents_share=self.question_sents_share,
        )

        # if no valid spans can be created, skip the text
        if c_q_spans is None:
            return None

        context_span, question_span = c_q_spans

        #  animate_matches_lemmas = self._get_animate_lemmas(doc)

        # Count frequency of matches over doc/context to find 'important' words
        #   that we'll use for our multiple choice option
        important_entities, lemmas = self.get_frequent_matches_in_span(
            doc if take_word_options_from_entire_doc else context_span,
        )

        # Skip adjectival stories if we detect them
        is_story_potential_adjectival = self.adjectival_names_story_filter(
            lemmas, important_entities
        )
        if is_story_potential_adjectival:
            return None

        #  b()
        # Find tokens replaceable by "____" inside the question span
        possible_gaps = self.get_possible_gaps(
            span=question_span,
            previous_text=context_span,
            anim_lemmas=anim_lemmas,
            min_num_occs_in_prev_text=self.num_occs_in_prev_text_for_gap,
        )

        if not possible_gaps:
            logger.error(f"Found no gaps-eable tokens")
            return None

        tasks = self.create_individual_tasks(
            possible_gaps=possible_gaps,
            important_entities=important_entities,
            lemmas=lemmas,
            context_span=context_span,
            doc_md=doc_md,
        )

        tasks_dedup = self.deduplicate_tasks(tasks)
        return tasks_dedup, doc_md

    def deduplicate_tasks(self, tasks, use_options: bool = False) -> list:
        """Since it's conceivable the same tokens belong to multiple classes,
        we might end up with duplicated tasks.
        """
        # TODO does this function belong one level up, over docs?
        #  key_fn = lambda x: (x.context, x.question, tuple(sorted(x.options)))
        key_fn = lambda x: (
            x.context,
            x.question,
            tuple(sorted(x.options)) if use_options else True,
        )
        dict_with_tasks = dict()
        for t in tasks:
            key = key_fn(t)
            if key in dict_with_tasks:
                logger.error(f"Found duplicated task! {t}")
                # TODO remove breakpoint
                #  b()
            dict_with_tasks[key] = t
        res = list(dict_with_tasks.values())
        return res

    def create_individual_tasks(
        self,
        possible_gaps: list[Span],
        important_entities: dict[str, dict[str, int]],
        lemmas: dict[str, Token],
        context_span: Span,
        doc_md: dict,
        max_n_of_options: int = 6,
        min_n_of_options: int = 4,
        allowed_pymorphy_pos: list[str] = [NOUN, PROPN, VERB],
        add_distractors_to_patterns: list[str] = [P_NAMED_ENTITY, P_COMMON_NOUN],
        #  replace_pron_with_nouns: bool = True,
    ):
        """create_individual_tasks.

        Args:
            possible_gaps (list[Span]): spans with tokens that are candidates
                to become "____"
            important_entities (dict[str, dict[str, int]]): important_entities
                that we can use as multiple choice answers
            lemmas (dict[str, Token]): lemmas
            context_span (Span): context_span
            n_most_common_options (int): n_most_common_options
        """
        individual_tasks = list()
        sequential_number: int = 0

        num_failures_total: int = 0
        num_failures_instance: int = 0

        #  b()
        # For all types of replacements
        for match_name in possible_gaps:
            #  b()
            # For all individual words that can be replaced
            for cand in possible_gaps[match_name]:
                ## Create a masked question span
                sent = cand.sent
                # Index within the question span!
                sent_index = list(cand.doc.sents).index(sent)  # +self.n_context_sents
                # the 'document' for replacement candidates is the question span, not the original document
                index_of_cand_in_q_span = cand.i
                new_question_span = (
                    cand.doc[0 : cand.i].text
                    + f" {self.replacement_token_symbol} "
                    + cand.doc[cand.i + 1 :].text
                )

                ## Find cool replacement options for our original word
                # Use the pymorphy-spacy disamb. package to get the best disambiguation option
                # b()
                cand_m = self.disamb.get_with_disambiguation(cand)

                ro_and_md = self.smart_find_replacement_options(
                    match_name=match_name,
                    important_entities=important_entities,
                    num_replacements=max_n_of_options,
                    lemmas=lemmas,
                    token_to_replace=cand,
                    #  add_distractors=bool(self.distractors),
                    add_distractors=bool(self.distractors)
                    and match_name in add_distractors_to_patterns,
                )
                if ro_and_md is None:
                    num_failures_instance += 1
                    continue
                replacement_options_raw, md = ro_and_md

                replacement_options_agreed_dict = self.agree_replacement_options_with_token(
                    token_to_replace=cand,
                    replacement_options_dict=replacement_options_raw,
                    lemmas=lemmas,
                    allowed_pymorphy_pos=allowed_pymorphy_pos,
                    fail_gracefully=False,
                    #  tokens_idx_to_leave_unchanged = md
                )
                if not replacement_options_agreed_dict:
                    num_failures_instance += 1
                    continue

                replacement_options_agreed, roa_md = self.build_final_options_list(
                    replacement_options_agreed_dict,
                    target_options=max_n_of_options,
                )
                # SH
                logger.debug(
                    f"{cand.sent.text.strip()}\n\t{cand} -> {replacement_options_agreed}"
                )
                if not replacement_options_agreed:
                    logger.error(
                        f"No replacement options generated for '{cand}' among {replacement_options_raw}"
                    )
                    num_failures_instance += 1
                    continue

                replacement_options_final = replacement_options_agreed

                if len(replacement_options_final) != len(
                    set(replacement_options_final)
                ):
                    #  b()
                    #  logger.error(
                    raise ValueError(
                        f"Duplicate repl options, should't happen! {replacement_options_final}"
                    )

                if len(replacement_options_final) < min_n_of_options:
                    logger.info(
                        f"Skipping {match_name} w/ {len(replacement_options_final)}<{min_n_of_options} replacement options. {replacement_options_final=} "
                    )

                additional_md = dict(
                    #  repl_type=match_name,
                    num_context_sents=len(list(context_span.sents)),
                    #  question_sents_n=len(list(new_question_span.sents)),
                    num_context_sents_tokens=len(context_span),
                    num_question_sents_tokens=len(new_question_span),
                    n_question_sents_share=self.question_sents_share,
                    #  num_options_total=len(replacement_options),
                    #  index_label=replacement_options_final.index(cand.text),
                    sequential_number=sequential_number,
                    #  _debug_num_failures=num_failures_instance,
                    **md,
                    **roa_md,
                )
                task = SingleTask(
                    context=context_span.text.strip(),
                    question=new_question_span.strip(),
                    options=replacement_options_final,
                    answer=cand.text,
                    task_type=match_name,
                    md=additional_md,
                )
                individual_tasks.append(task)
                sequential_number += 1

                num_failures_total += num_failures_instance
                num_failures_instance = 0
                #  b()
        #  b()
        if num_failures_total > 0:
            logger.info(f"Num failures in story: {num_failures_total}")
        doc_md["num_failures"] = num_failures_total
        return individual_tasks

    # TODO
    #  def make_words_agree_eo(w1:)

    def _str_to_spacy_token(self, string: str) -> Token:
        return self.nlp(string)[0]

    def get_distractors(
        self,
        species=None,
        gender=None,
        for_match_type: str = P_NAMED_ENTITY,
        to_spacy_token: bool = False,
        num_distractors: int = None,
    ) -> list[Token | str]:
        """Get distractors of correct (grammatical) gender and species if required.

        genders: male, female
        species: human, animal

        None means no limit by this criterium,  `all genders/species`
        """

        if for_match_type not in self.distractors:
            logger.error(
                f"{for_match_type} not in distractors keys: {self.distractors.keys()}"
            )
            return list()
        #  if for_match_type!=P_NAMED_ENTITY:
        #      b()

        distractors = self.distractors[for_match_type]

        # spacy morph returns ['Fem']
        if isinstance(gender, list):
            gender = gender[0]

        # TODO neutral also exists! But for our purposes we can assume male only
        if gender:
            if "f" in gender.lower():
                gender = "female"
            elif "n" in gender.lower():
                gender = "neutral"
            else:
                gender = "male"

        if gender and gender not in ["male", "female", "neutral"]:
            raise ValueError
        if species and species not in ["animal", "human"]:
            raise ValueError

        flat = flatten(distractors)

        l_genders = [gender] if gender else ["male", "female"]
        l_species = [species] if species else ["human", "animal"]

        filtered = list()
        if len(list(flat.keys())[0]) == 2:
            # animate
            for s in l_species:
                for g in l_genders:
                    filtered.extend(flat[(s, g)])
        else:
            # inan
            for g in l_genders:
                filtered.extend(flat[(g,)])

        # deduplicate just in case
        # and sort as a neat side effect
        res = list(set(filtered))
        if to_spacy_token:
            res = [self._str_to_spacy_token(x) for x in res]

        return res[:num_distractors]

    @staticmethod
    def _add_if_not_present(needle, haystack: list):
        if needle not in haystack:
            haystack.append(needle)
        return haystack

    #  def smart_filter_tokens_by_gender(tokens: list[str], gender: list[str]):
    #      """ More complex version of the former function to filter tokens
    #      by gender.
    #      Formerly, if lemmas[r].morph.get("Gender") == cand_gender.
    #
    #      BUT some of the nouns involved
    #      """
    #  @staticmethod
    def smart_find_replacement_options(
        self,
        match_name: str,
        important_entities: Counter[str, int],
        lemmas: dict[str, Token],
        token_to_replace: Token,
        num_replacements: Optional[int] = None,
        min_real_options: int = 3,
        same_gender: bool = True,
        add_distractors: bool = True,
        add_most_frequent_all_genders_distractor=True,
    ) -> Optional[tuple[list[str], dict]]:
        """Limits the replacement options to the 'relevant' ones."""
        md = dict()

        ro_bytype = defaultdict(list)

        # Even if we provide no n_most_common, we expect them to be ordererd
        #   by frequency.

        # ACTUALLY since we'll filter by gender then we might as well get all of them here
        replacement_options_raw = [
            x[0]
            for x in important_entities[match_name].most_common(
                #  num_replacements  # - 2 if add_distractors else num_replacements
            )
        ]

        cand_gender = token_to_replace.morph.get("Gender")

        # If we need gender and our token has gender, do gender
        # But if it's a verb don't filter by it, because we'll inflect it later
        if cand_gender and same_gender and token_to_replace.pos_ != VERB:
            use_gender = True
        else:
            use_gender = False

        # Filter these replacement options to pick only ones of the correct gender
        clean_ro = list()

        if use_gender:
            for rep in replacement_options_raw:
                if lemmas[rep].morph.get("Gender") == cand_gender:
                    clean_ro.append(rep)
        else:
            #  dists_to_use = distractors.animals.female + distractors.animals.male
            clean_ro = replacement_options_raw

        # And now AFTER filtering by gender we cut it to the needed size
        clean_ro = clean_ro[:num_replacements]

        # We add the most frequent fitting word to be a baseline
        #  baseline_most_frequent_option = clean_ro[0]
        #  md["opts_bl_most_frequent"] = baseline_most_frequent_option
        #  b()
        #  md["opts_num_from_text"] = len(clean_ro)
        if len(clean_ro) < min_real_options:
            logger.warning(
                f"Skipping story/instance w/ {len(clean_ro)}<{min_real_options} real options: {clean_ro=}"
            )
            return None
            #  b()
        if len(clean_ro) < 4:
            logger.debug(f"Instance has only {len(clean_ro)} real options: {clean_ro=}")

        distractors = None
        mf_ag_distractor = None
        if add_distractors:
            distractors = self.get_distractors(
                gender=cand_gender if use_gender else None,
                for_match_type=match_name,
            )
            if distractors and len(clean_ro) > num_replacements - 1:
                # add at least one distractor by cutting non-d items
                # if we have enough non-distractor items
                clean_ro = clean_ro[:-1]
                logger.debug(
                    f"Cutting clean_ro to {len(clean_ro)} (needed {num_replacements}) to add a distractor"
                )
            # Adding +1 replacement in case some are double
            num_needed = max(0, num_replacements - len(clean_ro) + 2)

            distractors = distractors[:num_needed]
            #  md["opts_dist_num_external"] = len(distractors)
            logger.debug(
                f"Needed {num_replacements} options, got {len(clean_ro)} from text and {len(distractors)} as distractors"
            )
            #  clean_ro.extend(distractors)

            if add_most_frequent_all_genders_distractor:
                if replacement_options_raw[0] not in clean_ro:
                    mf_ag_distractor = replacement_options_raw[0]
                    # Cut it if it's already close to the max otherwise just append
                    # NB here we are likely cutting one of the other distractors
                    #  clean_ro = clean_ro[:-1] if num_replacements>=len(clean_ro) else clean_ro

                    # We cut it because we expect the end to be other distractors anyway
                    #   And if not — we can sacrifice a 'good' option among many to add this distractor I guess
                    if num_needed == 0:
                        logger.warning(
                            f"Cutting a good option {clean_ro[-1]} to add most-frequent AG distractor {mf_ag_distractor}"
                        )
                    clean_ro = clean_ro[:-1]

                    logger.debug(
                        f"Added most frequent all genders distractor '{mf_ag_distractor}' to {clean_ro}"
                    )
                    #  clean_ro.append(mf_ag_distractor)
                    #  md["opts_bl_most_frequent_all_genders"] = mf_ag_distractor

        # TODO заєць / зайчик?
        # again basic deduplication if we have a distractor as an option
        if len(clean_ro) != len(set(clean_ro)):
            raise ValueError(f"Duplicate options, shouldn't happen.")
            # b()

        # Add our token to replace to options, we do it here because later randomization will happen
        #   and we need the indexes of the various distractors stable
        #  self._add_if_not_present(token_to_replace.text, clean_ro)
        #  clean_ro.append(token_to_replace.text)

        #  clean_ro = list(set(clean_ro))

        # Save indexes of baselines and distractors
        #  md["opts_bl_most_frequent_idx"] = clean_ro.index(baseline_most_frequent_option)
        #  md["answer_idx"] = clean_ro.index(token_to_replace.text)

        #  if add_distractors:
        #      # Indexes of the distractors in the sorted list of options
        #      distractors_indexes = [
        #          i for i, w in enumerate(clean_ro) if w in distractors
        #      ]
        #      md["opts_dist_external_idx"] = distractors_indexes
        #
        #      if mf_ag_distractor is not None:
        #          # If we needed to add a diff gender most frequent distractor specifically
        #          md["opts_dist_most_frequent_all_genders_idx"] = clean_ro.index(
        #              mf_ag_distractor
        #          )

        # If we have a limit set it here
        # They were and will be ordered by frequency either way
        #  clean_ro = clean_ro[:num_replacements]

        ro_bytype[ReplacementOptionTypes.TRUE_ANSWER] = [token_to_replace.text]
        ro_bytype[ReplacementOptionTypes.NORMAL] = clean_ro
        if distractors:
            ro_bytype[ReplacementOptionTypes.DISTRACTOR_EXTERN] = distractors
        ro_bytype[ReplacementOptionTypes.BASELINE_MOSTFREQUENT] = [clean_ro[0]]
        if mf_ag_distractor:
            ro_bytype[ReplacementOptionTypes.DISTRACTOR_MOSTFREQUENT_AG] = [
                mf_ag_distractor
            ]

        #  b()
        return ro_bytype, md
        #  pass

    def agree_replacement_options_with_token(
        self,
        token_to_replace: Token,
        #  replacement_options: list[str],
        replacement_options_dict: dict[str, list],
        lemmas: dict[str, Token],
        allowed_pymorphy_pos: list[str],
        #  tokens_idx_to_leave_unchanged: list[int] = list(),
        fail_gracefully: bool = False,
        word_blacklist: list[str] = WORD_BLACKLIST,
    ) -> Optional[list[str]]:
        """Make all replacement options (given as str LEMMAS)
        agree with token_to_replace.

        Args:
            token_to_replace (Token): token_to_replace, spacy Token
            allowed_pymorphy_pos (list[str]): token_to_replace of pymorphy POS
                other than this one won't be inflected, None will be returned
        """

        # Pick best pymorphy Parse for this spacy token
        cand_m = self.disamb.get_with_disambiguation(token_to_replace)

        cand_m_pos = cand_m.tag.POS
        cand_m_case = cand_m.tag.case
        cand_m_number = cand_m.tag.number
        cand_m_person = cand_m.tag.person

        # VERB BITS
        # perf/inperf
        cand_m_aspect = cand_m.tag.aspect
        cand_m_tense = cand_m.tag.tense
        cand_m_gender = cand_m.tag.gender

        cand_m_infl_params_raw_verb = [
            cand_m_aspect,
            cand_m_tense,
            cand_m_person,
            cand_m_number,
            cand_m_gender,
        ]

        # Some of these inflection params may be None
        cand_m_infl_params_raw = [
            cand_m_case,
            cand_m_person,
            cand_m_number,  # Fails because pymorphy2 bug often so we make it last
        ]

        cand_m_infl_params_noun = [x for x in cand_m_infl_params_raw if x is not None]
        cand_m_infl_params_verb = [
            x for x in cand_m_infl_params_raw_verb if x is not None
        ]

        cand_m_infl_params = (
            cand_m_infl_params_verb
            #  if str(cand_m_pos) in [VERB, "GRND"]
            if str(cand_m_pos) == VERB
            else cand_m_infl_params_noun
        )

        if not cand_m_infl_params:
            #  b()
            logger.warning(
                f"Couldn't find any inflection parameters for {token_to_replace} / {cand_m}!"
            )
            if fail_gracefully:
                return replacement_options_dict
            #  b()
            return None

        if str(cand_m_pos) not in allowed_pymorphy_pos:
            logger.error(
                f"Candidate-for-masking {token_to_replace.text} is not a valid pos ({allowed_pymorphy_pos}) according to pymorphy: {cand_m.tag}"
            )
            if fail_gracefully:
                return replacement_options_dict
            # b()
            return None

        #  ros_agreed = list()
        ros_agreed_dict = defaultdict(list)
        for ro_type, ro_list in replacement_options_dict.items():
            if ro_type is ReplacementOptionTypes.TRUE_ANSWER:
                ros_agreed_dict[ro_type] = ro_list
                # If it's the token we replace we add it as-is
                # because agreement might change it (подивилась/подивилася)
                continue
            for i, r in enumerate(ro_list):
                # Get this lemma as spacy token
                if r in lemmas:
                    ro_token = lemmas[r]
                else:
                    ro_token = self._str_to_spacy_token(r)

                if r in word_blacklist:
                    logger.debug(
                        f"Skipping blacklisted word {r}/{str(ro_token)}/ {ro_token.sent}"
                    )
                # Parse of the lemma of this replacement option
                # We use spacy token to get the correct lemma
                # But we inflect it based on its normal form so we get no stray
                #   morphology from wherever spacy took that lemma.
                ro_m_nn = self.disamb.get_with_disambiguation(ro_token)
                ro_m = ro_m_nn.normalized
                #  if ro_m.word!=ro_token.norm_:
                #      b()

                # Agree the option with the target token
                # Can't convert plur to sing because pymorphy2 bug but we don't need to anyway
                # sing->plur works
                ro_agreed = self._inflect(ro_m, cand_m_infl_params)
                # ro_agreed

                if ro_agreed:
                    ro_agreed = ro_agreed.word
                else:
                    logger.error(
                        f"Can't inflect {ro_m.word} ({ro_m.tag}) to params {token_to_replace.text} ({cand_m_infl_params})"
                    )
                    if fail_gracefully:
                        ro_agreed = ro_m.word
                    else:
                        # b()
                        continue

                logger.debug(f"{r} -> {ro_agreed}")

                # Lastly attempt to match word shape
                if token_to_replace.is_title:
                    ro_agreed = ro_agreed.capitalize()
                else:
                    ro_agreed = ro_agreed.lower()

                ros_agreed_dict[ro_type].append(ro_agreed)
        #  b()
        return ros_agreed_dict

    def build_final_options_list(
        self, ros_agreed_dict: dict, target_options: int
    ) -> tuple[str, dict]:
        ros_metadata = dict()
        final_options_list = list()

        # remove external distractors that match existing opts - we'll dedup anyway,
        #   but that way our metadata will be correct, no opt will be both a distractor and not one
        ext_dists = ros_agreed_dict[ReplacementOptionTypes.DISTRACTOR_EXTERN]
        text_opts = (
            ros_agreed_dict[ReplacementOptionTypes.TRUE_ANSWER]
            + ros_agreed_dict[ReplacementOptionTypes.NORMAL]
        )
        ext_dists = [e for e in ext_dists if e not in text_opts]
        ros_agreed_dict[ReplacementOptionTypes.DISTRACTOR_EXTERN] = ext_dists

        for ro_type, ro_list in ros_agreed_dict.items():
            final_options_list.extend(ro_list)
        # 'deduplication'
        final_options_list = list(set(final_options_list))

        num_toomuch = len(final_options_list) - target_options
        if num_toomuch:
            logger.debug(f"Have {len(final_options_list)}>{target_options} opts")
        for i in range(num_toomuch):
            try:
                choice = ros_agreed_dict[ReplacementOptionTypes.DISTRACTOR_EXTERN].pop()
                logger.debug(f"Removing distractor {choice} from list")
            except IndexError:
                choice = ros_agreed_dict[ReplacementOptionTypes.NORMAL].pop()
                logger.info(f"Removing option {choice} from list")

            final_options_list.remove(choice)

        #  for ro_type, ro_list in ros_agreed_dict.items():
        for ro_type in ReplacementOptionTypes:
            ro_list = ros_agreed_dict[ro_type]
            ros_metadata["opts_" + ro_type.value] = ro_list
            ros_metadata["opts_" + ro_type.value + "_idx"] = sorted(
                [final_options_list.index(x) for x in ro_list]
            )
            ros_metadata["opts_num_" + ro_type.value] = len(ro_list)
        ros_metadata["opts_num_options_total"] = len(final_options_list)

        return final_options_list, ros_metadata

    def find_occs_of_token_lemma(self, lemma: str, haystack: Span) -> list[Span]:
        """Find how many times lemma was found in the lemmas of the other tokens
        in haystack."""
        occs = list()
        prev_lemmas = [get_lemma(x) for x in haystack]
        num = prev_lemmas.count(lemma)
        return num

    def get_possible_gaps(
        self,
        span: Span,
        previous_text: Span,
        anim_lemmas: Optional[list[str]] = None,
        min_num_occs_in_prev_text: dict = dict(
            default=2, P_COMMON_NOUN=4
        ),
    ) -> list[Span]:
        """Find tokens inside span that are replaceable by "____".

        TODO: this can be improved by a lot,for now it just
            looks for tokens matching the matcher classes

            Options:
            - DONE only replace words' nth+ occurrence

        """

        matches = self._get_matches(span, anim_lemmas=anim_lemmas)
        prev_found_matches = list()
        #  b()

        for m in matches.matches:
            #  prev_found_matches.append(m)

            # Span containing all the story text up till and not including the sentence with the match
            # span.doc is the entire story document (TODO abstraction leak)
            #  span_with_entire_story_before_match_sentence =  span.doc[0:len(previous_text) + m.sent.start]
            num_occs = self.find_occs_of_token_lemma(get_lemma(m), previous_text)
            # we look for occurrences in current span as well, all sents up till and not including match sentence
            start_of_span = span[: m.sent.start]
            num_occs_in_span = self.find_occs_of_token_lemma(
                get_lemma(m), start_of_span
            )
            total_num_occs = num_occs + num_occs_in_span
            relevant_num_occs = min_num_occs_in_prev_text.get(m.label_, min_num_occs_in_prev_text['default'])
            logger.debug(f"{relevant_num_occs=} for {m.label_}")
            if total_num_occs > 0:
                logger.debug(
                    f"Found {num_occs}+{num_occs_in_span}={total_num_occs} occs of {m} in prev text + beg. of question span"
                )
            else:
                logger.debug(
                    f"Found NO occs of {m} in prev text + beg. of question span"
                )

            if num_occs < relevant_num_occs:
                continue

            # Go through filter  for additional cases
            if not self.filter_gaps_match(m):
                continue

            prev_found_matches.append(m)

        matches.matches = prev_found_matches

        #  b()
        ret = matches.as_type_dict(as_tokens=True)
        return ret

    @staticmethod
    def filter_gaps_match(match) -> bool:
        if match[0].is_punct:
            # TODO sometimes this happens, I have no idea.
            #  (Pdb++) x
            #  !
            #  (Pdb++) x.morph
            #  Animacy=Inan|Case=Nom|Gender=Fem|Number=Sing
            logger.debug(f"Skipping strange match '{match}' that is punctuation")
            return False
        return True

    def get_frequent_matches_in_span(self, span: Doc | Span):
        """Counts the frequency of normalized matched entities in span.label

        Returns together with dictionary of spacy tokens for each lemma
        """
        matches_h = self._get_matches(span)
        matches_h.matches = [m for m in matches_h.matches if self.filter_gaps_match(m)]

        matches_count, lemmas = self._count_normalized_matches(matches_h.matches)
        return matches_count, lemmas

    def adjectival_names_story_filter(
        self, lemmas, important_entities, pymorphy_adj_parsings_threshold=0.3
    ) -> bool:
        """Some stories have adjectival names that break everything, this function tries to detect such stories."""
        if P_NAMED_ENTITY not in important_entities:
            logger.info(
                f"Can't check for adjectival stories because not looking for NAMED_ENTITIES"
            )
            return False

        potential_names_raw = list(important_entities[P_NAMED_ENTITY].keys())
        potential_names = list()
        for pn in potential_names_raw:
            potential_names.append(lemmas[pn])
        num_total = len(potential_names)
        capitalized = [x for x in potential_names if x.is_title]
        len_capitalized = len(capitalized)
        adjs_according_to_pymorphy = list()
        for t in capitalized:
            p = self.disamb.pymorphy_analyzer.parse(t.text)
            is_adj_prob = len([pp for pp in p if pp.tag.POS == ADJF]) / len(p)
            is_adj = is_adj_prob > pymorphy_adj_parsings_threshold
            if is_adj:
                adjs_according_to_pymorphy.append(t)
        if len(adjs_according_to_pymorphy) > 2:
            logger.warning(
                f"Adjectival story: {is_adj_prob*100:.2f}% named entities are adjectival: {adjs_according_to_pymorphy}"
            )
            return True
        return False

    def _get_matches(
        self,
        #  doc,
        #  context_sents_span,
        doc: Span | Doc,
        anim_lemmas: list[str] = None,
    ) -> MatchesHelper:
        #  if isinstance(doc, Span):
        #  span = self.nlp(doc.text)

        # Matches among our question sentences
        # TODO: this is slow
        # https://chat.openai.com/share/2d404312-7dad-4e74-823e-ea885a3bc7af
        # matcher expects a Doc,  but span.as_doc() doesn't work, nlp(span(text)) does
        #  matches = self.matcher(self.nlp(span.text))
        matches = self.matcher(self.nlp(doc.text), as_spans=True)

        mhelper = MatchesHelper(matches=matches, nlp=self.nlp)

        # Decrease the amount of wrong animacy matches
        mhelper._clean_up_inanimate_matches(
            #  self._clean_up_inanimate_matches(
            anim_match_name=self.ANIM_PATTERNS_NAMES[0],
            inan_match_name=self.INAN_PATTERNS_NAMES[0],
            additional_anim_lemmas=anim_lemmas,
        )

        return mhelper

    def _count_normalized_matches(
        self, matches: list[Span]
    ) -> tuple[dict[str, dict[str, int]], dict[str, Token]]:
        """Normalizes the Spans (assumed to contain a single Token) and
        counts how many occurrences of the same basic entity are found.

        Returns Counter with the counts and a dict with a spacy Token for the
        normalized forms.
        """
        # Counter with frequencies of words
        kinds = defaultdict(Counter)
        # Has instances of spacy tokens for normalized forms, so that
        #   we can then use the spacy morphology to pick the correct pymorphy
        #   one for inflection
        lemmas = dict()
        kinds = defaultdict(Counter)

        for span in matches:
            token = span[0]
            assert len(span) == 1

            match_name = self.nlp.vocab.strings[span.label]
            match_word = self._normalize_morph_word(token)

            kinds[match_name][match_word] += 1
            lemmas[match_word] = token

        return kinds, lemmas

    def _normalize_morph_word(
        self, token: spacy.tokens.Token | str, to_lower=True
    ) -> str:
        """Returns normal for of word. Кіз->коза.

        Function because we might want to be fancier here later.
        """
        #  if isinstance(token, spacy.tokens.Token):
        #  token = token.text
        #  word = token
        #  if to_lower:
        #  word = word.lower()
        #  morph_word = self.morph.parse(word)[0]

        #  morph_word = self.disamb.get_with_disambiguation(token)
        #  normalized_word = morph_word.normal_form

        normalized_word = get_lemma(token)

        return normalized_word

    @staticmethod
    def _inflect(
        parse: pymorphy2.analyzer.Parse, new_grammemes: set | frozenset
    ) -> pymorphy2.analyzer.Parse:
        """Sometimes inflecting with the entire batch fails, but one by one
        works. This chains the grammemes for one inflection at a time.

        pymorphy Parse as `parse` and output type

        This is a workaround for a pymorphy bug:
        https://github.com/pymorphy2/pymorphy2/issues/169
        """
        new_parse = parse
        for g in new_grammemes:
            if new_parse.inflect({g}):
                new_parse = new_parse.inflect({g})
            else:
                continue
        return new_parse

    @staticmethod
    def _make_agree_with_number(
        parse: pymorphy2.analyzer.Parse, n: int
    ) -> pymorphy2.analyzer.Parse:
        """Inflect `parse` to agree in number with `n`.
        (Like singular/plural, except 2-4 and5 5+ are separate in Ukrainian)

        Fix for pymorphy bug its function for this.
        Pymorphy bug: https://github.com/pymorphy2/pymorphy2/issues/169


        Args:
            parse (Parse): parse object to inflect to match number
            n (int): n number to agree on.

        Returns:
            Parse:
        """
        grams = parse.tag.numeral_agreement_grammemes(n)
        new_parse = CBTTaskMaker._inflect(parse=parse, new_grammemes=grams)
        return new_parse


def run():
    model_name = MODEL_NAME
    model_name = "uk_core_news_trf"

    #  c = CBTTaskMaker(n_context_sents=20, n_question_sents=5, model_name=model_name)
    #  r = c.process_txt(SMALL_STORY)

    c = CBTTaskMaker(n_context_sents=6, n_question_sents=3, model_name=model_name)

    r, dm = c.process_txt(ANOTHER_STORY)
    print(r)
    b()
    #  print("run")
    #  items = read_epub()
    #  LIMIT = None
    #  console=Console()
    #  console.rule("Processing book")

    #  tasks = process_book(EPUB_FILE, lim=LIMIT)
    #  s = tasks.tasks[0][0]
    #  b()
    pass


def main():
    #  args = parse_args_get() if mode == "get" else parse_args_crawl()
    #  logger.setLevel(args.loglevel if args.loglevel else logging.INFO)
    #  logger.debug(args)

    USE_PBD = True
    USE_PBD = False
    try:
        run()
    except Exception as e:
        #  if args.pdb:
        # if True:
        if USE_PBD:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise e


if __name__ == "__main__":
    main()
