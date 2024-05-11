from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

#############
## FILE BITS
#############

# TODO clean all of this up
MODEL_NAME = "de_core_news_sm"
EPUB_FILE = Path(
    "./tests/assets/Richard Wilhelm - Chinesische Märchen_ 100"
    " Märchen aus China mit vielen Illustrationen.epub"
)

MODEL_NAME = "uk_core_news_sm"
MODEL_NAME = "uk_core_news_lg"
MODEL_NAME = "uk_core_news_trf"

EPUB_FILE = Path("./tests/assets/virm.epub")
EPUB_FILE = Path("./tests/assets/Українські народні казки book-11855819-6c9f9a.epub")

DEFAULT_OUTPUT_DIR = (
    # Path(__file__).parent.parent.parent.parent.parent / "data/CBT/tasks"
    Path(__file__).parent
    / "../../../artifacts/tasks/"
).resolve()

DEFAULT_INPUT_CSV_FILE = (
    #  Path(__file__).parent.parent.parent / "./tests/assets/csv/generated_new_stories.csv"
    # Path(__file__).parent.parent.parent.parent.parent / "data/CBT/generated5/corrected/latest.csv"
    Path(__file__).parent
    / ("../../../artifacts/stories/few_shot_split.csv")
).resolve()

DEFAULT_DATA_YAML_FILE = Path(__file__).parent / "words_and_data.yaml"

FILE_NAME_SPLIT_PAT = "stories_{}"
SPLIT_NAMES = ["train", "validation", "test"]

LIMIT = 2
LIMIT = None

CSV_STORIES_KEY = "fixed_story"
CSV_USABLE_KEY = "result"
CSV_TODO_KEY = "result"
CSV_MD_TARGET_KEY = "story_metadata"
CSV_LS_STORY_ID = "id"


############
# UGLY YAML DATA BITS
###########

YAML_DATA = OmegaConf.load(DEFAULT_DATA_YAML_FILE)

#############
## SPACY BITS
#############

REPLACEMENT_TOKEN_SYMBOL = "______"

## Patterns
#
#  # NB Animate can also be Adj,  and many animate things get detected as inan but not the other way around
#  PAT_NAMED_ENTITY = {
#      # See also: ent_type=="PER" and `NameType=Sur`, pos=='PROPN'
#      #  "POS": "NOUN",
#      "POS": {"IN": ["NOUN", "PROPN"]},
#      #  "TAG": "NOUN",
#      "MORPH": {
#          "IS_SUPERSET": [
#              #  "Number=Sing",
#              #  "Case=Nom",
#              "Animacy=Anim",
#          ]
#      },
#  }
#
#  PAT_COMMON_NOUN = {
#      #  "TAG": "NOUN",
#      "POS": "NOUN",
#      "MORPH": {
#          "IS_SUPERSET": [
#              #  "Number=Sing",
#              #  "Case=Nom",
#              "Animacy=Inan",
#          ]
# },
#  }
#
#  PAT_PRON_SG = {
#      #  "TAG": "NOUN",
#      "POS": "PRON",
#      "MORPH": {
#          "IS_SUPERSET": [
#              "PronType=Prs",
#              #  "Number=Sing",
#              #  "Case=Nom",
#              "Number=Sing",
#          ]
#      },
#  }
#
#
"""
Other options are:
неї -> Case=Gen|Gender=Fem|Number=Sing|Person=3|PronType=Prs

"""

#  if "uk" in MODEL_NAME:
#  PAT_NAMED_ENTITY["MORPH"]["IS_SUPERSET"].append("Animacy=Anim")
#  PAT_COMMON_NOUN["MORPH"]["IS_SUPERSET"].append("Animacy=Anim")

# TODO
#   - VERBS
#   - PREPOSITIONS
#       Направляється       :	VERB 	Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin

# 6/2 seems best

N_CONTEXT_SENTENCES = 6
N_QUESTION_SENTENCES = 6
N_CONTEXT_SENTENCES = None

N_CONTEXT_SENTENCES = 6
N_QUESTION_SENTENCES = 2
N_QUESTION_SENTENCES = None

###########

SMALL_STORY = "Жив був король. У нього було царство, де жило сто корів і тридцять кіз, 10 дерев, один камінь і один чарівник. Ще, у нього була дочка. Якось король у неї спитав, чи щаслива вона. Дочка не відповіла, лише сумно подивилась за вікно, на дерева та козу."

ANOTHER_STORY = """Колись давним-давно, в піщаних просторах пустелі, жив хитрий верблюд. Він був відомий своєю вмінням уникати праці, перекладаючи свої обов'язки на менш кмітливих сусідів - невеликого єнота та серйозного орла. Вони терпеливо виконували важку роботу, в той час як верблюд ласував найсолодшими пагонами.

Одного дня, коли вода в оазі на межі висихання, верблюд вирішив, що єнот і орел повинні відправитись у небезпечну подорож за новим джерелом. "Тільки ви маєте кмітливість і силу знайти воду," - лукаво мовив верблюд.

Єнот і орел, виснажені його маніпуляціями, нарешті усвідомили хитрість верблюда і вирішили діяти спільно. Вони пішли, обіцяючи верблюду привести воду, але насправді вони планували знайти нову оазу лише для себе.

Залишившись на самоті, верблюд швидко зрозумів, що його власна лінь і хитрість привели до катастрофи. Орел і єнот знайшли нове місце, а верблюд, не здатний самостійно вижити, був змушений мандрувати пустелею у пошуках води і допомоги.

Але пустеля була невблаганною, і верблюд, нарешті, зрозумів, що хитрість без мудрості і співпраці - це шлях до самотності та відчаю. Саме ця думка була його останньою, перш ніж пустеля поглинула його."""

######

#  JSON_DUMP_KWARGS = {'indent':4, 'ensure_ascii':False}
