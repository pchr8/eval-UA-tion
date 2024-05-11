# UA-CBT (Ukrainian Children's Book Test)

> [!IMPORTANT]
> The code is buggy, messy, many existing tests fail, many comments are lies, and the typing annotations aren't all correct — it works and worked to generate the dataset, but **here be dragons**. I didn't have time to fix it (still don't, sadly) — it's more useful out there in its imperfect state, "better done than perfect". (I may abstract some useful bits and make them into packages with higher standards later though.)
> 
> PRs and issues welcome.

## Basics
- **Datasets**
    - [shamotskyi/ua_cbt](https://huggingface.co/datasets/shamotskyi/ua_cbt) the task itself
    - [shamotskyi/ua_cbt_stories](https://huggingface.co/datasets/shamotskyi/ua_cbt_stories) before/after stories (private for now, shoot me an email to get access)

## Concept
1. `cbt_prompt_generation` takes a YAML with bits to create LLM prompts that are used to generate stories 
2. LLMs generate stories (separate package, **TODO**) — but `./artifacts/stories/few_shot_split.csv` is an example file with generated and manually fixed stories taken as input by `ua_cbt`.
3. `ua_cbt` takes the CSV with these generated stories and creates UA-CBT tasks out of it.

## Directories
- `ua_cbt` is a poetry projects with two packages:
    - `cbt_prompt_generation` takes a YAML and generates different LLM prompts out of it, which are used to get LLMs to generate stories
    - `ua_cbt` generates UA-CBT tasks based on prompts
- `./artifacts/` has random generated files — e.g. the generated LLM prompts — to illustrate the packages, but are not equal to the ones actually used to generate the dataset.
    - anything with `prompts` in the name refers to the LLM prompts used to generate stories.
    - `./artifacts/stories/`: `few_shot_split.csv` is a CSV file  with generated/fixed stories taken as input to create UA-CBT tasks.

## Usage 
### Installation
```bash 
poetry shell 
poetry install

# spacy models — check MODEL_NAME
python3 -m spacy download uk_core_news_sm
# or 
python3 -m spacy download uk_core_news_trf
```

### Generating LLM templates
- `poetry run gen` runs the LLM prompt template generation. 

### Generating UA-CBT tasks

- `poetry run ml`  generates dummy tasks out of a dummy story. You're dropped in a `pdbpp` shell at the end


#### Generating stories from story CSV 
Uses a CSV  to generate UA-CBT tasks from each story contained therein. This repo by default uses the `few_shot_split.csv` story, for the benchmark the code in the `ua_cbt_stories` dataset was used.


```bash
> poetry run run -h
/home/sh/uuni/eval-UA-tion/code/UA-CBT/ua_cbt/.venv/lib/python3.10/site-packages/torch/cuda/__init__.py:619: UserWarning: Can't initialize NVML
  warnings.warn("Can't initialize NVML")
usage: run [-h] [-input INPUT] [--output OUTPUT] [--limit LIMIT] [-splits SPLITS [SPLITS ...]] [--pdb] [-q] [-v]

options:
  -h, --help            show this help message and exit
  -input INPUT, -i INPUT
                        Input CSV file with stories (../../../artifacts/stories/few_shot_split.csv)
  --output OUTPUT, -o OUTPUT
                        Output for the dataset (../../../artifacts/tasks)
  --limit LIMIT, -l LIMIT
                        How many stories to parse (None)
  -splits SPLITS [SPLITS ...], -s SPLITS [SPLITS ...]
                        Share of all splits except train (None)
  --pdb, -P             Run PDB on exception
  -q                    Output only warnings
  -v, --verbose         Output more details
```

Running it generates the stories in multiple formats — you likely care about the json.

`stories_train.labelstudio.json` has additional metadata for each option, which was used in the [Label Studio](https://labelstud.io/) project to filter 'bad' task instances.

#### Sample task

```python
<SingleTask
    context='Колись давним-давно, в піщаних просторах пустелі, жив хитрий верблюд. Він був відомий своєю вмінням уникати праці,
перекладаючи свої обов\'язки на менш кмітливих сусідів - невеликого єнота та серйозного орла. Вони терпеливо виконували важку роботу, в той час
як верблюд ласував найсолодшими пагонами.\n\nОдного дня, коли вода в оазі на межі висихання, верблюд вирішив, що єнот і орел повинні
відправитись у небезпечну подорож за новим джерелом. "Тільки ви маєте кмітливість і силу знайти воду," - лукаво мовив верблюд.\n\nЄнот і орел,
виснажені його маніпуляціями, нарешті усвідомили хитрість верблюда і вирішили діяти спільно.'
    question='Вони пішли, обіцяючи верблюду привести воду, але насправді вони планували знайти нову оазу лише для себе.\n\nЗалишившись на
самоті, верблюд швидко зрозумів, що його власна лінь і хитрість привели до катастрофи. Орел і єнот знайшли нове місце, а ______ , не здатний
самостійно вижити, був змушений мандрувати пустелею у пошуках води і допомоги.'
    options=['сусід', 'півень', 'єнот', 'орел', 'верблюд', 'петро']
    answer='верблюд'
    story_id=None
    md={
        'num_context_sents': 6,
        'num_context_sents_tokens': 113,
        'num_question_sents_tokens': 341,
        'n_question_sents_share': -1,
        'sequential_number': 6,
        'opts_correct_answer': ['верблюд'],
        'opts_correct_answer_idx': [4],
        'opts_num_correct_answer': 1,
        'opts_replacement_option_from_text': ['верблюд', 'єнот', 'орел', 'сусід'],
        'opts_replacement_option_from_text_idx': [0, 2, 3, 4],
        'opts_num_replacement_option_from_text': 4,
        'opts_baseline_most_frequent': ['верблюд'],
        'opts_baseline_most_frequent_idx': [4],
        'opts_num_baseline_most_frequent': 1,
        'opts_distractor_external': ['півень', 'петро'],
        'opts_distractor_external_idx': [1, 5],
        'opts_num_distractor_external': 2,
        'opts_distractor_most_frequent_any_gender': [],
        'opts_distractor_most_frequent_any_gender_idx': [],
        'opts_num_distractor_most_frequent_any_gender': 0,
        'opts_num_options_total': 6
    }
>
```

## Known issues 
- No documentation
- All tests fail 
    - Many refer to getting stories out of EPUB files — not used  

## TODO 
- ~~Everything~~
- chiefly — sample files to demostrate basic I/O, esp. with LLM generated stories.
