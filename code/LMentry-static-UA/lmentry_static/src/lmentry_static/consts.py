from pathlib import Path

JSON_DUMP_KWARGS = {"ensure_ascii": False, "indent": 4}

## KEYS
TEMPLATE_N = "template_n"  # TODO use this
KEY_SYSTEM_PROMPT = "system_prompts"

OPTION_KEY = "option_{n}"
LABEL_KEY = "label"

## FILENAMES
TASK_OUTPUT_FN_FORMAT = "{}.json"
TASK_OUTPUT_CSV_FORMAT = "{}.csv"
#  TASK_OUTPUT_JSONL_FORMAT = "{}.openai.jsonl"
TASK_OUTPUT_JSONL_FORMAT = "{}.jsonl"
TASK_HF_OUTPUT_FORMAT = "hf_{}"
CONFIG_FN = "config_full.yaml"
CONFIG_FN = "config.yaml"

## PATHS
resources_path = Path(__file__).parent.parent.parent.parent / "resources"
templates_path = resources_path / "templates"
config_path = resources_path / CONFIG_FN

data_word_classes_yaml_path = resources_path / "word_classes.yaml"

# default_output_path = Path(__file__).parent.parent.parent.parent.parent.parent / "data/lmentry"
default_output_path = Path(__file__).parent.parent.parent.parent / "artifacts"

paths = [templates_path, config_path]
for p in paths:
    assert p.exists()

#  MODEL_NAME = "uk_core_news_sm"
