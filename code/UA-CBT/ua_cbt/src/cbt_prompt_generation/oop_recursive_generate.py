import logging
import pdb
import sys
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import itertools
from pathlib import Path
from random import choices
from typing import Optional

import pandas as pd
import yaml

b = breakpoint


ADDITIONAL_COLUMNS = ["story_fixed"]
ADDITIONAL_COLUMNS = list()

DATA_FOLDER = Path("../artifacts")
# assert DATA_FOLDER.exists()

NEW_STORIES_FN = "new_story_templates.csv"
#  YAML_FN = "input_yaml.yaml"


class PromptGenerator:
    """Generates CBT story prompts from all combinations of values
    provided in the input YAML.

    The YAML may contain instead of {format_strings_bits} UPPERCASE_BITS,
    which will get filled parametrically by the generator.
    """

    READING_LEVELS = [
        "graduate student",
        #  "high school",
        #  "middle school",
        #  "primary school",
    ]

    N_MAIN_CHARACTERS = ["3"]
    N_MINOR_CHARACTERS = ["5"]
    NUM_WORDS = ["500"]

    BAD_ENDING = [" The story should have an unhappy ending.", ""]
    #  BE_HOTFIX = lambda x: "unhappy" in x

    OTHER_CHARACTERS = [" but with different characters"]

    STORY_TEMPLATE_KEY = "story_template"

    #  def __init__(self, reading_levels: Optional[list[str]]=PromptGenerator.READING_LEVELS):
    def __init__(
        self,
        #  reading_levels: Optional[list[str]] = None,
        reading_level: Optional[str] = None,
        #  bad_ending: Optional[str] = None,
        other_characters: Optional[str] = None,
        num_stories: Optional[int] = 300,
        output_dir: Optional[str | Path] = DATA_FOLDER,
        output_fn: Optional[str] = NEW_STORIES_FN,
        #  output_file: Optional[str | Path] = DATA_FOLDER /  "story_prompts.csv",
        additional_columns: Optional[list[str]] = ADDITIONAL_COLUMNS,
    ):
        #  self.reading_level = (
        #      type(self).READING_LEVELS[0] if not reading_level else reading_level
        #  )

        #  self.bad_ending = "" if not bad_ending else type(self).BAD_ENDING[0]
        self.other_characters = (
            "" if not other_characters else type(self).OTHER_CHARACTERS[0]
        )

        self.num_stories = num_stories

        self.output_dir = Path(output_dir).resolve().absolute()
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self.output_fn = output_fn

        self.additional_columns = additional_columns

        # keys same format as inside template strings,  values are lists
        # Uppercase format markers mapped to the variable used to fill them

        self.KEYS = {
            #  "READING_LEVEL": self.reading_level,
            "READING_LEVEL": self.READING_LEVELS,
            "N_MAIN_CHARACTERS": self.N_MAIN_CHARACTERS,
            # TODO - should be MINOR, not recurring
            "N_MINOR_CHARACTERS": self.N_MINOR_CHARACTERS,
            "BAD_ENDING": self.BAD_ENDING,
            "NUM_WORDS": self.NUM_WORDS,
            #  "OTHER_CHARACTERS": self.other_characters,
        }

        logger.info(
            f"Running with settings {self.KEYS}, {self.num_stories=} {self.output_dir=}, {self.output_fn},  {self.additional_columns=}, {self.other_characters=}"
        )

    @staticmethod
    def make_param_temps(template_strings: list[str], keys=None) -> list[str]:
        # Transform "Some TEMPLATE" => "Some {TEMPLATE}"
        preformatted_templates = list()
        for new_temp in template_strings:
            for k in keys:
                if k in new_temp:
                    new_temp = new_temp.replace(k, "{" + k + "}")
            preformatted_templates.append(new_temp)
        return preformatted_templates

    @staticmethod
    def fill_templates_with_keys(
        preformatted_templates: dict[str, str], key_name: str, key_values: list[str]
    ):
        # For each dict like (t=template_text, **other_params)
        #   create multiple other dicts for diff key_values
        #  changed_pts = preformatted_templates.copy()
        changed_pts = list()
        for pt in preformatted_templates:
            #  b()
            if key_name in pt[PromptGenerator.STORY_TEMPLATE_KEY]:
                formatted = "{" + key_name + "}"
                if formatted in pt[PromptGenerator.STORY_TEMPLATE_KEY]:
                    # TODO understand why {{this}} happens
                    logger.debug("{{this}} happened")
                    #  b()
                else:
                    pt[PromptGenerator.STORY_TEMPLATE_KEY] = pt[
                        PromptGenerator.STORY_TEMPLATE_KEY
                    ].replace(key_name, formatted)
                for kv in key_values:
                    new_pt = pt.copy()
                    new_pt[PromptGenerator.STORY_TEMPLATE_KEY] = pt[
                        PromptGenerator.STORY_TEMPLATE_KEY
                    ].format(**{key_name: kv})
                    new_pt[key_name] = kv
                    changed_pts.append(new_pt)
        #  b()
        return changed_pts

    def fill_parametrized_gaps(
        self, template_strings: list[str], keys=None
    ) -> list[str]:
        """Fill any uppercase bits remaining in the templates with
        values given during init.

        TODO support multiple ones sometime, e.g. multiple reading levels
        """
        if keys is None:
            keys = self.KEYS
        #  results = list()

        #  preformatted_templates = self.make_param_temps(
        #      template_strings=template_strings, keys=keys
        #  )

        # Create a new version for each option in each of these keys
        new_pts = [{self.STORY_TEMPLATE_KEY: pt} for pt in template_strings]
        for k in keys:
            new_pts = self.fill_templates_with_keys(
                preformatted_templates=new_pts, key_name=k, key_values=keys[k]
            )
            #  b()
        #  print(new_pts[0])
        #  print(new_pts[1])
        return new_pts
        # Then fill it in
        #  for pt in new_pts:

        results: list[dict] = list()
        #  templates_with_values = list()
        for pt in preformatted_templates:
            template_with_values = pt
            twv_dict = dict()
            for k in keys:
                if k in template_with_values:
                    for v in self.KEYS[k]:
                        template_with_values = template_with_values.format(**{k: v})
                        twv_dict[k] = v
                else:
                    twv_dict[k] = None
            results.append(dict(story_template=template_with_values, **twv_dict))
        #  return results
        #
        #              #  b()
        #              #  for val inside self.KEYS[k]:
        #              new_temps = list()
        #              for v in self.KEYS[k]:
        #                  new_temp = new_temp.format(**{k: v})
        #                  new_temp_meta = dict(STORY_TEMPLATE=new_temp, )
        #                  new_temps.append(new_temp)
        #              results.extend(new_temps)
        #      #  results.append(new_temp)
        #      #  results.extend(new_temps)
        #  #  b()
        return results

    @staticmethod
    def generate_combinations(template_data):
        """Recursively generate all possible combinations inside template data.

        template_data can be:
            - a dict with options/parts keys:
                - then for each option we fill it with all possible combinations of the values
                for its parts by recursively calling this function
                - if no parts, just options, we return a list with all options
        """
        #  b()
        if "options" in template_data:
            # Generate combinations for parts if they exist
            if "parts" in template_data:
                parts_combinations = {
                    part: PromptGenerator.generate_combinations(sub_data)
                    for part, sub_data in template_data["parts"].items()
                }

                # Generate all combinations of parts
                all_part_combinations = list(
                    itertools.product(
                        *(parts_combinations[part] for part in parts_combinations)
                    )
                )

                # Generate full options by formatting them with each combination of parts
                return [
                    option.format(**dict(zip(parts_combinations, combo)))
                    for option in template_data["options"]
                    for combo in all_part_combinations
                ]
            else:
                return template_data["options"]
        else:
            # If no options, handle as a list of items (leaf node)
            if isinstance(template_data, list) and all(
                isinstance(item, str) for item in template_data
            ):
                return template_data
            else:
                raise ValueError("Template structure is not valid.")

    @staticmethod
    def generate_all_combinations(data: dict):
        """Generate all str story prompts based on all permutations of all
        list items inside data.

        `data` has to have the following structure, with two fields in the root:
        - main_template:
            - contains a string with the upper-level template data
        - templates
            - has a key for each template field in main_template
            - each key is a dict with two fields:
                - options
                    - a list of strings (or string templates)
                - parts
                    - for all keys in all arguments prement in `options`, has either
                        - a list of strings with possible values for the keys
                        - two keys that work identically to the already described ones
                            - options

        For example:
            TODO
        Args:
            data:
        """

        # Parse highest level of dictionary
        main_template = data["main_template"]
        templates_dict = data["templates"]

        # Parse the individual parts in template dict for the highest level
        # Create a list of str options for each highest-level keys
        # Recursively filling out the sub-format-fields for each
        parts_combinations = {
            part: PromptGenerator.generate_combinations(templates_dict[part])
            for part in templates_dict
        }
        logger.info(
            f"Generated {sum(len(x) for x in parts_combinations.values())} parts"
        )

        # All permutations of the three highest-level keys list values
        all_combinations = itertools.product(
            *(parts_combinations[part] for part in parts_combinations)
        )

        # Return all final versions of the main_template
        return [
            main_template.format(**dict(zip(parts_combinations, combo)))
            for combo in all_combinations
        ]

    def generate(
        self, yaml_content, fill_gaps: Optional[bool] = True
    ) -> list[dict[str, str]]:
        """Generate all story prompts."""
        combinations = list()

        # Load the YAML content
        data = yaml.safe_load(yaml_content)

        # Generate all possible combinations
        all_combinations = self.generate_all_combinations(data)

        # Print each combination
        for combination in all_combinations:
            combinations.append(combination)

        combinations = choices(combinations, k=self.num_stories)
        logger.info(f"Randomly selected {len(combinations)} from them.")
        #  b()
        if fill_gaps:
            logger.info(f"Filling parametrized gaps: {self.KEYS}")
            new_combinations = self.fill_parametrized_gaps(combinations)
            logger.info(f"Generated {len(combinations)} prompts!")
        #  b()
        return new_combinations

    @staticmethod
    def write_combinations(
        combinations: list[str],
        output_file: Path,
        params: Optional[dict] = None,
        additional_columns: Optional[list[str]] = None,
    ):
        #  df_dicts = list()
        #  for c in combinations:
        #      df_dicts.append(dict(story_template=c, **params))

        # Ugly hotfix to transform bad_ending into a bool
        for c in combinations:
            c["BAD_ENDING"] = bool(c["BAD_ENDING"])

        df = pd.DataFrame(combinations)

        if additional_columns:
            df[additional_columns] = ""

        df.to_csv(
            output_file,
            #  sep="\t",
            #  index=False,
        )
        logger.info(f"Written to {output_file}")
        return

    def run(self, yaml_content: str | Path):
        """Entry point of the program."""
        if isinstance(yaml_content, str):
            yc = yaml_content
        else:
            yc = yaml_content.read_text()
        res = self.generate(yc)
        output_file = self.output_dir / self.output_fn
        self.write_combinations(
            res,
            output_file=output_file,
            params=self.KEYS,
            additional_columns=self.additional_columns,
        )

        logger.info(f"Wrote stories CSV to {str(output_file)}")
        yaml_output_file = self.output_dir / (self.output_fn.split(".")[0] + ".yaml")
        yaml_output_file.write_text(yc, encoding="utf8")
        logger.info(f"Wrote YAML to {str(yaml_output_file)}")
        yaml_output_file_2 = self.output_dir / (
            self.output_fn.split(".")[0] + ".keys.yaml"
        )
        keys_as_yaml = yaml.dump(self.KEYS)
        yaml_output_file_2.write_text(keys_as_yaml, encoding="utf8")
        return res


def run():
    #  mode = "control"
    #  mode = "normal, control"
    mode = "normal"

    num_stories = 3
    num_stories = 300

    res = list()

    yaml_file = Path(__file__).parent / "prompt_v5.yaml"
    if "normal" in mode:
        # New stories
        #  output_file = DATA_FOLDER / "new_story_prompts.csv"
        #  output_file = DATA_FOLDER / "new_story_prompts.csv"
        p = PromptGenerator(
            output_fn="new_story_prompts_v6.csv",
            num_stories=num_stories,
        )
        res.extend(p.run(yaml_file))

    #  if "control" in mode:
    #      output_file = "./DATA/existing_story_templates.csv"
    #      p = PromptGenerator(
    #          output_file=output_file,
    #          #  bad_ending=False,
    #          other_characters=False,
    #          num_stories=num_stories,
    #      )
    #      res.extend(p.run(yaml_content_existing))

    #  pp(res)
    #  print(res)
    #  b()
    #  for r in res[:40]:
    #      print(r + "\n\n====\n")


def main():

    try:
        run()
    except Exception as e:
        #  if args.pdb:
        if True:
            extype, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
        else:
            raise e


if __name__ == "__main__":
    main()
