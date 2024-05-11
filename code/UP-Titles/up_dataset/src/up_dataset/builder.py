import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("__name__")


b = breakpoint
from rich import inspect, print

# This is the CSV for the https://huggingface.co/datasets/shamotskyi/ukr_pravda_2y dataset
UP_CSV_PATH = "/tmp/full_crawls/dataset_up_since_2022_2023-12-13_nocommit.csv"
# UP_CSV_PATH = "/home/sh/uuni/master/data/up_crawls/full_crawls/dataset_up_since_2022_2023-12-13_nocommit.csv"

NUM_ROWS = 5000
NUM_ROWS = None

NUM_DOCS_TO_SAMPLE = 5000
NUM_DOCS_TO_SAMPLE = 35000

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
# TODO - lower->more words, 0.004 => 124 words
VECT_MIN_DF = 0.0
VECT_MAX_FEATURES = 530

NUM_OPTS = 10
# DATASET_SIZE = 100
DATASET_SIZE = 5000

# DATASET_SIZE = 20
# NUM_DOCS_TO_SAMPLE = 15000
# NUM_ROWS = 10000


OUTPUT_DS_PATH = (
    # Path(__file__).parent.parent.parent.parent.parent.parent
    # / f"data/up_title_match/up_titles_{NUM_DOCS_TO_SAMPLE}-{DATASET_SIZE}-ukr.csv"
    Path(__file__).parent.parent.parent.parent
    / f"artifacts/up_titles_{NUM_DOCS_TO_SAMPLE}-{DATASET_SIZE}-ukr.csv"
    #  / f"data/up_title_match/up_titles_{NUM_DOCS_TO_SAMPLE}-{DATASET_SIZE}-eng.csv"
)


class UPDatasetBuilder:
    KEY_TITLE = "ukr_title"
    KEY_TEXT = "ukr_text"
    #  KEY_TITLE = "eng_title"
    #  KEY_TEXT = "eng_text"

    KEY_MASKED_TITLE = "masked_title"
    KEY_MASKED_TEXT = "masked_text"

    #  KEY_TAGS = "ukr_tags"
    KEY_TAGS = "tags"

    KEY_LABEL = "label"

    #  KEY_VECTOR_TAGS = "tags_binary"
    KEY_SIM_TITLES = "similar_titles"
    KEY_SIM_IDX = "similar_idx"

    def __init__(
        self,
        num_rows=NUM_ROWS,
        num_docs_to_sample=NUM_DOCS_TO_SAMPLE,
        input_csv=UP_CSV_PATH,
        output_csv=OUTPUT_DS_PATH,
        do_mask_numbers: bool = True,
        num_opts: int = NUM_OPTS,
        dataset_size: int = DATASET_SIZE,
    ):
        # N of rows to read from CSv
        self.num_rows = num_rows

        # N of docs to process from there
        #   >~40k size crashes similarity fn on my computer (TODO)
        self.num_docs_to_sample = num_docs_to_sample

        # Title options for each doc
        self.num_opts = num_opts
        # Total docs w/ titles in final dataset
        # NB titles will come from the larger (num-docs-to-sample) dataset
        self.dataset_size = dataset_size

        # If True, numbers will be replaced with XXX
        self.do_mask_numbers = do_mask_numbers

        self.input_csv = Path(input_csv).absolute()
        self.output_csv = Path(output_csv).absolute()
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        assert self.input_csv.exists()
        logger.info(
            f"Running with {self.num_docs_to_sample} docs on paths {self.input_csv=} {self.output_csv=}"
        )

    @staticmethod
    def parse_up_crawl(
        path: str | Path = UP_CSV_PATH,
        num_rows=NUM_ROWS,
        num_docs_to_sample=NUM_DOCS_TO_SAMPLE,
    ) -> pd.DataFrame:
        path = Path(path).absolute()
        assert path.exists()
        #  df = pd.read_csv(path, usecols=[self.TITLE_KEY, self.TEXT_KEY])
        df = pd.read_csv(
            path,
            usecols=[
                UPDatasetBuilder.KEY_TITLE,
                UPDatasetBuilder.KEY_TEXT,
                UPDatasetBuilder.KEY_TAGS,
            ],
            nrows=num_rows,
        )
        logger.info(f"Read {len(df)} rows, sampled {num_docs_to_sample}")
        df = df.dropna()
        df = df.sample(min(num_docs_to_sample, len(df))).reset_index()
        #  print(df[UPDatasetBuilder.KEY_TAGS].apply(lambda x: x.split(',')).explode().value_counts().to_dict())
        return df

    @staticmethod
    def mask_numbers(s: pd.Series, replace_with: str = "X") -> pd.Series:
        """
        Replace all numbers in Series with replace_with.

        Use-case: remove years, numbers of destroyed tanks and whatever
            to make it harder to match specific articles to specific titles.
        """
        # TODO sometime sth more advanced and spacy-y, incl. esp. w/ last names
        NUMBER_REGEX = r"[0-9]{1}"
        new_s = s.str.replace(NUMBER_REGEX, replace_with, regex=True)
        return new_s

    @staticmethod
    def vectorize_tags(
        #  df,
        #  tags_key: str = "ukr_tags",
        tags_series: pd.Series,
        #  tags_output_key: str = "tags_binary",
        vect_min_df=VECT_MIN_DF,
        vect_max_features=VECT_MAX_FEATURES,
    ) -> tuple[CountVectorizer, np.ndarray]:
        v = CountVectorizer(
            binary=True,
            tokenizer=lambda x: x.lower().split(","),
            stop_words=None,
            min_df=vect_min_df,
            max_features=vect_max_features,
            dtype=np.int8,
        )
        res = v.fit_transform(tags_series)
        logger.info(f"Tokenized shape {res.shape}")
        return v, res
        #  df[tags_output_key] = res.toarray().tolist()
        #  return df, v, res

    def get_sim_matrix(ar: np.ndarray) -> np.ndarray:
        """
        In: array of num_docs*num_tags
        """
        res = cosine_similarity(ar)
        return res

    @staticmethod
    def get_docs_most_similar_to(
        doc_index: int, sim_matrix: np.ndarray, num_docs: int = 10
    ) -> np.ndarray:
        #  return np.argsort(sim_matrix[doc_index])[-num_docs:-1]
        res = np.argsort(sim_matrix[doc_index])[-num_docs:]
        #  b()
        return res

    @staticmethod
    def vectorize_and_similarity(df: pd.DataFrame) -> np.ndarray:
        """
        Vectorizes tags in df tag column, and calculates docs similarity
        based on vectorized tags. Returns pairwise similarity ndarray,

        # and  sets tags into df.
        """
        #  df["tags_p"] = df.ukr_tags.apply(lambda x: x.lower().split(","))
        #  df, vectorizer, sim_array = UPDatasetBuilder.vectorize_tags(
        vectorizer, tags_matrix = UPDatasetBuilder.vectorize_tags(
            tags_series=df[UPDatasetBuilder.KEY_TAGS],
        )
        logger.info(f"Getting pairwise similarities...")
        sim_matrix = UPDatasetBuilder.get_sim_matrix(tags_matrix)
        #  df[UPDatasetBuilder.KEY_VECTOR_TAGS] = list(tags_matrix)
        #  b()
        return sim_matrix

    @staticmethod
    def build_ds(
        df: pd.DataFrame,
        sim_matrix: np.ndarray,
        num_opts_per_doc: int = 10,
        final_ds_size=1000,
        do_mask_numbers: bool = False,
    ) -> pd.DataFrame:
        """
        Goal: article text -> 10 article titles of SIMILAR articles.
        Similarity is measured by number of common tags.

        sim_matrix is a pairwise similarity ndarray of df documents.
        """
        ds = df.iloc[:final_ds_size]
        most_similar = list()

        # Since now pandas doesn't like setting wrong dtypes, we explicitly
        #   create these cols of object dtype to be allowed to put a list there...
        ds.loc[:, UPDatasetBuilder.KEY_SIM_TITLES] = pd.Series(
            index=ds.index, dtype=object
        )
        ds.loc[:, UPDatasetBuilder.KEY_SIM_IDX] = pd.Series(
            index=ds.index, dtype=object
        )

        col_title = UPDatasetBuilder.KEY_TITLE
        col_text = UPDatasetBuilder.KEY_TEXT

        if do_mask_numbers:
            # ds!
            ds[UPDatasetBuilder.KEY_MASKED_TEXT] = UPDatasetBuilder.mask_numbers(
                ds[UPDatasetBuilder.KEY_TEXT]
            )
            ds[UPDatasetBuilder.KEY_MASKED_TITLE] = UPDatasetBuilder.mask_numbers(
                ds[UPDatasetBuilder.KEY_TITLE]
            )
            # df!
            df[UPDatasetBuilder.KEY_MASKED_TITLE] = UPDatasetBuilder.mask_numbers(
                df[UPDatasetBuilder.KEY_TITLE]
            )
            col_title = UPDatasetBuilder.KEY_MASKED_TITLE
            col_text = UPDatasetBuilder.KEY_MASKED_TEXT

        for i, doc in ds.iterrows():
            ms = UPDatasetBuilder.get_docs_most_similar_to(
                doc_index=i, sim_matrix=sim_matrix, num_docs=num_opts_per_doc
            ).tolist()

            # If our article is not among the most similar ones
            #   (e.g. too many articles with similarity==1) we add it.
            if i not in ms:
                ms = [i] + ms[1:]

            docs_titles = df[col_title].loc[ms]
            #  b()

            correct_title = doc[col_title]
            shuffled_docs_titles = list(set(docs_titles))
            label = shuffled_docs_titles.index(correct_title)

            ds.at[i, UPDatasetBuilder.KEY_SIM_IDX] = list(ms)
            ds.at[i, UPDatasetBuilder.KEY_SIM_TITLES] = shuffled_docs_titles
            ds.at[i, UPDatasetBuilder.KEY_LABEL] = label
            #  b()

        UPDatasetBuilder.pp_titles(
            ds,
            col_title=UPDatasetBuilder.KEY_TITLE,
            col_titles=UPDatasetBuilder.KEY_SIM_TITLES,
            num=100,
        )
        #  b()
        return ds

    @staticmethod
    def pp_titles(df, col_title, col_titles, num=100):
        for i, r in df.iterrows():
            if i > num:
                break
            print(f"{r[col_title]} ->")
            for t in r[col_titles]:
                print(f"\t{t}")

    def run(
        self,
    ):
        df = self.parse_up_crawl(
            path=self.input_csv,
            num_rows=self.num_rows,
            num_docs_to_sample=self.num_docs_to_sample,
        )
        sim_matrix = self.vectorize_and_similarity(df=df)
        res = self.build_ds(
            df,
            sim_matrix=sim_matrix,
            num_opts_per_doc=self.num_opts,
            final_ds_size=self.dataset_size,
            do_mask_numbers=self.do_mask_numbers,
        )
        res.to_csv(self.output_csv, index=False)
        #  b()
        return res


def run():
    logging.basicConfig(level=logging.DEBUG)
    logger.setLevel(logging.INFO)
    up = UPDatasetBuilder()
    res = up.run()
    #  b()


if __name__ == "__main__":
    run()
