""" File contains the logic for parsing epub files and providing the stories 
as plaintext.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Union, Any

import logging

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from dataclasses import dataclass

logging.basicConfig()
logger = logging.getLogger(__package__)


def _normalize_txt(txt_raw: str):
    txt = txt_raw.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return txt


def read_epub(path: Path, min_size_chars: int = 2000) -> tuple[list[tuple[int,str]], dict]:
    """Read .epub file, return a list of large enough chapters as int, str + dict 
    metadata.

    A chapter is an ITEM_DOCUMENT.

    Args:
        path (Path): path to file
        min_size_chars (int): min number of chars in valid ITEM_DOCUMENTs.
    """
    book = epub.read_epub(path)
    items = enumerate(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
    items_txts = [(i, chapter_to_str(it)) for i,it in items]
    tales = [it for it in items_txts if len(it[1]) > min_size_chars]

    md = {'source': path.name, 'filename':str(path)}
    #  tales = [_normalize_txt(it) for it in items_txts]
    return tales, md


def chapter_to_str(
    chapter,
    #  sep: str = "\n",
    sep: str = " ",
) -> str:
    """Convert epub ITEM_DOCUMENTs to plaintext string.

    `sep` is the separator between text in <p> in the epub.

    https://andrew-muller.medium.com/getting-text-from-epub-files-in-python-fbfe5df5c2da
    """
    soup = BeautifulSoup(chapter.get_body_content(), "html.parser")
    text = [para.get_text().strip() for para in soup.find_all("p")]
    return sep.join(text)

