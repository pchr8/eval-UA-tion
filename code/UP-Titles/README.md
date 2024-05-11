# UP-Titles

> [!IMPORTANT]
> The code is buggy, messy, many existing tests fail, many comments are lies, and the typing annotations aren't all correct — it works and worked to generate the dataset, but **here be dragons**. I didn't have time to fix it (still don't, sadly) — it's more useful out there in its imperfect state, "better done than perfect". (I may abstract some useful bits and make them into packages with higher standards later though.)
> 
> PRs and issues welcome.

## Basics
- **Datasets**:
    - [shamotskyi/up_titles_masked](https://huggingface.co/datasets/shamotskyi/up_titles_masked) generated using this code. **Masked**: all digits replaced by `X` characters. ("234 Million Dollars" → "XXX Million Dollars")
        - uses CSV of <https://huggingface.co/datasets/shamotskyi/ukr_pravda_2y> as source
    - [anilev6/up_titles_unmasked](https://huggingface.co/datasets/anilev6/up_titles_unmasked) (sister dataset that uses different code, generated separately based on masked version and the same `ukr_pravda_2y` dataset)

## Directories
- `up_dataset` is the code used to generate `up_titles_masked`
- `./artifacts/` has a small demonstrative dataset 

