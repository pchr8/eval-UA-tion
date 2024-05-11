## Basics
This directory contains the (messy!) code used to generate and evaluate the tasks.

## Content
- Dataset creation:
    - `./UA-CBT` — Ukrainian Children's Book Test code (creating LLM prompts to generate stories & creating UA-CBT task instances from these stories)
    - `./UP-Titles/` — UP-Titles (masked) code
    - `./LMentry-static-UA/` 
- Evaluation:
    - `./lmeval_tasks/` contains the [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/) YAMLs used for evaluation.
