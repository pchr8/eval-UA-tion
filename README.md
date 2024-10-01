<!-- ![](./images/logos/l1.png)  -->
<p align="center">
  <img src="https://raw.githubusercontent.com/pchr8/eval-UA-tion/main/images/logos/l1.png" alt="Eval-UA-tion logo"/>
</p>

<!-- ![](./images/logos/l2.png)  -->
<!-- ![](./images/logos/l3.png)  -->

# Eval-UA-tion 1.0 

## Intro
This repository contains the scripts used to generate and evaluate the datasets from the Eval-UA-tion 1.0 benchmark for evaluating LLMs in Ukrainian.

It was initially my Master's Thesis (see [/other/MA](/other/MA)), then was accepted to [UNLP 2024 \| The Third Ukrainian Natural Language Processing Workshop](https://unlp.org.ua/) (publication: [Eval-UA-tion 1.0: Benchmark for Evaluating Ukrainian (Large) Language Models - ACL Anthology](https://aclanthology.org/2024.unlp-1.13/)).

## TL; DR
- Benchmark for evaluating LLMs in the Ukrainian language
- 3 novel tasks (9 datasets in total).
- All with human baselines, on (for now...) contamination-free data 
- For more, see
  - Presentation (using [quarto](https://quarto.org/docs/presentations/revealjs/))
    - Thesis defense (longer): [online](https://serhii.net/F/MA/presentation/) / [source](/other/MA/presentation)
    - UNLP (newer+shorter): [online](https://serhii.net/F/eval-ua-tion/presentation/#/) / [source](/other/UNLP/presentation)
  - UNLP video: <https://www.youtube.com/watch?v=O4cWIVPPhR4>

## This repository 
- [/code](/code) contains the (messy) code used to generate and evaluate most of the tasks
- [/other/MA](/other/MA) contains my Master's Thesis and defense presentation (that have more details than the paper)

## License 
- [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) for the code, [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) for the Thesis and presentation unless stated otherwise. 
- The presentation uses the MIT-licensed animation from the README of the excellent [anilev6/HumanResponseBot: a special research project](https://github.com/anilev6/HumanResponseBot).

## Thanks and acknowledgements
These awesome people helped proofread the stories, annotate the datasets and establish human baselines (in alphabetical order):
- Oleksii K.
- Viacheslav Kravchenko
- Daria Kravets
- Anna-Izabella Levbarg
- Lina Mykhailenko
- Mariia Tkachenko
- @arturius453

Special thanks to Anna-Izabella Levbarg ([anilev6](https://github.com/anilev6)), who wrote the [anilev6/HumanResponseBot](https://github.com/anilev6/HumanResponseBot) Telegram bot used for all human baselines and basically made them happen. 


## Citing
ACL Anthology: [Eval-UA-tion 1.0: Benchmark for Evaluating Ukrainian (Large) Language Models - ACL Anthology](https://aclanthology.org/2024.unlp-1.13/)

```bib
@inproceedings{hamotskyi-etal-2024-eval,
    title = "Eval-{UA}-tion 1.0: Benchmark for Evaluating {U}krainian (Large) Language Models",
    author = {Hamotskyi, Serhii  and
      Levbarg, Anna-Izabella  and
      H{\"a}nig, Christian},
    editor = "Romanyshyn, Mariana  and
      Romanyshyn, Nataliia  and
      Hlybovets, Andrii  and
      Ignatenko, Oleksii",
    booktitle = "Proceedings of the Third Ukrainian Natural Language Processing Workshop (UNLP) @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.unlp-1.13",
    pages = "109--119",
    abstract = "In this paper, we introduce Eval-UA-tion, a set of novel Ukrainian-language datasets aimed at evaluating the performance of language models on the Ukrainian language. The tasks include UA-CBT (inspired by the Children{'}s Book Test, a fill-in-the-gaps type task aimed at gauging the extent to which a story narrative is understood), UP-Titles (where the online newspaper \textit{Ukrainska Pravda}{`}s articles have to be matched to the correct title among 10 similar ones), and LMentry-static-UA/LMES (inspired by the LMentry benchmark, a set of tasks simple to solve for humans but hard for LMs, such as {`}which of these words is longer{'} and {`}what is the fifth word of this sentence{'}). With the exception of UP-Titles, the tasks are built in a way to minimize contamination and use material unlikely to be present in the training sets of language models, and include a split for few-shot model prompting use that minimizes contamination. For each task human and random baselines are provided.",
}
```



