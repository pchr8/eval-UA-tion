ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def options_to_alpha(strings):
    res = [f"{ALPHABET[i]}: {s}" for i, s in enumerate(strings)]
    return res


# For UA-CBT
def doc_to_text(doc):
    """From a list of strings in options create alphabetic thingy."""
    strings = doc["options"]
    opts_list = options_to_alpha(strings)
    options_string = "; ".join(opts_list)

    story_text = doc['context']+" "+doc['question']
    story = story_text.replace("\n", " ")

    template = f"{story}\nПИТАННЯ: Яке слово має бути замість ______? {options_string}\nВІДПОВІДЬ:"
    #  doc_to_text: "{{context.replace('\n',' ')}}\n{{question.replace('\n',' ')}}\n\nПИТАННЯ: Яке слово має бути замість______?\nВІДПОВІДЬ: "
    return template


# For UA-CBT
def doc_to_target(doc):
    strings = doc["options"]
    opts_list = options_to_alpha(strings)
    answer = doc['answer']
    answer_index = strings.index(answer)
    answer_letter = ALPHABET[answer_index]
    return answer_letter
