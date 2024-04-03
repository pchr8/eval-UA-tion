#  ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def options_to_alpha(strings):
    #  res = [f"{ALPHABET[i]}. '{s}'" for i, s in enumerate(strings)]
    res = [f"{i}) {s}" for i, s in enumerate(strings)]
    return res


def doc_to_text_unmasked(doc):
    """From a list of strings in options create alphabetic thingy."""
    # TODO ugly copying code later clean up
    strings = doc["similar_titles"]
    opts_list = options_to_alpha(strings)
    options_string = "\n".join(opts_list)

    text = doc['ukr_text']
    text = text.replace("\n", " ")

    template = f"{text}\nПИТАННЯ: Який з наведених заголовків найбільше підходить для цієї статті?\n{options_string}\nВІДПОВІДЬ:"
    #  doc_to_text: "{{context.replace('\n',' ')}}\n{{question.replace('\n',' ')}}\n\nПИТАННЯ: Яке слово має бути замість______?\nВІДПОВІДЬ: "
    return template

# For UA-CBT
def doc_to_target_unmasked(doc):
    #  strings = doc["similar_titles"]
    label= doc['label']
    return str(int(label))
    #  return strings[label]

    #  opts_list = options_to_alpha(strings)
    #  answer = doc['ukr_title']
    #  answer_index = strings.index(answer)
    #  answer_letter = ALPHABET[answer_index]
    #  return answer_letter
