import spacy

def _pretty_output_morph(s: spacy.tokens.Span):
    for t in s:
        if t.is_punct or t.pos_ == "SPACE":
            continue
        # there's also tag_
        print(f"{str(t):<20}:\t{t.pos_:5}\t{t.morph}")
    print("==")



