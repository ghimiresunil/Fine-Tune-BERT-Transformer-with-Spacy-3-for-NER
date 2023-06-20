import sys, fitz
import spacy

import transformers
import spacy_transformers

nlp = spacy.load("./models/model-best")

pdf_file = './testresume/Ali, Mohammad_Taha - 2022-06-23 07-36-29.pdf'

doc = fitz.open(pdf_file)

text = " "
for page in doc:
  text = text + str(page.get_text())

text = text.strip()
text = ' '.join(text.split())
doc = nlp(text)

for ent in doc.ents:
    print(ent.text, "-------------->", ent.label_)