import tempfile
import shutil
from datareader import pdf_to_text
from typing import Dict
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import sys, fitz
import spacy

import transformers
import spacy_transformers

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Ner Inference Pipeline"}


@app.post("/extract_ner/")
async def classifydoc(uploaded_file: UploadFile = File(...)):
    nlp = spacy.load("./models/model-best")
    textual_content = read_file(uploaded_file)
    # print(textual_content)
    doc = nlp(textual_content)

    final = {}
    temp = []

    for ent in doc.ents:
        if ent.label_ == "PER":
            final['Name'] = ent.text

        if ent.label_ == "EMAIL":
            final['EMAIL'] = ent.text

        if ent.label_ == "PHONE":
            final['Phone'] = ent.text

        if ent.label_ == "HARD SKILLS":
            temp.append(ent.text)
        
    final['Hard Skills'] = temp

    return {"parsed_output": final}


def read_file(FILE):
    try:
        suffix = Path(FILE.filename).suffix
        print(suffix)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(FILE.file, tmp)
            PATH = Path(tmp.name)
        textual_content = pdf_to_text(PATH, dolower=True)
    finally:
        FILE.file.close()
        tmp.close()
    return textual_content