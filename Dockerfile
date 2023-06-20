FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir pymupdf \
    && python -m spacy download en_core_web_sm

COPY api.py clean_text.py datareader.py main.py README.md ./
COPY models ./models

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "80"]
